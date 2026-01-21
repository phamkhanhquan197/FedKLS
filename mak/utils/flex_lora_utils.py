from __future__ import annotations

from typing import Callable, Dict, List, Optional

import copy
import numpy as np
import torch

from flwr.common.logger import log
from logging import INFO

from mak.utils.general import _slice_pad_lora_params


RankPolicy = Dict[str, int]
ClientRankPolicyMap = Dict[int, RankPolicy]
ClientTypeMap = Dict[int, int]


def resolve_layer_group(base: str) -> str:
    """Heuristic layer-group resolver for per-layer rank policies.

    Groups (paper Type 3 support):
    - attn: attention/self-attn layers
    - ffn:  feed-forward / MLP layers
    - all:  fallback
    """
    b = base.lower()
    if "attention" in b or "self_attn" in b:
        return "attn"
    if "ffn" in b or "mlp" in b or ".lin1" in b or ".lin2" in b or "lin1" in b or "lin2" in b:
        return "ffn"
    return "all"

# ---------------------------------------------------------------------
# Client type sampling (Figure-3 distributions only)
# ---------------------------------------------------------------------
def build_client_type_map(config: dict, num_clients: int) -> ClientTypeMap:
    """Build client_id -> LoRA type map (Type 1..4), paper-aligned.

    Supported distributions (Figure 3):
      - uniform
      - heavy_tail_light
      - heavy_tail_strong
      - normal

    Optional:
      - flex_lora_config.client_type_map for deterministic override
    """
    flex_cfg = config.get("flex_lora_config", {})
    seed = config.get("common", {}).get("seed", 42)
    rng = np.random.default_rng(seed)

    dist_name = flex_cfg.get("distribution", "uniform").lower()

    if dist_name == "uniform":
        types = [1, 2, 3, 4]
        probs = [0.25, 0.25, 0.25, 0.25]

    elif dist_name == "heavy_tail_light":
        # Figure 3: many low-resource clients
        types = [1, 2, 3, 4]
        probs = [0.80, 0.10, 0.05, 0.05]

    elif dist_name == "heavy_tail_strong":
        # Figure 3: many high-resource clients
        types = [1, 2, 3, 4]
        probs = [0.10, 0.05, 0.05, 0.80]

    elif dist_name == "normal":
        positions = np.array([1, 2, 3, 4])
        mu, sigma = 2.5, 0.7
        probs = np.exp(-0.5 * ((positions - mu) / sigma) ** 2)
        probs = probs / probs.sum()
        types = positions.tolist()

    else:
        raise ValueError(f"Unknown FlexLoRA distribution: {dist_name}")

    sampled = rng.choice(types, size=int(num_clients), replace=True, p=probs)

    return {cid: int(sampled[cid]) for cid in range(int(num_clients))}


def build_client_rank_policy_map(config: dict, client_type_map: ClientTypeMap) -> ClientRankPolicyMap:
    """Build client_id -> per-layer rank policy map (Table 1).
    """
    flex_cfg = config.get("flex_lora_config", {})

    # Defaults per paper Table 1
    type1_r = int(flex_cfg.get("type1_rank", 8))
    type2_r = int(flex_cfg.get("type2_rank", 30))
    type3_attn_r = int(flex_cfg.get("type3_attn_rank", 30))
    type3_ffn_r = int(flex_cfg.get("type3_ffn_rank", 200))
    type4_r = int(flex_cfg.get("type4_rank", 200))

    out: ClientRankPolicyMap = {}
    for cid, t in client_type_map.items():
        if t == 1:
            out[cid] = {"all": type1_r}
        elif t == 2:
            out[cid] = {"all": type2_r}
        elif t == 3:
            out[cid] = {
                "attn": type3_attn_r,
                "ffn": type3_ffn_r,
            }
        elif t == 4:
            out[cid] = {"all": type4_r}
        else:
            raise ValueError(f"Unknown client type {t} (cid={cid})")

    return out


def get_rank_for_base(rank_policy: RankPolicy, base: str) -> int:
    group = resolve_layer_group(base)
    if group in rank_policy:
        return int(rank_policy[group])
    if "all" in rank_policy:
        return int(rank_policy["all"])
    # fallback: choose max provided
    return int(max(rank_policy.values()))


def _slice_pad_lora_factor(t: torch.Tensor, local_rank: int, is_A: bool) -> torch.Tensor:
    """Slice/pad a LoRA factor to local_rank.

    A: [out, r] -> slice/pad columns
    B: [r, in]  -> slice/pad rows
    """
    if t.dim() != 2:
        return t

    if is_A:
        _, r = t.shape
        if r > local_rank:
            return t[:, :local_rank]
        if r < local_rank:
            pad_cols = local_rank - r
            return torch.nn.functional.pad(t, (0, pad_cols, 0, 0), mode="constant", value=0.0)
        return t

    r, _ = t.shape
    if r > local_rank:
        return t[:local_rank, :]
    if r < local_rank:
        pad_rows = local_rank - r
        return torch.nn.functional.pad(t, (0, 0, 0, pad_rows), mode="constant", value=0.0)
    return t


def _rebuild_svd_adapters_no_svd_policy(
    model: torch.nn.Module,
    base_config: dict,
    rank_policy: RankPolicy,
    rank_for_base: Optional[Callable[[RankPolicy, str], int]] = None,
) -> tuple[torch.nn.Module, int]:
    """Rebuild SVDAdapter modules per-layer using a rank policy (client-safe, no SVD).

    P1 mitigation (Step 1): preserve existing adapter state by projecting A/B to the
    desired rank (truncate or zero-pad) instead of re-initializing.
    """
    from mak.models.svd_model import SVDAdapter

    rank_for_base = rank_for_base or get_rank_for_base

    alpha_default = base_config.get("peft", {}).get("alpha", None)

    # Deterministic init if seed is set
    seed = base_config.get("common", {}).get("seed", None)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    rebuilt = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, SVDAdapter):
            continue

        # Determine desired rank for this adapter based on its base name
        # e.g. "distilbert.transformer.layer.0.attention.q_lin"
        base = name
        desired = int(rank_for_base(rank_policy, base))

        W_res = module.W_res
        d_out, d_in = int(W_res.shape[0]), int(W_res.shape[1])

        device = W_res.device
        dtype = W_res.dtype

        # --- P1 FIX: Project existing A/B instead of re-initializing ---
        A_old = module.A.clone().detach()
        B_old = module.B.clone().detach()

        A = _slice_pad_lora_factor(A_old, local_rank=desired, is_A=True)
        B = _slice_pad_lora_factor(B_old, local_rank=desired, is_A=False)
        # --- END P1 FIX ---

        # Preserve bias values if present
        if getattr(module, "bias", None) is not None:
            original_bias = module.bias.detach().clone()
        else:
            original_bias = None

        alpha = float(alpha_default) if alpha_default is not None else float(desired)

        new_layer = SVDAdapter(
            W_res=W_res.detach().clone(),
            A=A,
            B=B,
            alpha=alpha,
            rank=desired,
            original_bias=original_bias,
        )

        # Replace module in parent
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name

        setattr(parent, child_name, new_layer)
        rebuilt += 1

    return model, rebuilt


def ensure_local_rank_adapters(
    model: torch.nn.Module,
    base_config: dict,
    rank_policy: RankPolicy,
    rank_for_base: Optional[Callable[[RankPolicy, str], int]] = None,
) -> torch.nn.Module:
    """Ensure model adapter architecture matches the given rank policy (no client SVD)."""
    rank_for_base = rank_for_base or get_rank_for_base

    sd = model.state_dict()
    a_keys = [k for k in sd.keys() if k.endswith(".A") and isinstance(sd[k], torch.Tensor) and sd[k].dim() == 2]
    b_keys = [k for k in sd.keys() if k.endswith(".B") and isinstance(sd[k], torch.Tensor) and sd[k].dim() == 2]

    if a_keys or b_keys:
        # Check if all ranks match desired policy
        try:
            a_bad = 0
            b_bad = 0
            for k in a_keys:
                base = k[:-2]
                desired = int(rank_for_base(rank_policy, base))
                if int(sd[k].shape[1]) != desired:
                    a_bad += 1
            for k in b_keys:
                base = k[:-2]
                desired = int(rank_for_base(rank_policy, base))
                if int(sd[k].shape[0]) != desired:
                    b_bad += 1

            if a_bad == 0 and b_bad == 0:
                return model

            # log(
            #     INFO,
            #     f"[FlexLoRA] Adapter rank-policy mismatch detected. A_bad={a_bad}/{len(a_keys)} "
            #     f"B_bad={b_bad}/{len(b_keys)} -> rebuilding adapters (no SVD).",
            # )
        except Exception as e:
            log(INFO, f"[FlexLoRA] Adapter inspection failed ({e}); rebuilding adapters (no SVD).")

        # First try rebuilding SVDAdapter modules
        model, rebuilt = _rebuild_svd_adapters_no_svd_policy(
            model=model,
            base_config=base_config,
            rank_policy=rank_policy,
            rank_for_base=rank_for_base,
        )
        # log(INFO, f"[FlexLoRA] Rebuilt {rebuilt} SVDAdapter modules without SVD (rank_policy).")
        if rebuilt > 0:
            return model

        # Fallback: inject from Linear is not supported for per-layer rank policy in this baseline
        raise ValueError("[FlexLoRA] No SVDAdapter modules found to rebuild under rank policy")

    # No adapters found at all
    raise ValueError("[FlexLoRA] No adapters found; expected server-adapted model with SVDAdapters")


def slice_and_load_params(
    model: torch.nn.Module,
    params: List[np.ndarray],
    rank_policy: RankPolicy,
    device: str | torch.device,
    rank_for_base: Optional[Callable[[RankPolicy, str], int]] = None,
) -> None:
    """Load FULL downlink parameters into a per-layer-policy FlexLoRA model.

    The server sends a full state_dict corresponding to global_rank.
    Clients slice/pad A/B factors per adapter base name using `rank_policy`.

    This function assumes the model already has policy-shaped adapters.
    """
    rank_for_base = rank_for_base or get_rank_for_base

    model_state = model.state_dict()
    if len(params) != len(model_state):
        raise ValueError(
            f"slice_and_load_params expects FULL state_dict payload. got={len(params)} expected={len(model_state)}"
        )

    dev = torch.device(device) if isinstance(device, str) else device

    state_dict = {}
    for (k, _), arr in zip(model_state.items(), params):
        t = torch.from_numpy(np.asarray(arr)).to(device=dev)

        if k.endswith(".A"):
            base = k[:-2]
            desired = int(rank_for_base(rank_policy, base))
            t = _slice_pad_lora_factor(t, local_rank=desired, is_A=True)
        elif k.endswith(".B"):
            base = k[:-2]
            desired = int(rank_for_base(rank_policy, base))
            t = _slice_pad_lora_factor(t, local_rank=desired, is_A=False)

        state_dict[k] = t

    model.load_state_dict(state_dict, strict=False)


def load_server_eval_params_flex_lora(
    model: torch.nn.Module,
    parameters: List[np.ndarray],
    device: str | torch.device,
) -> None:
    """Load parameters for SERVER-side centralized evaluation for FlexLoRA."""
    # Lazy import to avoid circular dependency
    from mak.utils.helper import get_target_keys, get_config, parse_args

    dev = torch.device(device) if isinstance(device, str) else device
    config = get_config(parse_args().config)
    model_state = model.state_dict()

    # FULL payload
    if len(parameters) == len(model_state):
        full_sd = {}
        for k, arr in zip(model_state.keys(), parameters):
            full_sd[k] = torch.from_numpy(np.asarray(arr)).to(device=dev)
        model.load_state_dict(full_sd, strict=False)
        return

    # PARTIAL payload
    target_keys = get_target_keys(model, bias=config.get("peft", {}).get("bias", True))
    if len(parameters) != len(target_keys):
        raise ValueError(
            f"FlexLoRA server eval payload length mismatch: expected {len(target_keys)} got {len(parameters)}"
        )

    # Build partial update dict
    update = {}
    for k, arr in zip(target_keys, parameters):
        t = torch.from_numpy(np.asarray(arr)).to(device=dev)

        # Defensive: ensure A/B shapes match current global-rank model
        if k.endswith(".A") or k.endswith(".B"):
            if k.endswith(".A"):
                t = _slice_pad_lora_params(t, target_rank=int(t.shape[1]), param_type="A")
            else:
                t = _slice_pad_lora_params(t, target_rank=int(t.shape[0]), param_type="B")

        update[k] = t

    model_state.update(update)
    model.load_state_dict(model_state, strict=True)

def log_flexlora_assignment(
    client_type_map: dict[int, int],
    rank_policy_map: dict[int, dict],
    config: dict,
) -> None:
    from collections import Counter
    from flwr.common.logger import log
    from logging import INFO

    flex_cfg = config.get("flex_lora_config", {})

    # ----- Type descriptions -----
    type_desc = {
        1: f"Type-1: r={flex_cfg.get('type1_rank', 8)} on all layers",
        2: f"Type-2: r={flex_cfg.get('type2_rank', 30)} on all layers",
        3: (
            f"Type-3: r={flex_cfg.get('type3_attn_rank', 30)} on attention, "
            f"r={flex_cfg.get('type3_ffn_rank', 200)} on FFN"
        ),
        4: f"Type-4: r={flex_cfg.get('type4_rank', 200)} on all layers (server-equivalent)",
    }

    log(INFO, "========== FlexLoRA Client Type Definitions ==========")
    for k in sorted(type_desc):
        log(INFO, type_desc[k])

    # ----- Distribution summary -----
    counts = Counter(client_type_map.values())
    log(INFO, "========== FlexLoRA Client Type Distribution ==========")
    log(INFO, f"Distribution: {flex_cfg.get('distribution', 'uniform')}, Global rank: {flex_cfg.get('global_rank', 'N/A')}")
    for t in sorted(type_desc):
        log(INFO, f"Type-{t}: {counts.get(t, 0)} clients")

    # ----- Per-client assignment -----
    log(INFO, "========== FlexLoRA Client Assignments ==========")
    for cid in sorted(client_type_map):
        t = client_type_map[cid]
        policy = rank_policy_map[cid]

        if "all" in policy:
            policy_str = f"r={policy['all']} (all layers)"
        else:
            policy_str = ", ".join(
                f"{k}=r{v}" for k, v in policy.items()
            )

        log(INFO, f"Client {cid}: Type-{t} | {policy_str}")
