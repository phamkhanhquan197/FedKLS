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


def get_global_rank(config: dict) -> int:
    flex_cfg = config.get("flex_lora_config", {})
    return int(flex_cfg.get("global_rank", config.get("peft", {}).get("rank", 32)))


def build_client_type_map(config: dict, num_clients: int) -> ClientTypeMap:
    """Build deterministic client_id -> type_id map (Type 1..4).

    Supports:
    - flex_lora_config.client_type_map: explicit override mapping
    - flex_lora_config.client_type_distribution: list of {type, ratio}

    Notes:
    - This repo uses small smoke tests; determinism matters more than perfect sampling.
    - If no distribution is provided, defaults to Uniform over 4 types.
    """
    flex_cfg = config.get("flex_lora_config", {})
    seed = int(flex_cfg.get("seed", config.get("common", {}).get("seed", 42)))

    # Explicit override for smoke tests
    override = flex_cfg.get("client_type_map", None)
    if isinstance(override, dict) and override:
        out: ClientTypeMap = {}
        for k, v in override.items():
            out[int(k)] = int(v)
        # Fill missing cids with Type 1
        for cid in range(int(num_clients)):
            out.setdefault(int(cid), 1)
        return out

    dist = flex_cfg.get("client_type_distribution", None)
    if not dist:
        dist = [
            {"type": 1, "ratio": 0.25},
            {"type": 2, "ratio": 0.25},
            {"type": 3, "ratio": 0.25},
            {"type": 4, "ratio": 0.25},
        ]

    rng = np.random.default_rng(seed)
    types = [int(item["type"]) for item in dist]
    probs = np.asarray([float(item.get("ratio", 0.0)) for item in dist], dtype=np.float64)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)

    sampled = rng.choice(np.asarray(types), size=int(num_clients), replace=True, p=probs)


    return {i: int(sampled[i]) for i in range(int(num_clients))}


def build_client_rank_policy_map(config: dict, client_type_map: ClientTypeMap) -> ClientRankPolicyMap:
    """Build client_id -> rank policy map aligned with paper Table 1.

    global_rank is assumed to be max rank (e.g., 200).
    local ranks must satisfy local <= global.
    """
    flex_cfg = config.get("flex_lora_config", {})
    global_rank = int(flex_cfg.get("global_rank", 200))

    # Defaults per paper Table 1
    type1_r = int(flex_cfg.get("type1_rank", 8))
    type2_r = int(flex_cfg.get("type2_rank", 30))
    type3_attn_r = int(flex_cfg.get("type3_attn_rank", 30))
    type3_ffn_r = int(flex_cfg.get("type3_ffn_rank", 200))
    type4_r = int(flex_cfg.get("type4_rank", 200))

    def _check(r: int, cid: int) -> int:
        if r > global_rank:
            raise ValueError(
                f"FlexLoRA: local rank must be <= global_rank. cid={cid} rank={r} global_rank={global_rank}"
            )
        return int(r)

    out: ClientRankPolicyMap = {}
    for cid, t in client_type_map.items():
        tid = int(t)
        if tid == 1:
            out[int(cid)] = {"all": _check(type1_r, cid)}
        elif tid == 2:
            out[int(cid)] = {"all": _check(type2_r, cid)}
        elif tid == 3:
            out[int(cid)] = {
                "attn": _check(type3_attn_r, cid),
                "ffn": _check(type3_ffn_r, cid),
            }
        elif tid == 4:
            out[int(cid)] = {"all": _check(type4_r, cid)}
        else:
            raise ValueError(f"FlexLoRA: unknown client type: {tid} (cid={cid})")

    return out


def get_rank_for_base(rank_policy: RankPolicy, base: str) -> int:
    group = resolve_layer_group(base)
    if group in rank_policy:
        return int(rank_policy[group])
    if "all" in rank_policy:
        return int(rank_policy["all"])
    # fallback: choose max provided
    return int(max(rank_policy.values()))


def generate_rank_map(config: dict, num_clients: int) -> Dict[int, int]:
    """Backward-compatible: generate single-rank map.

    Kept for older configs; new implementation uses client types and rank policies.
    """
    flex_cfg = config.get("flex_lora_config", {})
    dist = flex_cfg.get("rank_distribution", [])
    global_rank = int(flex_cfg.get("global_rank", config.get("peft", {}).get("rank", 32)))
    seed = int(flex_cfg.get("seed", config.get("common", {}).get("seed", 42)))

    if not dist:
        return {i: global_rank for i in range(int(num_clients))}

    rng = np.random.default_rng(seed)

    ranks = [int(item["rank"]) for item in dist]
    probs = np.asarray([float(item["ratio"]) for item in dist], dtype=np.float64)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)

    sampled = rng.choice(np.asarray(ranks), size=int(num_clients), replace=True, p=probs)
    sampled[0] = global_rank

    return {i: int(sampled[i]) for i in range(int(num_clients))}


def setup_server_config(config: dict, global_rank: int) -> dict:
    """Create a server-side config copy with peft.rank overridden to global_rank."""
    server_cfg = copy.deepcopy(config)
    server_cfg.setdefault("peft", {})["rank"] = int(global_rank)
    return server_cfg


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


def _inject_lora_adapters_no_svd(
    model: torch.nn.Module,
    base_config: dict,
    local_rank: int,
) -> torch.nn.Module:
    """Inject LoRA/SVDAdapter modules WITHOUT running SVD (client-side safe).

    Policy:
    - A: small Gaussian init
    - B: zeros
    - W_res: original linear weight (frozen)

    This is used to create a local-rank architecture on the client side while
    respecting the constraint: "SVD must happen on the server, not on clients".

    NOTE: This injection path only works if the model still contains nn.Linear.
    If the model was already adapted (nn.Linear -> SVDAdapter), use the
    SVDAdapter rebuild path (_rebuild_svd_adapters_no_svd).
    """
    from mak.models.svd_model import SVDAdapter

    rank = int(local_rank)
    alpha = float(base_config.get("peft", {}).get("alpha", rank))

    # Deterministic init if seed is set
    seed = base_config.get("common", {}).get("seed", None)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    # Replace Linear layers similarly to apply_svd_to_model.extract_linear_layers
    def should_skip(name: str) -> bool:
        return name in ["pre_classifier", "classifier", "model.norm", "score"]

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if should_skip(name):
            continue

        d_out, d_in = module.weight.data.shape
        device = module.weight.data.device
        dtype = module.weight.data.dtype

        A = (torch.randn(d_out, rank, device=device, dtype=dtype) * 0.01)
        B = torch.zeros(rank, d_in, device=device, dtype=dtype)

        W_res = module.weight.data.clone().detach()
        original_bias = module.bias.data.clone().detach() if module.bias is not None else None

        new_layer = SVDAdapter(W_res=W_res, A=A, B=B, alpha=alpha, rank=rank, original_bias=original_bias)

        # Replace module in parent
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, new_layer)
        replaced += 1

    log(INFO, f"[FlexLoRA] Injected {replaced} LoRA adapters without SVD (rank={rank}).")
    return model


def _rebuild_svd_adapters_no_svd(
    model: torch.nn.Module,
    base_config: dict,
    local_rank: int,
) -> tuple[torch.nn.Module, int]:
    """Rebuild SVDAdapter modules to `local_rank` without SVD (client-safe)."""
    from mak.models.svd_model import SVDAdapter

    rank = int(local_rank)
    alpha = float(base_config.get("peft", {}).get("alpha", rank))

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

        # W_res is a buffer tensor with shape [d_out, d_in]
        W_res = module.W_res
        d_out, d_in = int(W_res.shape[0]), int(W_res.shape[1])

        device = W_res.device
        dtype = W_res.dtype

        A = (torch.randn(d_out, rank, device=device, dtype=dtype) * 0.01)
        B = torch.zeros(rank, d_in, device=device, dtype=dtype)

        # Preserve bias values if present
        if getattr(module, "bias", None) is not None:
            original_bias = module.bias.detach().clone()
        else:
            original_bias = None

        new_layer = SVDAdapter(
            W_res=W_res.detach().clone(),
            A=A,
            B=B,
            alpha=alpha,
            rank=rank,
            original_bias=original_bias,
        )

        # Replace module in parent
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            # Top-level module edge-case
            parent = model
            child_name = name

        setattr(parent, child_name, new_layer)
        rebuilt += 1

    return model, rebuilt


def _rebuild_svd_adapters_no_svd_policy(
    model: torch.nn.Module,
    base_config: dict,
    rank_policy: RankPolicy,
    rank_for_base: Optional[Callable[[RankPolicy, str], int]] = None,
) -> tuple[torch.nn.Module, int]:
    """Rebuild SVDAdapter modules per-layer using a rank policy (client-safe, no SVD)."""
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

        A = (torch.randn(d_out, desired, device=device, dtype=dtype) * 0.01)
        B = torch.zeros(desired, d_in, device=device, dtype=dtype)

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

            log(
                INFO,
                f"[FlexLoRA] Adapter rank-policy mismatch detected. A_bad={a_bad}/{len(a_keys)} "
                f"B_bad={b_bad}/{len(b_keys)} -> rebuilding adapters (no SVD).",
            )
        except Exception as e:
            log(INFO, f"[FlexLoRA] Adapter inspection failed ({e}); rebuilding adapters (no SVD).")

        # First try rebuilding SVDAdapter modules
        model, rebuilt = _rebuild_svd_adapters_no_svd_policy(
            model=model,
            base_config=base_config,
            rank_policy=rank_policy,
            rank_for_base=rank_for_base,
        )
        log(INFO, f"[FlexLoRA] Rebuilt {rebuilt} SVDAdapter modules without SVD (rank_policy).")
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
    from mak.utils.helper import get_ffa_target_keys

    dev = torch.device(device) if isinstance(device, str) else device

    model_state = model.state_dict()

    # FULL payload
    if len(parameters) == len(model_state):
        full_sd = {}
        for k, arr in zip(model_state.keys(), parameters):
            full_sd[k] = torch.from_numpy(np.asarray(arr)).to(device=dev)
        model.load_state_dict(full_sd, strict=False)
        return

    # PARTIAL payload
    target_keys = get_ffa_target_keys(model)
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
