from __future__ import annotations

from typing import Dict, List

import copy
import numpy as np
import torch

from flwr.common.logger import log
from logging import INFO

from mak.utils.general import _slice_pad_lora_params


def generate_rank_map(config: dict, num_clients: int) -> Dict[int, int]:
    """Generate deterministic client_id -> rank mapping."""
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


def ensure_local_rank_adapters(
    model: torch.nn.Module,
    base_config: dict,
    local_rank: int,
) -> torch.nn.Module:
    """Ensure model adapter architecture matches `local_rank` (no client SVD)."""
    desired = int(local_rank)

    sd = model.state_dict()
    a_keys = [k for k in sd.keys() if k.endswith(".A") and isinstance(sd[k], torch.Tensor) and sd[k].dim() == 2]
    b_keys = [k for k in sd.keys() if k.endswith(".B") and isinstance(sd[k], torch.Tensor) and sd[k].dim() == 2]

    if a_keys or b_keys:
        # Check if all ranks match
        try:
            a_bad = [k for k in a_keys if int(sd[k].shape[1]) != desired]
            b_bad = [k for k in b_keys if int(sd[k].shape[0]) != desired]
            if not a_bad and not b_bad:
                return model
            log(
                INFO,
                f"[FlexLoRA] Adapter rank mismatch detected. desired={desired} "
                f"A_bad={len(a_bad)}/{len(a_keys)} B_bad={len(b_bad)}/{len(b_keys)} -> rebuilding adapters (no SVD).",
            )
        except Exception as e:
            log(INFO, f"[FlexLoRA] Adapter inspection failed ({e}); rebuilding adapters (no SVD).")

        # First try rebuilding SVDAdapter modules (the common case when server has
        # already adapted the model).
        model, rebuilt = _rebuild_svd_adapters_no_svd(
            model=model,
            base_config=base_config,
            local_rank=desired,
        )
        log(INFO, f"[FlexLoRA] Rebuilt {rebuilt} SVDAdapter modules without SVD (rank={desired}).")
        if rebuilt > 0:
            return model

        # Fallback: if no SVDAdapter modules were found/rebuilt, try injecting from Linear.
        log(INFO, f"[FlexLoRA] No SVDAdapter modules found to rebuild; falling back to Linear injection (no SVD).")
        return _inject_lora_adapters_no_svd(model=model, base_config=base_config, local_rank=desired)

    # No adapters found at all
    log(INFO, f"[FlexLoRA] No adapters found; injecting adapters (no SVD) rank={desired}.")
    return _inject_lora_adapters_no_svd(model=model, base_config=base_config, local_rank=desired)


def slice_and_load_params(
    model: torch.nn.Module,
    params: List[np.ndarray],
    local_rank: int,
    device: str | torch.device,
) -> None:
    """Load FULL downlink parameters into a local-rank FlexLoRA model.

    The server sends a full state_dict corresponding to global_rank.
    Clients with local_rank < global_rank must slice A/B factors before loading.

    This function assumes the model already has local-rank adapters.
    """
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
            t = _slice_pad_lora_factor(t, local_rank=int(local_rank), is_A=True)
        elif k.endswith(".B"):
            t = _slice_pad_lora_factor(t, local_rank=int(local_rank), is_A=False)

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
