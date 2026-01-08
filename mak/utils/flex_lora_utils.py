from __future__ import annotations

from typing import Dict, List

import copy
import numpy as np
import torch

from mak.utils.general import _slice_pad_lora_params


def generate_rank_map(config: dict, num_clients: int) -> Dict[int, int]:
    """Generate a deterministic FlexLoRA rank map for all clients.

    The sampling follows config['flex_lora_config']['rank_distribution'].
    Ensures at least one client uses global_rank.
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


def ensure_local_rank_adapters(
    model: torch.nn.Module,
    base_config: dict,
    local_rank: int,
) -> torch.nn.Module:
    """Ensure the given model is adapted with LoRA/SVD adapters of rank==local_rank.

    Clean & safe approach for FlexLoRA:
    - If the model already has adapter factors (.A/.B), we DO NOT attempt to re-run
      apply_svd_to_model(), because that function only adapts torch.nn.Linear layers
      and will return 0 layers once the model has been wrapped by SVDAdapter.
    - Instead, we deterministically slice/pad all existing .A/.B tensors to the
      requested local_rank and load them back.
    - Only if the model has no adapters at all do we fall back to apply_svd_to_model.

    This avoids size-mismatch during round>1 partial updates and keeps other
    baselines untouched.
    """
    desired = int(local_rank)

    sd = model.state_dict()
    a_keys = [k for k in sd.keys() if k.endswith(".A")]
    b_keys = [k for k in sd.keys() if k.endswith(".B")]

    # Case 1) Model already has adapters -> normalize rank by slicing/padding in-place
    if a_keys or b_keys:
        # Fast path: already all OK
        try:
            all_ok = True
            for k in a_keys:
                v = sd[k]
                if isinstance(v, torch.Tensor) and v.dim() == 2 and int(v.shape[1]) != desired:
                    all_ok = False
                    break
            if all_ok:
                for k in b_keys:
                    v = sd[k]
                    if isinstance(v, torch.Tensor) and v.dim() == 2 and int(v.shape[0]) != desired:
                        all_ok = False
                        break
            if all_ok:
                return model
        except Exception:
            # If inspection fails, fall through to safe conversion
            pass

        # Convert all A/B to desired rank
        update = {}
        for k in a_keys:
            v = sd[k]
            if isinstance(v, torch.Tensor):
                update[k] = _slice_pad_lora_params(v, target_rank=desired, param_type="A")
        for k in b_keys:
            v = sd[k]
            if isinstance(v, torch.Tensor):
                update[k] = _slice_pad_lora_params(v, target_rank=desired, param_type="B")

        sd.update(update)
        model.load_state_dict(sd, strict=False)
        return model

    # Case 2) No adapters exist yet -> apply adapter injection
    # Lazy import to avoid circular dependency
    from mak.utils.helper import apply_svd_to_model

    cfg = copy.deepcopy(base_config)
    cfg.setdefault("peft", {})["rank"] = desired

    # Important: apply_svd_to_model mutates the model in-place
    return apply_svd_to_model(model=model, config=cfg)


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
