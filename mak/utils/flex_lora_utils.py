from __future__ import annotations

from typing import Dict, List

import copy
import numpy as np
import torch


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
        # Fallback: everyone uses global rank
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
        out, r = t.shape
        if r > local_rank:
            return t[:, :local_rank]
        if r < local_rank:
            pad_cols = local_rank - r
            return torch.nn.functional.pad(t, (0, pad_cols, 0, 0), mode="constant", value=0.0)
        return t

    # B
    r, inn = t.shape
    if r > local_rank:
        return t[:local_rank, :]
    if r < local_rank:
        pad_rows = local_rank - r
        return torch.nn.functional.pad(t, (0, 0, 0, pad_rows), mode="constant", value=0.0)
    return t


def slice_and_load_params(
    model: torch.nn.Module,
    params: List[np.ndarray],
    local_rank: int,
    device: str | torch.device,
) -> None:
    """Load FULL downlink parameters into a local-rank FlexLoRA model.

    The server sends a full state_dict corresponding to global_rank.
    Clients with local_rank < global_rank must slice A/B factors before loading.

    This function:
    1) Builds an OrderedDict from model.state_dict() keys aligned with params.
    2) For keys ending with .A or .B: slice/pad to local_rank.
    3) Loads with strict=False.
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
