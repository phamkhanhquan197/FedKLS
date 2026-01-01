from __future__ import annotations

from logging import INFO
from typing import Any, List

import numpy as np
import torch
from flwr.common.logger import log

from mak.clients.base_client import BaseClient
from mak.utils.helper import get_ffa_target_keys


class FFALoRAClient(BaseClient):
    """FFA-LoRA Client (Phase 1) - deterministic name-based mapping.

    Standards:
    - Communication is based on model.state_dict().
    - Client uplink ALWAYS returns PARTIAL tensors using get_ffa_target_keys(model).
    - Round 1 downlink is FULL state_dict values (handled via BaseClient super().set_parameters).
    - Round >1 downlink is PARTIAL tensors aligned with the same sorted keys.
    """

    def __repr__(self) -> str:
        return "FFA-LoRA client"

    def get_parameters(self, config: Any | None = None) -> List[np.ndarray]:
        """Always return PARTIAL parameters (deterministic order)."""
        keys = get_ffa_target_keys(self.model)
        sd = self.model.state_dict()
        return [sd[k].detach().cpu().numpy() for k in keys]

    def set_parameters(self, parameters: List[np.ndarray], config: Any | None = None) -> None:
        """Round 1: load FULL. Round >1: inject PARTIAL by sorted keys."""
        if parameters is None:
            return

        sd = self.model.state_dict()
        full_len = len(sd)
        incoming_len = len(parameters)

        # Round 1 (FULL): use BaseClient loading
        if incoming_len == full_len:
            super().set_parameters(parameters)

            # Safety lock: freeze all A matrices
            frozen = 0
            for name, p in self.model.named_parameters():
                if ("lora_A" in name) or (".A" in name):
                    if p.requires_grad:
                        frozen += 1
                    p.requires_grad = False

            log(
                INFO,
                f"Client {self.client_id}: Loaded FULL state_dict (len={incoming_len}). "
                f"Safety-lock applied (froze {frozen} A-params if they were trainable).",
            )
            return

        # Round > 1 (PARTIAL)
        keys = get_ffa_target_keys(self.model)
        if incoming_len != len(keys):
            raise ValueError(
                f"Client {self.client_id}: partial length mismatch. expected={len(keys)} got={incoming_len}"
            )

        with torch.no_grad():
            for k, v in zip(keys, parameters):
                t = torch.from_numpy(np.asarray(v)).to(device=sd[k].device, dtype=sd[k].dtype)
                if sd[k].shape != t.shape:
                    raise RuntimeError(
                        f"Client {self.client_id}: tensor shape mismatch for key '{k}': "
                        f"local={tuple(sd[k].shape)} incoming={tuple(t.shape)}"
                    )
                sd[k].copy_(t)

        log(INFO, f"Client {self.client_id}: Injected PARTIAL parameters (len={incoming_len}).")
