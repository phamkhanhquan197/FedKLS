from __future__ import annotations

from logging import INFO
from typing import Any, List

import numpy as np
import torch
from flwr.common.logger import log

from mak.clients.base_client import BaseClient


class FFALoRAClient(BaseClient):
    """FFA-LoRA client (Phase 1 refactor).

    Rules:
    - Inherit from BaseClient (no duplicated training loop).
    - Source of truth: requires_grad.
      * A must be frozen forever ("lora_A" or ".A" parameters).
      * Only trainable params are communicated after round 1.

    Protocol:
    - Round 1 downlink: FULL model parameters (all parameters).
    - Round >1 downlink: PARTIAL parameters (trainable-only).

    Note: We keep the logic robust by using a length-based handshake and
    a safety lock that forces A frozen immediately after any load.
    """

    def __repr__(self) -> str:
        return "FFA-LoRA client"

    def set_parameters(self, parameters: List[np.ndarray], config: Any | None = None) -> None:
        """Set parameters with Round-1 full-load vs later trainable-only injection.

        Round detection (as requested):
        - Round 1 iff len(parameters) == len(list(self.model.parameters()))

        Safety lock:
        - After loading, force any parameter with name containing "lora_A" or ".A"
          to have requires_grad=False.
        """
        if parameters is None:
            return

        incoming_len = len(parameters)
        total_param_len = len(list(self.model.parameters()))

        # Round 1: FULL parameters
        if incoming_len == total_param_len:
            super().set_parameters(parameters)

            # CRITICAL SAFETY LOCK: force-freeze A
            frozen = 0
            for name, param in self.model.named_parameters():
                if ("lora_A" in name) or (".A" in name):
                    if param.requires_grad:
                        frozen += 1
                    param.requires_grad = False

            log(
                INFO,
                f"Client {self.client_id}: Loaded FULL parameters (len={incoming_len}). "
                f"Safety-lock applied (froze {frozen} A-params if they were trainable).",
            )
            return

        # Round >1: PARTIAL parameters (trainable-only)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        expected = len(trainable_params)

        if incoming_len != expected:
            raise ValueError(
                f"Client {self.client_id}: size mismatch in partial update. "
                f"Client expects {expected} trainable tensors, Server sent {incoming_len}."
            )

        with torch.no_grad():
            for local_p, incoming_p in zip(trainable_params, parameters):
                # Ensure dtype/device match
                t = torch.from_numpy(np.asarray(incoming_p)).to(device=local_p.device, dtype=local_p.dtype)
                if local_p.data.shape != t.shape:
                    raise RuntimeError(
                        f"Client {self.client_id}: tensor shape mismatch while injecting trainable params: "
                        f"local={tuple(local_p.data.shape)} incoming={tuple(t.shape)}"
                    )
                local_p.data[:] = t

        # Safety lock again (paranoia): ensure A stays frozen
        for name, param in self.model.named_parameters():
            if ("lora_A" in name) or (".A" in name):
                param.requires_grad = False

        log(INFO, f"Client {self.client_id}: Injected trainable-only parameters (len={incoming_len}).")

    def get_parameters(self, config: Any | None = None) -> List[np.ndarray]:
        """Return trainable-only parameters based on requires_grad."""
        return [p.detach().cpu().numpy() for p in self.model.parameters() if p.requires_grad]

