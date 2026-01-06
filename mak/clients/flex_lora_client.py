from __future__ import annotations

from typing import Any, List

import numpy as np
import torch

from mak.clients.base_client import BaseClient
from mak.utils.general import set_params


class FlexLoRAClient(BaseClient):
    """FlexLoRA client.

    Design constraints:
    - Strict inheritance: subclass BaseClient only.
    - Strict reuse: parameter injection delegated to mak.utils.general.set_params.
    - Communication protocol: deterministic name-based sorted list.

    FlexLoRA trains and communicates both A and B factors.
    """

    def __init__(
        self,
        client_id: int,
        model,
        trainset,
        valset,
        config_sim: dict,
        device,
        save_dir,
        kl_norm=None,
        dataset=None,
        apply_transforms=None,
        data_scheduler=None,
        bias=None,
        rank_map: dict | None = None,
    ):
        super().__init__(
            client_id=client_id,
            model=model,
            trainset=trainset,
            valset=valset,
            config_sim=config_sim,
            device=device,
            save_dir=save_dir,
            dataset=dataset,
            apply_transforms=apply_transforms,
            data_scheduler=data_scheduler,
            bias=bias,
        )
        self.rank_map = rank_map or {}

    def __repr__(self) -> str:
        return " FlexLoRA client"

    def get_parameters(self, config: Any | None = None) -> List[np.ndarray]:
        """Return FlexLoRA uplink payload: all `.A` and `.B` tensors in sorted-key order."""
        sd = self.model.state_dict()
        params_to_send = {k: v for k, v in sd.items() if k.endswith(".A") or k.endswith(".B")}

        # Deterministic ordering
        return [tensor.detach().cpu().numpy() for _, tensor in sorted(params_to_send.items())]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load parameters using the shared set_params with FlexLoRA rank adaptation."""
        # We purposely reuse general.set_params; rank adaptation happens only when method='flex_lora'.
        set_params(
            self.model,
            parameters,
            method="flex_lora",
            bias=self.config_sim.get("peft", {}).get("bias", True),
            client_id=self.client_id,
            rank_map=self.rank_map,
            device=str(self.device),
        )

        # Ensure model on correct device
        self.model.to(self.device)

        # No additional logic here; BaseClient.fit/evaluate uses self.set_parameters() at round start.

