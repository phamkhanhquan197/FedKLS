from __future__ import annotations

from typing import Any, List

import numpy as np
import torch

from mak.clients.base_client import BaseClient
from mak.utils.general import set_params
from mak.utils.helper import get_ffa_target_keys


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
        """Return FlexLoRA uplink payload: all trainable parameters in deterministic order."""
        sd = self.model.state_dict()
        target_keys = get_ffa_target_keys(self.model)
        
        # Only include keys that exist in the state dict
        params_to_send = {k: v for k, v in sd.items() if k in target_keys}
        
        # Ensure all target keys are present (even if zero)
        for k in target_keys:
            if k not in params_to_send:
                raise ValueError(f"Parameter {k} not found in model state dict")
                
        # Return in deterministic order
        return [params_to_send[k].detach().cpu().numpy() for k in target_keys]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load parameters using the shared set_params with FlexLoRA rank adaptation."""
        # IMPORTANT:
        # Round 1 downlink is FULL state_dict (len(parameters)==len(model_state)).
        # Round >1 downlink is PARTIAL list aligned with get_ffa_target_keys(model).
        model_state = self.model.state_dict()
        target_keys = get_ffa_target_keys(self.model)

        # If full model update, delegate to BaseClient/set_params full-load path
        if len(parameters) == len(model_state):
            set_params(
                self.model,
                parameters,
                method="flex_lora",
                bias=self.config_sim.get("peft", {}).get("bias", True),
                client_id=self.client_id,
                rank_map=self.rank_map,
                device=str(self.device),
            )
            self.model.to(self.device)
            return

        # Otherwise, partial update must match our target key list
        if len(parameters) != len(target_keys):
            raise ValueError(
                f"Parameter count mismatch: expected {len(target_keys)}, got {len(parameters)}"
            )

        # Use set_params to handle the actual parameter loading (expects List[np.ndarray])
        set_params(
            self.model,
            parameters,
            method="flex_lora",
            bias=self.config_sim.get("peft", {}).get("bias", True),
            client_id=self.client_id,
            rank_map=self.rank_map,
            device=str(self.device),
        )

        self.model.to(self.device)

