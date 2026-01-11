from __future__ import annotations

from typing import Any, List

import numpy as np

from mak.clients.base_client import BaseClient
from mak.utils.general import set_params
from mak.utils.helper import get_ffa_target_keys
from mak.utils.flex_lora_utils import ensure_local_rank_adapters, slice_and_load_params


class FlexLoRAClient(BaseClient):
    """FlexLoRA client (heterogeneous LoRA ranks).

    - Round 1: receives full model and adapts A/B shapes to local rank.
    - Round >1: exchanges partial payload defined by `get_ffa_target_keys`.
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
        rank_policy_map: dict | None = None,
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
        self.rank_policy_map = rank_policy_map or {}
        self._policy_initialized = False

    def __repr__(self) -> str:
        return " FlexLoRA client"

    def get_parameters(self, config: Any | None = None) -> List[np.ndarray]:
        """Return FlexLoRA uplink payload: all trainable parameters in deterministic order."""
        sd = self.model.state_dict()
        target_keys = get_ffa_target_keys(self.model)

        params_to_send = {k: v for k, v in sd.items() if k in target_keys}

        for k in target_keys:
            if k not in params_to_send:
                raise ValueError(f"Parameter {k} not found in model state dict")

        return [params_to_send[k].detach().cpu().numpy() for k in target_keys]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load parameters.

        - Round 1: FULL state_dict downlink (global_rank). We slice A/B to local_rank.
        - Round > 1: PARTIAL downlink aligned with get_ffa_target_keys(model).
        """
        model_state = self.model.state_dict()
        target_keys = get_ffa_target_keys(self.model)

        # Round 1: FULL model update (global payload)
        if len(parameters) == len(model_state):
            if not self.rank_policy_map or int(self.client_id) not in self.rank_policy_map:
                raise ValueError("FlexLoRA requires rank_policy_map[client_id] for full update slicing")

            rank_policy = self.rank_policy_map[int(self.client_id)]

            # Ensure the client model is adapted with per-layer rank-policy adapters before loading.
            self.model = ensure_local_rank_adapters(
                model=self.model,
                base_config=self.config_sim,
                rank_policy=rank_policy,
            )

            slice_and_load_params(
                model=self.model,
                params=parameters,
                rank_policy=rank_policy,
                device=str(self.device),
            )
            self.model.to(self.device)
            self._policy_initialized = True
            return

        # Round > 1: partial update must match our target key list
        if len(parameters) != len(target_keys):
            raise ValueError(
                f"Parameter count mismatch: expected {len(target_keys)}, got {len(parameters)}"
            )

        # Round > 1: policy should already be established after Round 1.
        if not self.rank_policy_map or int(self.client_id) not in self.rank_policy_map:
            raise ValueError("FlexLoRA requires rank_policy_map[client_id] for partial update slicing")

        if not self._policy_initialized:
            # Fallback safety: enforce policy once (projection-based; should not reset after P1 Step 1).
            rank_policy = self.rank_policy_map[int(self.client_id)]
            self.model = ensure_local_rank_adapters(
                model=self.model,
                base_config=self.config_sim,
                rank_policy=rank_policy,
            )
            self._policy_initialized = True

        set_params(
            self.model,
            parameters,
            method="flex_lora",
            bias=self.config_sim.get("peft", {}).get("bias", True),
            client_id=self.client_id,
            rank_policy_map=self.rank_policy_map,
            device=str(self.device),
        )

        self.model.to(self.device)
