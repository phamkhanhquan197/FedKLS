from __future__ import annotations

from logging import INFO
from typing import List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import FitIns, Parameters
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FFALoRAStrategy(FedAvg):
    """FFA-LoRA Strategy (Phase 1 refactor) - requires_grad based index mapping.

    Goals:
    - Fix RuntimeError: size mismatch by:
      * Keeping a FULL parameter snapshot on server: self.current_full_parameters
      * Aggregating only trainable tensors (B, bias, etc.)
      * Reconstructing a FULL list after aggregation and returning it

    Rules of engagement:
    - Strict inheritance from FedAvg (use super().aggregate_fit).
    - Paper fidelity: A is frozen forever (validated in initialize_parameters).
    """

    def __init__(self, model, config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.config_sim = config

        # Full snapshot of global parameters as List[np.ndarray]
        self.current_full_parameters: Optional[List[np.ndarray]] = None

        # Indices of trainable parameters in the FULL list
        self.trainable_indices: Optional[List[int]] = None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global parameters and build index mapping.

        Safety check (critical): any lora_A/.A parameter must have requires_grad=False.
        """
        # Safety check: A must be frozen
        for name, p in self.model.named_parameters():
            if ("lora_A" in name) or (".A" in name):
                if p.requires_grad:
                    raise ValueError("Model configuration error: Matrix A is not frozen!")

        # Snapshot full parameters
        self.current_full_parameters = [p.detach().cpu().numpy() for p in self.model.parameters()]

        # Build trainable index map based on requires_grad
        self.trainable_indices = [
            i for i, p in enumerate(self.model.parameters()) if bool(getattr(p, "requires_grad", False))
        ]

        log(
            INFO,
            f"FFALoRAStrategy.initialize_parameters: full_len={len(self.current_full_parameters)} "
            f"trainable_len={len(self.trainable_indices)}",
        )

        return fl.common.ndarrays_to_parameters(self.current_full_parameters)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Send full params in round 1, trainable-only thereafter."""
        if self.current_full_parameters is None or self.trainable_indices is None:
            raise ValueError("FFALoRAStrategy not initialized: call initialize_parameters first")

        if server_round == 1:
            params_to_send = fl.common.ndarrays_to_parameters(self.current_full_parameters)
            log(INFO, "FFALoRAStrategy: Round 1 downlink sending FULL parameters")
        else:
            trainable_nd = [self.current_full_parameters[i] for i in self.trainable_indices]
            params_to_send = fl.common.ndarrays_to_parameters(trainable_nd)
            log(
                INFO,
                f"FFALoRAStrategy: Round {server_round} downlink sending TRAINABLE-ONLY params "
                f"(len={len(trainable_nd)})",
            )

        # FedAvg sampling logic
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        fit_config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_config = dict(fit_config)
        fit_config["server_round"] = server_round
        fit_config["round"] = server_round

        return [(client, FitIns(params_to_send, fit_config)) for client in clients]

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate trainable-only params via FedAvg, then reconstruct full list."""
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None:
            return None, metrics

        if self.current_full_parameters is None or self.trainable_indices is None:
            raise ValueError("FFALoRAStrategy not initialized: missing full snapshot/index map")

        aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

        if len(aggregated_ndarrays) != len(self.trainable_indices):
            raise ValueError(
                f"FFALoRAStrategy: aggregation size mismatch. "
                f"Expected {len(self.trainable_indices)} trainable tensors, got {len(aggregated_ndarrays)}."
            )

        # Merge into full snapshot
        new_full = list(self.current_full_parameters)
        for idx, new_val in zip(self.trainable_indices, aggregated_ndarrays):
            new_full[idx] = new_val

        self.current_full_parameters = new_full

        log(
            INFO,
            f"FFALoRAStrategy.aggregate_fit: Round {server_round} merged aggregated trainables into full snapshot. "
            f"full_len={len(new_full)} trainable_len={len(self.trainable_indices)}",
        )

        # Return FULL parameters so the server can keep a consistent global model state
        return fl.common.ndarrays_to_parameters(new_full), metrics

