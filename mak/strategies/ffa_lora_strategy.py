"""FFA-LoRA strategy (Phase 1) - Deterministic Name-Based Aggregation.

Key principles:
- Communication standard: model.state_dict() (NOT model.parameters()).
- Deterministic mapping: Name Filter -> Sort -> Key-based injection.
- Aggregation scope: only LoRA B + selected bias + classifier/head weights.

Round protocol:
- Round 1 downlink: FULL state_dict values.
- Round >=2 downlink: PARTIAL values filtered by get_ffa_target_keys(model).
- Client uplink: ALWAYS PARTIAL values in the same sorted-key order.

The server aggregates the PARTIAL tensors with FedAvg and injects them back into
its server-side model state_dict before returning a FULL snapshot.
"""

from __future__ import annotations

from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from flwr.common import FitRes, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from mak.utils.helper import get_ffa_target_keys


class FFALoRAStrategy(FedAvg):
    """FFA-LoRA Strategy with deterministic name-based parameter mapping."""

    def __init__(self, model, config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Server-side reference model used for key filtering + key injection
        self.model = model
        self.config_sim = config

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Return initial global model parameters as FULL state_dict values."""
        full = [val.detach().cpu().numpy() for val in self.model.state_dict().values()]
        return ndarrays_to_parameters(full)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Round 1: send FULL state_dict. Round>1: send PARTIAL by sorted keys."""
        if server_round == 1:
            params_to_send = ndarrays_to_parameters(
                [val.detach().cpu().numpy() for val in self.model.state_dict().values()]
            )
            log_msg = "FFALoRA configure_fit: Round 1 sending FULL state_dict"
        else:
            keys = get_ffa_target_keys(self.model)
            sd = self.model.state_dict()
            params_to_send = ndarrays_to_parameters([sd[k].detach().cpu().numpy() for k in keys])
            log_msg = f"FFALoRA configure_fit: Round {server_round} sending PARTIAL (len={len(keys)})"

        log(WARNING, log_msg)

        # FedAvg sampling logic
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        fit_config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_config = dict(fit_config)
        fit_config["server_round"] = server_round
        fit_config["round"] = server_round

        from flwr.common import FitIns

        return [(client, FitIns(params_to_send, fit_config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate partial tensors, inject into server model, return FULL snapshot."""
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None:
            return None, metrics_aggregated

        aggregated_ndarrays: NDArrays = parameters_to_ndarrays(aggregated_parameters)

        # Inject aggregated partials into server model by deterministic keys
        keys = get_ffa_target_keys(self.model)
        if len(keys) != len(aggregated_ndarrays):
            raise ValueError(
                f"FFALoRAStrategy: aggregated length mismatch. expected={len(keys)} got={len(aggregated_ndarrays)}"
            )

        sd = self.model.state_dict()
        with torch.no_grad():
            for k, v in zip(keys, aggregated_ndarrays):
                t = torch.from_numpy(np.asarray(v)).to(device=sd[k].device, dtype=sd[k].dtype)
                if sd[k].shape != t.shape:
                    raise RuntimeError(
                        f"FFALoRAStrategy: tensor shape mismatch for key '{k}': "
                        f"local={tuple(sd[k].shape)} incoming={tuple(t.shape)}"
                    )
                sd[k].copy_(t)

        # Return FULL snapshot for global sync
        full_snapshot = [val.detach().cpu().numpy() for val in self.model.state_dict().values()]
        return ndarrays_to_parameters(full_snapshot), metrics_aggregated
