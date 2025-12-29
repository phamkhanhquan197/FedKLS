# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FFA-LoRA strategy."""
from logging import WARNING, INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

class FFALoRAStrategy(FedAvg):
    """FFA-LoRA Strategy

    - Clients control uplink payload:
        * Round 1   : [A1, B1, A2, B2, ...]
        * Round > 1 : [B1, B2, ...]

    - Server aggregation:
        * Weighted average over received tensors
        * No A/B inspection

    - Server downlink:
        * Round 1   : full parameters
        * Round > 1 : B-only parameters
    """

    def __init__(self, config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Aggregation (uplink)
    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate whatever parameters the clients send (weighted average)."""
        if not results:
            log(WARNING, f"Round {server_round}: No results to aggregate")
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            log(WARNING, f"Round {server_round}: {len(failures)} client failures during fit")
            return None, {}
        
        log(INFO, f"Round {server_round}: Aggregated parameters from {len(results)} clients")
        
        # --------------------------------------------------------------
        # FedAvg (Flower reference implementation)
        # --------------------------------------------------------------
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        # Convert aggregated weights back to Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # --------------------------------------------------------------
        # Metrics aggregation
        # --------------------------------------------------------------
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated



    # @staticmethod
    # def _extract_b_from_full_ndarrays(full: List) -> List:
    #     # full order is [A1,B1,A2,B2,...] -> B at odd indices
    #     return [full[i] for i in range(1, len(full), 2)]

    # def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
    #     # Use initial_parameters passed in constructor (created in helper.get_strategy)
    #     # Cache full and b_only views
    #     if self.initial_parameters is None:
    #         return None

    #     full_nd = fl.common.parameters_to_ndarrays(self.initial_parameters)
    #     b_nd = self._extract_b_from_full_ndarrays(full_nd)

    #     self._global_full = self.initial_parameters
    #     self._global_b_only = fl.common.ndarrays_to_parameters(b_nd)

    #     log(INFO, f"FFALoRAStrategy: initialize_parameters cached full_len={len(full_nd)} b_len={len(b_nd)}")
    #     return self.initial_parameters

    # def configure_fit(
    #     self,
    #     server_round: int,
    #     parameters: Parameters,
    #     client_manager: ClientManager,
    # ) -> List[Tuple[ClientProxy, FitIns]]:
    #     # Build FitIns list similar to FedAvg but choose params based on round.
    #     if self._global_full is None or self._global_b_only is None:
    #         # fallback compute from passed parameters
    #         full_nd = fl.common.parameters_to_ndarrays(parameters)
    #         self._global_full = parameters
    #         self._global_b_only = fl.common.ndarrays_to_parameters(self._extract_b_from_full_ndarrays(full_nd))

    #     if server_round == 1:
    #         params_to_send = self._global_full
    #         log(INFO, "FFALoRAStrategy: Round 1 downlink sending FULL (A+B)")
    #     else:
    #         params_to_send = self._global_b_only
    #         log(INFO, f"FFALoRAStrategy: Round {server_round} downlink sending B-ONLY")

    #     # Let FedAvg decide which clients to sample
    #     sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
    #     clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

    #     fit_config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
    #     # Ensure client can decide round behavior
    #     fit_config = dict(fit_config)
    #     fit_config["server_round"] = server_round
    #     fit_config["round"] = server_round

    #     return [(client, FitIns(params_to_send, fit_config)) for client in clients]

    # def aggregate_fit(
    #     self,
    #     server_round: int,
    #     results,
    #     failures,
    # ):
    #     # Aggregate only B matrices.
    #     if not results:
    #         return None, {}

    #     # Convert each client result to ndarrays
    #     client_nds: List[List] = []
    #     weights: List[int] = []

    #     for _, fit_res in results:
    #         nds = fl.common.parameters_to_ndarrays(fit_res.parameters)
    #         client_nds.append(nds)
    #         weights.append(fit_res.num_examples)

    #     # Determine if clients returned full or b_only
    #     first_len = len(client_nds[0])
    #     if self._global_full is None:
    #         raise ValueError("FFALoRAStrategy: global parameters not initialized")
    #     global_full_nd = fl.common.parameters_to_ndarrays(self._global_full)
    #     global_b_nd = self._extract_b_from_full_ndarrays(global_full_nd)

    #     if first_len == len(global_b_nd):
    #         # already B-only
    #         b_updates = client_nds
    #     elif first_len == len(global_full_nd):
    #         # full, extract B
    #         b_updates = [self._extract_b_from_full_ndarrays(nd) for nd in client_nds]
    #     else:
    #         raise ValueError(
    #             f"FFALoRAStrategy: unexpected client param len={first_len}; "
    #             f"expected b_len={len(global_b_nd)} or full_len={len(global_full_nd)}"
    #         )

    #     # Weighted average elementwise over list positions
    #     total_examples = sum(weights)
    #     agg_b: List = []
    #     for j in range(len(global_b_nd)):
    #         weighted = None
    #         for nds, num_ex in zip(b_updates, weights):
    #             contrib = nds[j] * (num_ex / total_examples)
    #             weighted = contrib if weighted is None else (weighted + contrib)
    #         agg_b.append(weighted)

    #     # Update cached globals
    #     self._global_b_only = fl.common.ndarrays_to_parameters(agg_b)

    #     # Also update full by replacing B slots, keep A unchanged
    #     new_full = list(global_full_nd)
    #     for b_i, full_pos in zip(agg_b, range(1, len(new_full), 2)):
    #         new_full[full_pos] = b_i
    #     self._global_full = fl.common.ndarrays_to_parameters(new_full)

    #     log(INFO, f"FFALoRAStrategy: Round {server_round} aggregated B-only (b_len={len(agg_b)})")

    #     # Return parameters for next round (Flower passes into configure_fit)
    #     # We return full for consistency, configure_fit decides what to send.
    #     return self._global_full, {}
