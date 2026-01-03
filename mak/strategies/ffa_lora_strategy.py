# Copyright 2020 Flower Labs GmbH. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

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
