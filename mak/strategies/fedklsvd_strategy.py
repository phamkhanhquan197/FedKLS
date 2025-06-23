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
"""FedKL-SVD strategy."""
from logging import WARNING, INFO

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from functools import reduce
import numpy as np

class FedKLSVDStrategy(FedAvg):
    """Implement custom strategy for FedKL-SVD based on FedAvg class."""
    def __init__(
        self,
        alpha=0.5,  # Hyperparameter to control kl_norm influence
        *,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        inplace=True,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=inplace,
        )
        self.alpha = alpha 

    def aggregate_fit(
        self, 
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results for FedKL-SVD using weighted average."""
        if not results:
            log(WARNING, f"Round {server_round}: No results to aggregate")
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            log(WARNING, f"Round {server_round}: {len(failures)} client failures during fit")
            return None, {}
        log(INFO, f"Round {server_round}: Aggregated parameters from {len(results)} clients")
        

        #Extract parameters, num_examples, and kl_norm from results
        parameters = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        num_examples = [fit_res.num_examples for _, fit_res in results]
        kl_norms = [fit_res.metrics.get("kl_norm", 0.0) for _, fit_res in results]

        # Aggregate parameters using the custom aggregation function
        aggregated_parameters = self.fedklsvd_aggregate(parameters, num_examples, kl_norms)

        # Aggregate parameters using weighted averaging based on number of examples (IMPORTANT) (FEDAVG)
        # #Convert client parameters to Numpy arrays
        # weights_results = [
        #     (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        #     for _, fit_res in results
        # ]
        # # aggregated_parameters = aggregate(weights_results)



        # Convert aggregated weights back to Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_parameters)

        # Aggragate custom metrics if aggregation function was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1: # Only log this warning once
             log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
    def fedklsvd_aggregate(self, parameters: List[NDArrays], num_examples: List[int], kl_norms: List[float]) -> NDArrays:
        """Compute weighted average using num_examples and kl_norm."""
        # Compute weights as num_examples * (1 - alpha * kl_norm)
        weights = [
            n * (1.0 - self.alpha * kl_norm)
            for n, kl_norm in zip(num_examples, kl_norms)
        ]
        
        # Ensure weights are non-negative and sum to a positive value
        weights = [max(w, 0.0) for w in weights]
        weights_total = sum(weights)
        if weights_total <= 0:
            log(INFO, "Total weights are zero or negative, falling back to uniform weighting")
            weights = [1.0 / len(parameters)] * len(parameters)
            weights_total = sum(weights)

        # Log the weights for debugging
        log(INFO, f"Round aggregation weights (raw): {weights}")
        log(INFO, f"Round aggregation weights: {np.round(np.array(weights)/weights_total, 4)} (total: {weights_total})")
        # Compute weighted sum of parameters
        weighted_parameters = [
            [layer * w for layer in client_params]
            for client_params, w in zip(parameters, weights)
        ]

        # Compute the weighted average
        aggregated_parameters: NDArrays = [
            reduce(np.add, layer_updates) / weights_total
            for layer_updates in zip(*weighted_parameters)
        ]
        
        return aggregated_parameters
