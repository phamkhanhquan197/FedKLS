# mak/strategies/fedsa_lora_strategy.py
from logging import WARNING, INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


class FedSALoRAStrategy(FedAvg):
    """
    FedSA-LoRA Strategy

    - Clients uplink:
        * Always: A-only (and optional bias if enabled at client side)
    - Server aggregation:
        * Weighted average over received tensors
    - Server downlink:
        * Same tensors (A-only) are broadcast back
    """

    def __init__(self, config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            log(WARNING, f"Round {server_round}: No results to aggregate")
            return None, {}
        if not self.accept_failures and failures:
            log(WARNING, f"Round {server_round}: {len(failures)} client failures during fit")
            return None, {}

        log(INFO, f"Round {server_round}: Aggregated parameters from {len(results)} clients")

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated