from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import FitRes, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedASStrategy(FedAvg):
    """FedAS strategy: server injects per-client prev_state in FitIns.config and
    aggregates client updates using client-provided FIM-trace weights.
    """

    def __init__(
        self,
        *,
        fraction_fit=1,
        fraction_evaluate=1,
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

        # Mapping client_id -> last prev_state ndarrays sent to that client
        self.client_prev_states = {}
        self._last_sent_prev_states = {}

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, object]]:
        # Use base implementation to sample clients and create FitIns
        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        # Convert current global parameters to ndarrays to use as fallback
        global_nd = parameters_to_ndarrays(parameters) if parameters is not None else None

        # Inject per-client prev_state into each FitIns.config
        result = []
        for client_proxy, fit_ins in client_instructions:
            prev = self.client_prev_states.get(client_proxy.cid, global_nd)
            # Store what we are sending so we can update mapping later if needed
            self._last_sent_prev_states[client_proxy.cid] = prev
            if fit_ins.config is None:
                fit_ins.config = {}
            fit_ins.config["fedas.prev_state_ndarrays"] = prev
            fit_ins.config["fedas.prev_state_version"] = f"round_{server_round - 1}"
            result.append((client_proxy, fit_ins))

        return result

    def aggregate_fit(
        self,
        server_round: int,
        server_params: NDArrays,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # If no results, behave like base
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Convert each client's returned Parameters to ndarrays
        client_nds = [parameters_to_ndarrays(res.parameters) for _, res in results]

        # Compute weights: weight_i = num_examples_i * fim_trace_i (default fim_trace=1.0)
        weights = []
        for _, res in results:
            fim = 1.0
            try:
                fim = float(res.metrics.get("fedas.fim_trace", 1.0))
            except Exception:
                fim = 1.0
            weights.append(res.num_examples * fim)

        total_weight = sum(weights)
        if total_weight == 0:
            # Fallback to simple averaging
            averaged = np.mean(client_nds, axis=0)
        else:
            # Weighted average element-wise across ndarrays
            averaged = []
            for layer_idx in range(len(client_nds[0])):
                layer_sum = None
                for nd, w in zip(client_nds, weights):
                    arr = nd[layer_idx].astype(np.float64)
                    if layer_sum is None:
                        layer_sum = w * arr
                    else:
                        layer_sum = layer_sum + w * arr
                averaged_layer = (layer_sum / total_weight).astype(client_nds[0][layer_idx].dtype)
                averaged.append(averaged_layer)

        parameters_aggregated = ndarrays_to_parameters(averaged)

        # Update per-client stored prev_states to what we last sent (keeps mapping consistent)
        for client_proxy, _ in results:
            last_sent = self._last_sent_prev_states.get(client_proxy.cid, None)
            if last_sent is not None:
                self.client_prev_states[client_proxy.cid] = last_sent

        # Aggregate metrics using provided aggregation function if any
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated
