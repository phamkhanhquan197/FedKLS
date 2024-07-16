# # import os

# from logging import WARNING
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# from dataclasses import dataclass, asdict
# from functools import reduce
# from numpy import average, dtype, ndarray
# from numpy import array
# from numpy.typing import NDArray
# from functools import reduce
# import numpy as np

# from flwr.server.strategy.aggregate import aggregate
# from flwr.common.logger import log
# from flwr.server.client_proxy import ClientProxy
# from flwr.server.client_manager import ClientManager
# from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
# from flwr.server.strategy import FedAvg

# from flwr.common import (
#     EvaluateIns,
#     EvaluateRes,
#     FitIns,
#     FitRes,
#     MetricsAggregationFn,
#     NDArrays,
#     Parameters,
#     Scalar,
#     ndarrays_to_parameters,
#     parameters_to_ndarrays,
# )


# class FedAvgPR(FedAvg):
#     """FedAvg with Polyak-Rupert Averaging Scheme"""
#     def __init__(self, *, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, evaluate_fn: Callable[[int, List[ndarray[Any, dtype[Any]]], Dict[str, bool | bytes | float | int | str]], Tuple[float, Dict[str, bool | bytes | float | int | str]] | None] | None = None, on_fit_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, on_evaluate_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, accept_failures: bool = True, initial_parameters: Parameters | None = None, fit_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, evaluate_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, inplace: bool = True) -> None:
#         super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)

#         self.pr_weight = None
#         self.all_models = list()
#         self.rho = 0.5 # 1) 0.1, 2) 0.5, 3) 0.9, 4) 0.01
#         self.warmup_period = 280

#     def aggregate_fit(
#         self,
#         server_round: int, 
#         results: List[Tuple[ClientProxy | FitRes]], 
#         failures: List[Tuple[ClientProxy | FitRes] | BaseException]
#     ) -> Tuple[Parameters | None | Dict[str, bool | bytes | float | int | str]]:
#         """Aggregate fit results using weighted average."""
#         if not results:
#             return None, {}
#         # Do not aggregate if there are failures and failures are not accepted
#         if not self.accept_failures and failures:
#             return None, {}
        
#         # MISSING: if self.inplace ??

#         # client parameters to ndarray
#         weights_results = [
#             (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
#             for _, fit_res in results
#         ]
#         # aggregate local updates (new global model)
#         aggregated_ndarrays = aggregate(weights_results)

#         ### CUSTOM POLYAK AVG ###

#         # add the new global model to all global model list
#         # if server_round == 1:
#         #     # self.pr_weight = aggregated_ndarrays # save the first global model
#         #     # no need for PR-Avg
#         #     pass
#         if server_round == self.warmup_period:
#             # self.all_models.append(aggregated_ndarrays)
#             self.pr_weight = aggregated_ndarrays

#         elif server_round > self.warmup_period:
#             # self.all_models.append(aggregated_ndarrays)
#             # PR-Avg the new global model
            
#             aggregated_ndarrays = pr_aggregated_perlayer(self.pr_weight, aggregated_ndarrays, self.rho)
#             self.pr_weight = aggregated_ndarrays
#             # aggregated_ndarrays = pr_aggregated_iterative(self.all_models, self.rho)

#         # ndarray to parameters (PR-Avg new global model)
#         parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

#         # Aggregate custom metrics if aggregation fn was provided
#         metrics_aggregated = {}
#         if self.fit_metrics_aggregation_fn:
#             fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
#             metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
#         elif server_round == 1:  # Only log this warning once
#             log(WARNING, "No fit_metrics_aggregation_fn provided")

#         return parameters_aggregated, metrics_aggregated
        
# def pr_aggregated_perlayer(pr_weight, new_globalmodel, rho):
#     weights_prime: NDArrays = [
#         np.add(rho * layer_pr, (1 - rho) * layer_global) 
#         for layer_pr, layer_global in zip(pr_weight, new_globalmodel)
#     ]
#     return weights_prime

# def pr_aggregated_iterative(all_models, rho):
#     # get first global model as polyak-rupp avg model
#     weight_pr = all_models[0]
    
#     # iteratively perform pr-avg on all global models
#     for global_model in all_models[1:]:
#         # weight_pr = pr_aggregated_perlayer(weight_pr, global_model, rho)
#         weight_pr = [(pr_avg * rou) + (global_avg * (1 - rou)) 
#                      for pr_avg, global_avg in zip(weight_pr, global_model)]

#     return weight_pr

# # average the client local updates as usuall
# # check for pr_weight
#     # if none, initialize it with new aggregated weight (should be in np.arrays)
#     # else, average it with the new agg_weight (new global model)
# # convert to Parameters, and send the averaged global model weight to server

        
