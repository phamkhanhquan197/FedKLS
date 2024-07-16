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
# import random
# import csv

# import flwr as fl
# from flwr.server.strategy.aggregate import aggregate
# from flwr.common.logger import log
# from flwr.server.client_proxy import ClientProxy
# from flwr.server.client_manager import ClientManager
# from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
# from flwr.server.strategy import FedAvg, FedProx

# from mak.strategy.genetic_algo import GA

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

# # def configure_fit(
# #         self, server_round: int, parameters: Parameters, client_manager: ClientManager
# #     ) -> List[Tuple[ClientProxy, FitIns]]:
# #         """Configure the next round of training.

# #         Sends the proximal factor mu to the clients
# #         """
# #         # Get the standard client/config pairs from the FedAvg super-class
# #         client_config_pairs = super().configure_fit(
# #             server_round, parameters, client_manager
# #         )

# #         # Return client/config pairs with the proximal factor mu added
# #         return [
# #             (
# #                 client,
# #                 FitIns(
# #                     fit_ins.parameters,
# #                     {**fit_ins.config, "proximal_mu": self.proximal_mu},
# #                 ),
# #             )
# #             for client, fit_ins in client_config_pairs
# #         ]

# # average the client local updates as usuall
# # check for pr_weight
#     # if none, initialize it with new aggregated weight (should be in np.arrays)
#     # else, average it with the new agg_weight (new global model)
# # convert to Parameters, and send the averaged global model weight to server

# # ************************************************************************************************************
# # ************************************************************************************************************

# class FedAvgGA(FedAvg):
#     """FedAvg with CLient Selection using Genetic Algorithm"""
#     def __init__(self, *, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, evaluate_fn: Callable[[int, List[ndarray[Any, dtype[Any]]], Dict[str, bool | bytes | float | int | str]], Tuple[float, Dict[str, bool | bytes | float | int | str]] | None] | None = None, on_fit_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, on_evaluate_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, accept_failures: bool = True, initial_parameters: Parameters | None = None, fit_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, evaluate_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, inplace: bool = True) -> None:
#         super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)

#         # self.pr_weight = None
#         # self.rho = 0.5 # 1) 0.1, 2) 0.5, 3) 0.9, 4) 0.01
#         # self.warmup_period = 280
#         self.ga_selection = GA()

#     def configure_fit(
#         self, server_round: int, parameters: Parameters, client_manager: ClientManager
#     ) -> List[Tuple[ClientProxy, FitIns]]:
#         """Configure the next round of training.
        
#         Applies Genetic Algorithm for client selection
#         """
#         config = {}
#         if self.on_fit_config_fn is not None:
#             # Custom fit config function provided
#             config = self.on_fit_config_fn(server_round)
#         fit_ins = FitIns(parameters, config)

#         # Sample clients
#         sample_size, min_num_clients = self.num_fit_clients(
#             client_manager.num_available()
#         )

#         print("+++++++++++++ Inside Fed Genetic Algo +++++++++++++")
#         clients = client_manager.sample(
#             ga_class=self.ga_selection,
#             server_round=server_round,
#             num_clients=sample_size,
#             min_num_clients=min_num_clients,
#         )

#         print(f"+++++++++++++ Fed GENETIC New Seed {self.ga_selection.seed_number} +++++++++++++")

#         # Return client/config pairs
#         return [(client, fit_ins) for client in clients]
    

# # custom fedavg which also calculates cost
# class FedAvgMine(FedAvg):
#     """FedAvg with Polyak-Rupert Averaging Scheme"""
#     def __init__(self, *, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, evaluate_fn: Callable[[int, List[ndarray[Any, dtype[Any]]], Dict[str, bool | bytes | float | int | str]], Tuple[float, Dict[str, bool | bytes | float | int | str]] | None] | None = None, on_fit_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, on_evaluate_config_fn: Callable[[int], Dict[str, bool | bytes | float | int | str]] | None = None, accept_failures: bool = True, initial_parameters: Parameters | None = None, fit_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, evaluate_metrics_aggregation_fn: Callable[[List[Tuple[int, Dict[str, bool | bytes | float | int | str]]]], Dict[str, bool | bytes | float | int | str]] | None = None, inplace: bool = True) -> None:
#         super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)

#         self.ga_selection = GA()

#     def configure_fit(
#         self, server_round: int, parameters: Parameters, client_manager: ClientManager
#     ) -> List[Tuple[ClientProxy, FitIns]]:
#         """Configure the next round of training."""
#         config = {}
#         if self.on_fit_config_fn is not None:
#             # Custom fit config function provided
#             config = self.on_fit_config_fn(server_round)
#         fit_ins = FitIns(parameters, config)

#         # Sample clients
#         sample_size, min_num_clients = self.num_fit_clients(
#             client_manager.num_available()
#         )

#         print("+++++++++++++ Inside Fed Mine Algo +++++++++++++")
#         # clients = client_manager.sample(
#         #     num_clients=sample_size, min_num_clients=min_num_clients
#         # )

#         ### CUSTOM CHANGES ###
#         # fetch all the available clients (cids)
#         available_cids = list(client_manager.clients)
#         # choose clients
#         clients = client_manager.sample(
#             ga_class=None,
#             server_round=server_round,
#             num_clients=sample_size,
#             min_num_clients=min_num_clients
#         )

#         # COST calculation of chosen clients
#         ### Assumption is all clients are available
#         costs_all_clients = self.ga_selection.assign_costs(available_cids)
#         # list of client final chosen clients for training round
#         chosen_cids = [client_.cid for client_ in clients]
#         # total cost of final chosen clients for training round
#         chosen_cids_cost = sum(costs_all_clients[cid] for cid in chosen_cids)

#         # plot the fitness value of sample cids
#         plot_cost_each_round = '/home/ghani/Downloads/Codes/nclab_flwr_pytorch/flwr-torch/mak/strategy/fedmine_cost_noniid.csv'
#         field_names = ["round","total_cost"]
#         dict = {"round" : server_round, "total_cost" : chosen_cids_cost}
#         with open(plot_cost_each_round,'a') as f:
#             dictwriter_object = csv.DictWriter(f, fieldnames=field_names)
#             dictwriter_object.writerow(dict)
#             f.close()


#         print(f"+++++++++++++ Fed MINE New Seed {self.ga_selection.seed_number} +++++++++++++")

#         ### CHANGE FIT_INS OF CLIENTS HERE

#         # Return client/config pairs
#         return [(client, fit_ins) for client in clients]

