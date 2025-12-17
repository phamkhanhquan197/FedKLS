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
"""FedKL-SVD Flower server."""

import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    DisconnectRes,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from mak.strategies.fedklsvd_strategy import FedKLSVDStrategy

from mak.servers.custom_server import ServerSaveData, fit_clients

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class FedKLSVDServer(ServerSaveData):
    """FedKL-SVD server implementation, inheriting from ServerSaveData."""

    def __init__(
        self, *, client_manager, strategy=None, out_file_path=None, target_acc=0.99
    ):
        super().__init__(
            client_manager=client_manager,
            strategy=strategy,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )
        # If no strategy is provided, use the custom FedKLSVDStrategy
        if strategy is None:
            self.strategy = FedKLSVDStrategy()

        log(INFO, f"Using Custom Save Data Server (FedKL-SVD) with strategy: {self.strategy.__class__}")

    def fit_round(
            self,
            server_round: int,
            timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        
        """Perform a single round of federated learning."""
        #Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        #Track download size (server -> clients)
        param_size = sum(len(p) for p in self.parameters.tensors) / 1e9 # Convert to GB
        num_clients = len(client_instructions)
        self.comm_tracker.log_round(
            server_round=server_round,
            upload=0,
            download=param_size * num_clients)
        
        log(INFO, "======================================Round %s======================================", server_round)
        log(INFO, f"Model size: {param_size:.4f} GB = {param_size*1024:.4f} MB")
        if not client_instructions:
            log(INFO, "Start trainining: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "Start training: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        
        log(
            DEBUG,
            "Server evaluation with %s results and %s failures",
            len(results),
            len(failures),
        )

        #Log client training details
        for i in range(len(results)):
            client_id = results[i][1].metrics["client_id"]
            train_samples = results[i][1].num_examples
            num_class =  len(results[i][1].metrics["class_distribution"])
            class_dist = results[i][1].metrics["class_distribution"]

            log(INFO, "Client %s (Total training samples: %s, Class Distribution (%s classes): %s)", 
                client_id, train_samples, num_class, class_dist) 
            
        # Track upload size (clients -> server)
        upload_size = 0.0
        for client, fit_res in results:
            client_upload = sum(len(t) for t in fit_res.parameters.tensors) / 1e9
            self.comm_tracker.per_client[client.cid]["upload"] += client_upload
            upload_size += client_upload
        #Update tracker
        self.comm_tracker.per_round[server_round]["upload"] = upload_size
        self.comm_tracker.total_upload += upload_size
        self.comm_tracker.per_round[server_round]["download"] = param_size * num_clients
        
        log(INFO, f"Round {server_round} upload size: {upload_size:.4f} GB = {upload_size*1024:.4f} MB, " 
            f"download size: {param_size * num_clients:.4f} GB = {param_size * num_clients*1024:.4f} MB, "
            f"total: {upload_size + param_size * num_clients:.4f} GB = {(upload_size + param_size * num_clients)*1024:.4f} MB")

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)


        parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)




        
