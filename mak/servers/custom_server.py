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
"""Flower server."""


import concurrent.futures
import csv
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
from mak.utils.communication_tracker import CommunicationTracker
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters

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


class ServerSaveData:
    """Flower server customised to save data of rounds in csv files."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        out_file_path=None,
        target_acc=0.85,
        lora=False,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None
        self.out_file_path = out_file_path
        self.target_acc = target_acc
        self.lora = lora
        self.comm_tracker = CommunicationTracker()
        st = f"Using Custom Save Data Server with strategy : {self.strategy.__class__}"
        log(INFO, st)

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            curr_round_start_time = timeit.default_timer()
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "Global results: (round: %s, loss: %s, metric: %s, training + aggregate time: %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - curr_round_start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout, curr_round_start_time=curr_round_start_time )
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
            # Conclude round
            # Local results
            local_loss = res_fed[0] if res_fed is not None else None
            local_metric = res_fed[1] if res_fed is not None else None
            local_accuracy = local_metric["accuracy"] if local_metric is not None else None
            local_f1 = local_metric["f1_score"] if local_metric is not None else None
            # Global results
            global_loss = res_cen[0] if res_cen is not None else None
            global_metric = res_cen[1] if res_cen is not None else None
            global_accuracy = global_metric["accuracy"] if global_metric is not None else None
            global_f1 = global_metric["f1_score"] if global_metric is not None else None
            # log(INFO, f"Accuracy: {acc}")
            if self.out_file_path is not None:
                field_names = ["round", "global_accuracy", "global_f1_score", "global_loss", "local_accuracy", "local_f1", "local_loss", "processing_time", "upload_gb", "download_gb"]
                dict = {
                    "round": current_round,
                    "global_accuracy": global_accuracy,
                    "global_f1_score": global_f1,
                    "global_loss": global_loss,
                    "local_accuracy": local_accuracy,
                    "local_f1": local_f1,
                    "local_loss": local_loss,
                    "processing_time": timeit.default_timer() - curr_round_start_time,
                    "upload_gb": self.comm_tracker.per_round[current_round]["upload"],
                    "download_gb": self.comm_tracker.per_round[current_round]["download"],
                }
                with open(self.out_file_path, "a") as f:
                    dictwriter_object = csv.DictWriter(f, fieldnames=field_names)
                    dictwriter_object.writerow(dict)
                    f.close()
            if global_accuracy >= float(self.target_acc):
                log(
                    INFO,
                    f"Reached target accuracy so stopping further rounds: {self.target_acc}",
                )
                break

        # Total communication costs
        total_cost = self.comm_tracker.get_total_cost()
        log(INFO, "Total Communication Costs:")
        log(INFO, f"Upload: {total_cost['total_upload_gb']:.4f} GB = {total_cost['total_upload_gb'] / 1024:.4f} TB")
        log(INFO, f"Download: {total_cost['total_download_gb']:.4f} GB = {total_cost['total_download_gb'] / 1024:.4f} TB")
        log(INFO, f"Total: {total_cost['total_communication_gb']:.4f} GB = {total_cost['total_communication_gb'] / 1024:.4f} TB")

        # Total time
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s = %s minutes = %s hours", elapsed, elapsed / 60, elapsed / 3600)


        return history

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
        curr_round_start_time: float,
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        log(INFO, "**************************************************")
        if not client_instructions:
            log(INFO, "Start evaluating: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "Start evaluating: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            num_batches=2,  # Number of concurrent batches
           
        )
        log(
            DEBUG,
            "Client evaluation with %s results and %s failures",
            len(results),
            len(failures),
        )

        for i in range(len(results)):
            log(INFO, "Client %s (Total validation samples: %s, Accuracy: %s, F1_Score: %s, Loss: %s, Class Distribution (%s classes): %s)", results[i][1].metrics["client_id"], results[i][1].num_examples,
                results[i][1].metrics["accuracy"], results[i][1].metrics["f1_score"], results[i][1].loss, len(results[i][1].metrics["class_distribution"]), results[i][1].metrics["class_distribution"])      

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        log(
            INFO,
            "Local results (avg): (round %s: loss: %s, metric: %s, processing time: %s)",
            server_round,
            loss_aggregated,
            metrics_aggregated,
            timeit.default_timer() - curr_round_start_time,
        )
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float]) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
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
        log(INFO, f"Model size: {param_size:.4f} GB = {param_size*1e3:.4f} MB")
        if not client_instructions:
            log(INFO, "Start trainining: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "Start training: sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

        # # ------------------- START: Print client weight shapes -------------------
        # log(INFO, "--- Client Weight Shapes Received (Round %s) ---", server_round)
        # for client_proxy, fit_res in results:
        #     # fit_res is of type FitRes
        #     # fit_res.parameters is of type Parameters
        #     client_cid = client_proxy.cid # Get client ID for logging
        #     if fit_res.parameters and fit_res.parameters.tensors:
        #          # Convert Parameters (bytes) to a list of NumPy ndarrays
        #         client_weights_ndarrays = parameters_to_ndarrays(fit_res.parameters)
        #         log(INFO, f"Client {client_cid} (Num examples: {fit_res.num_examples}) sent {len(client_weights_ndarrays)} parameter layers/tensors:")
        #         for i, layer_weights in enumerate(client_weights_ndarrays):
        #             log(INFO, f"  Client {client_cid} - Layer {i}: shape {layer_weights.shape}, dtype {layer_weights.dtype}")

        #     else:
        #         log(INFO, f"Client {client_cid} did not return parameters or parameters.tensors was empty.")
        # # -------------------- END: Print client weight shapes --------------------


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
        

        log(INFO, f"Round {server_round} upload size: {upload_size:.4f} GB = {upload_size*1e3:.4f} MB, " 
            f"download size: {param_size * num_clients:.4f} GB = {param_size * num_clients*1e3:.4f} MB, "
            f"total: {upload_size + param_size * num_clients:.4f} GB = {(upload_size + param_size * num_clients)*1e3:.4f} MB")

        log(
            DEBUG,
            "Server evaluation with %s results and %s failures",
            len(results),
            len(failures),
        )
    
        for i in range(0, len(results)):
            log(INFO, "Client %s (Total training samples: %s, Class Distribution (%s classes): %s)", results[i][1].metrics["client_id"], results[i][1].num_examples, len(results[i][1].metrics["class_distribution"]), results[i][1].metrics["class_distribution"]) 

        # Standard aggregation for non-LoRA models
        parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        ##Check how many tensor the model performs aggregating
        # aggregated_ndarrays = parameters_to_ndarrays(parameters_aggregated)
        # log(INFO, f"Aggregated parameters ({len(aggregated_ndarrays)} tensors):")
        # for i, arr in enumerate(aggregated_ndarrays):
        #     log(INFO, f"  Aggregated Tensor {i}: shape {arr.shape}")


        return parameters_aggregated, metrics_aggregated, (results, failures)


    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        
        return get_parameters_res.parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)

def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
    num_batches: int = 2  # Parameter to control number of concurrent batches
) -> EvaluateResultsAndFailures:
    """Evaluate parameters on clients with dynamic batching."""
    # Calculate batch size based on number of clients and desired batches
    batch_size = max(1, len(client_instructions) // num_batches)
    
    # Create batches of client instructions
    batches = [
        client_instructions[i:i + batch_size]
        for i in range(0, len(client_instructions), batch_size)
    ]
    
    # Limit to num_batches (combine any extra batches into the last one)
    if len(batches) > num_batches and len(batches) > 1:
        last_batch = []
        for i in range(num_batches - 1, len(batches)):
            last_batch.extend(batches[i])
        batches = batches[:num_batches-1] + [last_batch]
    
    # Process each client in a batch sequentially, but run batches concurrently
    def process_batch(batch: List[Tuple[ClientProxy, EvaluateIns]]) -> List[Tuple[ClientProxy, EvaluateRes]]:
        batch_results = []
        batch_failures = []
        for client_proxy, ins in batch:
            try:
                result = client_proxy.evaluate(ins, timeout=timeout)
                batch_results.append((client_proxy, result))
            except Exception as e:
                batch_failures.append((client_proxy, e))
        return batch_results, batch_failures
    
    # Use ThreadPoolExecutor to run batches concurrently
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_batches) as executor:
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                batch_results, batch_failures = future.result()
                results.extend(batch_results)
                failures.extend(batch_failures)
            except Exception as e:
                # If the entire batch processing fails
                failures.append(e)
    
    return results, failures



def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
