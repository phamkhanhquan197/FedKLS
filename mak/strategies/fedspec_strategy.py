from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_manager import ClientManager
import copy

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class FedSpecStrategy(FedAvg):
    """FedSpec strategy"""

    def __init__(self, config: dict, model, initial_rank_dict:dict, initial_kl_dict: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config  # keep config reference, no new fields
        self.model = model  # keep model reference, no new fields
        self.initial_rank_dict  = initial_rank_dict  # keep rank dictionary reference, no new fields
        self.initial_kl_dict = initial_kl_dict 
        self.rank_dict = {}   # client_id -> rank (updated after round 1)
        self.rank_dict_current = {}  # snapshot of the current version before next update
        self.kl_dict = initial_kl_dict     # client_id -> KL norm 

    def store_initial_model(self, model, config):
        """Store initial model parameters as W0 for reconstruction."""
        from mak.utils.helper import extract_linear_layers
        layer_to_svd = extract_linear_layers(model, config)
        
        self.W_0 = {}
        for name, param in model.named_parameters():
            base_name = name.rsplit(".", 1)[0]
            if name.endswith(".weight") and base_name in layer_to_svd:
                self.W_0[name] = param.detach().cpu().numpy().copy()

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
            if server_round == 1:
                config["rank_dict"] = self.initial_rank_dict
                config["kl_dict"] = self.initial_kl_dict
            else:
                config["rank_dict"] = self.rank_dict
                config["kl_dict"] = self.kl_dict
        config["param_names"] = [name for name, _ in self.model.named_parameters()]

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
            config["current_rank_dict"] = self.rank_dict_current
            config["next_rank_dict"] = self.rank_dict
            config["kl_dict"] = self.kl_dict
        
        config["param_names"] = [name for name, _ in self.model.named_parameters()]

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average (FedSpec-compatible)."""
        if not results:
            log(WARNING, f"Round {server_round}: No results to aggregate")
            return None, {}
        if not self.accept_failures and failures:
            log(WARNING, f"Round {server_round}: {len(failures)} client failures during fit")
            return None, {}
        
        # Convert all client parameters once
        client_ndarrays = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Update rank_dict and kl_dict based on client metrics (for next round's configuration)
        self.rank_dict = self.initial_rank_dict if server_round == 1 else self.rank_dict
        self.rank_dict_current = self.rank_dict.copy()
        for client_proxy, fit_res in results:
            client_id = int(client_proxy.cid)
            metrics = fit_res.metrics
            if metrics is not None:
                self.rank_dict[client_id] = metrics.get("rank", None)
                self.kl_dict[client_id] = metrics.get("kl_norm", None)
            # OPTIONAL: This code to check the shapes of the received adapters from clients before aggregation
            # weights = parameters_to_ndarrays(fit_res.parameters)
            # shapes = [w.shape for w in weights]
            # print(f"[Round {server_round}] Client {client_id} BEFORE padding ({len(shapes)} adapters): {shapes}")


        
        # ---- Step 1: Infer global max_rank ----
        max_rank = max(min(w.shape) for weights, _ in client_ndarrays for w in weights if w.ndim == 2)

        # ---- Step 2: Zero padding to max_rank ----
        def _pad(w: np.ndarray) -> np.ndarray:
            if w.ndim != 2:
                return w

            r = min(w.shape)
            if r == max_rank:
                return w

            # LoRA B: (r, in_dim) → pad rows
            if w.shape[0] == r:
                return np.pad(w, ((0, max_rank - r), (0, 0)), mode="constant")

            # LoRA A: (out_dim, r) → pad cols
            return np.pad(w, ((0, 0), (0, max_rank - r)), mode="constant")
        
        weights_results = [([ _pad(w) for w in weights ], num_examples) for weights, num_examples in client_ndarrays]
        
        # OPTIONAL: Check shapes AFTER padding, do not use 
        for client_proxy, fit_res in results:
            client_id = int(client_proxy.cid)
            weights = parameters_to_ndarrays(fit_res.parameters)
            shapes = [w.shape for w in weights]
            print(f"[Round {server_round}] Client {client_id} AFTER padding ({len(shapes)} adapters): {shapes}")
        
        # ---- Step 3: Aggregate LoRA with max rank (A_global, B_global) ----
        aggregated_lora = aggregate(weights_results)
        # aggregated_lora = self.masked_aggregate(weights_results)
        # print(f"[Round {server_round}] Aggregated LoRA shape: {[w.shape for w in aggregated_lora]}")

        # ---- Step 4: Initialize W_0 with SVD components (only once) ----
        if server_round == 1 and not hasattr(self, "W_0"):
            self.store_initial_model(self.model, self.config)
            # print(f"[Round {server_round}] Initial W_0 shape: {[w.shape for w in self.W_0.values()]}")

        # ---- Step 5: Reconstruct FULL model weights W_global = W_res + A_global@B_global^T ----
        W_global = []
        lora_idx = 0
        for name, param in self.model.state_dict().items():
            
            if name in self.W_0:               
                # ---- LoRA reconstruction ----
                A_global = aggregated_lora[lora_idx]
                B_global = aggregated_lora[lora_idx + 1]

                delta = A_global @ B_global
                W = self.W_0[name] + delta 
                W_global.append(W)

                lora_idx += 3

            #Add the aggrageted bias term if it exists
            elif name.endswith(".bias") and (name.replace(".bias", ".weight") in self.W_0):
                # print(f"Processing bias layer: {name} with shape {param.shape}")
                W_global.append(param.detach().cpu().numpy())
            
            else:
                # Remaning layers 
                W_global.append(param.detach().cpu().numpy())
  
        # ---- Step 6: Convert to Parameters ----
        parameters_aggregated = ndarrays_to_parameters(W_global)

        # print(f"[Round {server_round}] Aggregated global model parameter shapes: {[w.shape for w in W_global]}")

        # ---- Step 7: Metrics ----
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
