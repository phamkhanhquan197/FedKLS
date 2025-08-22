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
"""FedAWA strategy."""
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import NDArrays, Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy
from typing import List, Dict, Optional, Tuple, Union
from flwr.common.logger import log
from logging import INFO, WARNING
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from mak.utils.general import set_params

class FedAWAStrategy(FedAvg):
    """FedAWA strategy implementing adaptive weight aggregation.
    FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors (CVPR-2025)
    https://arxiv.org/html/2503.15842v1#S4.E3
    Implementation based on https://github.com/ChanglongShi/FedAWA
    """
    
    def __init__(
        self, 
        model,
        test_data,
        config,
        device="cuda",
        apply_transforms_test = None,
        fraction_fit: float = 0.3,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 10,
        min_available_clients: int = 2,
        evaluate_fn = None,
        evaluate_metrics_aggregation_fn = None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        initial_parameters: Optional[Parameters] = None,
        **kwargs        
    ) -> None:

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            initial_parameters=initial_parameters,
        )

        self.model = model
        self.test_data = test_data
        self.device = device
        self.config = config['fedawa_config']
        self.config["dataset"] = config["common"]["dataset"]
        self.config["batch_size"] = config["client"]["batch_size"]
        self.config["peft_enabled"] = config["peft"]["enabled"]
        self.apply_transforms = apply_transforms_test
        self.valid_set = self._get_valid_set()
        self.global_parameters: NDArrays = None
        if initial_parameters is not None:
            self.global_parameters = fl.common.parameters_to_ndarrays(initial_parameters)
            log(INFO, "Initialized global parameters from provided initial_parameters")
            self.param_names = list(self.model.state_dict().keys())
            print(f"FedAWA Strategy initialized with {len(self.param_names)} parameters")

    def __repr__(self) -> str:
        return "FedAWA Strategy"

    def _get_valid_set(self):
        """Create validation set from test data."""
        client_dataset_splits = self.test_data.train_test_split(
            test_size=self.config["server_valid_ratio"]
        )
        valset = client_dataset_splits["test"]
        valset = valset.with_transform(self.apply_transforms)
        return DataLoader(valset, batch_size=self.config["batch_size"], shuffle=True)

    def _filter_adapter_parameters(self):
        """Filter global parameters to include only adapters (A, B, biases)."""
        if not self.config["peft_enabled"]:
            return
        if self.global_parameters is None or not self.param_names:
            log(WARNING, "No parameters or names to filter")
            return
        filtered_params = []
        filtered_names = []
        if any("distilbert." in n for n in self.param_names):
            filtered_params = [p for i, p in enumerate(self.global_parameters) if "lin" in self.param_names[i]]
            filtered_names = [n for n in self.param_names if "lin" in n]
        elif any("bert." in n for n in self.param_names):
            filtered_params = [p for i, p in enumerate(self.global_parameters) if "self" in self.param_names[i] or "dense" in self.param_names[i]]
            filtered_names = [n for n in self.param_names if "self" in n or "dense" in n]
        elif any("model." in n for n in self.param_names):
            filtered_params = [p for i, p in enumerate(self.global_parameters) if "self_attn" in self.param_names[i] or "mlp" in self.param_names[i]]
            filtered_names = [n for n in self.param_names if "self_attn" in n or "mlp" in n]
        else:
            log(WARNING, "Unknown model architecture for PEFT filtering")
            return
        if not filtered_params:
            log(WARNING, "No adapter parameters found. Falling back to full parameters.")
        else:
            self.global_parameters = filtered_params
            self.param_names = filtered_names
            log(INFO, f"Filtered global parameters for PEFT: {len(self.global_parameters)} parameters")

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """Initialize global parameters, restricting to adapters for PEFT."""
        if self.global_parameters is not None:
            log(INFO, "Returning pre-initialized global parameters")
            return fl.common.ndarrays_to_parameters(self.global_parameters)
        
        params = super().initialize_parameters(client_manager)
        if params is None:
            log(INFO, "No initial parameters from strategy, requesting from a random client")
            random_client = client_manager.sample(1)[0]
            ins = GetParametersIns(config={})
            try:
                get_parameters_res = random_client.get_parameters(ins=ins, timeout=None)
                params = get_parameters_res.parameters
            except Exception as e:
                log(WARNING, f"Failed to get parameters from client: {e}")
                return None
        if params is not None:
            self.global_parameters = fl.common.parameters_to_ndarrays(params)
            log(INFO, "Global parameters initialized from client")
        return params

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client parameters using FedAWA's adaptive weights."""
        if not results:
            log(WARNING, f"Round {server_round}: No results to aggregate")
            return None, {}
        if not self.accept_failures and failures:
            log(WARNING, f"Round {server_round}: {len(failures)} client failures during fit")
            return None, {}

        # Collect client parameters and dataset sizes
        client_params = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        dataset_sizes = [fit_res.num_examples for _, fit_res in results]
        total_size = sum(dataset_sizes)
        initial_weights = [size / total_size for size in dataset_sizes]

        # Compute client vectors: τ_k = θ_k - θ_g
        if server_round == 1 and self.config["peft_enabled"] == True:
            self._filter_adapter_parameters()

        client_vectors = [
            [local - global_p for local, global_p in zip(c_params, self.global_parameters)]
            for c_params in client_params
        ]

        # Optimize aggregation weights
        optimized_weights = self.optimize_weights(client_vectors, client_params, initial_weights)
 
        # Aggregate parameters: θ_g = Σ λ_k θ_k
        aggregated_params = [np.zeros_like(p) for p in client_params[0]]
        for lambda_k, c_params in zip(optimized_weights, client_params):
            for i, (agg, c_p) in enumerate(zip(aggregated_params, c_params)):
                aggregated_params[i] += lambda_k * c_p

        # Update global parameters
        self.global_parameters = aggregated_params

        # Convert to Parameters
        parameters_aggregated = fl.common.ndarrays_to_parameters(aggregated_params)

        # Aggregate custom metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        log(INFO, f"Round {server_round}: Aggregated parameters from {len(results)} clients, weights: {np.round(optimized_weights, 4)}")
        return parameters_aggregated, {"lambda_weights": optimized_weights}


    def optimize_weights(
            self,
            client_vectors: List[NDArrays],
            client_params: List[NDArrays],
            initial_weights: List[float]
        ) -> List[float]:
            """Optimize aggregation weights using validation loss."""
            cohort_size = len(client_params)
            optimizees = torch.tensor(
                [torch.log(torch.tensor(w)) for w in initial_weights],
                device=self.device,
                requires_grad=True
            )
            optimizer = optim.Adam([optimizees], lr=self.config["server_lr"], betas=(0.5, 0.999))
            if self.config["server_optimizer"] == "sgd":
                optimizer = optim.SGD([optimizees], lr=self.config["server_lr"], momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            softmax = nn.Softmax(dim=0)

            self.model.train()
            for _ in range(self.config["server_epochs"]):
                for batch in self.valid_set:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    target = batch["labels"].to(self.device)

                    model_params = None
                    for i in range(cohort_size):
                        weight = softmax(optimizees)[i]
                        c_params = [torch.tensor(p, device=self.device) for p in client_params[i]]
                        if i == 0:
                            model_params = [weight * p for p in c_params]
                        else:
                            model_params = [mp + weight * cp for mp, cp in zip(model_params, c_params)]

                    #Convert model_params to NumPy arrays before passing to set_params
                    model_params_np = [param.detach().cpu().numpy() for param in model_params]
                    set_params(self.model, model_params_np)

                    optimizer.zero_grad()
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                    loss = F.cross_entropy(output, target)

                    tau_g = [np.zeros_like(v) for v in client_vectors[0]]
                    for w, tau_k in zip(softmax(optimizees), client_vectors):
                        for i, (g, t) in enumerate(zip(tau_g, tau_k)):
                            g += w.cpu().detach().numpy() * t

                    term1 = 0.0
                    for w, tau_k in zip(softmax(optimizees), client_vectors):
                        diff_norm = sum(np.linalg.norm(t_k - t_g) for t_k, t_g in zip(tau_k, tau_g))
                        term1 += w.cpu().detach().numpy() * diff_norm

                    cos_sim = sum(
                    np.sum(a.detach().cpu().numpy() * g) / (np.linalg.norm(a.detach().cpu().numpy()) * np.linalg.norm(g) + 1e-10)
                    for a, g in zip(model_params, self.global_parameters)
                    )

                    term2 = 1 - cos_sim / len(model_params)

                    total_loss = loss + self.config["gamma"] * torch.tensor(term1 + term2, device=self.device)
                    total_loss.backward()
                    optimizer.step()
                scheduler.step()

            optimized_weights = softmax(optimizees).detach().cpu().numpy().tolist()
            return optimized_weights


    

    



