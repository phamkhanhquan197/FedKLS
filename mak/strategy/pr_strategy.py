from typing import Dict, List, Tuple, Union, Optional
from logging import WARNING
import flwr as fl
from dataclasses import dataclass, asdict
import json
from functools import reduce
import numpy as np
import tensorflow as tf

from flwr.common import (
    Scalar,
    FitRes,
    FitIns,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
import os
from tensorflow.keras.models import load_model

from numpy import average
from numpy import array
from numpy.typing import NDArray
from functools import reduce


class FedPR(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model,
        pr_alpha,
        models_path,
        fraction_fit: float,
        fraction_evaluate: float,
        min_fit_clients: int,
        min_evaluate_clients : int,
        min_available_clients : int,
        evaluate_fn,
        initial_parameters,
        evaluate_metrics_aggregation_fn,
        on_fit_config_fn = None

    ) -> None:
        super().__init__(fraction_fit=fraction_fit,
                         fraction_evaluate = fraction_evaluate,
                         min_fit_clients = min_fit_clients,
                         min_evaluate_clients = min_evaluate_clients,
                         min_available_clients = min_available_clients,
                         evaluate_fn = evaluate_fn,
                         initial_parameters= initial_parameters,
                         on_fit_config_fn = on_fit_config_fn)
        print(f"++++++++++++++++ Using FedPR Strategy with PR alpha = {pr_alpha}+++++++++++++++++++++++++++")
        self.model = model
        self.models_path = models_path
        self.pr_alpha = pr_alpha
        self.fraction_fit = fraction_fit
        self.evaluate_fn = evaluate_fn
        self.initial_parameters = initial_parameters
        self.on_fit_config_fn = on_fit_config_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.all_model_weights = []

    def __repr__(self) -> str:
        return "FedPR"

    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        # self.model.set_weights(parameters_to_ndarrays(parameters_aggregated))

        self.all_model_weights.append(parameters_to_ndarrays(parameters_aggregated))
        # append the parameters here
        if len(self.all_model_weights) > 1:
            parameters_aggregated = ndarrays_to_parameters(self.pr_aggregate_v3(all_models=self.all_model_weights,alpha=self.pr_alpha))


        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
  

    def pr_aggregate(self,all_models, alpha):

        averaged_weights = all_models[-1]

        for i in range(1, len(all_models)):
            current_weights = all_models[i]
            averaged_weights = [alpha * avg + (1 - alpha) * cur for avg, cur in zip(averaged_weights, current_weights)]
        return averaged_weights

    # def pr_aggregate(self, all_models, alpha):
    #     averaged_weights = all_models[-1].copy()

    #     for i in range(len(all_models) - 2, -1, -1):
    #         current_weights = all_models[i]

    #         for j in range(len(averaged_weights)):
    #             averaged_weights[j] = alpha * averaged_weights[j] + (1 - alpha) * current_weights[j]

    #     return averaged_weights


    def _save_model(self):
        self.model.save()

    def pr_aggregate_v2(self,all_models,alpha = 0.1):
        # machine learning mastery
        # https://machinelearningmastery.com/polyak-neural-network-model-weight-ensemble/
        # prepare an array of equal weights
        n_models = len(all_models)
        # weights = [1/n_models for i in range(1, n_models+1)]
        weights = [alpha * (1 - alpha)**i for i in range(n_models)]
        # determine how many layers need to be averaged
        n_layers = len(all_models[-1])
        # create an set of average model weights
        avg_model_weights = list()
        for layer in range(n_layers):
            # collect this layer from each model
            layer_weights = array([model[layer] for model in all_models])
            # weighted average of weights for this layer
            avg_layer_weights = average(layer_weights, axis=0, weights=weights)
            # store average layer weights
            avg_model_weights.append(avg_layer_weights)
        return avg_model_weights
    
    def pr_aggregate_v3(self, all_models, alpha : float = 0.9) -> NDArray:
        """Compute Polyak-Ruppert weighted average."""
        # Initialize the average weights with the weights from the first result
        weights_prime = all_models[0].copy()
        
        for weights in all_models[1:]:
            # Update the average weights using Polyak-Ruppert averaging
            # weights_prime = alpha * weights_prime + (1 - alpha) * weights
            weights_prime = [alpha * avg + (1 - alpha) * cur for avg, cur in zip(weights_prime, weights)]
            
        return weights_prime
    

    def layerwise_polyak_ruppert_average(self, weights_list: List[NDArray], alpha: float = 0.9) -> NDArray:
        """Compute layer-wise Polyak-Ruppert weighted average for a list of weights."""
        
        # Check if the weights_list is not empty
        if not weights_list:
            raise ValueError("Input weights_list should not be empty.")
        
        # Get the number of layers and the size of each layer
        num_layers = len(weights_list[0])
        layer_size = len(weights_list[0][0])

        # Initialize the average weights with the weights from the first result
        averaged_weights = np.zeros_like(weights_list[0])

        for layer in range(num_layers):
            layer_weights = [weights[layer] for weights in weights_list]
            
            # Perform Polyak-Ruppert averaging for the current layer
            averaged_layer = layer_weights[0].copy()
            for weights in layer_weights[1:]:
                averaged_layer = alpha * averaged_layer + (1 - alpha) * weights
            
            averaged_weights[layer] = averaged_layer

        return averaged_weights