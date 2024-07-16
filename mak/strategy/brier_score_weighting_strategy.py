from typing import Dict, List, Tuple, Union, Optional
from logging import WARNING
import flwr as fl
from dataclasses import dataclass, asdict
import json
from functools import reduce
import numpy as np
from flwr.server.strategy.aggregate import aggregate
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.common import NDArray, NDArrays
from flwr.common import (
    Scalar,
    FitRes,
    FitIns,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from sklearn.metrics import brier_score_loss
from tensorflow.keras.models import load_model
import tensorflow as tf

from mak.utils.utils import get_model


class BrierScoreWeightingStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model,
        test_data,
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
        print("++++++++++++++++ Using Brier Score based weighting Strategy +++++++++++++++++++++++++++")
        self.model = model
        self.x_test, self.y_test = test_data
        self.fraction_fit = fraction_fit
        self.evaluate_fn = evaluate_fn
        self.initial_parameters = initial_parameters
        self.on_fit_config_fn = on_fit_config_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        

    def __repr__(self) -> str:
        return " Brier Score based weighting Strategy"

    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using brier score weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results 
        #weighted results based on the loss
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, self.evaluate_client(parameters_to_ndarrays(fit_res.parameters)))
            for _, fit_res in results
        ]
        print(f"++++++++++ weights results done aggregating parameters")
        parameters_aggregated = ndarrays_to_parameters(self._aggregate_greedy(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    

    # def evaluate_client(self, client_parameters):
    #     """Evaluates client model and return brier score"""
    #     client_model = get_model()
    #     client_model.set_weights(client_parameters)
    #     loss, acc = client_model.evaluate(
    #         self.x_test, self.y_test, batch_size=32, verbose = 0,
    #     )
    #     return 1/loss

    def evaluate_client(self, client_parameters):
        """Evaluates client model and return brier score"""
        client_model = get_model()
        client_model.set_weights(client_parameters)
        
        # Get predictions on the evaluation set
        predictions = client_model.predict(self.x_test)
        
        # Convert predictions to probabilities using softmax
        probabilities = tf.nn.softmax(predictions, axis=1).numpy()
        # Calculate Brier Score for each class
        brier_scores = []
        true_labels = tf.argmax(self.y_test, axis=1).numpy()
        for class_index in range(10):  # Assuming there are 10 classes
            class_labels = np.array([1 if label == class_index else 0 for label in true_labels])
            class_probabilities = probabilities[:, class_index]
            brier_score = brier_score_loss(class_labels, class_probabilities)
            brier_scores.append(brier_score)

        # Return the mean Brier score
        mean_brier_score = np.mean(brier_scores)
        # return 1/mean_brier_score   for brier score based approach
        return mean_brier_score # for greedy approach


    def _aggregate(self, results: List[Tuple[NDArrays, int, float]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        # print(f"@@@@++++++ inside _aggregate+++++++++")
        num_examples_total = sum([num_examples for _, num_examples, _ in results])
        total_brier_score = sum([brier_score for _ , _, brier_score in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * brier_score for layer in weights] for weights, num_examples, brier_score in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / total_brier_score
            for layer_updates in zip(*weighted_weights)
        ]
        # print("+++++++++ _aggregate done")
        return weights_prime
    
    def _aggregate_greedy(self, results: List[Tuple[NDArrays, int, float]]) -> NDArrays:
        """Compute weighted average based on greedy approach."""
        # Calculate the total number of examples used during training
        print(f"@@@@++++++ inside _aggregate_greedy+++++++++")
        #1 calculate loss or use loss that is already in the results
        #2 sort the results based on loss values
        ranked_clients = sorted(range(len(results)), key=lambda i: results[i][2])
        print(f"Ranked clients : {ranked_clients}")
        # Start the soup by using the first ingredient.
        best_index = ranked_clients[0]
        greedy_soup_ingredients = [best_index]
        greedy_soup_params = results[best_index][0]
        best_val_loss_so_far = results[best_index][2]
        print(f"Best val loss so far : {best_val_loss_so_far}")

        for i in range(1, len(results)):
            # print(f'Testing client {i} of {len(results)}')
            # Get the potential greedy soup, which consists of the greedy soup with the new model added.
            new_ingredient_params = results[ranked_clients[i]][0]
            num_ingredients = len(greedy_soup_ingredients)
            # print(f"++++++++ num ingredients : {num_ingredients}")

            # Initialize an empty ndarray to store updated parameters
            potential_greedy_soup_params = []

            # Iterate over each index in the range of the length of new_ingredient_params
            for k in range(len(new_ingredient_params)):
                # print(f"++++++++++++ k : {k}")
                # Update parameters based on the number of ingredients
                old_param_update = greedy_soup_params[k].copy() * (num_ingredients / (num_ingredients + 1.))
                # print('f++++++ old update done')
                new_param_update = new_ingredient_params[k].copy() * (1. / (num_ingredients + 1))
                # print('f++++++ new param update done')
                
                # Combine the updated parameters and store in the new ndarray
                updated_param = old_param_update + new_param_update
                potential_greedy_soup_params.append(updated_param)
                # print('f++++++ greedy soup update done')


            # potential_greedy_soup_params = {
            #     k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
            #         new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
            #     for k in new_ingredient_params
            # }
            # Run the potential greedy soup on the held-out val set.
            # print(f"++++++++ type of pweights : {type(potential_greedy_soup_params)}")
            held_out_val_loss = self.evaluate_client(potential_greedy_soup_params)
            # If loss on the held-out val set decreases, add the new model to the greedy soup.
            print(f'Potential greedy soup val loss {held_out_val_loss}, best so far {best_val_loss_so_far}.')
            if held_out_val_loss < best_val_loss_so_far:
                greedy_soup_ingredients.append(ranked_clients[i])
                best_val_loss_so_far = held_out_val_loss
                greedy_soup_params = potential_greedy_soup_params
                print(f'Adding to soup. New soup is {greedy_soup_ingredients}')





            

        #3 create weight list
        # num_examples_total = sum([num_examples for _, num_examples, _ in results])
        # total_loss = sum([loss for _ , _, loss in results])

        # # Create a list of weights, each multiplied by the related number of examples
        # weighted_weights = [
        #     [layer * eval_loss for layer in weights] for weights, num_examples, eval_loss in results
        # ]

        # # Compute average weights of each layer
        # weights_prime: NDArrays = [
        #     reduce(np.add, layer_updates) / total_loss
        #     for layer_updates in zip(*weighted_weights)
        # ]
        # # print("+++++++++ _aggregate done")
        return greedy_soup_params

