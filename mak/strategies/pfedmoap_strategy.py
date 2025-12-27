from typing import Dict, List, Tuple, Optional
import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


class PFedMoAPStrategy(FedAvg):
    def __init__(
        self,
        *,
        config,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = config
        self.num_experts = config["pfedmoap_config"]["num_experts"]
        self.prompt_pool: Dict[int, np.ndarray] = {}

    # -------------------
    # Init global prompt
    # -------------------
    def initialize_parameters(self, client_manager) -> Parameters:
        prompt_len = self.cfg["pfedmoap_config"]["prompt_len"]
        prompt_dim = self.cfg["pfedmoap_config"]["prompt_dim"]
        prompt0 = 0.02 * np.random.randn(prompt_len, prompt_dim).astype(np.float32)
        return ndarrays_to_parameters([prompt0])

    # -------------------
    # Expert selection
    # -------------------
    def _select_experts(self, cid: int) -> List[np.ndarray]:
        if cid not in self.prompt_pool:
            return []

        current = self.prompt_pool[cid]
        distances = []
        for k, v in self.prompt_pool.items():
            if k == cid:
                continue
            dist = np.linalg.norm(current - v)
            distances.append((dist, k))

        distances.sort()
        selected = [self.prompt_pool[k] for _, k in distances[: self.num_experts - 1]]
        return selected

    # -------------------
    # Send fit instructions
    # -------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ):
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )

        global_prompt = parameters_to_ndarrays(parameters)[0]

        fit_ins = []
        for client in clients:
            cid = int(client.cid)
            experts = self._select_experts(cid)

            config = {
                "round": server_round,
                "has_experts": len(experts) > 0,
            }

            fit_ins.append(
                (
                    client,
                    (
                        ndarrays_to_parameters([global_prompt]),
                        {
                            **config,
                            "experts": experts,
                        },
                    ),
                )
            )
        return fit_ins

    # -------------------
    # Aggregate prompts
    # -------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        if not results:
            return None, {}

        prompts = []
        weights = []

        for client, fit_res in results:
            prompt = parameters_to_ndarrays(fit_res.parameters)[0]
            prompts.append(prompt)
            weights.append(fit_res.num_examples)

            # update pool
            self.prompt_pool[int(client.cid)] = prompt

        aggregated = aggregate(list(zip(prompts, weights)))
        return ndarrays_to_parameters([aggregated]), {}
