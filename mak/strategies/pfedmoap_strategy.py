# mak/strategies/pfedmoap_strategy.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


class PFedMoAPStrategy(FedAvg):
    """
    Federated Prompt Learning strategy for CLIP (pFedMoAP).
    Server state:
      1) global prompt (Flower Parameters)
      2) prompt_pool: cid -> prompt ndarray (prompt_len, prompt_dim)
    Round behavior:
      1) configure_fit sends global prompt and per client nonlocal expert prompts
      2) aggregate_fit aggregates only prompt and updates prompt_pool
    """

    def __init__(self, *, config: Dict, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config
        pf = config["pfedmoap_config"]

        self.prompt_len = int(pf["prompt_len"])
        self.prompt_dim = int(pf["prompt_dim"])

        self.num_experts = int(pf.get("num_experts", pf.get("K", 4)))
        if self.num_experts < 1:
            self.num_experts = 1

        # selection mode: "knn" or "random"
        self.selection = str(pf.get("selection", "knn")).lower()
        self.seed = int(pf.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        # server side prompt pool
        self.prompt_pool: Dict[int, np.ndarray] = {}

        # optional: warmup behavior, if pool too small
        self.allow_random_when_insufficient = bool(pf.get("allow_random_when_insufficient", True))

    def _ensure_prompt_shape(self, prompt: np.ndarray) -> np.ndarray:
        prompt = np.asarray(prompt, dtype=np.float32)
        if prompt.shape != (self.prompt_len, self.prompt_dim):
            raise ValueError(
                f"Prompt shape mismatch, got {prompt.shape}, expected {(self.prompt_len, self.prompt_dim)}"
            )
        return prompt

    def _select_expert_prompts(self, cid: int) -> List[np.ndarray]:
        # If pool not ready, return empty
        if len(self.prompt_pool) <= 1:
            return []

        # If cid missing in pool, cold start
        if cid not in self.prompt_pool:
            if not self.allow_random_when_insufficient:
                return []
            # random selection from pool excluding cid (cid not in pool anyway)
            keys = list(self.prompt_pool.keys())
            if not keys:
                return []
            k = min(self.num_experts - 1, len(keys))
            if k <= 0:
                return []
            chosen = self.rng.choice(keys, size=k, replace=False)
            return [self.prompt_pool[int(x)] for x in chosen]

        # Normal case: cid exists in pool
        if self.num_experts <= 1:
            return []

        keys = [k for k in self.prompt_pool.keys() if k != cid]
        if not keys:
            return []

        k_need = min(self.num_experts - 1, len(keys))
        if k_need <= 0:
            return []

        if self.selection == "random":
            chosen = self.rng.choice(keys, size=k_need, replace=False)
            return [self.prompt_pool[int(x)] for x in chosen]

        # KNN selection with L2 distance between flattened prompts
        q = self.prompt_pool[cid].reshape(-1)
        dists: List[Tuple[float, int]] = []
        for k in keys:
            p = self.prompt_pool[k].reshape(-1)
            dist = float(np.linalg.norm(q - p))
            dists.append((dist, k))
        dists.sort(key=lambda x: x[0])
        chosen_ids = [k for _, k in dists[:k_need]]
        return [self.prompt_pool[int(x)] for x in chosen_ids]

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        # In your pipeline you already pass initial_parameters from get_strategy().
        # This is a safe fallback if initial_parameters is not provided.
        prompt0 = (0.02 * np.random.randn(self.prompt_len, self.prompt_dim)).astype(np.float32)
        return ndarrays_to_parameters([prompt0])

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Let FedAvg decide sampled clients (fraction_fit, min_fit_clients, etc.)
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        if not client_instructions:
            return []

        global_prompt = parameters_to_ndarrays(parameters)[0]
        global_prompt = self._ensure_prompt_shape(global_prompt)

        new_instructions: List[Tuple[ClientProxy, FitIns]] = []
        for client, fitins in client_instructions:
            cid = int(client.cid)

            expert_prompts = self._select_expert_prompts(cid)
            expert_prompts = [self._ensure_prompt_shape(p) for p in expert_prompts]

            cfg = dict(fitins.config) if fitins.config is not None else {}
            cfg["current_round"] = server_round
            cfg["pfedmoap_has_experts"] = bool(len(expert_prompts) > 0)
            cfg["pfedmoap_expert_prompts"] = [p.tolist() for p in expert_prompts]
            cfg["pfedmoap_num_experts"] = len(expert_prompts) + 1  # local plus nonlocal

            new_fitins = FitIns(
                parameters=ndarrays_to_parameters([global_prompt]),
                config=cfg,
            )
            new_instructions.append((client, new_fitins))

        return new_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if (not self.accept_failures) and failures:
            return None, {}

        # Extract prompt updates (single tensor per client)
        prompts_and_weights: List[Tuple[np.ndarray, int]] = []
        for client, fit_res in results:
            cid = int(client.cid)
            nds = parameters_to_ndarrays(fit_res.parameters)
            if len(nds) != 1:
                raise ValueError(f"PFedMoAP expects 1 tensor (prompt) from client {cid}, got {len(nds)}")

            prompt_i = self._ensure_prompt_shape(nds[0])
            prompts_and_weights.append((prompt_i, fit_res.num_examples))

            # Update server prompt pool with latest client prompt
            self.prompt_pool[cid] = prompt_i

        # FedAvg on prompt only
        aggregated_prompt = aggregate(prompts_and_weights)
        aggregated_parameters = ndarrays_to_parameters([aggregated_prompt])

        # Metrics
        metrics: Dict[str, Scalar] = {}
        metrics["pfedmoap_pool_size"] = int(len(self.prompt_pool))
        metrics["pfedmoap_round"] = int(server_round)

        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            try:
                metrics.update(self.fit_metrics_aggregation_fn(fit_metrics))
            except Exception as e:
                log("WARNING", f"fit_metrics_aggregation_fn failed: {e}")

        return aggregated_parameters, metrics
