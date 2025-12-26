# mak/strategies/pfedmoap_strategy.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy


class PFedMoAP(fl.server.strategy.FedAvg):
    """
    pFedMoAP strategy (Phase 2 server side):
      - global model parameters in Flower = global prompt only
      - maintain prompt_pool[cid] = last uploaded local prompt
      - configure_fit sends non-local prompts to each client via FitIns.config
      - aggregate_fit performs FedAvg on prompts to get new global prompt
      - gating stays local, never aggregated
    """

    def __init__(
        self,
        *,
        config_sim: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.config_sim = config_sim
        cfg = config_sim.get("pfedmoap_config", {})
        self.prompt_len = int(cfg.get("prompt_len", cfg.get("num_tokens", 16)))
        self.prompt_dim = int(cfg.get("prompt_dim", 768))

        # How many non-local experts to send
        self.non_local_k = int(cfg.get("non_local_k", 4))

        # prompt_pool stores numpy arrays with shape [L, D]
        self.prompt_pool: Dict[str, np.ndarray] = {}

        # Cache latest global prompt as numpy array [L, D]
        self.global_prompt: Optional[np.ndarray] = None

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        # Use initial_parameters passed from helper.get_strategy
        return self.initial_parameters

    def _params_to_prompt(self, params: Parameters) -> np.ndarray:
        nds = fl.common.parameters_to_ndarrays(params)
        if len(nds) != 1:
            raise ValueError(f"PFedMoAP expects exactly 1 ndarray (prompt), got {len(nds)}")
        prompt = nds[0]
        if tuple(prompt.shape) != (self.prompt_len, self.prompt_dim):
            raise ValueError(
                f"Prompt shape mismatch. Expected {(self.prompt_len, self.prompt_dim)}, got {tuple(prompt.shape)}"
            )
        return prompt

    def _prompt_to_params(self, prompt: np.ndarray) -> Parameters:
        return fl.common.ndarrays_to_parameters([prompt.astype(np.float32, copy=False)])

    def _select_non_local_prompts(self, target_cid: str) -> np.ndarray:
        """
        Select K prompts from pool excluding target_cid.
        Return shape [K, L, D], may be empty [0, L, D] if pool is empty.
        """
        # candidates are other clients that already have a prompt in pool
        candidates = [cid for cid in self.prompt_pool.keys() if cid != target_cid]
        if len(candidates) == 0 or self.non_local_k <= 0:
            return np.zeros((0, self.prompt_len, self.prompt_dim), dtype=np.float32)

        # simple deterministic selection: sort then take first K
        candidates = sorted(candidates)
        chosen = candidates[: min(self.non_local_k, len(candidates))]
        stacked = np.stack([self.prompt_pool[cid] for cid in chosen], axis=0).astype(np.float32, copy=False)
        return stacked

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Let FedAvg pick clients and build FitIns (this keeps your pipeline unchanged)
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)

        # Update cached global prompt
        try:
            self.global_prompt = self._params_to_prompt(parameters)
        except Exception:
            # If this happens, it means helper initial_parameters is still wrong
            self.global_prompt = None

        # Inject non-local prompts into config per client
        out: List[Tuple[ClientProxy, FitIns]] = []
        for client, fitins in fit_instructions:
            cid = str(client.cid)

            non_local = self._select_non_local_prompts(target_cid=cid)

            # Important: do not overwrite existing config fields
            new_cfg = dict(fitins.config)
            # nested lists to stay JSON serializable
            new_cfg["pfedmoap_non_local_prompts"] = non_local.tolist()

            out.append((client, FitIns(fitins.parameters, new_cfg)))
        return out

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # First let FedAvg aggregate the prompt
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        # Update pool with each participating client's uploaded prompt (local prompt)
        for client, fitres in results:
            cid = str(client.cid)
            try:
                prompt = self._params_to_prompt(fitres.parameters)
                self.prompt_pool[cid] = prompt
            except Exception:
                # skip malformed
                continue

        # Update cached global prompt
        if aggregated_params is not None:
            try:
                self.global_prompt = self._params_to_prompt(aggregated_params)
            except Exception:
                pass

        return aggregated_params, metrics
