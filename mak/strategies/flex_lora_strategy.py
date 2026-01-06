from __future__ import annotations

from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


class FlexLoRAStrategy(FedAvg):
    """FlexLoRA Strategy.

    Key constraints:
    - Strict inheritance: subclass FedAvg.
    - Math fidelity: client updates -> pad -> aggregate -> SVD -> (A,B) reprojection.

    Note: This implementation follows the Phase-2 plan and uses a simplified,
    model-agnostic approach: it assumes client payload is an ordered list of
    tensors containing only LoRA factors (A/B) in deterministic sorted-key order.

    For the first engineering iteration, we SVD-merge each corresponding A/B pair
    independently per tensor pair (A_i, B_i) by reconstructing DeltaW.

    The server returns aggregated parameters as a flat list in the same order.
    """

    def __init__(self, *, config: dict, model, rank_map: dict[int, int], global_rank: int, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config
        self.model = model
        self.rank_map = rank_map
        self.global_rank = int(global_rank)

        # Cache ordered keys for deterministic A/B mapping from server model
        self._ab_keys = [k for k in self.model.state_dict().keys() if k.endswith(".A") or k.endswith(".B")]
        self._ab_keys = sorted(self._ab_keys)

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        # Send FULL state_dict on round 1 (consistent with project baseline behavior)
        full = [val.detach().cpu().numpy() for val in self.model.state_dict().values()]
        return ndarrays_to_parameters(full)

    def _pad_to_global(self, t: torch.Tensor, key: str) -> torch.Tensor:
        """Pad A/B tensor to global rank.

        A: [out, r] -> pad columns to [out, R]
        B: [r, in]  -> pad rows to [R, in]
        """
        if key.endswith(".A"):
            out, r = t.shape
            if r >= self.global_rank:
                return t[:, : self.global_rank]
            pad_cols = self.global_rank - r
            return torch.nn.functional.pad(t, (0, pad_cols, 0, 0), mode="constant", value=0.0)
        if key.endswith(".B"):
            r, inn = t.shape
            if r >= self.global_rank:
                return t[: self.global_rank, :]
            pad_rows = self.global_rank - r
            return torch.nn.functional.pad(t, (0, 0, 0, pad_rows), mode="constant", value=0.0)
        return t

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            log(WARNING, f"Round {server_round}: No results to aggregate")
            return None, {}
        if not self.accept_failures and failures:
            log(WARNING, f"Round {server_round}: {len(failures)} client failures during fit")
            return None, {}

        # Convert each client payload into padded arrays for FedAvg
        padded_weights_results = []
        for client, fit_res in results:
            cid = int(client.cid)
            nds = parameters_to_ndarrays(fit_res.parameters)

            # If full model update is sent (round 1), fallback to FedAvg behavior
            if len(nds) == len(self.model.state_dict()):
                padded_weights_results.append((nds, fit_res.num_examples))
                continue

            # Otherwise assume it's A/B-only payload in server key order
            if len(nds) != len(self._ab_keys):
                raise ValueError(
                    f"FlexLoRA: payload length mismatch. expected={len(self._ab_keys)} got={len(nds)} for cid={cid}"
                )

            padded = []
            for key, arr in zip(self._ab_keys, nds):
                t = torch.from_numpy(np.asarray(arr)).float()
                t = self._pad_to_global(t, key)
                padded.append(t.cpu().numpy())

            padded_weights_results.append((padded, fit_res.num_examples))

        # FedAvg aggregation over padded tensors
        aggregated = aggregate(padded_weights_results)

        # SVD merge step (per A/B pair): reconstruct DeltaW, SVD, reproject
        # We operate on global-rank tensors.
        agg_tensors = [torch.from_numpy(np.asarray(x)).float() for x in aggregated]

        # Build mapping for aggregated A/B
        agg_sd = {}
        for k, t in zip(self._ab_keys, agg_tensors):
            agg_sd[k] = t

        # Reproject each (A,B) by SVD on DeltaW = A@B
        for key in list(agg_sd.keys()):
            if not key.endswith(".A"):
                continue
            base = key[:-2]
            a_key = base + ".A"
            b_key = base + ".B"
            if b_key not in agg_sd:
                continue

            A = agg_sd[a_key]
            B = agg_sd[b_key]
            delta = A @ B
            # SVD
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            r = min(self.global_rank, S.shape[0])
            Sr = S[:r]
            sqrtS = torch.diag(torch.sqrt(Sr + 1e-12))
            A_new = U[:, :r] @ sqrtS
            B_new = sqrtS @ Vh[:r, :]
            agg_sd[a_key] = A_new
            agg_sd[b_key] = B_new

        # Inject back into server model (only if these keys exist in model)
        model_sd = self.model.state_dict()
        with torch.no_grad():
            for k in self._ab_keys:
                if k not in model_sd:
                    continue
                t = agg_sd[k].to(device=model_sd[k].device, dtype=model_sd[k].dtype)
                # Align to server model's current rank (may differ); slice if needed
                if model_sd[k].shape != t.shape:
                    # minimal safe alignment
                    if k.endswith(".A"):
                        t = t[:, : model_sd[k].shape[1]]
                    elif k.endswith(".B"):
                        t = t[: model_sd[k].shape[0], :]
                model_sd[k].copy_(t)

        # Return FULL model snapshot to keep global sync consistent with existing infra
        full_snapshot = [val.detach().cpu().numpy() for val in self.model.state_dict().values()]
        return ndarrays_to_parameters(full_snapshot), {}

