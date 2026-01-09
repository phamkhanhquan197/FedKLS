from __future__ import annotations

from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
class FlexLoRAStrategy(FedAvg):
    """FlexLoRA FedAvg strategy.

    Round 1: send full model.
    Round >1: aggregate partial payload defined by `get_ffa_target_keys(model)`.

    LoRA aggregation is done in ΔW-space: ΔW_i = A_i @ B_i, then SVD(ΔW_agg) on server.
    """

    def __init__(self, *, config: dict, model, rank_map: dict[int, int], global_rank: int, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config
        self.model = model
        self.rank_map = rank_map
        self.global_rank = int(global_rank)


    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        # Send FULL state_dict on round 1 (consistent with project baseline behavior)
        full = [val.detach().cpu().numpy() for val in self.model.state_dict().values()]
        return ndarrays_to_parameters(full)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates for FlexLoRA.

        Protocol (as confirmed):
        - Round 1 downlink: FULL state_dict (handled by initialize_parameters / BaseClient).
        - Round >1 uplink: client sends ALL trainable params as a *partial* named-less list.
          This includes: .A, .B, and standard trainables (bias/head/classifier).
        - Round >1 downlink: server sends PARTIAL list for the same target keys.

        Math fix (critical):
        - Do NOT avg(A) and avg(B) independently.
        - Correct: per layer compute ΔW_i = A_i @ B_i, then ΔW_agg = Σ w_i ΔW_i,
          then SVD(ΔW_agg) and energy-preserving reprojection to (A_new, B_new).
        """
        if not results:
            log(WARNING, f"Round {server_round}: No results to aggregate")
            return None, {}
        if not self.accept_failures and failures:
            log(WARNING, f"Round {server_round}: {len(failures)} client failures during fit")
            return None, {}

        # Round 1: clients typically return FULL state_dict. Use vanilla FedAvg.
        # (We keep this for compatibility with the existing infra.)
        first_nds = parameters_to_ndarrays(results[0][1].parameters)
        if len(first_nds) == len(self.model.state_dict()):
            aggregated, metrics = super().aggregate_fit(server_round, results, failures)
            return aggregated, metrics

        # Round > 1: partial payload aligned with get_ffa_target_keys(self.model)
        # Lazy import to avoid circular imports (helper -> server -> strategy -> helper).
        from mak.utils.helper import get_ffa_target_keys

        target_keys = get_ffa_target_keys(self.model)

        # Total examples for weighting
        n_total = sum(fit_res.num_examples for _, fit_res in results)
        if n_total <= 0:
            log(WARNING, f"Round {server_round}: Total num_examples is 0")
            return None, {}

        # Parse each client's payload into a dict[key] = ndarray for O(1) access.
        client_payloads: List[Tuple[int, int, Dict[str, np.ndarray]]] = []
        for client, fit_res in results:
            cid = int(client.cid)
            nds = parameters_to_ndarrays(fit_res.parameters)

            if len(nds) != len(target_keys):
                raise ValueError(
                    f"FlexLoRA: payload length mismatch. expected={len(target_keys)} got={len(nds)} for cid={cid}"
                )

            payload = {k: np.asarray(v) for k, v in zip(target_keys, nds)}
            client_payloads.append((cid, fit_res.num_examples, payload))

        # Split target keys into LoRA A/B keys and standard keys
        a_keys = [k for k in target_keys if k.endswith(".A")]
        b_keys_set = {k for k in target_keys if k.endswith(".B")}
        lora_bases = []
        for ak in a_keys:
            base = ak[:-2]
            bk = base + ".B"
            if bk in b_keys_set:
                lora_bases.append(base)

        lora_keys_set = set()
        for base in lora_bases:
            lora_keys_set.add(base + ".A")
            lora_keys_set.add(base + ".B")

        standard_keys = [k for k in target_keys if k not in lora_keys_set]

        # --- 1) Standard aggregation (bias/head/classifier/etc.) ---
        standard_agg: Dict[str, np.ndarray] = {}
        for k in standard_keys:
            agg = None
            for _, n_i, payload in client_payloads:
                w = n_i / n_total
                arr = payload[k]
                if agg is None:
                    agg = (arr.astype(np.float32) * w)
                else:
                    agg += (arr.astype(np.float32) * w)
            standard_agg[k] = agg
            del agg

        # --- 2) LoRA aggregation (SVD(Avg(A@B))) - layer-wise to save RAM ---
        lora_agg: Dict[str, np.ndarray] = {}

        # Choose device for SVD merge (GPU if available, else CPU)
        svd_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for base in lora_bases:
            a_key = base + ".A"
            b_key = base + ".B"

            delta_w_agg = None  # torch.Tensor [d_out, d_in]

            # Layer-wise accumulation over clients
            for _, n_i, payload in client_payloads:
                w = n_i / n_total

                A_i = torch.from_numpy(payload[a_key]).to(device=svd_device, dtype=torch.float32)
                B_i = torch.from_numpy(payload[b_key]).to(device=svd_device, dtype=torch.float32)

                # ΔW_i = A_i @ B_i
                dwi = A_i @ B_i

                if delta_w_agg is None:
                    delta_w_agg = dwi.mul(w)
                else:
                    delta_w_agg.add_(dwi, alpha=float(w))

                # free temps ASAP
                del A_i, B_i, dwi

            if delta_w_agg is None:
                continue

            # SVD merge on aggregated update
            U, S, Vh = torch.linalg.svd(delta_w_agg, full_matrices=False)

            r = min(int(self.global_rank), int(S.shape[0]))
            Ur = U[:, :r]
            Sr = S[:r]
            Vhr = Vh[:r, :]

            # Energy-preserving reprojection
            sqrtS = torch.diag(torch.sqrt(Sr + 1e-12))
            A_new = Ur @ sqrtS
            B_new = sqrtS @ Vhr

            # Store as numpy (server downlink is partial list)
            lora_agg[a_key] = A_new.detach().cpu().numpy()
            lora_agg[b_key] = B_new.detach().cpu().numpy()

            # free per-layer tensors
            del delta_w_agg, U, S, Vh, Ur, Sr, Vhr, sqrtS, A_new, B_new

        # --- 3) Build final aggregated payload in target_keys order ---
        out_nds: List[np.ndarray] = []
        for k in target_keys:
            if k in lora_agg:
                out_nds.append(lora_agg[k])
            else:
                out_nds.append(standard_agg[k])

        return ndarrays_to_parameters(out_nds), {}

