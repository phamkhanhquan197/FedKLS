"""FedSVD strategy (Flower version).

This ports the *aggregation logic* from the 3rd-party `fed-svd` implementation
into Flower's Strategy API.

Key idea in upstream code:
- Clients train LoRA parameters.
- Server aggregates either LoRA deltas (FedAvg) or only a subset (FFA), with
  optional FLOra/FedEx-style cross-term updates.

Implementation notes for this repo:
- We aggregate over a *parameter subset* based on name matching ("lora_A"/
  "lora_B").
- Clients may send either full adapter params or adapter *diffs*; this strategy
  assumes clients send full params by default (FedAvg-like). If you want diff
  semantics, set `fedsvd.send_deltas: true` and ensure the client implements it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _is_lora_a(name: str) -> bool:
    return name.endswith(".A") or ("lora_A" in name) or ("lora_embedding_A" in name)


def _is_lora_b(name: str) -> bool:
    return name.endswith(".B") or ("lora_B" in name) or ("lora_embedding_B" in name)


def _is_classifier(name: str) -> bool:
    return "classifier" in name


class FedSVDStrategy(FedAvg):
    """FedSVD aggregation over LoRA parameters.

    Parameters
    ----------
    mode:
        - "fedavg": aggregate LoRA A and B (standard FedAvg on adapter params)
        - "ffa":    aggregate only LoRA B (Freeze-A, train/share B)
    agg_flora / agg_fedex:
        Optional aggregation variants from upstream fed-svd.
        These require clients to send *deltas* for A/B (theta_diff) computed
        against the round-start weights. Until the client is ported to provide
        that, keep them False.
    """

    def __init__(
        self,
        *,
        mode: str = "fedavg",
        agg_flora: bool = False,
        agg_fedex: bool = False,
        send_deltas: bool = False,
        recalculate_svd_period: int = 0,
        svd_warmup_steps: int = 0,
        include_classifier: bool = True,
        param_name_fn: Optional[Callable[[], List[str]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode.lower()
        self.agg_flora = agg_flora
        self.agg_fedex = agg_fedex
        self.send_deltas = send_deltas
        self.recalculate_svd_period = int(recalculate_svd_period or 0)
        self.svd_warmup_steps = int(svd_warmup_steps or 0)
        self.include_classifier = include_classifier
        self.param_name_fn = param_name_fn

        self._round_start: Optional[List[np.ndarray]] = None

        if self.mode not in {"fedavg", "ffa"}:
            raise ValueError(f"FedSVDStrategy: unsupported mode '{mode}'")

        if self.agg_flora and self.agg_fedex:
            raise ValueError("FedSVDStrategy: only one of agg_flora/agg_fedex can be true")

    def _reinit_svd(self, server_round: int, aggregated: List[np.ndarray], names: Optional[List[str]]) -> List[np.ndarray]:
        """Optionally rerun SVD to reinitialize LoRA A/B (upstream fed-svd behavior).

        Upstream logic (3rd-party/fed-svd): after aggregating and updating the global model,
        the server periodically calls `reinit_lora(model)`:
          - compute prod = B @ A
          - SVD(prod) = V S U^T
          - set A <- U^T[:r]
          - set B <- V[:,:r] diag(S[:r])

        In this Flower port we usually don't hold the actual torch model on the server
        Strategy, so we apply the same transformation directly on the aggregated
        parameter list using name-based pairing.
        """

        if self.recalculate_svd_period <= 0:
            return aggregated
        if server_round <= self.svd_warmup_steps:
            return aggregated
        if (server_round % self.recalculate_svd_period) != 0:
            return aggregated
        if not names:
            # Can't identify LoRA A/B tensors without names.
            return aggregated

        # Aggregated list may be a prefix of `names` (PEFT partial params). Clamp for safety.
        eff_len = min(len(aggregated), len(names))
        names_eff = names[:eff_len]
        out = list(aggregated[:eff_len])

        # Build quick index of A/B tensors by their "base" prefix.
        idx_a: Dict[str, int] = {}
        idx_b: Dict[str, int] = {}
        for i, nm in enumerate(names_eff):
            if "lora_A" in nm or nm.endswith(".A") or "lora_embedding_A" in nm:
                idx_a[nm.rsplit(".", 1)[0]] = i
            elif "lora_B" in nm or nm.endswith(".B") or "lora_embedding_B" in nm:
                idx_b[nm.rsplit(".", 1)[0]] = i

        # Reinit each pair that exists.
        for base, ia in idx_a.items():
            ib = idx_b.get(base)
            if ib is None:
                continue

            A = out[ia]
            B = out[ib]

            # Only handle 2D matrices (LoRA weights). Skip embeddings/odd shapes safely.
            if not (isinstance(A, np.ndarray) and isinstance(B, np.ndarray)):
                continue
            if A.ndim != 2 or B.ndim != 2:
                continue
            if B.shape[1] != A.shape[0]:
                # Expected shapes: A (r, in), B (out, r)
                continue

            prod = B @ A  # (out, in)
            try:
                U, S, Vt = np.linalg.svd(prod, full_matrices=False)
            except Exception:
                continue

            r = A.shape[0]
            # U: (out, k), S: (k,), Vt: (k, in)
            Ur = U[:, :r]
            Sr = S[:r]
            Vtr = Vt[:r, :]

            A_new = Vtr
            B_new = Ur @ (np.diag(Sr).astype(B.dtype, copy=False))

            out[ia] = A_new.astype(A.dtype, copy=False)
            out[ib] = B_new.astype(B.dtype, copy=False)

        # Keep any tail (when aggregated is longer than names, which shouldn't happen)
        if len(aggregated) > eff_len:
            out.extend(aggregated[eff_len:])
        return out

    def configure_fit(self, server_round, parameters, client_manager):
        # Capture round-start global params (needed if clients send deltas)
        try:
            self._round_start = parameters_to_ndarrays(parameters)
        except Exception:
            self._round_start = None
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Client results
        weights: List[List[np.ndarray]] = [parameters_to_ndarrays(r.parameters) for _, r in results]
        num_examples = [r.num_examples for _, r in results]

        names = self.param_name_fn() if self.param_name_fn else None
        aggregated = self._aggregate_selected(weights, num_examples, names, delta_mode=self.send_deltas)

        if self.send_deltas and self._round_start is not None and len(self._round_start) == len(aggregated):
            updated = [base + delta for base, delta in zip(self._round_start, aggregated)]
            updated = self._reinit_svd(server_round, updated, names)
            return ndarrays_to_parameters(updated), {}

        aggregated = self._reinit_svd(server_round, aggregated, names)

        return ndarrays_to_parameters(aggregated), {}

    def _aggregate_selected(
        self,
        weights: List[List[np.ndarray]],
        num_examples: List[int],
        names: Optional[List[str]],
        *,
        delta_mode: bool,
    ) -> List[np.ndarray]:
        """Aggregate ndarrays with selection mask based on parameter names."""

        # Clients in this repo often send *partial* parameter lists when PEFT is enabled.
        # That means `len(weights[i])` can differ across clients and also differ from
        # `len(names)` (which comes from the full `model.state_dict()` on the server).
        #
        # To avoid misalignment and IndexError, we only aggregate indices that are
        # present for *all* clients. For the rest we emit:
        # - zeros (delta_mode) so base weights stay unchanged
        # - first-available tensor (param_mode) as a best-effort fallback
        #
        # The robust solution is to also send parameter *names* from clients; but until
        # then, this keeps the simulation running and preserves semantics when clients
        # return identical partial vectors.
        min_len = min(len(w) for w in weights) if weights else 0

        # Fall back to plain FedAvg if we don't have parameter names.
        # (Without names we can't reliably select LoRA A/B subsets.)
        if names is None:
            total = float(sum(num_examples))
            ratios = [n / total for n in num_examples]
            if min_len == 0:
                return []
            avg = [sum(ratios[i] * weights[i][j] for i in range(len(weights))) for j in range(min_len)]
            if delta_mode and self._round_start is not None and len(self._round_start) == len(avg):
                # If we can't select by name, treat all layers as deltas.
                return avg
            return avg

        # If we have names but some clients returned fewer tensors than `names`,
        # clamp the aggregation to the intersection prefix to avoid index errors.
        names_eff = names[:min_len]

        adapter_prefixes = {
            nm.rsplit(".", 1)[0]
            for nm in names_eff
            if nm.endswith(".A") or nm.endswith(".B")
        }

        def _is_adapter_bias(nm: str) -> bool:
            return nm.endswith(".bias") and nm.rsplit(".", 1)[0] in adapter_prefixes

        # Build selection mask
        def should_aggregate(nm: str) -> bool:
            if _is_adapter_bias(nm):
                return True
            if self.include_classifier and _is_classifier(nm):
                return True
            if self.mode == "fedavg":
                return _is_lora_a(nm) or _is_lora_b(nm)
            # ffa
            return _is_lora_b(nm)

        sel = [should_aggregate(nm) for nm in names_eff]

        total = float(sum(num_examples))
        ratios = [n / total for n in num_examples]

        # Weighted average for selected params.
        # For non-selected params:
        # - delta_mode: emit zeros (so base weights stay unchanged)
        # - param_mode: copy from first client
        out: List[np.ndarray] = []
        for j, do_aggr in enumerate(sel):
            if do_aggr:
                layer = sum(ratios[i] * weights[i][j] for i in range(len(weights)))
                out.append(layer)
            else:
                out.append(np.zeros_like(weights[0][j]) if delta_mode else weights[0][j])
        return out
