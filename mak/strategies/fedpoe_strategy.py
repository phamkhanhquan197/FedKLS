from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _num_fit_clients(num_available: int, fraction_fit: float, min_fit_clients: int) -> Tuple[int, int]:
    sample_size = max(int(num_available * fraction_fit), min_fit_clients)
    return sample_size, min_fit_clients


def _num_evaluate_clients(
    num_available: int, fraction_evaluate: float, min_evaluate_clients: int
) -> Tuple[int, int]:
    sample_size = max(int(num_available * fraction_evaluate), min_evaluate_clients)
    return sample_size, min_evaluate_clients


class FedPOEStrategy(FedAvg):
    """Fed-POE (Hedge mixture) strategy.

    This is a Flower-side port of the *mixture weight update* idea from the
    3rd-party Fed-POE code (e.g. cifar10.py): each client maintains two weights
    (a,b) which represent trust in:
      - the federated/global model (a)
      - the client's local/personalized model (b)

    Each round, clients report two losses in EvaluateRes.metrics:
      - loss_fed
      - loss_loc
    and we update:
      a <- a * exp(-eta * loss_fed)
      b <- b * exp(-eta * loss_loc)

    We also compute a mixed loss proxy:
      loss_mix = (a*loss_fed + b*loss_loc)/(a+b)
    and return it as an aggregated metric.

    NOTE: This strategy still performs standard parameter aggregation for
    training (FedAvg). The mixture is only used for reporting/selection.
    """

    def __init__(
        self,
        *,
        eta: float = 0.0,
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eta = float(eta)
        self.eps = float(eps)
        # Per-client weights (initialized lazily)
        self._a: Dict[str, float] = {}
        self._b: Dict[str, float] = {}

    def _ensure_client(self, cid: str) -> None:
        if cid not in self._a:
            self._a[cid] = 1.0
        if cid not in self._b:
            self._b[cid] = 1.0

    @staticmethod
    def _safe_float(x) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        # Let FedAvg compute the standard aggregated loss/metrics
        loss_avg, metrics_avg = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            return loss_avg, metrics_avg

        # Update per-client (a,b) and compute mixed-loss per client
        mix_losses = []
        fed_losses = []
        loc_losses = []

        for client, res in results:
            cid = client.cid
            self._ensure_client(cid)

            loss_fed = self._safe_float(res.metrics.get("loss_fed")) if res.metrics else None
            loss_loc = self._safe_float(res.metrics.get("loss_loc")) if res.metrics else None
            if loss_fed is None:
                loss_fed = self._safe_float(res.loss)
            if loss_loc is None:
                # If local loss isn't provided, fall back to fed loss
                loss_loc = loss_fed

            if loss_fed is None or loss_loc is None:
                continue

            # Hedge update
            if self.eta > 0:
                try:
                    self._a[cid] *= math.exp(-self.eta * loss_fed)
                    self._b[cid] *= math.exp(-self.eta * loss_loc)
                except OverflowError:
                    # If overflow happens, renormalize down
                    self._a[cid] = max(self._a[cid], self.eps)
                    self._b[cid] = max(self._b[cid], self.eps)

            denom = max(self._a[cid] + self._b[cid], self.eps)
            loss_mix = (self._a[cid] * loss_fed + self._b[cid] * loss_loc) / denom

            mix_losses.append((res.num_examples, loss_mix))
            fed_losses.append((res.num_examples, loss_fed))
            loc_losses.append((res.num_examples, loss_loc))

        def wavg(pairs: List[Tuple[int, float]]) -> Optional[float]:
            if not pairs:
                return None
            total = sum(n for n, _ in pairs)
            if total <= 0:
                return None
            return sum(n * v for n, v in pairs) / total

        poe_loss_mix = wavg(mix_losses)
        poe_loss_fed = wavg(fed_losses)
        poe_loss_loc = wavg(loc_losses)

        out = dict(metrics_avg)
        if poe_loss_mix is not None:
            out["poe_loss_mix"] = poe_loss_mix
        if poe_loss_fed is not None:
            out["poe_loss_fed"] = poe_loss_fed
        if poe_loss_loc is not None:
            out["poe_loss_loc"] = poe_loss_loc

        return loss_avg, out


class FedPOERegressionTextStrategy(fl.server.strategy.Strategy):
    """Fed-POE regression-style strategy for text.

    Server state is a single parameter vector: theta (flattened all kernels).

    Clients return (in FitRes.metrics):
      - poe_grads: np.ndarray shape (K, theta_dim)
      - poe_w:     np.ndarray shape (K,)

    The server aggregates grads (weighted by client examples and optionally w)
    and performs one simple SGD step:
      theta <- theta - lr * grad

    We re-use `server_learning_rate` from config or default to 1e-1.
    """

    def __init__(
        self,
        *,
    initial_theta: Optional[object] = None,
        server_learning_rate: float = 0.1,
        eta: float = 0.0,
        lam: float = 0.0,
        num_kernels: int = 4,
        n_components: int = 256,
        pooling: str = "auto",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        # Local import so fedpoe_strategy.py stays lightweight for users who
        # don't enable FedPOERegressionText.
        import numpy as np  # type: ignore
        from flwr.common.parameter import parameters_to_ndarrays

        self.server_learning_rate = float(server_learning_rate)
        self.eta = float(eta)
        self.lam = float(lam)
        self.num_kernels = int(num_kernels)
        self.n_components = int(n_components)
        self.pooling = str(pooling)

        self.fraction_fit = float(fraction_fit)
        self.fraction_evaluate = float(fraction_evaluate)
        self.min_fit_clients = int(min_fit_clients)
        self.min_evaluate_clients = int(min_evaluate_clients)
        self.min_available_clients = int(min_available_clients)
        self.accept_failures = bool(accept_failures)

        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn

        self._theta: Optional[np.ndarray] = None
        if initial_theta is not None:
            self._theta = np.asarray(initial_theta, dtype=np.float32)
        elif initial_parameters is not None and len(initial_parameters.tensors) > 0:
            arrs = parameters_to_ndarrays(initial_parameters)
            if arrs:
                self._theta = np.asarray(arrs[0], dtype=np.float32)

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        from flwr.common.parameter import ndarrays_to_parameters

        if self._theta is None:
            # Unknown until first client creates head; send empty.
            return ndarrays_to_parameters([])
        return ndarrays_to_parameters([self._theta])

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        sample_size, min_num_clients = _num_fit_clients(
            client_manager.num_available(), self.fraction_fit, self.min_fit_clients
        )
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config: Dict[str, Scalar] = {"round": server_round, "strategy": "FedPOERegressionText"}
        if self.on_fit_config_fn is not None:
            config.update(self.on_fit_config_fn(server_round))

        return [(c, fl.common.FitIns(parameters, config)) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        # Local import so strategy file doesn't hard-depend on numpy unless used.
        import numpy as np  # type: ignore
        from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays

        if not results:
            return None, {}
        if failures and not self.accept_failures:
            return None, {}

        # Initialize theta from first client if needed
        if self._theta is None:
            first_params = parameters_to_ndarrays(results[0][1].parameters)
            if first_params:
                self._theta = np.asarray(first_params[0], dtype=np.float32)

        if self._theta is None:
            return None, {}

        total_examples = 0
        grad_accum = None

        for _, fit_res in results:
            n = int(getattr(fit_res, "num_examples", 0) or 0)
            total_examples += n

            m = fit_res.metrics or {}
            grads = m.get("poe_grads")
            w = m.get("poe_w")

            if grads is None:
                continue

            grads = np.asarray(grads, dtype=np.float32)  # (K, theta_dim)
            if w is not None:
                w = np.asarray(w, dtype=np.float32).reshape(-1, 1)
                if w.shape[0] == grads.shape[0]:
                    grads = grads * w

            # Sum over kernels to get global grad vector
            g = grads.sum(axis=0)  # (theta_dim,)

            if grad_accum is None:
                grad_accum = n * g
            else:
                grad_accum += n * g

        if grad_accum is None or total_examples <= 0:
            return ndarrays_to_parameters([self._theta]), {}

        grad_avg = grad_accum / float(total_examples)
        self._theta = self._theta - self.server_learning_rate * grad_avg

        metrics: Dict[str, Scalar] = {"server_lr": self.server_learning_rate}
        return ndarrays_to_parameters([self._theta]), metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        sample_size, min_num_clients = _num_evaluate_clients(
            client_manager.num_available(), self.fraction_evaluate, self.min_evaluate_clients
        )
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config: Dict[str, Scalar] = {"round": server_round, "strategy": "FedPOERegressionText"}
        if self.on_evaluate_config_fn is not None:
            config.update(self.on_evaluate_config_fn(server_round))

        return [(c, fl.common.EvaluateIns(parameters, config)) for c in clients]

    def aggregate_evaluate(self, server_round: int, results, failures):
        # Weighted average evaluation loss
        if not results:
            return None, {}
        total = sum(r.num_examples for _, r in results)
        if total <= 0:
            return None, {}

        loss = sum(r.num_examples * r.loss for _, r in results) / total
        # pass through any scalar metrics that look like accuracy
        accs = []
        for _, r in results:
            if r.metrics and "accuracy" in r.metrics:
                try:
                    accs.append((r.num_examples, float(r.metrics["accuracy"])))
                except Exception:
                    pass
        metrics: Dict[str, Scalar] = {}
        if accs:
            metrics["accuracy"] = sum(n * a for n, a in accs) / total
        return loss, metrics

    def evaluate(self, server_round: int, parameters: Parameters):
        from flwr.common.parameter import parameters_to_ndarrays

        if self.evaluate_fn is None:
            return None
        theta = parameters_to_ndarrays(parameters)
        if not theta:
            return None
        return self.evaluate_fn(server_round, theta, {})
