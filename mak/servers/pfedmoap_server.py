# mak/servers/pfedmoap_server.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from flwr.server.history import History
import timeit
import numpy as np

from logging import INFO, DEBUG
from flwr.common import Parameters, Scalar
from flwr.common.logger import log
from flwr.server.server import FitResultsAndFailures

from mak.servers.custom_server import ServerSaveData, fit_clients


def _bytes_of_expert_prompts_from_config(cfg: Dict[str, Any]) -> int:
    prompts = cfg.get("pfedmoap_expert_prompts", None)
    if not prompts:
        return 0
    total = 0
    for p in prompts:
        try:
            total += int(np.asarray(p, dtype=np.float32).nbytes)
        except Exception:
            continue
    return total


class PFedMoAPServer(ServerSaveData):
    """
    Same as ServerSaveData, but communication tracking includes:
      - broadcast parameters bytes
      - extra per-client config bytes (pfedmoap_expert_prompts)

    Also prints logs similar to custom_server.fit_round.
    """

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        curr_round_start_time = timeit.default_timer()

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        # If no clients selected
        if not client_instructions:
            log(INFO, "======================================Round %s======================================", server_round)
            log(INFO, "Start trainining: no clients selected, cancel")
            return None

        # -------------------------
        # Compute download payload
        # -------------------------
        # params bytes (server -> clients)
        param_bytes = sum(len(t) for t in self.parameters.tensors)

        # extra cfg bytes per client (expert prompts)
        cfg_bytes_total = 0
        for _, fitins in client_instructions:
            cfg_bytes_total += _bytes_of_expert_prompts_from_config(fitins.config)

        # total download = params broadcast to each client + cfg bytes
        num_clients = len(client_instructions)
        download_gb = (param_bytes * num_clients + cfg_bytes_total) / 1e9
        param_size_gb = param_bytes / 1e9

        # -------------------------
        # Tracker: ensure keys exist
        # -------------------------
        if server_round not in self.comm_tracker.per_round:
            self.comm_tracker.per_round[server_round] = {"upload": 0.0, "download": 0.0}
        else:
            self.comm_tracker.per_round[server_round].setdefault("upload", 0.0)
            self.comm_tracker.per_round[server_round].setdefault("download", 0.0)

        # set download explicitly
        self.comm_tracker.per_round[server_round]["download"] = float(download_gb)
        self.comm_tracker.total_download += float(download_gb)

        # -------------------------
        # Logs similar to custom_server
        # -------------------------
        log(INFO, "======================================Round %s======================================", server_round)
        log(INFO, f"Model size: {param_size_gb:.6f} GB = {param_size_gb*1024:.6f} MB")
        log(
            INFO,
            "Round %s download: params_bytes=%s, cfg_bytes=%s, total=%.6f GB",
            server_round,
            param_bytes,
            cfg_bytes_total,
            download_gb,
        )
        log(
            DEBUG,
            "Start training: sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # -------------------------
        # Fit selected clients
        # -------------------------
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            num_threads=self.num_train_thread,
        )

        # -------------------------
        # Compute upload payload
        # -------------------------
        upload_bytes_total = 0
        upload_size_gb = 0.0

        for client_proxy, fit_res in results:
            # per-client upload tracking (if structure exists)
            client_upload_gb = 0.0
            if fit_res.parameters is not None and fit_res.parameters.tensors is not None:
                client_upload_gb = sum(len(t) for t in fit_res.parameters.tensors) / 1e9

            upload_bytes_total += int(client_upload_gb * 1e9)
            upload_size_gb += client_upload_gb

            # update comm_tracker.per_client if available
            try:
                if hasattr(self.comm_tracker, "per_client"):
                    if client_proxy.cid in self.comm_tracker.per_client:
                        self.comm_tracker.per_client[client_proxy.cid].setdefault("upload", 0.0)
                        self.comm_tracker.per_client[client_proxy.cid]["upload"] += client_upload_gb
            except Exception:
                pass

        # set upload explicitly
        self.comm_tracker.per_round[server_round]["upload"] = float(upload_size_gb)
        self.comm_tracker.total_upload += float(upload_size_gb)

        log(
            INFO,
            f"Round {server_round} upload size: {upload_size_gb:.6f} GB = {upload_size_gb*1024:.6f} MB, "
            f"download size: {download_gb:.6f} GB = {download_gb*1024:.6f} MB, "
            f"total: {(upload_size_gb + download_gb):.6f} GB = {(upload_size_gb + download_gb)*1024:.6f} MB"
        )

        log(
            DEBUG,
            "Server training with %s results and %s failures",
            len(results),
            len(failures),
        )
        
  
        for i in range(len(results)):
            client_proxy = results[i][0]
            fit_res = results[i][1]
            metrics = fit_res.metrics or {}

            client_id = metrics.get("client_id", None)
            if client_id is None:
                client_id = getattr(client_proxy, "cid", "unknown")

            train_samples = fit_res.num_examples

            class_dist = metrics.get("class_distribution", None)
            if class_dist is None:
                log(INFO, "Client %s (Total training samples: %s, Class Distribution: N/A)", client_id, train_samples)
                continue

            try:
                num_class = len(class_dist)
            except Exception:
                num_class = "unknown"

            log(
                INFO,
                "Client %s (Total training samples: %s, Class Distribution (%s classes): %s)",
                client_id,
                train_samples,
                num_class,
                class_dist,
            )

        # -------------------------
        # Aggregate
        # -------------------------
        parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(
            server_round, results, failures
        )

        # You can add this line if you want an explicit processing time log for training round:
        # log(INFO, "Round %s fit_round time: %.6fs", server_round, timeit.default_timer() - curr_round_start_time)

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Override to return History (Flower expects History, not tuple)."""
        res = super().fit(num_rounds=num_rounds, timeout=timeout)

        # Your ServerSaveData.fit returns (History, total_time) in this repo
        if isinstance(res, tuple):
            history = res[0]
            # optional: keep total time for later use
            try:
                self.last_fit_seconds = float(res[1])
            except Exception:
                self.last_fit_seconds = None
            return history

        # In case upstream changes and returns History directly
        return res