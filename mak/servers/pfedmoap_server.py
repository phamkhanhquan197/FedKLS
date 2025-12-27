# mak/servers/pfedmoap_server.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from flwr.common import Parameters, Scalar
from flwr.common.logger import log
from logging import INFO

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
    """

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ):
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "Start training: no clients selected, cancel")
            return None

        # Download bytes:
        # 1) parameters broadcast (global prompt)
        param_bytes = sum(len(t) for t in self.parameters.tensors)
        # 2) config payload per client (expert prompts)
        cfg_bytes_total = 0
        for _, fitins in client_instructions:
            cfg_bytes_total += _bytes_of_expert_prompts_from_config(fitins.config)

        download_gb = (param_bytes * len(client_instructions) + cfg_bytes_total) / 1e9
        self.comm_tracker.log_round(server_round=server_round, upload=0, download=download_gb)

        log(
            INFO,
            "Round %s download: params_bytes=%s, cfg_bytes=%s, total=%.6f GB",
            server_round,
            param_bytes,
            cfg_bytes_total,
            download_gb,
        )

        # Use the original parallel fit implementation from ServerSaveData
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            num_threads=self.num_train_thread,
        )

        # Upload bytes: sum of returned parameters from all successful clients
        upload_bytes_total = 0
        for _, fit_res in results:
            upload_bytes_total += sum(len(t) for t in fit_res.parameters.tensors)

        upload_gb = upload_bytes_total / 1e9
        self.comm_tracker.per_round[server_round]["upload"] = upload_gb
        self.comm_tracker.total_upload += upload_gb

        log(INFO, "Round %s upload: %.6f GB", server_round, upload_gb)

        parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        return parameters_aggregated, metrics_aggregated, (results, failures)
