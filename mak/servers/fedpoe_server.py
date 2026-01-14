from __future__ import annotations

from typing import Optional

from flwr.server.client_manager import ClientManager

from mak.servers.custom_server import ServerSaveData
from mak.strategies.fedpoe_strategy import FedPOEStrategy


class FedPOEServer(ServerSaveData):
    """Server wrapper for FedPOE.

    Today it reuses the existing `ServerSaveData` orchestration.
    We keep this class so `helper.get_server(...)` can select a dedicated server
    type (mirrors how other algorithms in this repo have custom servers).
    """

    def __init__(
        self,
        *,
        strategy: FedPOEStrategy,
        client_manager: ClientManager,
        out_file_path=None,
        target_acc=0.99,
        num_train_thread: int = 1,
        num_test_thread: int = 1,
    ) -> None:
        super().__init__(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )


class FedPOERegressionTextServer(ServerSaveData):
    """Server wrapper for FedPOERegressionText.

    Reuses the existing ServerSaveData orchestration.
    """

    def __init__(
        self,
        *,
        strategy,
        client_manager: ClientManager,
        out_file_path=None,
        target_acc=0.99,
        num_train_thread: int = 1,
        num_test_thread: int = 1,
    ) -> None:
        super().__init__(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
