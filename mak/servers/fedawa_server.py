from mak.servers.custom_server import ServerSaveData
from mak.strategies.fedawa_strategy import FedAWAStrategy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from typing import Optional

class FedAWAServer(ServerSaveData):
    """FedAWA server implementation inheriting from ServerSaveData."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        model,
        test_data,
        apply_transforms_test,
        config,
        device="cuda",
        out_file_path=None,
        target_acc=0.99,
        num_train_thread=1,
        num_test_thread=1,
    ) -> None:
        # Initialize FedAWA strategy if none provided
        if strategy is None:
            strategy = FedAWAStrategy(
                model=model,
                test_data=test_data,
                apply_transforms_test=apply_transforms_test,
                config=config,
                device=device,
                fraction_fit=config["server"]["fraction_fit"],
                fraction_evaluate=config["server"]["fraction_evaluate"],
                min_fit_clients=config["server"]["min_fit_clients"],
                min_evaluate_clients=config["server"]["min_evaluate_clients"],
            )
        super().__init__(
            client_manager=client_manager,
            strategy=strategy,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )