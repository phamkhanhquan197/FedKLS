from mak.clients.base_client import BaseClient

class FedAvgClient(BaseClient):
    """
    Simple flwr client implementation using basic fedavg approach
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir, kl_norm=None
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir
        )

    def __repr__(self) -> str:
        return " FedAvg client"

