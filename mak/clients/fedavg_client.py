from mak.clients.base_client import BaseClient

class FedAvgClient(BaseClient):
    """
    Simple flwr client implementation using basic fedavg approach
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir, kl_norm=None, dataset=None, apply_transforms=None
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir, dataset=dataset, apply_transforms=apply_transforms
        )

    def __repr__(self) -> str:
        return " FedAvg client"

