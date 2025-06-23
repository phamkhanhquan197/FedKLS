from mak.clients.base_client import BaseClient

class FedAWAClient(BaseClient):
    """FedAWA client implementation iheriting from BaseClient."""

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir, kl_norm=None
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir
        )


    def __repr__(self) -> str:
        return "FedAWA client"

