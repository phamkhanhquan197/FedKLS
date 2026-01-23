from mak.clients.base_client import BaseClient

class FedAWAClient(BaseClient):
    """FedAWA client implementation iheriting from BaseClient."""

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir, kl_norm=None, dataset=None, apply_transforms=None, data_scheduler=None, bias=None, rank_policy_map=None,
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir, dataset=dataset, apply_transforms=apply_transforms, data_scheduler=data_scheduler, bias=bias, rank_policy_map=rank_policy_map,  
        )


    def __repr__(self) -> str:
        return "FedAWA client"


