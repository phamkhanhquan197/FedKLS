from torch.utils.data import DataLoader

from mak.clients.base_client import BaseClient
from mak.utils.helper import get_optimizer
import torch

class FedAvgClient(BaseClient):
    """
    Simple flwr client implementation using basic fedavg approach
    """

    def __init__(
        self, client_id, model, trainset, valset, config_sim, device, save_dir
    ):
        super().__init__(
            client_id, model, trainset, valset, config_sim, device, save_dir
        )

    def __repr__(self) -> str:
        return " FedAvg client"

