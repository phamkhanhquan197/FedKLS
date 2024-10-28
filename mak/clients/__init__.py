import flwr as fl
from flwr_datasets import FederatedDataset

from mak.clients.fedavg_client import FedAvgClient
from mak.clients.fedprox_client import FedProxClient


def get_client_fn(
    config_sim: dict, dataset: FederatedDataset, model, device, apply_transforms
):
    strategy = config_sim["server"]["strategy"]
    client_class = get_client_class(strategy)
    train_batch_size = config_sim["client"]["batch_size"]
    test_batch_size = config_sim["client"]["test_batch_size"]

    def client_fn(cid: str) -> fl.client.Client:
        client_dataset = dataset.load_partition(int(cid), "train")
        client_dataset_splits = client_dataset.train_test_split(test_size=0.15)
        trainset = client_dataset_splits["train"].with_transform(apply_transforms)
        valset = client_dataset_splits["test"].with_transform(apply_transforms)
        return client_class(
            model=model,
            trainset=trainset,
            valset=valset,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            device=device,
        ).to_client()

    return client_fn


def get_client_class(strategy: str):
    if strategy == "fedprox":
        return FedProxClient
    else:
        return FedAvgClient
