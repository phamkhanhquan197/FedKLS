import flwr as fl
from flwr_datasets import FederatedDataset

from mak.clients.fedavg_client import FedAvgClient
from mak.clients.fednova_client import FedNovaClient
from mak.clients.fedprox_client import FedProxClient
from mak.clients.scaffold_client import ScaffoldClient
from collections import Counter
from logging import INFO
from flwr.common.logger import log


def get_client_fn(
    config_sim: dict,
    dataset: FederatedDataset,
    model,
    device,
    apply_transforms,
    save_dir,
):
    strategy = config_sim["server"]["strategy"].lower()
    client_class = get_client_class(strategy)
    num_clients = config_sim["server"]["num_clients"]

    # Precompute distributions once (thread-safe for Flower simulations)
    client_distributions = {}
    log(INFO, "=>>>>> CLASS DISTRIBUTIONS OF ALL CLIENTS <<<<<<=")
    for cid in range(num_clients):
        client_data = dataset.load_partition(cid)
        labels = [item['label'] for item in client_data]
        client_distributions[cid] = dict(sorted(Counter(labels).items()))
        log(INFO, f"Client {cid} ({len(client_distributions[cid])} classes, {len(client_data)} samples) : {client_distributions[cid]}")

    def client_fn(cid: str) -> fl.client.Client:
        #Access precomputed client partitions
        client_dataset_total = dataset.load_partition(partition_id = int(cid))
        client_dataset_splits = client_dataset_total.train_test_split(test_size=0.2, seed=config_sim["common"]["seed"])
        
        trainset = client_dataset_splits["train"].with_transform(apply_transforms)
        valset = client_dataset_splits["test"].with_transform(apply_transforms)

        return client_class(
            client_id=int(cid),
            model=model,
            trainset=trainset,
            valset=valset,
            config_sim=config_sim,
            device=device,
            save_dir=save_dir,
        ).to_client()

    return client_fn


def get_client_class(strategy: str):
    if strategy == "fedprox":
        return FedProxClient
    elif strategy == "scaffold":
        return ScaffoldClient
    elif strategy == "fednova":
        return FedNovaClient
    else:
        return FedAvgClient
