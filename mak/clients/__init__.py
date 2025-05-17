import flwr as fl
from flwr_datasets import FederatedDataset

from mak.clients.fedavg_client import FedAvgClient
from mak.clients.fednova_client import FedNovaClient
from mak.clients.fedprox_client import FedProxClient
from mak.clients.scaffold_client import ScaffoldClient
from collections import Counter
from logging import INFO
from flwr.common.logger import log
from mak.utils.dataset_info import dataset_info
import math


def compute_KL_divergence(client_distributions: dict, num_classes: int) -> dict:
    # Step 1: Compute global/ideal IID label distribution
    global_prob = 1/num_classes

    # Step 2: Compute local distribution on each client
    local_prob = {}
    for client, dist in client_distributions.items():
        total_sample = sum([s for s in dist.values()])
        local_prob[client] = {cls: s/total_sample for cls, s in dist.items()}

    # Step 3: Compute KL divergence per client
    kl_values = {}
    for client, dist in local_prob.items():
        kl = 0
        for c in range(num_classes):
            p_ic = dist.get(c, 0) # P_i(c), 0 if class c not present
            q_c = global_prob     # Q(c), ideal IID distribution

            if p_ic > 0: # skip zero to avoid log(0)
                kl += p_ic * math.log(p_ic/q_c)
        kl_values[client] = kl

    # Step 4: Normalize using Min-Max
    kl_min = min(kl_values.values())
    kl_max = max(kl_values.values())
    # Prevent divide-by-zero in case all KLs are the same
    if kl_max == kl_min:
        kl_normalized = {client: 0.0 for client in kl_values}
    else:
        kl_normalized = {
            client: (kl - kl_min) / (kl_max - kl_min)
            for client, kl in kl_values.items()
        }

    return kl_normalized  

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
    dataset_name = config_sim["common"]["dataset"]
    num_classes = dataset_info[dataset_name]["num_classes"]
    

    # Precompute distributions once (thread-safe for Flower simulations)
    client_distributions = {}
    log(INFO, "=>>>>> CLASS DISTRIBUTIONS OF ALL CLIENTS <<<<<<=")
    for cid in range(num_clients):
        client_data = dataset.load_partition(cid)
        labels = [item['label'] for item in client_data]
        client_distributions[cid] = dict(sorted(Counter(labels).items()))
        log(INFO, f"Client {cid} ({len(client_distributions[cid])} classes, {len(client_data)} samples) : {client_distributions[cid]}")
    log(INFO, "*"*200)
    kl_normalized_per_client  = compute_KL_divergence(client_distributions, num_classes)
    for cid_int, kl_norm_val in kl_normalized_per_client.items():
        log(INFO, f"Client {cid_int}: Normalized KL Divergence = {kl_norm_val:.4f}")

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
