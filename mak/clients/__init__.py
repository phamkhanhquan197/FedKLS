import flwr as fl
from flwr_datasets import FederatedDataset
from mak.clients.fedavg_client import FedAvgClient
from mak.clients.fednova_client import FedNovaClient
from mak.clients.fedprox_client import FedProxClient
from mak.clients.scaffold_client import ScaffoldClient
from mak.clients.fedklsvd_client import FedKLSVDClient
from mak.clients.fedawa_client import FedAWAClient
from logging import INFO
from flwr.common.logger import log

def get_client_fn(
    config_sim: dict,
    dataset: FederatedDataset,
    model,
    device,
    apply_transforms,
    save_dir,
    kl_norm_dict: dict = None, #Precomputed KL divergence values from server, if available
):
    strategy = config_sim["server"]["strategy"].lower()
    client_class = get_client_class(strategy)
    num_clients = config_sim["server"]["num_clients"]
    method = config_sim["peft"]["method"]
    
    # Use precomputed kl_norm values if provided by server, otherwise compute them
    if method == "fedkls" and kl_norm_dict is None:
        log(INFO, "No precomputed KL divergence values provided. Computing client distributions and kl_norm...")
        from mak.utils.helper import compute_KL_divergence, compute_client_distributions
        from mak.utils.dataset_info import dataset_info

        dataset_name = config_sim["common"]["dataset"]
        num_classes = dataset_info[dataset_name]["num_classes"]

        # Precompute distributions once (thread-safe for Flower simulations)
        client_distributions = compute_client_distributions(config_sim, dataset, num_clients)
        kl_normalized_per_client = compute_KL_divergence(client_distributions, num_classes)
        for cid, kl_norm_val in kl_normalized_per_client.items():
            log(INFO, f"Client {cid}: Normalized KL Divergence = {kl_norm_val:.4f}")

    elif method == "fedkls" and kl_norm_dict is not None:
        kl_normalized_per_client = kl_norm_dict
        # log(INFO, "Using precomputed kl_norm values for clients from server.")
    else:
        # log(INFO, f"Method is {method}. Skipping KL divergence computation.")
        pass


    def client_fn(cid: str) -> fl.client.Client:
        #Access precomputed client partitions
        client_dataset_total = dataset.load_partition(partition_id = int(cid))
        client_dataset_splits = client_dataset_total.train_test_split(test_size=0.2, seed=config_sim["common"]["seed"])
        
        trainset = client_dataset_splits["train"].with_transform(apply_transforms)
        valset = client_dataset_splits["test"].with_transform(apply_transforms)

        #Pass the normalized KL divergence to the client
        kl_norm = kl_normalized_per_client[int(cid)] if method == "fedkls" else 0.0
        client = client_class(
            client_id=int(cid),
            model=model, 
            trainset=trainset,
            valset=valset,
            config_sim=config_sim, 
            device=device,
            save_dir=save_dir,
            kl_norm=kl_norm,  
        )
        return client.to_client()
    

    return client_fn


def get_client_class(strategy: str):
    if strategy == "fedprox":
        return FedProxClient
    elif strategy == "scaffold":
        return ScaffoldClient
    elif strategy == "fednova":
        return FedNovaClient
    elif strategy == "fedklsvd":
        return FedKLSVDClient
    elif strategy == "fedawa":
        return FedAWAClient
    else:
        return FedAvgClient
