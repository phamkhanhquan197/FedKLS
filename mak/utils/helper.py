import argparse
import csv
import json
import os
import random
from datetime import date, datetime
from logging import INFO
from typing import Dict

import flwr as fl
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr.common import Scalar
from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader

import mak
from mak.servers.custom_server import ServerSaveData
from mak.servers.fedklsvd_server import FedKLSVDServer
from mak.servers.fednova_server import FedNovaServer
from mak.servers.scaffold_server import ScaffoldServer
from mak.strategies.fednova_strategy import FedNovaStrategy
from mak.strategies.scaffold_strategy import ScaffoldStrategy
from mak.strategies.fedklsvd_strategy import FedKLSVDStrategy
from mak.utils.dataset_info import dataset_info
from mak.utils.general import set_params, test, weighted_average
from mak.models.svd_model import SVDAdapter
import math
from collections import Counter

def get_device_and_resources(config_sim):
    # Check if GPU is available
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config_sim["client"]["gpu"] else "cpu"
    )

    # Assign GPU and CPU resources
    if device.type == "cuda":
        # Assign GPU resources
        num_gpus_total = config_sim["client"]["total_gpus"]
        if num_gpus_total > 0:
            ray_init_args = {
                "num_cpus": config_sim["client"]["total_cpus"],
                "num_gpus": num_gpus_total,
            }
        else:
            ray_init_args = {
                "num_cpus": config_sim["client"]["total_cpus"],
                "num_gpus": 0,
            }
    else:
        # Assign CPU resources
        ray_init_args = {"num_cpus": config_sim["client"]["total_cpus"], "num_gpus": 0}

    # Assign client resources
    client_res = {
        "num_cpus": config_sim["client"]["num_cpus"],
        "num_gpus": config_sim["client"]["num_gpus"] if device.type == "cuda" else 0.0,
    }
    if config_sim["common"]["multi_node"]:
        ray_init_args = {}
        ray_init_args["address"] = "auto"
        ray_init_args["runtime_env"] = {"py_modules": [mak]}
    return device, ray_init_args, client_res


def gen_dir_outfile_server(config):
    # generates the basic directory structure for out data and the header for file
    today = date.today()
    BASE_DIR = "output"
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

    # create a date wise folder
    if not os.path.exists(os.path.join(BASE_DIR, str(today))):
        os.mkdir(os.path.join(BASE_DIR, str(today)))

    # create saperate folder based on strategy
    if not os.path.exists(
        os.path.join(BASE_DIR, str(today), config["server"]["strategy"])
    ):
        os.mkdir(os.path.join(BASE_DIR, str(today), config["server"]["strategy"]))

    # create saperate folder based on data distribution type
    if not os.path.exists(
        os.path.join(
            BASE_DIR,
            str(today),
            config["server"]["strategy"],
            config["common"]["data_type"],
        )
    ):
        os.mkdir(
            os.path.join(
                BASE_DIR,
                str(today),
                config["server"]["strategy"],
                config["common"]["data_type"],
            )
        )

    dirs = os.listdir(
        os.path.join(
            BASE_DIR,
            str(today),
            config["server"]["strategy"],
            config["common"]["data_type"],
        )
    )
    final_dir_path = os.path.join(
        BASE_DIR,
        str(today),
        config["server"]["strategy"],
        config["common"]["data_type"],
        str(len(dirs)),
    )

    if not os.path.exists(final_dir_path):
        os.mkdir(final_dir_path)
    if not os.path.exists(os.path.join(final_dir_path, "clients")):
        os.mkdir(os.path.join(final_dir_path, "clients"))
    # models_dir = os.path.join(final_dir_path,'models')
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    # save all confugration file as json file
    json_file_name = f"config.json"
    with open(os.path.join(final_dir_path, json_file_name), "w") as fp:
        json.dump(config, fp, indent=4)
    dataset_str = config["common"]["dataset"].replace("/", "_")
    file_name = f"{config['server']['strategy']}_{dataset_str}_{config['common']['data_type']}_{config['client']['batch_size']}_{config['client']['lr']}_{config['client']['epochs']}"
    file_name = f"{file_name}.csv"
    out_file_path = os.path.join(final_dir_path, file_name)
    # create empty server history file
    if not os.path.exists(out_file_path):
        with open(out_file_path, "w", encoding="UTF8") as f:
            # create the csv writer
            header = ["round", "global_accuracy", "global_f1_score", "global_loss", "local_accuracy", "local_f1", "local_loss", "processing_time", "upload_gb", "download_gb"]
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()
    return out_file_path, final_dir_path


def get_partitioner(config_sim):
    num_clients = config_sim["server"]["num_clients"]
    if config_sim["common"]["data_type"] == "dirichlet_niid":
        # alpha value
        dirchlet_alpha = config_sim["common"]["dirichlet_alpha"]
        # dataset
        dataset_name = config_sim["common"]["dataset"]
        # dataset's label column
        label = dataset_info[dataset_name]["output_column"]
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=label,
            alpha=dirchlet_alpha,
            min_partition_size=2, #minimum number of samples in each partition
            self_balancing=True,
            shuffle=True,
            seed=config_sim["common"]["seed"],
        )
        
    else:
        partitioner = IidPartitioner(num_partitions=num_clients)
    # return train data
    return {"train": partitioner}


def get_dataset(config_sim):
    partitioner = get_partitioner(config_sim=config_sim)
    dataset_name = config_sim["common"]["dataset"]
    if dataset_name not in dataset_info.keys():
        raise Exception(f"Dataset name should be among : {list(dataset_info.keys())}")
    else:
        fds = FederatedDataset(dataset=dataset_name, partitioners=partitioner)
        # get test column name
        test_set = dataset_info[dataset_name]["test_set"]
        centralized_testset = fds.load_split(test_set)
        return fds, centralized_testset

def extract_linear_layers(model):
    """Return a dict of {layer_name: layer_module} for all linear layers in the model.
    Optionally skips layers specified in layers_to_skip.
    """
    linear_layers = {}

    for name, module in model.named_modules():
        # Check if the module is a Linear layer
        if isinstance(module, torch.nn.Linear):
            if name in ["pre_classifier","classifier"]: # Check if any part of the layer_to_skip is in the current layer's name
                continue
            linear_layers[name] = module

    return linear_layers

def apply_svd_to_model(model, config, kl_norm = None, client_id = None):
    """
    Apply SVD to the specified linear layers of the model, replacing them with SVDAdapter.
    The SVDAdapter class itself ensures W_res is frozen (as a buffer).
    Args:
        model (nn.Module): The model to modify.
        config (dict): Configuration dictionary, must contain config["lora"]["rank"],
                       config["lora"]["alpha"], and config["lora"]["method"].
        layers_to_skip_svd (list, optional): List of string name parts of layers
                                             to skip during SVD adaptation.
                                             E.g., ["classifier", "pre_classifier"].
    Returns:
        nn.Module: The modified model.
    """    
    linear_layers = extract_linear_layers(model)
    log(INFO, f"Found {len(linear_layers)} linear layers to adapt with SVD.")

    for name, layer in linear_layers.items():
        weight_matrix = layer.weight.data
        original_bias = layer.bias.data if layer.bias is not None else None
        rank = config["lora"]["rank"]
        alpha = config["lora"]["alpha"]
        method = config["lora"]["method"]

        if method == 'lora':
            # Original LoRA: Random initialization without SVD
            d_out, d_in = weight_matrix.shape
            A = torch.randn(d_out, rank, device=weight_matrix.device) * 0.01  # Gaussian init
            B = torch.zeros(rank, d_in, device=weight_matrix.device)  # Zero init
            W_res = weight_matrix
            log(INFO, f"Layer {name}: Applied LoRA with rank {rank}.")
        else:
            # Perform SVD for other methods
            U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
            max_possible_rank = S.size(0)
            if rank > max_possible_rank:
                log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                rank = max_possible_rank

            # Select components based on method
            if method == 'pissa':
                # Principal component as adapter (PiSSA)
                U_select = U[:, :rank]
                S_select = S[:rank]
                Vt_select = Vt[:rank, :]

            elif method == 'milora':
                # Minor component as adapter (MiLoRA)
                U_select = U[:, -rank:]
                S_select = S[-rank:]
                Vt_select = Vt[-rank:, :]

            elif method == 'middle':
                middle_index_start = math.floor(max_possible_rank/2)
                middle_index_end = middle_index_start + rank
                U_select = U[:, middle_index_start:middle_index_end]
                S_select = S[middle_index_start:middle_index_end]
                Vt_select = Vt[middle_index_start:middle_index_end, :]

            elif method == 'fedkl_svd':
                index_start = math.floor(kl_norm * (max_possible_rank - rank)) if kl_norm is not None else 0
                index_end = index_start + rank
                if client_id is not None:
                    log(INFO, f"Client {client_id}: SVD applied with index range {index_start} to {index_end} with rank {rank} for layer {name}.")
                U_select = U[:, index_start:index_end]
                S_select = S[index_start:index_end]
                
                Vt_select = Vt[index_start:index_end, :]
            else:
                raise ValueError(f"Unknown method: {method}")

            W_res = weight_matrix - (U_select @ torch.diag(S_select) @ Vt_select)
            A = U_select @ torch.diag(torch.sqrt(S_select))
            B = torch.diag(torch.sqrt(S_select)) @ Vt_select

        # Create the SVDAdapter
        new_layer = SVDAdapter(W_res=W_res, A=A, B=B, alpha=alpha, rank=rank, original_bias=original_bias)
        
        # Split layer name and replace the original layer
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, new_layer)  
    
    return model

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


def compute_client_distributions(dataset, num_clients: int) -> dict:
    """
    Compute the label distribution for each client in the federated dataset.
    
    Args:
        dataset: FederatedDataset object (e.g., from flwr_datasets)
        num_clients: Number of clients in the federated dataset
    
    Returns:
        dict: Mapping of client IDs to their label distributions
    """
    client_distributions = {}
    
    log(INFO, "=>>>>> CLASS DISTRIBUTIONS OF ALL CLIENTS <<<<<<=")
    for cid in range(num_clients):
        client_data = dataset.load_partition(cid)
        labels = [item['label'] for item in client_data]
        client_distributions[cid] = dict(sorted(Counter(labels).items()))
        log(INFO, f"Client {cid} ({len(client_distributions[cid])} classes, {len(client_data)} samples) : {client_distributions[cid]}")
    log(INFO, "*" * 150)
    
    return client_distributions

# def compute_class_weights(trainset, num_classes: int):
#     class_counts = np.zeros(num_classes)
#     for batch in trainset:
#         # Extract labels from the dictionary
#         target = batch["labels"]
#         # If target is a tensor, convert to numpy or handle as needed
#         if isinstance(target, torch.Tensor):
#             target = target.numpy()
#         # Handle batched or single labels
#         target = np.array(target).flatten()
#         for label in target:
#             class_counts[int(label)] += 1
#     class_weights = 1.0 / (class_counts + 1e-10)
#     class_weights = class_weights / class_weights.sum() * num_classes
#     return torch.tensor(class_weights, dtype=torch.float)

# def apply_global_lora_freezing_policy(
#     model, 
#     svd_adapter_class_name: str = "SVDAdapter", 
#     train_final_classifier: bool = True
# ):
#     """
#     Applies a global freezing policy for LoRA-style fine-tuning.
#     It freezes all parameters by default, then unfreezes:
#     1. 'A', 'B', and 'bias' nn.Parameters within modules of type svd_adapter_class_name.
#     2. All parameters of the module named 'classifier' if train_final_classifier is True.

#     Args:
#         model (nn.Module): The model to apply the policy to.
#         svd_adapter_class_name (str): The string name of your SVD adapter class.
#         train_final_classifier (bool): Whether to make the final classifier head trainable.

#     Returns:
#         nn.Module: The model with updated requires_grad flags.
#     """
#     print(f"\nApplying global freezing policy (Adapter class: '{svd_adapter_class_name}', Train classifier: {train_final_classifier})...")
    
#     num_total_params = 0
#     num_trainable_params = 0
#     trainable_param_names = []

#     for name, param in model.named_parameters():
#         num_total_params += param.numel()
#         param.requires_grad = False  # Freeze all parameters by default

#         # Check if the parameter belongs to an SVDAdapter module
#         module_path_parts = name.split('.')
#         current_param_short_name = module_path_parts[-1] # e.g., 'A', 'B', 'bias', 'weight'
        
#         is_svd_adapter_trainable_part = False
#         if len(module_path_parts) > 1: # Indicates a nested parameter
#             parent_module_path = '.'.join(module_path_parts[:-1])
#             try:
#                 parent_module = model.get_submodule(parent_module_path)
#                 if parent_module.__class__.__name__ == svd_adapter_class_name:
#                     if current_param_short_name in ['A', 'B', 'bias']:
#                         # Ensure the attribute on the parent is this exact parameter object
#                         # and that it's an nn.Parameter (already true from named_parameters)
#                         if getattr(parent_module, current_param_short_name, None) is param:
#                             param.requires_grad = True
#                             is_svd_adapter_trainable_part = True
#             except AttributeError:
#                 # This submodule path doesn't exist, should not happen if 'name' is valid from named_parameters
#                 pass 
        
#         # Optionally, make the final classifier head fully trainable
#         # This condition is separate to allow classifier to be trained even if it's not an SVDAdapter
#         if train_final_classifier and name.startswith("classifier."):
#             # If the classifier was already made trainable as an SVDAdapter part, this is fine.
#             # If the classifier is a standard nn.Linear, this will make its .weight and .bias trainable.
#             param.requires_grad = True
        
#         if param.requires_grad:
#             if not is_svd_adapter_trainable_part and name.startswith("classifier."): # For logging distinct cases
#                  pass # Already handled by train_final_classifier logic
#             num_trainable_params += param.numel()
#             if name not in trainable_param_names: # Avoid duplicates if classifier itself was an SVD adapter
#                 trainable_param_names.append(name)

#     print(f"  Total model parameters: {num_total_params}")
#     print(f"  Trainable parameters after policy: {num_trainable_params}")
#     if num_total_params > 0 :
#         print(f"  Percentage of trainable parameters: {(num_trainable_params / num_total_params) * 100:.4f}%")
#     if num_trainable_params == 0 and num_total_params > 0 :
#         print("  WARNING: No parameters are trainable after freezing policy! Model will not learn.")
#     # else:
#     #     print("  Trainable parameter names:")
#     #     for tp_name in sorted(trainable_param_names): # Sort for consistent logging
#     #         print(f"    - {tp_name}")
            
#     return model

def get_model(config, shape):
    model_name = config["common"]["model"]
    # get num_classes
    dataset_name = config["common"]["dataset"]
    num_classes = dataset_info[dataset_name]["num_classes"]

    # check if model is from huggingface
    if model_name in ["distilbert-base-uncased", "albert-base-v2"]:  # Add more as needed
        from transformers import AutoModelForSequenceClassification
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )


        return base_model

    # check custom models 
    model = getattr(__import__("mak.models", fromlist=[model_name]), model_name)(
        num_classes=num_classes, input_shape=shape
    )

    return model


def get_evaluate_fn(
    centralized_testset: Dataset,
    config_sim,
    device,
    save_model_dir,
    metrics_file,
    apply_transforms_test,
    model,
):
    """Return an evaluation function for centralized evaluation."""
    dataset_name = config_sim["common"]["dataset"]
    # shape = dataset_info[dataset_name]["input_shape"]

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):  
        ## model = get_model(config=config_sim, shape=shape)
        ## model = apply_svd_to_model(model=model, config=config_sim)
        set_params(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms_test)

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()

        testloader = DataLoader(testset, batch_size=config_sim["client"]["test_batch_size"])

        feature_key = dataset_info[dataset_name]["feature_key"]
        loss, accuracy, f1 = test(model, testloader, device=device, feature_key=feature_key)
        metrics_df = pd.read_csv(metrics_file)
        if metrics_df["global_loss"].min() > loss:
            log(
                INFO,
                f" =>>>>> Min Loss improved from {metrics_df['global_loss'].min()} to : {loss} =>>>>> Saving best model with accuracy : {accuracy}, f1_score : {f1}", 
            )
            torch.save(
                model.state_dict(), os.path.join(save_model_dir, "saved_best_model.pth")
            )

        if server_round == config_sim["server"]["num_rounds"]:
            torch.save(
                model.state_dict(),
                os.path.join(save_model_dir, "saved_final_model.pth"),
            )
        return loss, {"accuracy": accuracy, "f1_score": f1}
    
        

    return evaluate


def save_simulation_history(hist: fl.server.history.History, path):
    losses_distributed = hist.losses_distributed
    losses_centralized = hist.losses_centralized
    metrics_distributed_fit = hist.metrics_distributed_fit
    metrics_distributed = hist.metrics_distributed
    metrics_centralized = hist.metrics_centralized

    rounds = []
    losses_centralized_dict = {}
    losses_distributed_dict = {}
    accuracy_distributed_dict = {}
    accuracy_centralized_dict = {}
    f1_score_distributed_dict = {}
    f1_score_centralized_dict = {}

    for loss in losses_centralized:
        c_rnd = loss[0]
        rounds.append(c_rnd)
        losses_centralized_dict[c_rnd] = loss[1]

    for loss in losses_distributed:
        c_rnd = loss[0]
        losses_distributed_dict[c_rnd] = loss[1]
    if "accuracy" in metrics_distributed.keys():
        for acc in metrics_distributed["accuracy"]:
            c_rnd = acc[0]
            accuracy_distributed_dict[c_rnd] = acc[1]
    if "accuracy" in metrics_centralized.keys():
        for acc in metrics_centralized["accuracy"]:
            c_rnd = acc[0]
            accuracy_centralized_dict[c_rnd] = acc[1]
    if "f1_score" in metrics_distributed.keys():
        for f1 in metrics_distributed["f1_score"]:
            c_rnd = f1[0]
            f1_score_distributed_dict[c_rnd] = f1[1]
    if "f1_score" in metrics_centralized.keys():
        for f1 in metrics_centralized["f1_score"]:
            c_rnd = f1[0]
            f1_score_centralized_dict[c_rnd] = f1[1]


    if len(metrics_distributed_fit) != 0:
        pass  # TODO  check its implemetation later

    data = {
        "round": rounds,
        "global_loss": losses_centralized_dict,
        "local_loss": losses_distributed_dict,
        "global_accuracy": accuracy_centralized_dict,
        "local_accuracy": accuracy_distributed_dict,
        "global_f1_score": f1_score_centralized_dict,
        "local_f1_score": f1_score_distributed_dict,
    }

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate over each key in the data dictionary
    for key in data.keys():
        # If the key is 'rounds', set the 'rounds' column of the DataFrame to the rounds list
        if key == "round":
            df["round"] = data[key]
        # Otherwise, create a new column in the DataFrame with the key as the column name
        else:
            column_data = []
            # Iterate over each round in the 'rounds' list and add the corresponding value for the current key
            for round_num in data["round"]:
                # If the round number does not exist in the current key's dictionary, set the value to None
                if round_num not in data[key]:
                    column_data.append(None)
                else:
                    column_data.append(data[key][round_num])
            df[key] = column_data
    df.to_csv(os.path.join(path), index=False)


def get_server(strategy, client_manager, out_file_path, target_acc):
    if isinstance(strategy, ScaffoldStrategy):
        return ScaffoldServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )
    elif isinstance(strategy, FedNovaStrategy):
        return FedNovaServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )
    elif isinstance(strategy, FedKLSVDStrategy):
        return FedKLSVDServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )
    else:
        return ServerSaveData(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
        )


def get_strategy(
    config,
    test_data,
    save_model_dir,
    out_file_path,
    device,
    apply_transforms_test,
    size_weights,
    model,
):
    STRATEGY = config["server"]["strategy"]
    MIN_CLIENTS_FIT = config["server"]["min_fit_clients"]
    MIN_CLIENTS_EVAL = config["server"]["min_evaluate_clients"]
    NUM_CLIENTS = config["server"]["num_clients"]
    FRACTION_FIT = config["server"]["fraction_fit"]
    FRACTION_EVAL = config["server"]["fraction_evaluate"]

    kwargs = {
        "FedAvgM": {
            "server_learning_rate": 1.0,
            "server_momentum": 0.2,
        },
        "FedAdam": {
            "eta": 1e-1,
            "eta_l": 1e-1,
            "beta_1": 0.9,
            "beta_2": 0.99,
            "tau": 1e-9,
        },
        "FedOpt": {
            "eta": 1e-1,
            "eta_l": 1e-1,
            "beta_1": 0.0,
            "beta_2": 0.0,
            "tau": 1e-9,
        },
        "FedProx": {
            "proximal_mu": config["fedprox"]["proximal_mu"],
        },
        "FedLaw": {
            "config": config,
            "model": model,
            "test_data": test_data,
            "size_weights": size_weights,
            "apply_transforms": apply_transforms_test,
            "apply_transforms_test": apply_transforms_test,
        },
        "PowD": {
            "candidate_client_set": config["powd_config"]["candidate_client_set"],
        },
    } 
    print(getattr(__import__("mak.strategies", fromlist=[STRATEGY]), STRATEGY))
    return getattr(__import__("mak.strategies", fromlist=[STRATEGY]), STRATEGY)(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVAL,
        min_fit_clients=MIN_CLIENTS_FIT,
        min_evaluate_clients=MIN_CLIENTS_EVAL,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(
            centralized_testset=test_data,
            config_sim=config,
            save_model_dir=save_model_dir,
            metrics_file=out_file_path,
            device=device,
            apply_transforms_test=apply_transforms_test,
            model=model,
        ),
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_fit_config_fn(config_sim=config),
        initial_parameters=fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in model.state_dict().items()]),
        **kwargs.get(STRATEGY, {}),
    )


def set_seed(seed: int = 13):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set the seed for CUDA operations (if using GPU)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log(INFO, f"All random seeds set to {seed}")


def get_config(file_path):
    # Open the YAML file
    with open(file_path, "r") as file:
        # Parse the YAML data
        config = yaml.safe_load(file)
        return config


def get_fit_config_fn(config_sim):
    def fit_config(server_round: int):
        """Return training configuration dict for each round.

        passes the current round number to the client
        """
        config = {
            "round": server_round,
            "batch_size": config_sim["client"]["batch_size"],
            "epochs": config_sim["client"]["epochs"],
            "lr": config_sim["client"]["lr"],
            "optimizer": config_sim["common"]["optimizer"],
            "sgd_momentum": config_sim["common"]["sgd_momentum"],
            "strategy": config_sim["server"]["strategy"],
            "proximal_mu": config_sim["fedprox"]["proximal_mu"],
            "loss": config_sim["client"]["loss"],
        }
        return config

    return fit_config


def get_mode_and_shape(partition):
    data_set_keys = list(partition.features.keys())
    x_column = data_set_keys[0]
    shape = partition[x_column][0].size
    mode = partition[x_column][0].mode
    if mode == "RGB":
        channel = 3
    else:
        channel = 1
    return (channel, shape[0], shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLNCLAB")

    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="path to the config.yaml file.",
    )
    parser.add_argument("--strategy", type=str, help="FL Strategy/algorithm")
    parser.add_argument("--seed", type=int, help="Seed for randomness")
    parser.add_argument(
        "--noise", type=float, default=None, help="add dp noise to data or not"
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=None,
    )

    args = parser.parse_args()
    return args


def get_optimizer(model, client_config):
    if client_config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=client_config["lr"])
    else:
        return torch.optim.SGD(
            model.parameters(),
            lr=client_config["lr"],
            momentum=client_config["sgd_momentum"],
        )


# for fedlaw
def get_size_weights(federated_dataset, num_clients):
    sample_size = []
    for i in range(num_clients):
        sample_size.append(len(federated_dataset.load_partition(i)))
    size_weights = [i / sum(sample_size) for i in sample_size]
    return size_weights
