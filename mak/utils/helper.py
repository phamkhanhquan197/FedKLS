import argparse
import csv
import json
import os
import random
from datetime import date, datetime
from logging import INFO
from typing import Dict, List, Optional

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
from sklearn.model_selection import train_test_split

import mak
from mak.servers.custom_server import ServerSaveData
from mak.servers.fedklsvd_server import FedKLSVDServer
from mak.servers.ffa_lora_server import FFALoRAServer
from mak.servers.fednova_server import FedNovaServer
from mak.servers.scaffold_server import ScaffoldServer
from mak.servers.pfedmoap_server import PFedMoAPServer
from mak.servers.fedpoe_server import FedPOEServer
from mak.servers.fedpoe_server import FedPOERegressionTextServer
from mak.servers.fedsa_lora_server import FedSALoRAServer
from mak.servers.flex_lora_server import FlexLoRAServer

from mak.strategies.fednova_strategy import FedNovaStrategy
from mak.strategies.scaffold_strategy import ScaffoldStrategy
from mak.strategies.fedklsvd_strategy import FedKLSVDStrategy
from mak.strategies.ffa_lora_strategy import FFALoRAStrategy
from mak.strategies.pfedmoap_strategy import PFedMoAPStrategy
from mak.strategies.fedpoe_strategy import FedPOEStrategy
from mak.strategies.fedpoe_strategy import FedPOERegressionTextStrategy
from mak.strategies.fedsa_lora_strategy import FedSALoRAStrategy
from mak.strategies.flex_lora_strategy import FlexLoRAStrategy

from mak.utils.dataset_info import dataset_info
from mak.utils.general import set_params, test, weighted_average
from mak.models.svd_model import SVDAdapter, ConvAdapter
import math
from collections import Counter
import torch.nn.init as init
from datasets import load_dataset

def get_target_keys(model, bias=True) -> List[str]:
    """Return deterministic sorted list of target parameter names for FFA/Flex LoRA.

    Includes:
    1) Adapter matrices (.A, .B)
    2) Biases after A and B, not all bias (.bias)

    This function is intentionally model-agnostic and must remain deterministic.
    """
    model_state = model.state_dict()

    if any(key.startswith("distilbert.") for key in model_state.keys()):
        if bias:
            lora_keys = [k for k in model_state.keys() if ("lin" in k)]
        else:
            lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
    elif any(key.startswith("roberta.") for key in model_state.keys()):
        if bias:
            lora_keys = [k for k in model_state.keys() if ("self" in k or ("dense" in k and "classifier" not in k))]
        else:
            lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
    elif any(key.startswith("bert.") for key in model_state.keys()):
        if bias:
            lora_keys = [k for k in model_state.keys() if ("self" in k or "dense" in k)]
        else:
            lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
    elif any(key.startswith("model.") for key in model_state.keys()):
        if bias:
            lora_keys = [k for k in model_state.keys() if ("self_attn" in k or "mlp" in k)]
        else:
            lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
    elif any(key.startswith("layer") for key in model_state.keys()): #RESNET models
        if bias:
            lora_keys = [k for k in model_state.keys() if ("conv" in k)]
        else:
            lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]

    # Unique + deterministic order
    return sorted(set(lora_keys))

def _resolve_ray_tmp_dir(config_sim: dict) -> Optional[str]:
    common_cfg = (config_sim or {}).get("common", {})
    candidate = (
        common_cfg.get("ray_tmp_dir")
        or os.environ.get("FEDKLS_RAY_TMPDIR")
        or os.environ.get("RAY_TMPDIR")
        or os.environ.get("TMPDIR")
    )
    if not candidate:
        return None
    ray_tmp_dir = os.path.abspath(os.path.expanduser(str(candidate)))
    os.makedirs(ray_tmp_dir, exist_ok=True)
    return ray_tmp_dir


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

    # Ray writes session + spill data under /tmp by default. Allow redirecting this
    # to a larger disk to avoid GCS/raylet crashes when /tmp is full.
    if not config_sim["common"].get("multi_node", False):
        ray_tmp_dir = _resolve_ray_tmp_dir(config_sim)
        if ray_tmp_dir:
            ray_init_args["_temp_dir"] = ray_tmp_dir
            # Only set spilling config if the installed Ray supports it.
            # Some Ray versions reject unknown kwargs (RuntimeError: Unknown keyword argument(s)).
            try:
                import inspect
                import ray

                if "object_spilling_config" in inspect.signature(ray.init).parameters:
                    spill_dir = os.path.join(ray_tmp_dir, "spill")
                    os.makedirs(spill_dir, exist_ok=True)
                    ray_init_args.setdefault(
                        "object_spilling_config",
                        json.dumps(
                            {"type": "filesystem", "params": {"directory_path": spill_dir}}
                        ),
                    )
            except Exception:
                pass

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
        dirichlet_alpha = config_sim["common"]["dirichlet_alpha"]
        # dataset
        dataset_name = config_sim["common"]["dataset"]
        # dataset's label column
        label = dataset_info[dataset_name]["output_column"]
        # create partitioner
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=label,
            alpha=dirichlet_alpha,
            min_partition_size=1,  # minimum number of samples in each partition
            self_balancing=False,
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
        # get test split name (can be None for datasets with only 'train')
        test_set = dataset_info[dataset_name].get("test_set")
        centralized_testset = None
        if test_set is not None:
            centralized_testset = fds.load_split(test_set)

        # get class names for pFedMoAP
        out_col = dataset_info[dataset_name]["output_column"]
        if centralized_testset is not None:
            feat = centralized_testset.features.get(out_col, None)
            if feat is not None and hasattr(feat, "names") and feat.names:
                classnames = list(feat.names)
            else:
                num_classes = dataset_info[dataset_name]["num_classes"]
                classnames = [f"class{i}" for i in range(num_classes)]
        else:
            # Fallback: if there's no centralized testset, we can't read label names
            num_classes = dataset_info[dataset_name]["num_classes"]
            classnames = [f"class{i}" for i in range(num_classes)]

        return fds, centralized_testset, classnames
    
def extract_linear_layers(model, config):
    """Return a dict of {layer_name: layer_module} for all linear layers in the model.
    Optionally skips layers specified in layers_to_skip.
    """
    linear_layers = {}
    skip_layer_names = ["pre_classifier", "classifier", "model.norm", "score", "classifier.dense", "classifier.out_proj"]
    attenion_layer_names = ["self_attn", "attn", "attention"]

    # Optional: match PEFT-style target modules (e.g., ["query", "value"]) like 3rd-party/fed-svd.
    # For DistilBERT, HF uses q_lin/k_lin/v_lin/out_lin instead of query/key/value.
    target_modules = (config or {}).get("peft", {}).get("target_modules", None)
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    if target_modules is not None:
        target_modules = [str(x) for x in target_modules if str(x).strip()]

    # Implicit default: when running FedSVD with fedsvd_lora, target query/value only
    # to match the 3rd-party fed-svd PEFT LoRA placement.
    if not target_modules:
        strategy_name = str((config or {}).get("server", {}).get("strategy", "")).lower()
        peft_method = str((config or {}).get("peft", {}).get("method", "")).lower()
        if strategy_name == "fedsvd" and peft_method == "fedsvd_lora":
            target_modules = ["query", "value"]

    model_state = None
    try:
        model_state = model.state_dict()
    except Exception:
        model_state = {}

    is_distilbert = any(k.startswith("distilbert.") for k in model_state.keys())
    if target_modules and is_distilbert:
        # Minimal mapping to emulate PEFT target_modules=["query","value"] on DistilBERT.
        # query -> q_lin, value -> v_lin
        mapping = {
            "query": "q_lin",
            "key": "k_lin",
            "value": "v_lin",
            "out": "out_lin",
        }
        resolved = []
        for tm in target_modules:
            resolved.append(mapping.get(tm, tm))
        target_modules = resolved

    for name, module in model.named_modules():
        # Check if the module is a Linear layer
        if isinstance(module, torch.nn.Linear):
            if name in skip_layer_names: # Check if any part of the layer_to_skip is in the current layer's name
                continue

            # If target_modules are specified, select only those submodules.
            # Example: BERT/Roberta: ...attention.self.query / ...attention.self.value
            #          DistilBERT:   ...attention.q_lin / ...attention.v_lin
            if target_modules:
                leaf = name.split(".")[-1]
                if leaf in set(target_modules):
                    linear_layers[name] = module
                continue

            # Fallback: legacy selection by broad attention path matching
            if config["peft"]["layer"] == "attention_only":
                if any(att_name in name for att_name in attenion_layer_names):
                    linear_layers[name] = module
            else:
                linear_layers[name] = module

    return linear_layers

def extract_conv2_layers(model):
    conv2_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and (name.endswith("conv2") or name.endswith(".conv1")):
            conv2_layers[name] = module
    return conv2_layers

def apply_svd_to_model(model, config, kl_norm = None, client_id = None):
    """
    Apply SVD to the specified linear layers of the model, replacing them with SVDAdapter.
    The SVDAdapter class itself ensures W_res is frozen (as a buffer).
    Args:
        model (nn.Module): The model to modify.
        config (dict): Configuration dictionary, must contain config["peft"]["rank"],
                       config["peft"]["alpha"], and config["peft"]["method"].
        layers_to_skip_svd (list, optional): List of string name parts of layers
                                             to skip during SVD adaptation.
                                             E.g., ["classifier", "pre_classifier"].
    Returns:
        nn.Module: The modified model.
    """    
    if config["common"]["model"] in ["Resnet18", "ResNet18Pretrained", "ResNet34", "ResNet34Pretrained"]:
        layers_to_svd = extract_conv2_layers(model) 
        log(INFO, f"Found {len(layers_to_svd)} conv2 layers to adapt with SVD.")
    else:
        layers_to_svd = extract_linear_layers(model, config) 
        log(INFO, f"Found {len(layers_to_svd)} linear layers to adapt with SVD.")

    rank = config["peft"]["rank"]
    alpha = config["peft"]["alpha"]
    method = config["peft"]["method"]
    dp_fedsvd = False
    if config["fedsvd_config"]:
        dp_fedsvd = config["fedsvd_config"]["dp"]["enabled"]

    for name, layer in layers_to_svd.items():
        weight_matrix = layer.weight.data
        original_bias = layer.bias.data if layer.bias is not None else None

        if method == 'lora' or method == 'fedsvd_lora':
            # LoRA initialization - PEFT convention: A(r, in), B(out, r)
            # Forward: ΔW = B @ A
            fedsvd_init = None
            if method == 'fedsvd_lora':
                fedsvd_cfg = config.get("fedsvd_config", {})
                fedsvd_init = fedsvd_cfg.get("init_method", "kaiming")
            
            if isinstance(layer, torch.nn.Conv2d):
                c_out, c_in, k1, k2 = weight_matrix.shape
                d_in = c_in * k1 * k2
                
                # A: (r, in) - Kaiming init
                A = torch.empty(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                if fedsvd_init == "gaussian":
                    init.normal_(A, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(A, a=math.sqrt(5))
                
                # B: (out, r) - Zero init
                B = torch.zeros(c_out, rank, device=weight_matrix.device, dtype=weight_matrix.dtype)
                W_res = weight_matrix
            else:
                d_out, d_in = weight_matrix.shape
                
                # A: (r, in) - Kaiming init
                A = torch.empty(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                if fedsvd_init == "gaussian":
                    init.normal_(A, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(A, a=math.sqrt(5))
                
                # B: (out, r) - Zero init
                B = torch.zeros(d_out, rank, device=weight_matrix.device, dtype=weight_matrix.dtype)
                W_res = weight_matrix
            
            init_name = fedsvd_init if fedsvd_init else "kaiming"
            log(INFO, f"Layer {name}: Applied LoRA with rank {rank} (init={init_name}, A({rank},{d_in}), B({d_out if not isinstance(layer, torch.nn.Conv2d) else c_out},{rank})).")

        elif method == 'ffa_lora':
            # FFA-LoRA: Initialize A (configurable), B = 0, freeze A forever (external control)
            ffa_cfg = config.get("ffa_lora_config", {})
            init_method = ffa_cfg.get("init_method", "kaiming")

            if init_method not in {"kaiming", "gaussian", "orthogonal", "svd"}:
                raise ValueError(
                    f"Unknown init_method: {init_method}. Options: kaiming | gaussian | orthogonal | svd"
                )
            # --------------------------------------------------
            # Conv2d
            # --------------------------------------------------
            if isinstance(layer, torch.nn.Conv2d):
                c_out, c_in, k1, k2 = weight_matrix.shape
                d_in = c_in * k1 * k2
                W_flat = weight_matrix.view(c_out, -1)

                # A init
                if init_method =="svd":
                    U, S, Vt = torch.linalg.svd(W_flat, full_matrices=False)
                    max_possible_rank = S.size(0)
                    if rank > max_possible_rank:
                        log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                        rank = max_possible_rank
                    U_select = U[:, :rank]
                    S_select = S[:rank]
                    Vt_select = Vt[:rank, :]
                    
                    A = U_select @ torch.diag(torch.sqrt(S_select))  # Shape: [c_out, rank]
                    B = torch.diag(torch.sqrt(S_select)) @ Vt_select  # Shape: [rank, d_in]
                    W_res = weight_matrix - (U_select @ torch.diag(S_select) @ Vt_select).view(c_out, c_in, k1, k2)
                else:
                    A = torch.empty(c_out, rank, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    if init_method == "kaiming":
                        init.kaiming_normal_(A, mode="fan_out", nonlinearity="relu")
                    elif init_method == "gaussian":
                        init.normal_(A, mean=0.0, std=0.01)
                    elif init_method == "orthogonal":
                        init.orthogonal_(A)
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix
            # --------------------------------------------------
            # Linear
            # --------------------------------------------------
            else:
                d_out, d_in = weight_matrix.shape

                # A init
                if init_method =="svd":
                    U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
                    max_possible_rank = S.size(0)
                    if rank > max_possible_rank:
                        log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                        rank = max_possible_rank
                    U_select = U[:, :rank]
                    S_select = S[:rank]
                    Vt_select = Vt[:rank, :]
                    
                    A = U_select @ torch.diag(torch.sqrt(S_select))  # Shape: [d_out, rank]
                    B = torch.diag(torch.sqrt(S_select)) @ Vt_select  # Shape: [rank, d_in]
                    W_res = weight_matrix - (U_select @ torch.diag(S_select) @ Vt_select)
                else:
                    A = torch.empty(d_out, rank, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    if init_method == "kaiming":
                        init.kaiming_normal_(A, mode="fan_out", nonlinearity="relu")
                    elif init_method == "gaussian":
                        init.normal_(A, mean=0.0, std=0.01)
                    elif init_method == "orthogonal":
                        init.orthogonal_(A)
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix

            log(INFO, f"Layer {name}: Applied FFA-LoRA with rank {rank} (A init={init_method}, B=zero, A frozen).")

        elif method == "fedsa_lora":
            # FedSA-LoRA: Train both A and B locally, but only A is aggregated (handled in client/strategy)
            # Init: A configurable (default kaiming), B = 0, W_res = W

            init_method = config["fedsa_lora_config"]["init_method"]
            
            if init_method not in {"kaiming", "gaussian", "orthogonal", "svd"}:
                raise ValueError(
                    f"Unknown init_method: {init_method}. Options: kaiming | gaussian | orthogonal | svd"
                )

            # --------------------------
            # Conv2d
            # --------------------------
            if isinstance(layer, torch.nn.Conv2d):
                c_out, c_in, k1, k2 = weight_matrix.shape
                d_in = c_in * k1 * k2
                W_flat = weight_matrix.view(c_out, -1)

                if init_method == "svd":
                    U, S, Vt = torch.linalg.svd(W_flat, full_matrices=False)
                    max_possible_rank = S.size(0)

                    if rank > max_possible_rank:
                        log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                        rank = max_possible_rank
                    U_select = U[:, :rank]
                    S_select = S[:rank]
                    Vt_select = Vt[:rank, :]

                    A = U_select @ torch.diag(torch.sqrt(S_select))  # [c_out, r]
                    # FedSA default: B zero, do not preload low-rank recon into adapter
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix
                else:
                    A = torch.empty(c_out, rank, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    if init_method == "kaiming":
                        init.kaiming_normal_(A, mode="fan_out", nonlinearity="relu")
                    elif init_method == "gaussian":
                        init.normal_(A, mean=0.0, std=0.01)
                    elif init_method == "orthogonal":
                        init.orthogonal_(A)
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix

            # --------------------------
            # Linear
            # --------------------------
            else:
                d_out, d_in = weight_matrix.shape

                if init_method == "svd":
                    U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
                    max_possible_rank = S.size(0)

                    if rank > max_possible_rank:
                        log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                        rank = max_possible_rank
                    U_select = U[:, :rank]
                    S_select = S[:rank]
                    # A from SVD, B zero
                    A = U_select @ torch.diag(torch.sqrt(S_select))  # [d_out, r]
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix
                else:
                    A = torch.empty(d_out, rank, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    if init_method == "kaiming":
                        init.kaiming_normal_(A, mode="fan_out", nonlinearity="relu")
                    elif init_method == "gaussian":
                        init.normal_(A, mean=0.0, std=0.01)
                    elif init_method == "orthogonal":
                        init.orthogonal_(A)
                    B = torch.zeros(rank, d_in, device=weight_matrix.device, dtype=weight_matrix.dtype)
                    W_res = weight_matrix

            log(INFO, f"Layer {name}: Applied FedSA-LoRA with rank {rank} (A init={init_method}, B=zero).")

        else: # Other SVD-based methods
            if isinstance(layer, torch.nn.Conv2d): #Conv2d layer SVD
                # weight_matrix = weight_matrix.view(weight_matrix.size(0), -1)  # Flatten Conv2d weights
                c_out, c_in, k1, k2 = weight_matrix.shape
                W_flat = weight_matrix.view(c_out, -1)  # Shape: [c_out, c_in * k1 * k2]

                # SVD decompistion
                U, S, Vt = torch.linalg.svd(W_flat, full_matrices=False)
                max_possible_rank = S.size(0)
                if rank > max_possible_rank:
                    log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                    rank = max_possible_rank
                
                # Select components based on method
                if method == 'pissa' or method == "flex_lora":
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

                elif method == 'fedkls':
                    index_start = math.floor(kl_norm * (max_possible_rank - rank)) if kl_norm is not None else 0
                    index_end = index_start + rank
                    if client_id is not None:
                        log(INFO, f"Client {client_id}: SVD applied with index range {index_start} to {index_end} with rank {rank} for layer {name}.")
                    U_select = U[:, index_start:index_end]
                    S_select = S[index_start:index_end]
                    Vt_select = Vt[index_start:index_end, :]

                else:
                    raise ValueError(f"Unknown method: {method}")
            
                # Construct A and B matrices for Conv2d layer 
                W_res = weight_matrix - (U_select @ torch.diag(S_select) @ Vt_select).view(c_out, c_in, k1, k2)
                A = U_select @ torch.diag(torch.sqrt(S_select))  # Shape: [c_out, rank]
                B = torch.diag(torch.sqrt(S_select)) @ Vt_select  # Shape: [rank, c_in * k1 * k2]

                # ----- Compute relative differences -----
                rel_recon_error = torch.norm(weight_matrix - (A @ B).view(c_out, c_in, k1, k2)) / torch.norm(weight_matrix)
                print(f"Relative reconstruction error (W vs ΔW): {rel_recon_error:.6f}")

            else: #Linear layer SVD
                U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False) 
                max_possible_rank = S.size(0)
                if rank > max_possible_rank:
                    log(INFO, f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
                    rank = max_possible_rank

                # Select components based on method
                if method == 'pissa' or method == "flex_lora":
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

                elif method == 'fedkls':
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

                rel_recon_error = torch.norm(weight_matrix - A @ B) / torch.norm(weight_matrix)
                log(INFO, f"Layer {name}: Relative reconstruction error (W vs ΔW): {rel_recon_error:.6f}")


            log(INFO, f"Layer {name}: Applied {method} with rank {rank}.")

        # Create appropriate adapter
        if isinstance(layer, torch.nn.Conv2d):
            new_layer = ConvAdapter(original_conv=layer, W_res=W_res, A=A, B=B, alpha=alpha, rank=rank, use_dp = dp_fedsvd)
        else:
            new_layer = SVDAdapter(W_res=W_res, A=A, B=B, alpha=alpha, rank=rank, original_bias=original_bias, use_dp = dp_fedsvd)
        
        # Freeze A for FFA-LoRA (external control)
        if method == "ffa_lora":
            try:
                new_layer.A.requires_grad = False
            except Exception as e:
                log(INFO, f"Warning: Could not freeze A for layer {name}: {e}")

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

def compute_client_distributions(config, dataset, num_clients: int) -> dict:
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
        if config["common"]["dataset"] == "pranavmr/MM-IMDb":
            client_data = dataset[cid]["train"]
        else:
            client_data = dataset.load_partition(cid)
        dataset_name = config["common"]["dataset"]
        output_column = dataset_info[dataset_name]["output_column"]
        labels = [item[output_column] for item in client_data]
        client_distributions[cid] = dict(sorted(Counter(labels).items()))
        log(INFO, f"Client {cid} ({len(client_distributions[cid])} classes, {len(client_data)} samples) : {client_distributions[cid]}")
    log(INFO, f"Total samples from all clients: {sum([sum(dist.values()) for dist in client_distributions.values()])}")
    log(INFO, "*" * 150)
    
    return client_distributions

def get_model(config, shape, classnames=None):
    model_name = config["common"]["model"]
    # get num_classes
    dataset_name = config["common"]["dataset"]
    num_classes = dataset_info[dataset_name]["num_classes"]

    TEXT_ONLY_DATASETS = {"SetFit/20_newsgroups", "legacy-datasets/banking77", "fancyzhx/dbpedia_14"}
    # PFedMoAP CLIP guard
    if model_name == "Clip":
        if dataset_name in TEXT_ONLY_DATASETS:
            raise ValueError(f"PFedMoAP CLIP requires image dataset, got text dataset: {dataset_name}")

        pf = config["pfedmoap_config"]
        if classnames is None:
            classnames = [f"class{i}" for i in range(num_classes)]
        model = getattr(__import__("mak.models", fromlist=[model_name]), model_name)(
            num_classes=num_classes,
            input_shape=shape,
            backbone_name=pf.get("backbone_name", "ViT-B/32"),
            classnames=classnames,
            prompt_len=pf["prompt_len"],
            num_experts=pf.get("num_experts", 4),
            dgating=pf["dgating"],
            gating_heads=pf.get("heads", pf.get("gating_heads", 4)),
            lambda_local=pf.get("lambda_local", 1.0),
            freeze_text=True,
        )
        return model
    # check if model is from huggingface
    elif model_name in [
        "distilbert-base-uncased", 
        "bert-base-uncased", 
        "roberta-base",
        "roberta-large",
        "microsoft/deberta-v3-base",
        "Qwen/Qwen1.5-0.5B", 
        "meta-llama/Llama-2-7b-hf",
        "openai/clip-vit-base-patch32",
        ]:  # Add more as needed
        from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, CLIPModel
        if model_name == "Qwen/Qwen1.5-0.5B": #Need to check again when applying the quantization -> still error
            quantization_8_bit_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            quantization_4_bit_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # Use 4-bit quantization
                    bnb_4bit_quant_type="nf4",  # Normal float 4-bit quantization
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Optimize compute dtype
                    bnb_4bit_use_double_quant=True,  # Enable double quantization
                )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                # quantization_config=quantization_8_bit_config,
            )
            # Set pad_token_id to eos_token_id
            if base_model.config.pad_token_id is None:
                base_model.config.pad_token_id = base_model.config.eos_token_id
        elif model_name == "openai/clip-vit-base-patch32": #For multimodal dataset MM-IMDB
            clip_model = CLIPModel.from_pretrained(model_name)
            class CustomCLIP(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.vision_model = clip_model.vision_model
                    self.text_model = clip_model.text_model
                    self.visual_projection = clip_model.visual_projection
                    self.text_projection = clip_model.text_projection
                    embed_dim = clip_model.config.projection_dim #512
                    self.classifier = torch.nn.Linear(embed_dim*2, num_classes) # Concatenate vision + text embeds
                def forward(self, pixel_values, input_ids, attention_mask):
                    vision_outputs = self.vision_model(pixel_values=pixel_values)
                    text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
                    vision_embeds = self.visual_projection(vision_outputs.pooler_output)
                    text_embeds = self.text_projection(text_outputs.pooler_output)
                    # Concatenate vision and text embeddings
                    combined = torch.cat([vision_embeds, text_embeds], dim=1)
                    logits = self.classifier(combined)
                    return logits
            base_model = CustomCLIP()
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

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
        
        strategy = config_sim.get("server", {}).get("strategy", "")
        method = config_sim.get("peft", {}).get("method", "")
        bias = config_sim.get("peft", {}).get("bias", "")

        # Handle FedSVD strategy - need special handling for parameter mapping
        if strategy == "FedSVD":
            fedsvd_mode = config_sim.get("fedsvd_config", {}).get("mode", "fedavg")
            if fedsvd_mode == "ffa":
                method = "ffa_lora"  # Treat as FFA-LoRA (only B matrices)
            else:
                method = "lora"  # FedAvg mode: both A and B

        if strategy == "PFedMoAP" or method == "pfedmoap":
            if len(parameters) != 1:
                raise ValueError(f"PFedMoAP centralized eval expects 1 prompt, got {len(parameters)}")

            prompt = torch.from_numpy(np.asarray(parameters[0])).to(device=device)
            # Use model API directly
            if hasattr(model, "set_prompt"):
                model.set_prompt(prompt)
            if hasattr(model, "clear_nonlocal"):
                model.clear_nonlocal()
        elif strategy == "FlexLoRA" or method == "flex_lora":
            # Lazy import to avoid circular dependency
            from mak.utils.flex_lora_utils import load_server_eval_params_flex_lora
            load_server_eval_params_flex_lora(
                model=model,
                parameters=parameters,
                device=device,
            )
        else:
            set_params(model, parameters, method=method, bias=bias)

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
        #     torch.save(
        #         model.state_dict(), os.path.join(save_model_dir, "saved_best_model.pth")
        #     )

        # if server_round == config_sim["server"]["num_rounds"]:
        #     torch.save(
        #         model.state_dict(),
        #         os.path.join(save_model_dir, "saved_final_model.pth"),
        #     )
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


def get_server(strategy, client_manager, out_file_path, target_acc, num_train_thread, num_test_thread):
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
    elif isinstance(strategy, FFALoRAStrategy):
        return FFALoRAServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, PFedMoAPStrategy):
        return PFedMoAPServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, FedPOEStrategy):
        return FedPOEServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, FedPOERegressionTextStrategy):
        return FedPOERegressionTextServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, PFedMoAPStrategy):
        return PFedMoAPServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, FedPOEStrategy):
        return FedPOEServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, FedPOERegressionTextStrategy):
        return FedPOERegressionTextServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, PFedMoAPStrategy):
        return PFedMoAPServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, FedSALoRAStrategy):
        return FedSALoRAServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    elif isinstance(strategy, FlexLoRAStrategy):
        return FlexLoRAServer(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
        )
    else:           
        return ServerSaveData(
            strategy=strategy,
            client_manager=client_manager,
            out_file_path=out_file_path,
            target_acc=target_acc,
            num_train_thread=num_train_thread,
            num_test_thread=num_test_thread,
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
        "FedAWA": {
            "config": config,
            "model": model,
            "test_data": test_data,
            "apply_transforms_test": apply_transforms_test,
        },
        "FFALoRA": {
            "config": config,
        },
        "PFedMoAP": {
            "config": config,
        },
        "FedSALoRA": {
            "config": config,
        },
        "FlexLoRA": {
            "config": config,
            "model": model,
            "global_rank": config.get("flex_lora_config", {}).get(
                "global_rank", config.get("peft", {}).get("rank", 32)
            ),
        },
        "PowD": {
            "candidate_client_set": config["powd_config"]["candidate_client_set"],
        },
    } 

    # FedPOE configuration (Hedge mixture weights a/b updated from client-reported losses)
    if STRATEGY == "FedPOE":
        fedpoe_cfg = config.get("fedpoe_config", {}) or {}
        kwargs["FedPOE"] = {
            "eta": float(fedpoe_cfg.get("eta", 0.0) or 0.0),
        }

    # FedPOERegressionText configuration (Fed-POE regression-style over text embeddings)
    if STRATEGY == "FedPOERegressionText":
        poe_cfg = config.get("fedpoe_regression_text_config", {}) or {}
        kwargs["FedPOERegressionText"] = {
            "eta": float(poe_cfg.get("eta", 0.0) or 0.0),
            "lam": float(poe_cfg.get("lam", 0.0) or 0.0),
            "num_kernels": int(poe_cfg.get("num_kernels", 4) or 4),
            "n_components": int(poe_cfg.get("n_components", 256) or 256),
            "pooling": str(poe_cfg.get("pooling", "auto") or "auto"),
        }

    # Centralized evaluation:
    # - For most strategies: require a test split (test_data must not be None).
    # - For FedPOE / FedPOERegressionText: if the dataset has no test split
    #   (test_data is None), skip centralized evaluation because these methods
    #   rely on client-side eval signals.
    if test_data is None and STRATEGY in {"FedPOE", "FedPOERegressionText"}:
        log(INFO, f"{STRATEGY}: centralized testset is None -> skipping centralized evaluation.")
        evaluate_fn = None
    else:
        evaluate_fn = get_evaluate_fn(
            centralized_testset=test_data,
            config_sim=config,
            save_model_dir=save_model_dir,
            metrics_file=out_file_path,
            device=device,
            apply_transforms_test=apply_transforms_test,
            model=model,
        )

    # FedSVD configuration (matches config.yaml:fedsvd_config)
    # NOTE: The strategy can aggregate subsets (LoRA A/B) only if it can map
    # incoming ndarrays to parameter names. For standard PyTorch models we can
    # derive names from `model.state_dict()`. For other protocols, the strategy
    # falls back to aggregating all arrays.
    if STRATEGY == "FedSVD":
        fedsvd_cfg = config.get("fedsvd_config", {}) or {}
        kwargs["FedSVD"] = {
            "mode": fedsvd_cfg.get("mode", "fedavg"),
            "send_deltas": bool(fedsvd_cfg.get("send_deltas", False)),
            "agg_flora": bool(fedsvd_cfg.get("agg_flora", False)),
            "agg_fedex": bool(fedsvd_cfg.get("agg_fedex", False)),
            "recalculate_svd_period": int(fedsvd_cfg.get("recalculate_svd_period", 0) or 0),
            "svd_warmup_steps": int(fedsvd_cfg.get("svd_warmup_steps", 0) or 0),
            "debug": bool(fedsvd_cfg.get("debug", False)),
            "bias": bool((config.get("peft", {}) or {}).get("bias", True)),
            # Provide parameter names so the strategy can select LoRA A/B.
            # IMPORTANT: must match the same deterministic order used by `initial_parameters` below.
            "param_name_fn": (lambda: [k for k, _ in sorted(model.state_dict().items())]) if model is not None else None,
        }

    if STRATEGY == "PFedMoAP":
        prompt_len = config["pfedmoap_config"]["prompt_len"]
        
        prompt_dim = int(model.prompt_learner.ctx.shape[1])
        config["pfedmoap_config"]["prompt_dim"] = prompt_dim 

        # init global prompt
        prompt0 = (0.02 * np.random.randn(prompt_len, prompt_dim)).astype(np.float32)
        init_params = fl.common.ndarrays_to_parameters([prompt0])
    elif STRATEGY == "FedPOERegressionText":
        # This strategy only exchanges a lightweight theta vector for an RFF head.
        # The theta shape is derived client-side from embedding dim, so we start empty.
        init_params = fl.common.ndarrays_to_parameters([])
    else:
        # Sort keys for deterministic order (critical for proper parameter loading)
        sorted_state_dict = sorted(model.state_dict().items())
        init_params = fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in sorted_state_dict]
        )

    return getattr(__import__("mak.strategies", fromlist=[STRATEGY]), STRATEGY)(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVAL,
        min_fit_clients=MIN_CLIENTS_FIT,
        min_evaluate_clients=MIN_CLIENTS_EVAL,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_fit_config_fn(config_sim=config),
        on_evaluate_config_fn=get_evaluate_config_fn(config_sim=config),
        initial_parameters=init_params,
        **kwargs.get(STRATEGY, {}),
    )


def set_seed(seed: int = 13):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
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
            "current_round": server_round, #Add current_round for dynamic data updates
            "batch_size": config_sim["client"]["batch_size"],
            "epochs": config_sim["client"]["epochs"],
            "lr": config_sim["client"]["lr"],
            "optimizer": config_sim["common"]["optimizer"],
            "sgd_momentum": config_sim["common"]["sgd_momentum"],
            "strategy": config_sim["server"]["strategy"],
            "proximal_mu": config_sim["fedprox"]["proximal_mu"],
            "loss": config_sim["client"]["loss"],
        }
        # P2 FIX: Add explicit payload kind for FlexLoRA
        if config["strategy"] == "FlexLoRA":
            if server_round == 1:
                config["payload_kind"] = "full"
            else:
                config["payload_kind"] = "partial"
        return config

    return fit_config

def get_evaluate_config_fn(config_sim):
    def evaluate_config(server_round: int):
        """Return evaluation configuration dict for each round.
        
        passes the current round number to the client
        """
        config = {
            "round": server_round,
            "current_round": server_round,  # Add current_round for dynamic data updates
        }
        return config
    
    return evaluate_config

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
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method for SVD adaptation (e.g., 'lora', 'pissa', 'milora', 'middle', 'fedkls')",
    )
    parser.add_argument(
        "--enabled",
        type=bool,
        default=None,
        help="Enable or disable the method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to be used for the simulation",
    )

    args = parser.parse_args()
    return args


def get_optimizer(model, client_config):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        # Safety net for FedSVD: if adapter params exist but were accidentally
        # frozen, unfreeze them by name to avoid hard-crashing.
        if client_config.get("strategy") == "FedSVD":
            has_adapters = False
            for n, p in model.named_parameters():
                if n.endswith(".A") or n.endswith(".B"):
                    has_adapters = True
                    p.requires_grad = True
            if has_adapters:
                params = [p for p in model.parameters() if p.requires_grad]

        if len(params) == 0:
            raise ValueError("No trainable parameters found (all requires_grad=False)")

    if client_config["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=client_config["lr"])
    else:
        return torch.optim.SGD(
            params,
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
