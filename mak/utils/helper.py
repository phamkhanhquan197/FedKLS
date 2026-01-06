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
from sklearn.model_selection import train_test_split

import mak
from mak.servers.custom_server import ServerSaveData
from mak.servers.fedklsvd_server import FedKLSVDServer
from mak.servers.ffa_lora_server import FFALoRAServer
from mak.servers.flex_lora_server import FlexLoRAServer
from mak.servers.fednova_server import FedNovaServer
from mak.servers.scaffold_server import ScaffoldServer
from mak.servers.pfedmoap_server import PFedMoAPServer
from mak.strategies.fednova_strategy import FedNovaStrategy
from mak.strategies.scaffold_strategy import ScaffoldStrategy
from mak.strategies.fedklsvd_strategy import FedKLSVDStrategy
from mak.strategies.ffa_lora_strategy import FFALoRAStrategy
from mak.strategies.flex_lora_strategy import FlexLoRAStrategy
from mak.strategies.pfedmoap_strategy import PFedMoAPStrategy
from mak.utils.dataset_info import dataset_info
from mak.utils.general import set_params, test, weighted_average
from mak.models.svd_model import SVDAdapter, ConvAdapter
import math
from collections import Counter
import torch.nn.init as init
from datasets import load_dataset


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
        # get test column name
        test_set = dataset_info[dataset_name]["test_set"]
        centralized_testset = fds.load_split(test_set)

        # get class names for pFedMoAP
        out_col = dataset_info[dataset_name]["output_column"]
        feat = centralized_testset.features.get(out_col, None)
        if feat is not None and hasattr(feat, "names") and feat.names:
            classnames = list(feat.names)
        else:
            num_classes = dataset_info[dataset_name]["num_classes"]
            classnames = [f"class{i}" for i in range(num_classes)]

        return fds, centralized_testset, classnames


# ... (rest of the existing helper.py remains unchanged)


def get_ffa_target_keys(model) -> list[str]:
    """Return deterministic target keys for FFA-LoRA / FlexLoRA partial communication.

    Robust filtering rules:
    - LoRA factors: all params ending with `.A` or `.B`
    - Biases: all params ending with `.bias`
    - Classifier head: `.weight` params whose layer name contains one of:
      ["classifier", "head", "fc", "score", "linear"]

    Returns:
        List[str]: sorted list of parameter names.
    """
    sd = model.state_dict()
    head_keywords = ("classifier", "head", "fc", "score", "linear")

    keys = []
    for k in sd.keys():
        if k.endswith(".A") or k.endswith(".B"):
            keys.append(k)
            continue
        if k.endswith(".bias"):
            keys.append(k)
            continue
        if k.endswith(".weight"):
            lk = k.lower()
            if any(word in lk for word in head_keywords):
                keys.append(k)
                continue

    # Deterministic ordering
    return sorted(set(keys))
