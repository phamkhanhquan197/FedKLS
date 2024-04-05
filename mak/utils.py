import yaml
import numpy as np
import random
import argparse
from collections import OrderedDict
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import Scalar
from logging import INFO, DEBUG
from flwr.common.logger import log

from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from datetime import datetime, date
import random, csv, os, json
from torchvision.models import resnet18
from mak.strategy.is_strategy import ImportanceSamplingStrategyLoss
import pandas as pd
from mak.client import FlowerClient
from mak.training import test, weighted_average, set_params
import mak.models as custom_models
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

from torchvision.models.resnet import resnet18 as resnet18_torch

import mak


def get_device_and_resources(config_sim):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() and config_sim['client']['gpu'] else "cpu")

    # Assign GPU and CPU resources
    if device.type == 'cuda':
        # Assign GPU resources
        num_gpus_total = config_sim['client']['total_gpus']
        if num_gpus_total > 0:
            ray_init_args = {'num_cpus': config_sim['client']['total_cpus'], 'num_gpus': num_gpus_total}
        else:
            ray_init_args = {'num_cpus': config_sim['client']['total_cpus'], 'num_gpus': 0}
    else:
        # Assign CPU resources
        ray_init_args = {'num_cpus': config_sim['client']['total_cpus'], 'num_gpus': 0}

    # Assign client resources
    client_res = {'num_cpus': config_sim['client']['num_cpus'], 'num_gpus': config_sim['client']['num_gpus'] if device.type == 'cuda' else 0.0}
    if config_sim['common']['multi_node']:
        ray_init_args["address"] = "auto"
        ray_init_args["runtime_env"] = {"py_modules" : [mak]} 
    return device, ray_init_args, client_res

def gen_dir_outfile_server(config):
    # generates the basic directory structure for out data and the header for file
    today = date.today()
    BASE_DIR = "out"
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

    # create a date wise folder
    if not os.path.exists(os.path.join(BASE_DIR, str(today))):
        os.mkdir(os.path.join(BASE_DIR, str(today)))

    # create saperate folder based on strategy
    if not os.path.exists(os.path.join(BASE_DIR, str(today), config['server']['strategy'])):
        os.mkdir(os.path.join(BASE_DIR, str(today), config['server']['strategy']))

    # create saperate folder based on data distribution type
    if not os.path.exists(os.path.join(BASE_DIR, str(today), config['server']['strategy'], config['common']['data_type'])):
        os.mkdir(os.path.join(BASE_DIR, str(today),config['server']['strategy'], config['common']['data_type']))

    dirs = os.listdir(os.path.join(BASE_DIR, str(today),
                        config['server']['strategy'], config['common']['data_type']))
    final_dir_path = os.path.join(BASE_DIR, str(
        today), config['server']['strategy'], config['common']['data_type'], str(len(dirs)))

    if not os.path.exists(final_dir_path):
        os.mkdir(final_dir_path)
    # if not os.path.exists(os.path.join(final_dir_path,'models')):
    #     os.mkdir(os.path.join(final_dir_path,'models'))
    # models_dir = os.path.join(final_dir_path,'models')
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    #save all confugration file as json file
    json_file_name = f"config.json"
    with open(os.path.join(final_dir_path,json_file_name), 'w') as fp:
        json.dump(config, fp,indent=4)
    dataset_str = config['common']['dataset'].replace('/','_')
    file_name = f"{config['server']['strategy']}_{dataset_str}_{config['common']['data_type']}_{config['client']['batch_size']}_{config['client']['lr']}_{config['client']['epochs']}"
    file_name = f"{file_name}.csv"
    out_file_path = os.path.join(
        final_dir_path, file_name)
    # create empty server history file
    if not os.path.exists(out_file_path):
        with open(out_file_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            header = ["round", "accuracy", "loss", "time"]
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()
    return out_file_path, final_dir_path

def get_partitioner(config_sim):
    num_clients = config_sim['server']['num_clients']
    if config_sim['common']['data_type'] == 'dirichlet_niid':
        dirchlet_alpha = config_sim['common']['dirichlet_alpha']
        partitioner = DirichletPartitioner(num_partitions=num_clients, partition_by="label",
                                           alpha=dirchlet_alpha, min_partition_size=5,
                                           self_balancing=True)
    else:
        partitioner = IidPartitioner(num_partitions=num_clients)
    return {"train":partitioner}

def get_dataset(config_sim):
    dataset_name=config_sim['common']['dataset']
    supported_datasets = ['mnist', 'cifar10', 'fashion_mnist', 'sasha/dog-food', 'zh-plus/tiny-imagenet']
    partitioner = get_partitioner(config_sim=config_sim)
    if dataset_name not in supported_datasets:
        raise Exception(f"Dataset name should be among : {supported_datasets}")
    else:
        fds = FederatedDataset(dataset=dataset_name, partitioners=partitioner)
        if dataset_name == 'zh-plus/tiny-imagenet':
            centralized_testset = fds.load_split("valid")
        else:
            centralized_testset = fds.load_split("train")
        return fds, centralized_testset

def get_model(config):
    model_name = config['common']['model']
    num_classes = 10
    if model_name == 'resnet18':
        return custom_models.resnet18(num_classes = num_classes)
    elif model_name == 'resnet18_pretrained':
        mod = resnet18_torch(weights='DEFAULT')
        mod.fc = nn.Linear(mod.fc.in_features, num_classes)
        return mod
    elif model_name == 'net':
        return custom_models.Net()
    elif model_name == 'cifarnet':
        return custom_models.CifarNet()
    else:
        raise Exception(f"No model found named : {model_name}")

def get_evaluate_fn(
    centralized_testset: Dataset,config_sim,device,save_model_dir,metrics_file, apply_transforms,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        model = get_model(config=config_sim)
        set_params(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms)

        # Disable tqdm for dataset preprocessing
        disable_progress_bar()

        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)
        metrics_df = pd.read_csv(metrics_file)
        if metrics_df['loss'].min() > loss:
            log(INFO,f" =>>>>> Min Loss improved from {metrics_df['loss'].min()} to : {loss} Saving best model having accuracy : {accuracy}")
            torch.save(model.state_dict(), os.path.join(save_model_dir,'saved_best_model.pth'))
            
        
        if server_round == config_sim['server']['num_rounds']:
            torch.save(model.state_dict(), os.path.join(save_model_dir,'saved_model_final.pth'))
            print(f"++++++++++ final accuracy : {accuracy} and loss : {loss}")

        return loss, {"accuracy": accuracy}

    return evaluate

def save_simulation_history(hist : fl.server.history.History, path):
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

    for loss in losses_centralized:
        c_rnd = loss[0]
        rounds.append(c_rnd)
        losses_centralized_dict[c_rnd] = loss[1]

    for loss in losses_distributed:
        c_rnd = loss[0]
        losses_distributed_dict[c_rnd] = loss[1]
    if 'accuracy' in metrics_distributed.keys():
        for acc in metrics_distributed['accuracy']:
            c_rnd = acc[0]
            accuracy_distributed_dict[c_rnd] = acc[1]
    if 'accuracy' in metrics_centralized.keys():
        for acc in metrics_centralized['accuracy']:
            c_rnd = acc[0]
            accuracy_centralized_dict[c_rnd] = acc[1]

    if len(metrics_distributed_fit) != 0:
        pass # TODO  check its implemetation later

    data = {"rounds" :rounds, "losses_centralized":losses_centralized_dict,"losses_distributed":losses_distributed_dict,
                 "accuracy_distributed": accuracy_distributed_dict,"accuracy_centralized" :accuracy_centralized_dict}
    
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate over each key in the data dictionary
    for key in data.keys():
        # If the key is 'rounds', set the 'rounds' column of the DataFrame to the rounds list
        if key == 'rounds':
            df['rounds'] = data[key]
        # Otherwise, create a new column in the DataFrame with the key as the column name
        else:
            column_data = []
            # Iterate over each round in the 'rounds' list and add the corresponding value for the current key
            for round_num in data['rounds']:
                # If the round number does not exist in the current key's dictionary, set the value to None
                if round_num not in data[key]:
                    column_data.append(None)
                else:
                    column_data.append(data[key][round_num])
            df[key] = column_data
    df.to_csv(os.path.join(path))


def get_strategy(config,test_data,save_model_dir,out_file_path, device,apply_transforms):
    STRATEGY = config['server']['strategy']
    model = get_model(config=config)
    MIN_CLIENTS_FIT = config['server']['min_fit_clients']
    MIN_CLIENTS_EVAL = 2
    NUM_CLIENTS = config['server']['num_clients']
    FRACTION_FIT = config['server']['fraction_fit']
    FRACTION_EVAL = config['server']['fraction_evaluate']
    if STRATEGY == 'weighted_loss':
        strategy = ImportanceSamplingStrategyLoss(
            model=model,
            test_data= test_data,
            fraction_fit=FRACTION_FIT,  # Sample 10% of available clients for training
            fraction_evaluate=FRACTION_EVAL,  # Sample 5% of available clients for evaluation
            min_fit_clients=MIN_CLIENTS_FIT,  # Never sample less than 2 clients for training
            min_evaluate_clients=MIN_CLIENTS_EVAL,  # Never sample less than 2 clients for evaluation
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=get_evaluate_fn(centralized_testset=test_data,config_sim=config,save_model_dir = save_model_dir,metrics_file = out_file_path,device=device,apply_transforms=apply_transforms),
            evaluate_metrics_aggregation_fn=weighted_average,
            device=device,
            on_fit_config_fn=get_fit_config_fn(config_sim=config),
    )
    elif STRATEGY == "fedprox": #from flwr 1.XX
        strategy = fl.server.strategy.FedProx(
            fraction_fit=FRACTION_FIT,
            fraction_evaluate= FRACTION_EVAL,
            min_fit_clients=MIN_CLIENTS_FIT,
            min_evaluate_clients=MIN_CLIENTS_EVAL,
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=get_evaluate_fn(centralized_testset=test_data,config_sim=config,save_model_dir = save_model_dir,metrics_file = out_file_path,device=device,apply_transforms=apply_transforms),
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=get_fit_config_fn(config_sim=config),
            proximal_mu = 0.5,
        )
    elif STRATEGY == "fedavgm":
        strategy = fl.server.strategy.FedAvgM(
            fraction_fit=FRACTION_FIT,
            fraction_evaluate= FRACTION_EVAL,
            min_fit_clients=MIN_CLIENTS_FIT,
            min_evaluate_clients=MIN_CLIENTS_EVAL,
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=get_evaluate_fn(centralized_testset=test_data,config_sim=config,save_model_dir = save_model_dir,metrics_file = out_file_path,device=device,apply_transforms=apply_transforms),
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=get_fit_config_fn(config_sim=config),
            server_learning_rate=1.0,
            server_momentum=0.2,
        )
    else: #default fedavg strategy
        strategy = fl.server.strategy.FedAvg(
        fraction_fit= FRACTION_FIT ,  # Sample 10% of available clients for training
        fraction_evaluate= FRACTION_EVAL,  # Sample 5% of available clients for evaluation
        min_fit_clients=MIN_CLIENTS_FIT,  # Never sample less than 2 clients for training
        min_evaluate_clients=MIN_CLIENTS_EVAL,  # Never sample less than 2 clients for evaluation
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(centralized_testset=test_data,config_sim=config,save_model_dir = save_model_dir,metrics_file = out_file_path,device=device,apply_transforms=apply_transforms),
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_fit_config_fn(config_sim=config),
    )
    
    return strategy


def set_seed(seed : int = 13):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set the seed for CUDA operations (if using GPU)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}")


def get_config(file_path):
    # Open the YAML file
    with open(file_path, 'r') as file:
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
            "batch_size" : config_sim['client']['batch_size'],
            "epochs" : config_sim['client']['epochs'],
            "lr" : config_sim['client']['lr']
        }
        return config
    return fit_config


