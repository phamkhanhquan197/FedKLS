import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
from logging import INFO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, logging as hf_logging
from datasets import load_dataset
from flwr_datasets.partitioner import DirichletPartitioner
from typing import List, Tuple, Dict, OrderedDict, Union
import numpy as np
import math
from collections import Counter
import copy

# Suppress Hugging Face logging for cleaner output during simulation
hf_logging.set_verbosity_error()

# Configuration (Simplified for this test)
CONFIG = {
    "common": {
        "seed": 42,
        "num_total_classes": 20, # For 20 Newsgroups
    },
    "server": {
        "num_rounds": 50, # Keep it short for testing
        "num_clients": 10,
        "min_fit_clients": 10, # Train both clients each round
        "min_available_clients": 10,
    },
    "client": {
        "epochs": 1, # Local epochs
        "batch_size": 32,
        "lr": 5e-4, # May need tuning for LoRA
    },
    "lora": {
        "rank": 32,
        "alpha": 32, # Set alpha = rank for initial scaling of 1.0 in SVDAdapter
        "train_classifier_head": True, # Crucial to train the head
        # "method" will be implicitly handled by kl_norm for this test
    }
}


class SVDAdapter(nn.Module):
    def __init__(self, W_res, A, B, alpha, rank, original_bias=None):
        super().__init__()
        self.A = nn.Parameter(A.clone().detach()) # Trainable
        self.B = nn.Parameter(B.clone().detach()) # Trainable
        self.alpha = alpha # LoRA scaling factor
        self.rank = rank
        self.scaling = alpha/rank
        self.bias = nn.Parameter(original_bias.clone().detach())
        self.W_res = W_res.cuda()
        self.W_res.require_grad = False #Freeze the residual matrix

    def forward(self, x):
        """
        Performs the forward pass of the SVDAdapter.

        The computation is equivalent to:
        Output = x @ (W_res + scaling * A @ B)^T + bias
               = F.linear(x, W_res + scaling * A @ B, bias)

        Args:
            x (torch.Tensor): Input tensor.
                              Expected shape: [batch_size, ..., in_features]
                              where in_features must match self.W_res.shape[1],
                              self.B.shape[1].

        Returns:
            torch.Tensor: Output tensor.
                          Shape: [batch_size, ..., out_features]
                          where out_features is self.W_res.shape[0], self.A.shape[0].
        """
        effective_weight = self.W_res + self.scaling * (self.A @ self.B)
        output = F.linear(x, effective_weight, self.bias)
        return output
    

    def __repr__(self):
        bias_shape = list(self.bias.shape)
        bias_trainable = self.bias.requires_grad
        bias_info = f", bias={bias_shape} (trainable: {bias_trainable})"

        return (
            f"{self.__class__.__name__}("
            f"W_res: {list(self.W_res.shape)} (buffer, frozen), "
            f"A: {list(self.A.shape)} (trainable: {self.A.requires_grad}), "
            f"B: {list(self.B.shape)} (trainable: {self.B.requires_grad}), "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
            f"{bias_info})"
        )
    

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

def apply_svd_to_model(model, config):
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
        # log(INFO, f"  Adapting layer: {name}")
        # Get the weight matrix of the linear layer
        weight_matrix = layer.weight.data
        # Get the original bias
        original_bias = layer.bias.data

        # Perform SVD
        U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)
        rank = config["lora"]["rank"]
        alpha = config["lora"]["alpha"]


        max_possible_rank = S.size(0)
        if rank > max_possible_rank:
            log(INFO,f"Warning: Requested rank {rank} for layer {name} > max possible rank {max_possible_rank}.")
            rank = max_possible_rank


        kl_norm = config["lora"]["kl_norm"]
        index_start = math.floor(kl_norm * (max_possible_rank - rank))
        index_end = index_start + rank
        U_select = U[:,index_start:index_end]
        S_select = S[index_start:index_end]
        Vt_select = Vt[index_start:index_end,:]

        W_res = weight_matrix - (U_select @ torch.diag(S_select) @ Vt_select)

        # Initialize the adapter matrices with SVD components
        A = U_select @ torch.diag(torch.sqrt(S_select))
        B = torch.diag(torch.sqrt(S_select)) @ Vt_select

        # 1. Create the original layer with the SVDApapter
        new_layer = SVDAdapter(W_res=W_res, A=A, B=B, alpha=alpha, rank=rank, original_bias=original_bias)
        
        # 2. Split layer name into parent and child components
        # Example: "transformer.layer.0.attention.q_lin" becomes:
        # parent_name = "transformer.layer.0.attention"
        # child_name = "q_lin". 

        # 3. Get the parent module containing the original layer
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)

        # 4. Replace the original layer with the new layer
        setattr(parent, child_name, new_layer)  
    
    return model



# --- Client Definition --- ~~ class BaseClient in FedEasy
class SvdTestClient(fl.client.NumPyClient):
    def __init__(self, cid, base_model_constructor, train_loader, val_loader, client_specific_config, device):
        self.cid = cid
        self.config = client_specific_config # This contains ["lora"]["kl_norm"]
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # log(INFO, f"[Client {self.cid}] Initializing with LoRA config: {self.config['lora']}")
        fresh_base_model = base_model_constructor()
        self.model = apply_svd_to_model(fresh_base_model, self.config)     
        self.model.to(self.device)

    def get_parameters(self, config):
        # params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if "lin" in name}
        # return [tensor.cpu().numpy() for tensor in params_to_send.values()]
        # print(len(params_to_send))
        # print(len([p.detach().cpu().numpy() for p in self.model.parameters() if p.requires_grad]))
        return [p.detach().cpu().numpy() for p in self.model.parameters() if p.requires_grad]

    def set_parameters(self, parameters):
        self.set_params(self.model, parameters)

    def set_params(self, model, parameters):
        # trainable_model_params = [p for p in self.model.parameters() if p.requires_grad]
        # for model_p, server_p_np in zip(trainable_model_params, parameters):
        #     model_p.data.copy_(torch.from_numpy(server_p_np).to(model_p.device))
        """Set model weights from a list of NumPy ndarrays."""
        model_state = model.state_dict()
        if len(model_state.items()) != len(parameters): # Handle LoRA parameter update
            lora_keys = [k for k in model_state.keys() 
                        if ("lin" in k)]

            # Create state dict with only LoRA parameters
            lora_params = OrderedDict()
            for key, array in zip(lora_keys, parameters):
                lora_params[key] = torch.from_numpy(array)
            
            # Update model with LoRA parameters only
            model_state.update(lora_params)
            model.load_state_dict(model_state, strict=True)

            
        else: #Full parameter update
            params_dict = zip(model_state.keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config_from_server): # config_from_server from strategy's on_fit_config_fn
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.config["client"]["lr"])
        
        for epoch in range(self.config["client"]["epochs"]):
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config_from_server):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item() * labels.size(0)
                
                logits = outputs.logits
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else float('inf')
        # log(INFO, f"Client {self.cid} Eval: Acc={accuracy:.4f}, Loss={avg_loss:.4f}")
        return avg_loss, total, {"accuracy": accuracy}

# --- Main Simulation ---
def main():
    torch.manual_seed(CONFIG["common"]["seed"])
    np.random.seed(CONFIG["common"]["seed"])
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Using device: {DEVICE}")

    # Load 20 Newsgroups dataset
    log(INFO, "Loading 20 Newsgroups dataset...")
    dataset_20news = load_dataset("SetFit/20_newsgroups", split="train")
    # For simplicity, use a small subset for train and test for this example
    # And create a dummy federated structure
    
    # Create tokenizer
    MODEL_NAME = "distilbert-base-uncased" # Fixed for this test
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    dataset_20news = dataset_20news.map(tokenize_function, batched=True)
    dataset_20news = dataset_20news.rename_column("label", "labels") # For HF Trainer compatibility
    dataset_20news.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Split dataset for two clients and a centralized test set
    # This is a simplified non-IID split for demonstration
    # Client 0 gets more of first half, Client 1 gets more of second half
    data_len = len(dataset_20news)
    indices = list(range(data_len))
    np.random.shuffle(indices) # Shuffle once for all splits

    # Simplified split:
    # Client 0: samples 0-399, Client 1: samples 400-799, Test: 800-999 (small test)
    # To make it somewhat non-IID based on original ordering (which might have topic clusters)
    # let's assign blocks without shuffling first, then take subsets for testing.
    full_train_data = load_dataset("SetFit/20_newsgroups", split="train")
    full_test_data = load_dataset("SetFit/20_newsgroups", split="test") # Use official test set

    def client_data_transform(data_subset):
        data_subset = data_subset.map(tokenize_function, batched=True, remove_columns=["text", "label_text"])
        data_subset = data_subset.rename_column("label", "labels")
        data_subset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return data_subset
    

    # Create data partitioning for 10 clients
    dataset_size = len(full_train_data)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    # Partition data for 10 clients
    client_datasets = []
    samples_per_client = dataset_size // CONFIG["server"]["num_clients"]
    
    for i in range(CONFIG["server"]["num_clients"]):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_data = Subset(full_train_data, indices[start_idx:end_idx])
        client_datasets.append(client_data)
    
    centralized_test_dataset = client_data_transform(full_test_data.select(range(1000))) # Use 1000 samples for test

    
    # Transform and prepare client datasets
    client_train_datasets = []
    for client_data in client_datasets:
        dataset = client_data_transform(client_data.dataset.select(client_data.indices))
        client_train_datasets.append(dataset)
    
    # Create dataloaders for all clients
    client_train_loaders = [
        DataLoader(dataset, batch_size=CONFIG["client"]["batch_size"], shuffle=True)
        for dataset in client_train_datasets
    ]
    client_val_loaders = [
        DataLoader(dataset, batch_size=CONFIG["client"]["batch_size"])
        for dataset in client_train_datasets  # Using same data for validation
    ]


    def model_constructor():
        return DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=CONFIG["common"]["num_total_classes"])

    # Client function
    def client_fn(cid: str):
        client_specific_config = copy.deepcopy(CONFIG) # Base config
        if "lora" not in client_specific_config: client_specific_config["lora"] = {}

        kl_norm_values = {
            "0": 0.0000,
            "1": 0.5738,
            "2": 0.2761,
            "3": 0.1272,
            "4": 0.6142,
            "5": 0.3575,
            "6": 1.0000,
            "7": 0.4382, 
            "8": 0.6864,
            "9": 0.8523
        }
        
        client_specific_config["lora"]["kl_norm"] = kl_norm_values[cid]
        log(INFO, f"Client {cid} using kl_norm: {client_specific_config['lora']['kl_norm']}")
        
        # Use the appropriate dataloader for this client
        client_id = int(cid)
        return SvdTestClient(
            cid, 
            model_constructor, 
            client_train_loaders[client_id], 
            client_val_loaders[client_id], 
            client_specific_config, 
            DEVICE
        )

    # Server-side evaluation function
    def get_evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Union[Tuple[float, Dict[str, fl.common.Scalar]], None]:
        if server_round == 0 and not parameters: # Initial parameters might be empty before first fit
            # Or, if FedAvg starts with initial_parameters, this branch might not be needed.
            # For this test, let's ensure the model used for eval always gets some params.
            # If FedAvg has initial_parameters, it will pass them here for round 0.
            log(INFO, "Round 0 evaluation: No aggregated parameters yet (or using initial_parameters directly).")

        eval_model_base = model_constructor()
        # Adapt server model with a default (e.g., PiSSA-like) for consistent structure evaluation
        server_eval_config = copy.deepcopy(CONFIG)
        if "lora" not in server_eval_config: server_eval_config["lora"] = {}
        server_eval_config["lora"]["kl_norm"] = 0.0 # Consistent SVD adaptation for eval model structure
        
        eval_model= apply_svd_to_model(eval_model_base, server_eval_config)
        eval_model.to(DEVICE)

        # Set current global parameters
        trainable_model_params = [p for p in eval_model.parameters() if p.requires_grad]
        if parameters and len(trainable_model_params) == len(parameters):
            for model_p, server_p_np in zip(trainable_model_params, parameters):
                model_p.data.copy_(torch.from_numpy(server_p_np).to(model_p.device))
        elif not parameters and server_round > 0 : # No parameters received for update means issue
             log(INFO, "Warning: Evaluate_on_server received no parameters for update in later rounds.")
             return None # Cannot evaluate meaningfully
        elif parameters and len(trainable_model_params) != len(parameters):
            log(INFO, f"Warning: Param mismatch in evaluate_on_server. Model trainable: {len(trainable_model_params)}, Received: {len(parameters)}. Cannot reliably set.")
            return None


        eval_model.eval()
        correct = 0
        total = 0
        total_loss_val = 0.0
        test_loader_central = DataLoader(centralized_test_dataset, batch_size=CONFIG["client"]["batch_size"])
        
        with torch.no_grad():
            for batch in test_loader_central:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                outputs = eval_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss_val += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss_val / total if total > 0 else float('inf')
        log(INFO, f"********************ROUND {server_round} EVAL********************")
        log(INFO, f"Server Evaluation (Round {server_round}): Accuracy={accuracy:.4f}, Avg Loss={avg_loss:.4f}")
        return avg_loss, {"accuracy": accuracy}

    # Initial parameters for FedAvg strategy
    # Create a template model, SVD-adapt it (e.g., PiSSA-like), freeze, get trainable params
    init_model_base = model_constructor()
    init_model_config = copy.deepcopy(CONFIG)
    if "lora" not in init_model_config: init_model_config["lora"] = {}
    init_model_config["lora"]["kl_norm"] = 0.0 # PiSSA-like for initial structure
    
    init_model_svd = apply_svd_to_model(init_model_base, init_model_config)

    initial_parameters = fl.common.ndarrays_to_parameters(
        [p.detach().cpu().numpy() for p in init_model_svd.parameters() if p.requires_grad]
    )
    if not [p for p in init_model_svd.parameters() if p.requires_grad]:
        log(INFO, "CRITICAL: Initial parameters for strategy are empty! No trainable parts found in template.")


    # Strategy ~~ get_strategy() in FedEasy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, # Train all clients
        min_fit_clients=CONFIG["server"]["min_fit_clients"],
        min_available_clients=CONFIG["server"]["num_clients"],
        fraction_evaluate=0.0, # No client-side evaluation in this simple test
        min_evaluate_clients=0,
        evaluate_fn=get_evaluate_fn, # Server-side evaluation
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda server_round: CONFIG["client"] # Send client training hyperparams
    )

    # Start simulation
    client_resources = {
    "num_cpus": 2,  # Adjust as needed
    "num_gpus": 1 # 
    }
    log(INFO, "Starting Flower simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=CONFIG["server"]["num_clients"],
        config=fl.server.ServerConfig(num_rounds=CONFIG["server"]["num_rounds"]),
        strategy=strategy,
        client_resources=client_resources,
        
    )

if __name__ == "__main__":
    main()