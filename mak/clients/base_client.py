import os

import flwr as fl
from torch.utils.data import DataLoader
import torch
from mak.utils.general import set_fedspec_params, set_params, test
from mak.utils.helper import get_optimizer
from mak.utils.dataset_info import dataset_info
from flwr.common.logger import log
from torch.utils.data import ConcatDataset
from logging import INFO

class BaseClient(fl.client.NumPyClient):
    """flwr base client implementaion"""

    def __init__(
        self,
        client_id,
        model, 
        trainset,
        valset,
        config_sim,
        device,
        save_dir,
        dataset=None, # NEW: FederatedDataset reference
        kl_norm=None, # NEW: Store KL divergence value for FedSpec
        apply_transforms=None, # NEW: transform function
        data_scheduler=None, # NEW: DynamicDataScheduler for round-aware allocation
        bias=None,
        rank_policy_map: dict | None = None,
        clip_collator=None,  # NEW: CLIPCollator for multimodal datasets
    ):
        self.client_id = client_id
        self.config_sim = config_sim
        self.trainset = trainset
        self.valset = valset
        self.model = model
        self.device = device
        self.train_batch_size = self.config_sim["client"]["batch_size"]
        self.test_batch_size = config_sim["client"]["test_batch_size"]
        self.save_dir = os.path.join(save_dir, "clients")
        self.dataset_name = self.config_sim["common"]["dataset"]
        self.feature_key = dataset_info[self.dataset_name]["feature_key"]
        self.output_column = dataset_info[self.dataset_name]["output_column"]
        self.bias = self.config_sim.get("peft", {}).get("bias", True)
        #NEW: Store dataset reference and transform function for dynamic reload
        self.dataset = dataset
        self.apply_transforms = apply_transforms
        self.partition_id = client_id
        self.data_scheduler = data_scheduler # NEW: Store scheduler
        self.clip_collator = clip_collator  # NEW: Store collator for multimodal
        self.kl_norm = kl_norm # NEW: Store KL divergence value for FedSpec

        self.optimizer = None
    def __repr__(self) -> str:
        return " Flwr base client"

    def get_parameters(self, config): #Client -> Server
        if self.config_sim["peft"]["enabled"] == True:
            #Only send the A, B and bias parameters to the server 
            if any(key.startswith("distilbert.") for key in self.model.state_dict().keys()):
                if self.bias:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if "lin" in name}
                else:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if name.endswith(".B") or name.endswith(".A")}
            elif any(key.startswith("roberta.") for key in self.model.state_dict().keys()):
                if self.bias:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if "self" in name or ("dense" in name and "classifier" not in name)}
                else:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if name.endswith(".B") or name.endswith(".A")}
            elif any(key.startswith("bert.") for key in self.model.state_dict().keys()):
                if self.bias:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if "self" in name or "dense" in name}
                else:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if name.endswith(".B") or name.endswith(".A")}
            elif any(key.startswith("model.") for key in self.model.state_dict().keys()):
                if self.bias:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if "self_attn" in name or "mlp" in name}
                else:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if name.endswith(".B") or name.endswith(".A")}
            elif self.config_sim["common"]["model"] in ["Resnet18", "Resnet34","ResNet18Pretrained", "ResNet34Pretrained"]:
                if self.bias:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if "conv" in name}
                else:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if name.endswith(".B") or name.endswith(".A")}
            elif any(key.startswith("vision_model") for key in self.model.state_dict().keys()) or any(key.startswith("text_model") for key in self.model.state_dict().keys()): #CustomCLIP
                if self.bias:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if ("self_attn" in name or "mlp" in name)}
                else:
                    params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if name.endswith(".B") or name.endswith(".A")}
            else:
                #Need to revise
                # For other models (e.g., ResNet, CNN), send all parameters if PEFT is enabled
                # This handles cases where the model doesn't match the above patterns
                # params_to_send = {name: tensor for name, tensor in self.model.state_dict().items()}
                raise ValueError("PEFT parameter extraction not defined for this model architecture.")

            # Print parameter names and shapes
            # print("\n=== Parameters Sent to Server ===")
            # for name, tensor in params_to_send.items():
            #     print(f"{name}: {tuple(tensor.shape)}")
            # print("=================================\n")

            # Convert to numpy arrays (preserving order)
            return [tensor.cpu().numpy() for tensor in params_to_send.values()]
        else: 
            # Send full model parameters to server
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def reload_dataset(self, mode: str, round_num: int=1):
        """
        Reload client-side dataset without touching model parameters.
        Uses DynamicDataScheduler for round-aware, disjoint allocation.

        Args:
            mode: "replace" | "append"
                - "replace": Drop toàn bộ dataset cũ, load dataset mới từ nguồn dữ liệu
                - "append": Giữ dataset cũ và thêm dữ liệu mới
            mode: "replace" | "append" (legacy, kept for compatibility)
                - "replace": Drop entire old dataset, load new dataset from scheduler
                - "append": For incremental mode, dataset size increases monotonically
            round_num: Current round number for schedule lookup
        """
        # Use scheduler if available (new approach)
        if self.data_scheduler is not None:
            trainset, valset = self.data_scheduler.get_client_round_datasets(
                client_id=self.client_id,
                round_num=round_num,
                apply_transforms=self.apply_transforms if self.clip_collator is None else None  # No transform for multimodal
            )
            self.trainset = trainset
            self.valset = valset
            return
        
        # Fallback to old approach if scheduler not available
        if self.dataset is None or self.apply_transforms is None:
            raise RuntimeError("Dataset reference or transform function not provided.")

        # For multimodal: apply_transforms can be None (collator handles it)
        if self.clip_collator is None and self.apply_transforms is None:
            raise RuntimeError("Transform function not provided for non-multimodal dataset.")
        
        client_dataset_total = self.dataset.load_partition(
            partition_id=self.partition_id
        )

        splits = client_dataset_total.train_test_split(
            test_size=0.2,
            seed=self.config_sim["common"]["seed"],
        )

        # For multimodal: keep dataset raw (no transform), collator handles processing
        if self.clip_collator is None:
            new_trainset = splits["train"].with_transform(self.apply_transforms)
            new_valset = splits["test"].with_transform(self.apply_transforms)
        else:
            new_trainset = splits["train"]  # Raw dataset
            new_valset = splits["test"]     # Raw dataset

        if mode == "append":
            self.trainset = ConcatDataset([self.trainset, new_trainset])
            self.valset = ConcatDataset([self.valset, new_valset])
        else:
            self.trainset = new_trainset
            self.valset = new_valset

    def set_parameters(self, parameters):
        method = self.config_sim["peft"]["method"] if self.config_sim["peft"]["enabled"] else None
        bias = self.config_sim["peft"]["bias"] if self.config_sim["peft"]["enabled"] else None
        if method == "fedspec":
            set_fedspec_params(self.model, parameters, bias=bias)
        else:
            set_params(self.model, parameters, method=method, bias=bias)

    def count_class_distribution(self, dataset):
        """Count the class distribution in the dataset."""
        class_counts = {}
        for batch_data in dataset:
            if self.feature_key in ["text", "content", "sentence"] or self.feature_key == ["image", "text"]:
                labels = batch_data["labels"].to(self.device)
            else:
                labels = batch_data[self.output_column].to(self.device)
                
            # Count the occurrences of each class in the batch
            unique, counts = torch.unique(labels, return_counts=True)
            
            for class_id, count in zip(unique.tolist(), counts.tolist()):
                class_counts[class_id] = class_counts.get(class_id, 0) + count

        # Sort the class counts by class ID
        return dict(sorted(class_counts.items()))
    
    def fit(self, parameters, config):
        """
        Fit with dynamic dataset updates and strict model inheritance.
        """
        self.set_parameters(parameters)

        # Read dynamic data config
        dyn_cfg = self.config_sim.get("dynamic_data", {})
        enabled = dyn_cfg.get("enabled", False)
        mode = dyn_cfg.get("mode", "incremental")
        round_step = dyn_cfg.get("round_step", None)

        current_round = config.get("current_round", 0)

        # Decide whether to update dataset
        if enabled:
            # Always reload dataset to get round-specific indices and validation split
            # This ensures validation size changes when train size changes
            if self.data_scheduler is not None:
                self.reload_dataset(mode=mode, round_num=current_round)
            else:
                # Fallback to old approach
                if round_step is not None and (current_round % round_step == 0 or current_round == 1):
                    if mode == "reset":
                        self.reload_dataset(mode="replace", round_num=current_round)
                    elif mode == "incremental":
                        self.reload_dataset(mode="append", round_num=current_round)

        batch, epochs = (
            config["batch_size"],
            config["epochs"]
        )
        # Create a DataLoader for the training set
        # For multimodal: use collate_fn, enable num_workers for faster processing
        if self.clip_collator is not None:
            # Multimodal: use collator, enable workers
            trainloader = DataLoader(
                self.trainset,
                batch_size=batch,
                shuffle=True,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=True if min(4, os.cpu_count() or 1) > 0 else False,
                prefetch_factor=2,
                collate_fn=self.clip_collator,
            )
        else:
            # Non-multimodal: check if local function (legacy)
            is_local_function = '<locals>' in self.apply_transforms.__qualname__ if self.apply_transforms else False
            num_workers = 0 if is_local_function else min(4, os.cpu_count() or 1)
            trainloader = DataLoader(
                self.trainset,
                batch_size=batch,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        # Count the class distribution in the training set
        class_counts = self.count_class_distribution(trainloader)
  
        self.optimizer = get_optimizer(model=self.model, client_config=config)
        self.train(
            net=self.model,
            trainloader=trainloader,
            optim=self.optimizer,
            epochs=epochs,
            device=self.device,
            config=config,
        )

        params_to_send = self.get_parameters({})
        num_examples = len(trainloader.dataset)
        metrics = {"client_id": self.client_id, "class_distribution": class_counts}
        
        # Add kl_norm to metrics if available (for FedSpec)
        if hasattr(self, 'kl_norm') and self.kl_norm is not None:
            metrics["kl_norm"] = self.kl_norm
        
        return params_to_send, num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Reload dataset to ensure validation size is updated for current round
        # This is necessary because evaluate() may be called after fit() in the same round
        # but with different dataset allocations
        dyn_cfg = self.config_sim.get("dynamic_data", {})
        enabled = dyn_cfg.get("enabled", False)
        if enabled and self.data_scheduler is not None:
            # Try to get current_round from config, fallback to "round" key or 0
            current_round = config.get("current_round", config.get("round", 0))
            self.reload_dataset(mode=dyn_cfg.get("mode", "incremental"), round_num=current_round)

        # Create validation DataLoader with collator for multimodal
        if self.clip_collator is not None:
            valloader = DataLoader(
                self.valset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=True if min(4, os.cpu_count() or 1) > 0 else False,
                prefetch_factor=2,
                collate_fn=self.clip_collator,
            )
        else:
            is_local_function = '<locals>' in self.apply_transforms.__qualname__ if self.apply_transforms else False
            num_workers = 0 if is_local_function else min(4, os.cpu_count() or 1)
            valloader = DataLoader(
                self.valset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )

        # Count the class distribution in the validation set
        class_counts = self.count_class_distribution(valloader)
        loss, accuracy, f1 = self.test(self.model, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"client_id": self.client_id, 
        "accuracy": float(accuracy), "f1_score": float(f1), "class_distribution": class_counts}

    def get_loss(self, loss):
        return getattr(__import__("mak.losses", fromlist=[loss]), loss)()

    def train(self, net, trainloader, optim, epochs, device: str, config: dict):
        """Train the network on the training set."""
        criterion = self.get_loss(loss=config["loss"])
        net.train()
        for _ in range(epochs):
            trainloader_iter = trainloader
            for batch in trainloader_iter:
                # Check if multimodal (feature_key is a list with both image and text)
                if isinstance(self.feature_key, list) and "image" in self.feature_key and "text" in self.feature_key:
                    # CLIPCollator outputs "pixel_values", not "image"
                    if "pixel_values" in batch:
                        pixel_values = batch["pixel_values"].to(device)
                    else:
                        # Fallback for old transform-based approach
                        pixel_values = batch["image"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # labels = batch[self.output_column].to(device)
                    labels = batch["labels"].to(device)  # CLIPCollator always outputs "labels"
                    optim.zero_grad()
                    logits = net(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                elif self.feature_key in ["text", "content", "sentence"]:
                    # Text-specific forward pass
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    optim.zero_grad()
                    outputs = net(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                else:
                    # For image datasets, we can use the standard loss function
                    keys = list(batch.keys())
                    x_label, y_label = keys[0], keys[1]
                    images, labels = batch[x_label].to(device), batch[y_label].to(device)
                    optim.zero_grad()
                    loss = criterion(net(images), labels)
                # Backpropagation    
                loss.backward()
                optim.step()



    def test(self, net, testloader, device: str):
        return test(net=net, testloader=testloader, device=device, feature_key=self.feature_key, dataset_name=self.dataset_name)