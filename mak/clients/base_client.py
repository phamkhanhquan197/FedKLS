import os

import flwr as fl
from torch.utils.data import DataLoader
import torch
from mak.utils.general import set_params, test
from mak.utils.helper import get_optimizer
from mak.utils.dataset_info import dataset_info
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        apply_transforms=None, # NEW: transform function
        data_scheduler=None, # NEW: DynamicDataScheduler for round-aware allocation
        bias=None,
        rank_policy_map: dict | None = None,
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

        self.optimizer = None
        self.scheduler = None
        self.previous_val_loss = None

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
                apply_transforms=self.apply_transforms
            )
            self.trainset = trainset
            self.valset = valset
            return
        
        # Fallback to old approach if scheduler not available
        if self.dataset is None or self.apply_transforms is None:
            raise RuntimeError("Dataset reference or transform function not provided.")

        client_dataset_total = self.dataset.load_partition(
            partition_id=self.partition_id
        )

        splits = client_dataset_total.train_test_split(
            test_size=0.2,
            seed=self.config_sim["common"]["seed"],
        )

        new_trainset = splits["train"].with_transform(self.apply_transforms)
        new_valset = splits["test"].with_transform(self.apply_transforms)

        if mode == "append":
            self.trainset = ConcatDataset([self.trainset, new_trainset])
            self.valset = ConcatDataset([self.valset, new_valset])
        else:
            self.trainset = new_trainset
            self.valset = new_valset

    def set_parameters(self, parameters):
        method = self.config_sim["peft"]["method"] if self.config_sim["peft"]["enabled"] else None
        bias = self.config_sim["peft"]["bias"] if self.config_sim["peft"]["enabled"] else None
        set_params(self.model, parameters, method=method, bias=bias)

    def count_class_distribution(self, dataset):
        """Count the class distribution in the dataset."""
        class_counts = {}
        for batch_data in dataset:
            if self.feature_key in ["text", "content", "sentence"]:
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
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        # Count the class distribution in the training set
        class_counts = self.count_class_distribution(trainloader)
        # # Reuse or initialize optimizer
        # if self.optimizer is None:
        #     self.optimizer = get_optimizer(model=self.model, client_config=config)
        #     print(f"Client {self.client_id}, Initialized new optimizer with LR = {self.optimizer.param_groups[0]['lr']:.6f}")
        # else:
        #     print(f"Client {self.client_id}, Reusing optimizer with LR = {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # # Reuse or initialize scheduler
        # if self.scheduler is None:
        #     self.scheduler = ReduceLROnPlateau(
        #         self.optimizer,
        #         mode="min",
        #         factor=0.5,  # Reduce LR by factor of 10
        #         patience=3,  # Reduce immediately if no improvement
        #         verbose=True,
        #         min_lr=1e-6,
        #         threshold=0.05,  # 5% relative improvement
        #         threshold_mode='rel',
        #     )
        # if self.previous_val_loss is not None:
        #     self.scheduler.best = self.previous_val_loss
        #     print(f"Client {self.client_id}, Reusing scheduler with Best Loss = {self.scheduler.best:.6f}")
        # else:
        #     print(f"Client {self.client_id}, Reusing scheduler with Best Loss = {self.scheduler.best:.6f}")

        # self.load_state()  # Load state at start
        self.optimizer = get_optimizer(model=self.model, client_config=config)
        self.train(
            net=self.model,
            trainloader=trainloader,
            optim=self.optimizer,
            epochs=epochs,
            device=self.device,
            config=config,
            scheduler=self.scheduler, #Enable later if necessary
        )
        # # Store best loss and save state
        # self.previous_val_loss = self.scheduler.best
        # self.save_state()  # Save state after training

        params_to_send = self.get_parameters({})
        num_examples = len(trainloader.dataset)
        metrics = {"client_id": self.client_id, "class_distribution": class_counts}
        
        # Add kl_norm to metrics if available (for FedMoKLS)
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

        valloader = DataLoader(self.valset, batch_size=self.test_batch_size)
        # Count the class distribution in the validation set
        class_counts = self.count_class_distribution(valloader)
        loss, accuracy, f1 = self.test(self.model, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"client_id": self.client_id, 
        "accuracy": float(accuracy), "f1_score": float(f1), "class_distribution": class_counts}

    def get_loss(self, loss):
        return getattr(__import__("mak.losses", fromlist=[loss]), loss)()

    def train(self, net, trainloader, optim, epochs, device: str, config: dict, scheduler):
        """Train the network on the training set."""
        criterion = self.get_loss(loss=config["loss"])
        net.train()
        valloader = DataLoader(self.valset, batch_size=self.test_batch_size)

        # Optional DP-SGD for FedSVD (Opacus). This is best-effort and depends on
        # model/layer compatibility with Opacus.
        dp_cfg = (self.config_sim.get("fedsvd_config", {}) or {}).get("dp", {}) or {}
        dp_enabled = bool(dp_cfg.get("enabled", False)) and config.get("strategy") == "FedSVD"
        if dp_enabled:
            try:
                from opacus import PrivacyEngine  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "DP is enabled (fedsvd_config.dp.enabled=true) but 'opacus' is not installed. "
                    "Install it (e.g., pip install opacus) or disable DP."
                ) from e

            # Compute / use noise multiplier
            noise_multiplier = dp_cfg.get("noise_multiplier", None)
            if noise_multiplier is None:
                try:
                    from opacus.accountants.utils import get_noise_multiplier  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        "fedsvd_config.dp.noise_multiplier is null, but this Opacus version does not "
                        "provide get_noise_multiplier(). Please set fedsvd_config.dp.noise_multiplier explicitly."
                    ) from e

                dataset_size = max(1, len(trainloader.dataset))
                sample_rate = float(trainloader.batch_size) / float(dataset_size)
                noise_multiplier = float(
                    get_noise_multiplier(
                        target_epsilon=float(dp_cfg.get("eps", 8.0)),
                        target_delta=float(dp_cfg.get("delta", 1e-5)),
                        sample_rate=sample_rate,
                        epochs=float(epochs),
                    )
                )

            max_grad_norm = float(dp_cfg.get("max_grad_norm", 1.0))
            secure_rng = bool(dp_cfg.get("secure_rng", False))
            grad_sample_mode = str(dp_cfg.get("grad_sample_mode", "hooks"))

            # Create PrivacyEngine with compatibility across Opacus versions
            try:
                privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=secure_rng)
            except TypeError:
                privacy_engine = PrivacyEngine(accountant="rdp", secure_rng=secure_rng)

            # Make private.
            # NOTE: "ghost" enables fast gradient clipping but can raise
            # `AssertionError: loss_reduction ...` depending on Opacus version/model.
            # Default to "hooks" for broader compatibility.
            try:
                res = privacy_engine.make_private(
                    module=net,
                    optimizer=optim,
                    data_loader=trainloader,
                    noise_multiplier=float(noise_multiplier),
                    max_grad_norm=max_grad_norm,
                    grad_sample_mode=grad_sample_mode,
                )
            except TypeError:
                res = privacy_engine.make_private(
                    module=net,
                    optimizer=optim,
                    data_loader=trainloader,
                    noise_multiplier=float(noise_multiplier),
                    max_grad_norm=max_grad_norm,
                )

            # Opacus versions differ in return signature.
            # Common: (module, optimizer, data_loader)
            # Some:   (module, optimizer, data_loader, privacy_engine)
            if not isinstance(res, tuple):
                raise RuntimeError(f"Unexpected PrivacyEngine.make_private() return type: {type(res)}")
            if len(res) == 3:
                net, optim, trainloader = res
            elif len(res) == 4:
                net, optim, trainloader, _privacy_engine = res
            elif len(res) == 5:
                net, optim, trainloader, _privacy_engine, _criterion = res
            else:
                raise RuntimeError(f"Unexpected PrivacyEngine.make_private() return arity: {len(res)}")

        for _ in range(epochs):
            for batch in trainloader:
                if self.feature_key in ["text", "content", "sentence"]:
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
        # # Compute validation loss for scheduler
        # val_loss, _, _ = self.test(net=net, testloader=valloader, device=device)
        # print(f"Client {self.client_id}, Before Scheduler Step: Val Loss = {val_loss:.6f}, "
        #     f"Num Bad Epochs = {scheduler.num_bad_epochs}, Best Loss = {scheduler.best:.6f}, "
        #     f"Current LR = {optim.param_groups[0]['lr']:.6f}")
        # scheduler.step(val_loss)
        # print(f"Client {self.client_id}, After Scheduler Step: Val Loss = {val_loss:.6f}, "
        #     f"Num Bad Epochs = {scheduler.num_bad_epochs}, Best Loss = {self.scheduler.best:.6f}, "
        #     f"New LR = {optim.param_groups[0]['lr']:.6f}")


    def test(self, net, testloader, device: str):
        return test(net=net, testloader=testloader, device=device, feature_key=self.feature_key)

    # def save_state(self):
    #     torch.save({
    #         'optimizer': self.optimizer.state_dict(),
    #         'scheduler': self.scheduler.state_dict(),
    #         'best_loss': self.previous_val_loss
    #     }, os.path.join(self.save_dir, f"client_{self.client_id}_state.pt"))

    # def load_state(self):
    #     state_path = os.path.join(self.save_dir, f"client_{self.client_id}_state.pt")
    #     if os.path.exists(state_path):
    #         state = torch.load(state_path)
    #         self.optimizer.load_state_dict(state['optimizer'])
    #         self.scheduler.load_state_dict(state['scheduler'])
    #         self.previous_val_loss = state['best_loss']
    #         print(f"Client {self.client_id}, Loaded optimizer and scheduler state")
