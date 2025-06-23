import os

import flwr as fl
from torch.utils.data import DataLoader
import torch
from mak.utils.general import set_params, test
from mak.utils.helper import get_optimizer
from mak.utils.dataset_info import dataset_info
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        
        self.optimizer = None
        self.scheduler = None
        self.previous_val_loss = None

    def __repr__(self) -> str:
        return " Flwr base client"

    def get_parameters(self, config):
        if self.config_sim["peft"]["enabled"] == True:
            #Only send the A, B and bias parameters to the server 
            if any(key.startswith("distilbert.") for key in self.model.state_dict().keys()):
                params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if "lin" in name}
            elif any(key.startswith("bert.") for key in self.model.state_dict().keys()):
                params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if "self" in name or "dense" in name}
            elif any(key.startswith("model.") for key in self.model.state_dict().keys()):
                params_to_send = {name: tensor for name, tensor in self.model.state_dict().items() if "self_attn" in name or "mlp" in name}

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


    def set_parameters(self, parameters):
        set_params(self.model, parameters)

    def count_class_distribution(self, dataset):
        """Count the class distribution in the dataset."""
        class_counts = {}
        for batch_data in dataset:
            if self.feature_key == "text" or self.feature_key == "content":  
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
        self.set_parameters(parameters)

        batch, epochs, learning_rate = (
            config["batch_size"],
            config["epochs"],
            config["lr"],
        )
        # Create a DataLoader for the training set
        trainloader  = DataLoader(self.trainset, batch_size=batch, shuffle=True)
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

        return self.get_parameters({}), len(trainloader.dataset), {"client_id": self.client_id, "class_distribution": class_counts}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
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

        for _ in range(epochs):
            for batch in trainloader:
                if self.feature_key == "text" or self.feature_key == "content":
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