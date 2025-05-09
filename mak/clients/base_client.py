import os

import flwr as fl
from torch.utils.data import DataLoader
import torch
from mak.utils.general import set_params, test
from mak.utils.helper import get_optimizer
from mak.utils.dataset_info import dataset_info
from collections import OrderedDict
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
        self.model.to(self.device)
        self.save_dir = os.path.join(save_dir, "clients")
        self.dataset_name = self.config_sim["common"]["dataset"]
        self.feature_key = dataset_info[self.dataset_name]["feature_key"]
        self.output_column = dataset_info[self.dataset_name]["output_column"]

    def __repr__(self) -> str:
        return " Flwr base client"

    def get_parameters(self, config):
        if self.config_sim["lora"]["enabled"]:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items() if "lora" in _]
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        if self.config_sim["lora"]["enabled"]:
            # Load only LoRA parameters
            lora_params = {k: v for k, v in zip(self.model.state_dict().keys(), parameters) 
                        if "lora" in k}
            self.model.load_state_dict(lora_params, strict=False)
        else:
            # Original full parameter loading
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
            self.model.load_state_dict(state_dict, strict=False)
        # else:
        #     set_params(self.model, parameters)

    def count_class_distribution(self, dataset):
        """Count the class distribution in the dataset."""
        class_counts = {}
        for batch_data in dataset:
            if self.feature_key == "text":  
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
        optimizer = get_optimizer(model=self.model, client_config=config)
        self.train(
            net=self.model,
            trainloader=trainloader,
            optim=optimizer,
            epochs=epochs,
            device=self.device,
            config=config,
        )
        # print(f"Client {self.client_id} class distribution: {self._class_distribution}")
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

    def train(self, net, trainloader, optim, epochs, device: str, config: dict):
        """Train the network on the training set."""
        criterion = self.get_loss(loss=config["loss"])
        net.train()

        for _ in range(epochs):
            for batch in trainloader:
                if self.feature_key == "text":
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
        return test(net=net, testloader=testloader, device=device, feature_key=self.feature_key)
