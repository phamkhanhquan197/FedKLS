import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr_datasets import FederatedDataset
from mak.utils.helper import get_optimizer
from mak.utils.pytorch_transformations import get_transformations
from mak.training import set_params, train, test

# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model,trainset, valset, device):
        self.trainset = trainset
        self.valset = valset

        # Instantiate model
        self.model = model

        # Determine device
        self.device = device
        self.model.to(self.device) 

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Read from config
        if config["lr_scheduler"] == True and config["round"] > 1:
            config["lr"] = config["lr"] * (0.99 ** config["round"])
        batch, epochs, learning_rate = config["batch_size"], config["epochs"], config["lr"]
        # Construct dataloader
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        # Define optimizer
        optimizer = get_optimizer(model = self.model, config_client = config)

        # Train
        train(net=self.model, trainloader=trainloader, optim= optimizer, epochs=epochs, device=self.device)

        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)

        # Construct dataloader
        valloader = DataLoader(self.valset, batch_size=64)

        # Evaluate
        loss, accuracy = test(self.model, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def get_client_fn(dataset: FederatedDataset, model,device,apply_transforms):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        client_dataset = dataset.load_partition(int(cid), "train")

        # Now let's split it into train (90%) and validation (15%)
        client_dataset_splits = client_dataset.train_test_split(test_size=0.15)

        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]

        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)

        # Create and return client
        return FlowerClient(model=model,trainset=trainset,valset=valset,device=device).to_client()

    return client_fn