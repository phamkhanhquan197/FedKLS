import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr_datasets import FederatedDataset
from mak.utils.helper import get_optimizer
from mak.utils.pytorch_transformations import get_transformations
from mak.training import set_params, test
import mak



class BaseClient(fl.client.NumPyClient):
    """flwr base client implementaion """
    def __init__(self, model, trainset, valset, train_batch_size, test_batch_size, device):
        self.trainset = trainset
        self.valset = valset
        self.model = model
        self.device = device
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.model.to(self.device)

    def __repr__(self) -> str:
        return " Flwr base client"

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        set_params(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        if config["lr_scheduler"] == True and config["round"] > 1:
            config["lr"] = config["lr"] * (0.99 ** config["round"])
            
        batch, epochs, learning_rate = config["batch_size"], config["epochs"], config["lr"]
        
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        optimizer = get_optimizer(model=self.model, client_config=config)
        self.train(net=self.model, trainloader=trainloader, optim=optimizer, epochs=epochs, device=self.device, config = config)

        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        valloader = DataLoader(self.valset, batch_size=self.test_batch_size)
        loss, accuracy = self.test(self.model, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
    
    def train(self, net, trainloader, optim, epochs, device: str, config : dict):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        net.train()

        for _ in range(epochs):
            for batch in trainloader:
                keys = list(batch.keys())
                x_label, y_label = keys[0], keys[1]
                images, labels = batch[x_label].to(device), batch[y_label].to(device)
                optim.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                optim.step()


    def test(self, net, testloader, device: str):
       return test(net=net,testloader=testloader,device=device)



