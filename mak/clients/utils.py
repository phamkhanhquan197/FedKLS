
import flwr as fl
from mak.clients.fedavg_client import FedAvgClient
from mak.clients.fedprox_client import FedProxClient
from flwr_datasets import FederatedDataset
from flwr.common import Metrics
import torch
from collections import OrderedDict
from typing import Tuple, List
from flwr_datasets import FederatedDataset


def get_client_fn(config_sim: dict, dataset: FederatedDataset, model, device, apply_transforms):
    strategy = config_sim['server']['strategy']
    client_class = get_client_class(strategy)
    train_batch_size = config_sim['client']['batch_size']
    test_batch_size = config_sim['client']['test_batch_size']

    def client_fn(cid: str) -> fl.client.Client:
        client_dataset = dataset.load_partition(int(cid), "train")
        client_dataset_splits = client_dataset.train_test_split(test_size=0.15)
        trainset = client_dataset_splits["train"].with_transform(apply_transforms)
        valset = client_dataset_splits["test"].with_transform(apply_transforms)
        return client_class(model=model, trainset=trainset, valset=valset, train_batch_size=train_batch_size, test_batch_size=test_batch_size, device=device).to_client()
    return client_fn

def get_client_class(strategy: str):
    if strategy == 'fedprox':
        return FedProxClient
    else:
        return FedAvgClient
    
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    
    with torch.no_grad():
        for data in testloader:
            keys = list(data.keys())
            x_label, y_label = keys[0], keys[1]
            images, labels = data[x_label].to(device), data[y_label].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
