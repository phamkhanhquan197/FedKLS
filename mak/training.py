import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import Scalar
import torch
from collections import OrderedDict
from typing import Dict, Tuple, List
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from torch.utils.data import DataLoader
import numpy as np
# def train(
#     net, trainloader, valloader, epochs, device: torch.device = torch.device("cpu")
# ):
#     """Train the network on the training set."""
#     print("Starting training...")
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.SGD(
#         net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
#     )
#     net.train()
#     for _ in range(epochs):
#         for batch in trainloader:
#             images, labels = batch["image"], batch["label"]
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()

#     net.to("cpu")  # move model back to CPU

#     train_loss, train_acc = test(net, trainloader)
#     val_loss, val_acc = test(net, valloader)

#     results = {
#         "train_loss": train_loss,
#         "train_accuracy": train_acc,
#         "val_loss": val_loss,
#         "val_accuracy": val_acc,
#     }
#     return results


# def test(
#     net, testloader, steps: int = None, device: torch.device = torch.device("cpu")
# ):
#     """Validate the network on the entire test set."""
#     print("Starting evalutation...")
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     net.eval()
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(testloader):
#             images, labels = batch["image"], batch["label"]
#             images, labels = images.to(device), labels.to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == labels).sum().item()
#             if steps is not None and batch_idx == steps:
#                 break
#     accuracy = correct / len(testloader.dataset)
#     net.to("cpu")  # move model back to CPU
#     return loss, accuracy
# borrowed from Pytorch quickstart example
def train(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            keys = list(batch.keys())
            x_label, y_label = keys[0], keys[1]
            images, labels = batch[x_label].to(device), batch[y_label].to(device)
            # images, labels = batch["img"].to(device), batch["label"].to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optim.step()


# borrowed from Pytorch quickstart example
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

# def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
#     """Set model weights from a list of NumPy ndarrays."""
#     params_dict = zip(model.state_dict().keys(), params)
#     state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#     # print(f"+++++++++++++++ state dict : {state_dict}")
#     model.load_state_dict(state_dict, strict=True)

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
            {
                k: torch.tensor(np.atleast_1d(v))
                for k, v in zip(model.state_dict().keys(), params)
            }
        )
    model.load_state_dict(state_dict, strict=True)



def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# def apply_transforms(batch):
#     """Apply transforms to the partition from FederatedDataset."""
#     pytorch_transforms = Compose(
#         [
#             # Resize(256),
#             # CenterCrop(224),
#             ToTensor(),
#             # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )
#     batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#     return batch

# def get_evaluate_fn(
#     centralized_testset: Dataset,
# ):
#     """Return an evaluation function for centralized evaluation."""

#     def evaluate(
#         server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
#     ):
#         """Use the entire CIFAR-10 test set for evaluation."""

#         # Determine device
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         model = Net()
#         set_params(model, parameters)
#         model.to(device)

#         # Apply transform to dataset
#         testset = centralized_testset.with_transform(apply_transforms)

#         # Disable tqdm for dataset preprocessing
#         disable_progress_bar()

#         testloader = DataLoader(testset, batch_size=50)
#         loss, accuracy = test(model, testloader, device=device)

#         return loss, {"accuracy": accuracy}

#     return evaluate