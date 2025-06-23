import copy
from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import torch
from flwr.common import Metrics
from mak.utils.dataset_info import dataset_info
from sklearn.metrics import f1_score

# Testing if the dataset is text or image
def test(net, testloader, device: str, feature_key: str) -> Tuple[float, float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total = 0
    all_labels = []
    all_preds = []
    
    # Set the network to evaluation mode
    net.eval()

    if feature_key == "text" or feature_key == "content":
        #for text datasets, we need to use a different loss function
        with torch.no_grad():
            for batch in testloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = net(input_ids, attention_mask=attention_mask, labels=labels)
                loss += outputs.loss.item()
                logits = outputs.logits
                predicted = torch.argmax(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                #Collect for F1 score
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return loss, accuracy, f1
    #for image datasets, we can use the standard loss function
    else:
        with torch.no_grad():
            for data in testloader:
                keys = list(data.keys())
                x_label, y_label = keys[0], keys[1]
                images, labels = data[x_label].to(device), data[y_label].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                #Collect for F1 score
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        accuracy = correct / len(testloader.dataset)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return loss, accuracy, f1

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays], device: str = "cuda"):

    """Set model weights from a list of NumPy ndarrays."""
    model_state = model.state_dict()
    if len(model_state.items()) != len(params): # Handle LoRA parameter update
        if any(key.startswith("distilbert.") for key in model_state.keys()):
            lora_keys = [k for k in model_state.keys() 
                        if ("lin" in k)]
        elif any(key.startswith("bert.") for key in model_state.keys()):
            lora_keys = [k for k in model_state.keys() 
                        if ("self" in k or "dense" in k)]
        elif any(key.startswith("model.") for key in model_state.keys()):
            lora_keys = [k for k in model_state.keys() 
                        if ("self_attn" in k or "mlp" in k)]

        # Create state dict with only LoRA parameters
        lora_params = OrderedDict()
        for key, array in zip(lora_keys, params):
            lora_params[key] = torch.from_numpy(array)
        
        # Update model with LoRA parameters only
        model_state.update(lora_params)
        model.load_state_dict(model_state, strict=True)


    else: #Full parameter update
        params_dict = zip(model_state.keys(), params)
        # state_dict = OrderedDict({k: torch.tensor(v, device=device).clone().detach() for k, v in params_dict})
        state_dict = OrderedDict({k: v.clone().detach().to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
                                  for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)



def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples),
            "f1_score": sum(f1_scores) / sum(examples)}
