import copy
from collections import OrderedDict
from typing import List, Tuple
import numpy as np

import flwr as fl
import torch
from flwr.common import Metrics
from mak.utils.dataset_info import dataset_info
import torch.nn.functional as F
from sklearn.metrics import f1_score


# Testing if the dataset is text or image
def test(net, testloader, device: str, feature_key, dataset_name: str = None) -> Tuple[float, float, float]:
    """Validate the network on the entire test set.
    
    Args:
        feature_key: Can be str (for text/image-only) or list (for multimodal)
        dataset_name: Dataset name to determine multi-label vs single-label
    """
    correct, loss = 0, 0.0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Set the network to evaluation mode
    net.eval()
    
    # Determine if multi-label based on dataset_info
    is_multi_label = False
    output_column = "label"
    if dataset_name and dataset_name in dataset_info:
        is_multi_label = dataset_info[dataset_name].get("multi_label", False)
        output_column = dataset_info[dataset_name].get("output_column", "label")

    # Check if multimodal (feature_key is a list with both image and text)
    if isinstance(feature_key, list) and "image" in feature_key and "text" in feature_key:
        # Multimodal evaluation
        if is_multi_label:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in testloader:
                pixel_values = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch[output_column].to(device)
                logits = net(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                
                if is_multi_label:
                    # Multi-label: use sigmoid and threshold
                    loss += criterion(logits, labels.float()).item()
                    probs = torch.sigmoid(logits)
                    predicted = (probs > 0.5).int()
                    # For multi-label, accuracy is computed differently (exact match or hamming)
                    # Using exact match for now
                    correct += (predicted == labels.int()).all(dim=1).sum().item()
                else:
                    # Single-label: use softmax and argmax
                    loss += criterion(logits, labels).item()
                    probs = F.softmax(logits, dim=1)
                    predicted = torch.argmax(logits, dim=1)
                    correct += (predicted == labels).sum().item()
                
                total += labels.size(0)
                # Collect for F1 score
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0.0
        # For multi-label, use appropriate F1 metric
        if is_multi_label:
            f1 = f1_score(all_labels, all_preds, average='micro')  # micro-averaged for multi-label
        else:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return loss, accuracy, f1
    
    elif feature_key == "text" or feature_key == "content":
        #for text datasets, we need to use a different loss function
        with torch.no_grad():
            for batch in testloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = net(input_ids, attention_mask=attention_mask, labels=labels)
                loss += outputs.loss.item()
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)  # probability per class
                predicted = torch.argmax(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                #Collect for F1 score
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return loss, accuracy, f1
    #for image datasets, we can use the standard loss function
    else:
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for data in testloader:
                keys = list(data.keys())
                x_label, y_label = keys[0], keys[1]
                images, labels = data[x_label].to(device), data[y_label].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                probs = F.softmax(outputs, dim=1)  # probability per class
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                #Collect for F1 score
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        accuracy = correct / len(testloader.dataset)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return loss, accuracy, f1

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays], 
               device: str = "cuda", method: str = None, bias: str = True):

    """Set model weights from a list of NumPy ndarrays."""
    model_state = model.state_dict()
    if params is None:
        return  # Skip if parameters is None

    # print(f"len(params): {len(params)}") #108 -> For round > 1 -> this shows # of layers sent by server
    # print(f"len(model_state.items()): {len(model_state.items())}") #140 all the times -> this shows # of layers in local model
    if len(model_state.items()) == len(params): #Full model update (Round = 1)
        params_dict = zip(model_state.keys(), params)
        state_dict = OrderedDict({k: v.clone().detach().to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
                                  for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        if method == "ffa_lora": #Freeze all A adapters after full model update
            [p.__setattr__("requires_grad", False) for name, p in model.named_parameters() if name.endswith(".A")]
        return

    # Handle LoRA-only update (Round > 1)
    elif len(model_state.items()) != len(params) and method != "ffa_lora": # Handle normal LoRA parameter update
        if any(key.startswith("distilbert.") for key in model_state.keys()):
            if bias:
                lora_keys = [k for k in model_state.keys() 
                        if ("lin" in k)]
            else:
                lora_keys = [k for k in model_state.keys() 
                        if k.endswith(".B") or k.endswith(".A")]
                
        elif any(key.startswith("bert.") for key in model_state.keys()):
            if bias:
                lora_keys = [k for k in model_state.keys() 
                        if ("self" in k or "dense" in k)]
            else:
                lora_keys = [k for k in model_state.keys() 
                        if k.endswith(".B") or k.endswith(".A")]
        elif any(key.startswith("model.") for key in model_state.keys()):
            if bias:
                lora_keys = [k for k in model_state.keys() 
                        if ("self_attn" in k or "mlp" in k)]
            else:
                lora_keys = [k for k in model_state.keys() 
                        if k.endswith(".B") or k.endswith(".A")]

    elif len(model_state.items()) != len(params) and method == "ffa_lora": #Handle FFA-LoRA parameter update
        if any(k.startswith("distilbert.") for k in model_state.keys()):
            if bias:
                lora_keys = [
                    k for k in model_state.keys()
                    if k.endswith(".B") or (k.endswith(".bias") and "lin" in k)
                ]
            else:
                lora_keys = [
                    k for k in model_state.keys()
                    if k.endswith(".B")
                ]
        elif any(k.startswith("bert.") for k in model_state.keys()):
            if bias:
                lora_keys = [
                    k for k in model_state.keys()
                    if (
                        k.endswith(".B")
                        or (k.endswith(".bias") and "self" in k)
                        or (k.endswith(".bias") and "dense" in k)
                    )
                ]
            else:
                lora_keys = [
                    k for k in model_state.keys()
                    if k.endswith(".B")
                ]
        elif any(k.startswith("model.") for k in model_state.keys()):
            if bias:
                lora_keys = [
                    k for k in model_state.keys()
                    if (
                        k.endswith(".B")
                        or (k.endswith(".bias") and "self_attn" in k)
                        or (k.endswith(".bias") and "mlp" in k)
                    )
                ]
            else:
                lora_keys = [
                    k for k in model_state.keys()
                    if k.endswith(".B")
                ]
    
    # Create state dict with only LoRA-B parameters
    lora_params = OrderedDict()
    for key, array in zip(lora_keys, params):
        lora_params[key] = torch.from_numpy(array)
    # Update model with LoRA-B parameters only
    model_state.update(lora_params)
    model.load_state_dict(model_state, strict=True)




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
