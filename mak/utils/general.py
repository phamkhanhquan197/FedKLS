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

    if feature_key == "text":
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

# def set_params(model: torch.nn.Module, params_to_set: List[fl.common.NDArrays]):
#     """
#     Sets model parameters.
#     If 'params_to_set' matches the count of trainable parameters (requires_grad=True),
#     it updates only those.
#     Otherwise, if 'params_to_set' matches the count of all parameters, it attempts a full update.
#     Ends by calling model.load_state_dict() with the updated state.
#     """
    
#     current_model_state = model.state_dict() # Get a copy to modify
    
#     # Get names and references of trainable parameters in the model
#     trainable_param_names_ordered = []
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             trainable_param_names_ordered.append(name)

#     if len(params_to_set) == len(trainable_param_names_ordered):
#         # This is the LoRA/PEFT case: update only the trainable parameters
#         # print(f"set_params: Updating {len(params_to_set)} trainable parameters.")
#         if not trainable_param_names_ordered and params_to_set:
#             print("set_params WARNING: Received parameters to set, but model has no trainable parameters.")
#             # model.load_state_dict(current_model_state) # Load original state (effectively no change)
#             return # Or raise error, as this is an inconsistent state

#         updated_trainable_params = OrderedDict()
#         for name, new_value_np in zip(trainable_param_names_ordered, params_to_set):
#             # Ensure the tensor is on the correct device and has the correct dtype
#             # This should ideally match the existing parameter's properties
#             target_device = current_model_state[name].device
#             target_dtype = current_model_state[name].dtype
#             updated_trainable_params[name] = torch.from_numpy(new_value_np).to(target_device).to(target_dtype)
            
#             if current_model_state[name].shape != updated_trainable_params[name].shape:
#                 print(f"set_params ERROR: Shape mismatch for trainable parameter '{name}'. "
#                       f"Model: {current_model_state[name].shape}, Update: {updated_trainable_params[name].shape}.")
#                 # This is a critical error, do not proceed with load_state_dict with this inconsistent entry
#                 # For safety, you might want to return here or raise an exception.
#                 # As a fallback, load the original state (no change).
#                 # model.load_state_dict(model.state_dict()) # Re-load original to be safe
#                 return


#         # Update the full state dictionary with the new values for trainable parameters
#         current_model_state.update(updated_trainable_params)
#         # Load the modified full state dictionary.
#         # strict=True is generally safer if you expect all keys to be present,
#         # which they will be since we started with model.state_dict() and updated parts of it.
#         model.load_state_dict(current_model_state, strict=True)

#     elif len(params_to_set) == len(current_model_state):
#         # Full parameter update case
#         # print(f"set_params: Performing full update with {len(params_to_set)} parameters.")
#         full_update_state_dict = OrderedDict()
#         for (name, _), new_value_np in zip(current_model_state.items(), params_to_set):
#             target_device = current_model_state[name].device
#             target_dtype = current_model_state[name].dtype
#             full_update_state_dict[name] = torch.from_numpy(new_value_np).to(target_device).to(target_dtype)

#             if current_model_state[name].shape != full_update_state_dict[name].shape:
#                 print(f"set_params ERROR: Shape mismatch for full update parameter '{name}'. "
#                       f"Model: {current_model_state[name].shape}, Update: {full_update_state_dict[name].shape}.")
#                 return
#         model.load_state_dict(full_update_state_dict, strict=True)
#     else:
#         print(f"set_params ERROR: Mismatch in parameter list lengths. Cannot determine update strategy.")
#         print(f"  Received {len(params_to_set)} parameters to set.")
#         print(f"  Model has {len(current_model_state)} total parameters.")
#         print(f"  Model has {len(trainable_param_names_ordered)} trainable parameters.")
#         # Not loading anything to prevent potential corruption
#         # model.load_state_dict(model.state_dict()) # Re-load original to be safe if needed

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):

    """Set model weights from a list of NumPy ndarrays."""
    model_state = model.state_dict()
    if len(model_state.items()) != len(params): # Handle LoRA parameter update
        lora_keys = [k for k in model_state.keys() 
                    if ("lin" in k)]

        # Create state dict with only LoRA parameters
        lora_params = OrderedDict()
        for key, array in zip(lora_keys, params):
            lora_params[key] = torch.from_numpy(array)
        
        # Update model with LoRA parameters only
        model_state.update(lora_params)
        model.load_state_dict(model_state, strict=True)

        
    else: #Full parameter update
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    ###########################################################################
    # for name, tensor in model.state_dict().items():
    #     print(f"{name}: shape {tuple(tensor.shape)}")  
    # print(f"=>>>>>>>>>>>>>>>>>Number of layers: {len(model.state_dict())}")    


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


def get_unique_classes(dataloader):
    all_labels = []
    for batch in dataloader:
        keys = list(batch.keys())
        x_label, y_label = keys[0], keys[1]
        labels = batch[y_label]
        all_labels.extend(labels.numpy())  # Assuming labels are in tensor format

    unique_classes = list(set(all_labels))
    return unique_classes


def random_pertube(model, gamma):
    new_model = copy.deepcopy(model)
    for p in new_model.parameters():
        gauss = torch.normal(mean=torch.zeros_like(p), std=1)
        if p.grad is None:
            p.grad = gauss
        else:
            p.grad.data.copy_(gauss.data)

    norm = torch.norm(
        torch.stack(
            [p.grad.norm(p=2) for p in new_model.parameters() if p.grad is not None]
        ),
        p=2,
    )

    with torch.no_grad():
        scale = gamma / (norm + 1e-12)
        scale = torch.clamp(scale, max=1.0)
        for p in new_model.parameters():
            if p.grad is not None:
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)

    return new_model
