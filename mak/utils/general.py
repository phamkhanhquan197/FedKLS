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
def test(net, testloader, device: str, feature_key: str) -> Tuple[float, float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
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

def _slice_pad_lora_params(t: torch.Tensor, target_rank: int, param_type: str) -> torch.Tensor:
    """Slice or zero-pad LoRA factor to match target_rank.

    Args:
        t: Source tensor.
        target_rank: Desired rank dimension.
        param_type: "A" or "B".
            - A has shape [out, r] -> rank axis = 1
            - B has shape [r, in]  -> rank axis = 0

    Returns:
        Tensor with rank dimension adapted to target_rank.
    """
    if param_type == "A":
        if t.dim() != 2:
            return t
        out, r = t.shape
        if r > target_rank:
            return t[:, :target_rank]
        if r < target_rank:
            pad_cols = target_rank - r
            return F.pad(t, (0, pad_cols, 0, 0), mode="constant", value=0.0)
        return t

    if param_type == "B":
        if t.dim() != 2:
            return t
        r, inn = t.shape
        if r > target_rank:
            return t[:target_rank, :]
        if r < target_rank:
            pad_rows = target_rank - r
            return F.pad(t, (0, 0, 0, pad_rows), mode="constant", value=0.0)
        return t

    return t


def set_params(
    model: torch.nn.ModuleList,
    params: List[fl.common.NDArrays],
    device: str = "cuda",
    method: str = None,
    bias: str = True,
    rank_map: dict | None = None,
    client_id: int | None = None,
):

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
    
    # Build partial update state_dict
    lora_params = OrderedDict()

    # FlexLoRA rank adaptation (safe extend): only active when method == 'flex_lora'
    target_rank = None
    if method == "flex_lora":
        if rank_map is None or client_id is None:
            raise ValueError("FlexLoRA set_params requires rank_map and client_id")
        target_rank = int(rank_map[int(client_id)])

    for key, array in zip(lora_keys, params):
        t = torch.from_numpy(np.asarray(array))

        if method == "flex_lora" and (key.endswith(".A") or key.endswith(".B")):
            param_type = "A" if key.endswith(".A") else "B"
            t = _slice_pad_lora_params(t, target_rank=target_rank, param_type=param_type)

        lora_params[key] = t

    # Update model with partial parameters only
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
