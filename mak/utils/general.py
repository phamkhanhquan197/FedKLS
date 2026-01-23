from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import flwr as fl
import torch
from flwr.common import Metrics
from flwr.common.logger import log
from logging import INFO
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
from mak.utils.dataset_info import dataset_info


# Testing if the dataset is text or image
def test(net, testloader, device: str, feature_key, dataset_name: str = None) -> Tuple[float, float, float]:
    """Validate the network on the entire test set.
    
    Args:
        feature_key: Can be str (for text/image-only) or list (for multimodal)
        dataset_name: Dataset name to determine multi-label vs single-label
    """
    correct, loss = 0, 0.0
    total = 0
    num_batches = 0
    all_labels = []
    all_preds = []
    
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
            pbar = tqdm(testloader, desc="Evaluating", unit="batch", leave=True)
            for batch in pbar:
                pixel_values = batch["image"].to(device)
                # Check if input_ids and attention_mask exist (text might be missing from some examples)
                if "input_ids" in batch and "attention_mask" in batch:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    logits = net(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                else:
                    # If text is missing, use image-only forward pass (if model supports it)
                    # Otherwise, create dummy tensors
                    batch_size = pixel_values.size(0)
                    # Create dummy input_ids and attention_mask with padding tokens
                    # Assuming max_seq_length is 77 (CLIP standard) - adjust if needed
                    max_seq_length = 77
                    input_ids = torch.zeros((batch_size, max_seq_length), dtype=torch.long, device=device)
                    attention_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.long, device=device)
                    logits = net(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                labels = batch[output_column].to(device)
                
                if is_multi_label:
                    # Multi-label: use sigmoid and threshold
                    batch_loss = criterion(logits, labels.float()).item()
                    loss += batch_loss
                    probs = torch.sigmoid(logits)
                    predicted = (probs > 0.5).int()
                    # For multi-label, accuracy is computed differently (exact match or hamming)
                    # Using exact match for now
                    batch_correct = (predicted == labels.int()).all(dim=1).sum().item()
                    correct += batch_correct
                else:
                    # Single-label: use softmax and argmax
                    batch_loss = criterion(logits, labels).item()
                    loss += batch_loss
                    probs = F.softmax(logits, dim=1)
                    predicted = torch.argmax(logits, dim=1)
                    batch_correct = (predicted == labels).sum().item()
                    correct += batch_correct
                
                total += labels.size(0)
                num_batches += 1
                # Collect for F1 score
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                # Update progress bar with current metrics
                current_acc = correct / total if total > 0 else 0.0
                current_loss = loss / num_batches if num_batches > 0 else 0.0
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}", "samples": total})
        
        accuracy = correct / total if total > 0 else 0.0
        # For multi-label, use appropriate F1 metric
        if is_multi_label:
            f1 = f1_score(all_labels, all_preds, average='micro')  # micro-averaged for multi-label
        else:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return loss, accuracy, f1

    elif feature_key in ["text", "content", "sentence"]:
        #for text datasets, we need to use a different loss function
        with torch.no_grad():
            pbar = tqdm(testloader, desc="Evaluating", unit="batch", leave=True)
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = net(input_ids, attention_mask=attention_mask, labels=labels)
                batch_loss = outputs.loss.item()
                loss += batch_loss
                logits = outputs.logits
                predicted = torch.argmax(logits, dim=1)
                batch_correct = (predicted == labels).sum().item()
                correct += batch_correct
                total += labels.size(0)
                num_batches += 1
                #Collect for F1 score
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                # Update progress bar with current metrics
                current_acc = correct / total if total > 0 else 0.0
                current_loss = loss / num_batches if num_batches > 0 else 0.0
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}", "samples": total})
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return loss, accuracy, f1
    #for image datasets, we can use the standard loss function
    else:
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            pbar = tqdm(testloader, desc="Evaluating", unit="batch", leave=True)
            for data in pbar:
                keys = list(data.keys())
                x_label, y_label = keys[0], keys[1]
                images, labels = data[x_label].to(device), data[y_label].to(device)
                outputs = net(images)
                batch_loss = criterion(outputs, labels).item()
                loss += batch_loss
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                correct += batch_correct
                total += labels.size(0)
                num_batches += 1
                #Collect for F1 score
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                # Update progress bar with current metrics
                current_acc = correct / total if total > 0 else 0.0
                current_loss = loss / num_batches if num_batches > 0 else 0.0
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}", "samples": total})
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
        _, r = t.shape
        if r > target_rank:
            return t[:, :target_rank]
        if r < target_rank:
            pad_cols = target_rank - r
            return F.pad(t, (0, pad_cols, 0, 0), mode="constant", value=0.0)
        return t

    if param_type == "B":
        if t.dim() != 2:
            return t
        r, _ = t.shape
        if r > target_rank:
            return t[:target_rank, :]
        if r < target_rank:
            pad_rows = target_rank - r
            return F.pad(t, (0, 0, 0, pad_rows), mode="constant", value=0.0)
        return t

    return t

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays], 
               device: str = "cuda", method: str = None, bias: bool = True,
               rank_policy_map: dict | None = None, client_id: str | None = None):

    """Set model weights from a list of NumPy ndarrays."""
    model_state = model.state_dict()
    if params is None:
        return  # Skip if parameters is None

    if len(model_state.items()) == len(params): #Full model update (Round = 1 or full finetune)
        params_dict = zip(model_state.keys(), params)
        state_dict = OrderedDict({k: v.clone().detach().to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
                                  for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        if method == "ffa_lora": #Freeze all A adapters after full model update
            [p.__setattr__("requires_grad", False) for name, p in model.named_parameters() if name.endswith(".A")]
        return
    else:
        if method == "ffa_lora": #Send and receive only LoRA B adapters
            if any(k.startswith("distilbert.") for k in model_state.keys()):
                if bias:
                    lora_keys = [
                        k for k in model_state.keys()
                        if k.endswith(".B") or (k.endswith(".bias") and "lin" in k)
                    ]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".B")]
            elif any(k.startswith("roberta.") for k in model_state.keys()):
                if bias:
                    lora_keys = [
                        k for k in model_state.keys()
                        if k.endswith(".B")
                        or (k.endswith(".bias") and "self" in k)
                        or (k.endswith(".bias") and "dense" in k and "classifier" not in k)
                    ]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".B")]
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
                    lora_keys = [k for k in model_state.keys() if k.endswith(".B")]
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
                    lora_keys = [k for k in model_state.keys() if k.endswith(".B")]
        
        elif method == "fedsa_lora": #Send and receive only LoRA A adapters
            if any(k.startswith("distilbert.") for k in model_state.keys()):
                if bias:
                    lora_keys = [
                        k for k in model_state.keys()
                        if k.endswith(".A") or (k.endswith(".bias") and "lin" in k)
                    ]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".A")]
            elif any(k.startswith("roberta.") for k in model_state.keys()):
                if bias:
                    lora_keys = [
                        k for k in model_state.keys()
                        if k.endswith(".A")
                        or (k.endswith(".bias") and "self" in k)
                        or (k.endswith(".bias") and "dense" in k and "classifier" not in k)
                    ]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".A")]
            elif any(k.startswith("bert.") for k in model_state.keys()):
                if bias:
                    lora_keys = [
                        k for k in model_state.keys()
                        if (
                            k.endswith(".A")
                            or (k.endswith(".bias") and "self" in k)
                            or (k.endswith(".bias") and "dense" in k)
                        )
                    ]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".A")]

            elif any(k.startswith("model.") for k in model_state.keys()):
                if bias:
                    lora_keys = [
                        k for k in model_state.keys()
                        if (
                            k.endswith(".A")
                            or (k.endswith(".bias") and "self_attn" in k)
                            or (k.endswith(".bias") and "mlp" in k)
                        )
                    ]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".A")]

        # FlexLoRA partial update (Round > 1)
        elif method == "flex_lora":
            # Lazy import to avoid circular dependency (helper imports general)
            from mak.utils.helper import get_target_keys
            from mak.utils.flex_lora_utils import get_rank_for_base

            target_keys = get_target_keys(model, bias)
            if len(params) != len(target_keys):
                raise ValueError(
                    f"FlexLoRA set_params expects {len(target_keys)} params (target keys), got {len(params)}"
                )

            if rank_policy_map is None or client_id is None:
                raise ValueError("FlexLoRA set_params requires rank_policy_map and client_id")

            cid = int(client_id)
            if cid not in rank_policy_map:
                raise ValueError(f"FlexLoRA missing rank policy for client_id={cid}")
            rank_policy = rank_policy_map[cid]

            # Ensure tensors are created on the requested device
            dev = torch.device(device) if isinstance(device, str) else device

            update = OrderedDict()
            for key, array in zip(target_keys, params):
                t = torch.from_numpy(np.asarray(array)).to(device=dev)

                # Slice/pad LoRA factors per-layer according to rank policy
                if key.endswith(".A"):
                    base = key[:-2]
                    target_rank = int(get_rank_for_base(rank_policy, base))
                    t = _slice_pad_lora_params(t, target_rank=target_rank, param_type="A")
                elif key.endswith(".B"):
                    base = key[:-2]
                    target_rank = int(get_rank_for_base(rank_policy, base))
                    t = _slice_pad_lora_params(t, target_rank=target_rank, param_type="B")

                update[key] = t

            model_state.update(update)
            model.load_state_dict(model_state, strict=False)
            return 
        
        else: #Other methods (send and receive A+B)
            if any(key.startswith("distilbert.") for key in model_state.keys()):
                if bias:
                    lora_keys = [k for k in model_state.keys() if ("lin" in k)]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
            elif any(key.startswith("roberta.") for key in model_state.keys()):
                if bias:
                    lora_keys = [k for k in model_state.keys() if ("self" in k or ("dense" in k and "classifier" not in k))]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
            elif any(key.startswith("bert.") for key in model_state.keys()):
                if bias:
                    lora_keys = [k for k in model_state.keys() if ("self" in k or "dense" in k)]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
            elif any(key.startswith("model.") for key in model_state.keys()):
                if bias:
                    lora_keys = [k for k in model_state.keys() if ("self_attn" in k or "mlp" in k)]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]
            elif any(key.startswith("layer") for key in model_state.keys()): #RESNET models
                if bias:
                    lora_keys = [k for k in model_state.keys() if ("conv" in k)]
                else:
                    lora_keys = [k for k in model_state.keys() if k.endswith(".B") or k.endswith(".A")]

        # Create state dict with only LoRA parameters
        lora_params = OrderedDict()
        for key, array in zip(lora_keys, params):
            lora_params[key] = torch.from_numpy(array)
        # Update model with LoRA parameters only
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
