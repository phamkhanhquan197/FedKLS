from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import flwr as fl
import torch
import os
import sys
from flwr.common import Metrics
import torch.nn.functional as F
from sklearn.metrics import f1_score
from mak.utils.dataset_info import dataset_info
from tqdm import tqdm

# Testing if the dataset is text or image
def test(net, testloader, device: str, feature_key: str, dataset_name: str = None, desc: str = "Evaluating") -> Tuple[float, float, float]:
    """Validate the network on the entire test set.
    
    Args:
        feature_key: Can be str (for text/image-only) or list (for multimodal)
        dataset_name: Dataset name to determine multi-label vs single-label
    """

    correct = 0
    loss = 0.0
    total = 0
    num_batches = 0
    all_labels = []
    all_preds = []
    
    # Set the network to evaluation mode
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # Determine if multi-label based on dataset_info
    output_column = dataset_info[dataset_name].get("output_column", "label")

    # Avoid noisy progress bars in Ray workers / non-TTY outputs (they print one line per update)
    in_ray_worker = os.environ.get("RAY_WORKER_ID") is not None
    is_tty = False
    try:
        is_tty = sys.stdout.isatty()
    except Exception:
        is_tty = False
    disable_pbar = in_ray_worker or (not is_tty)

    # =========================
    # Multimodal (image + text)
    # =========================
    if isinstance(feature_key, list) and "image" in feature_key and "text" in feature_key:
        with torch.no_grad():
            pbar = tqdm(
                testloader,
                desc=desc,
                unit="batch",
                leave=False,
                mininterval=1.0,
                disable=disable_pbar,
            )
            for batch in pbar:
                # CLIPCollator outputs "pixel_values", not "image"
                if "pixel_values" in batch:
                    pixel_values = batch["pixel_values"].to(device)
                else:
                    # Fallback for old transform-based approach
                    pixel_values = batch["image"].to(device)
                
                # CLIPCollator always outputs "labels"
                if "labels" in batch:
                    labels = batch["labels"].to(device)
                else:
                    labels = batch[output_column].to(device)

                if "input_ids" in batch and "attention_mask" in batch:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                else:
                    batch_size = pixel_values.size(0)
                    max_seq_length = 77
                    input_ids = torch.zeros(
                        (batch_size, max_seq_length),
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.zeros_like(input_ids)

                logits = net(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                batch_loss = criterion(logits, labels)
                loss += batch_loss.item()

                # Handle multi-label vs single-label
                is_multi_label = dataset_info.get(dataset_name, {}).get("multi_label", False)
                if is_multi_label:
                    # Multi-label: threshold-based prediction
                    predicted = (torch.sigmoid(logits) > 0.5).float()
                    correct += (predicted == labels).all(dim=1).sum().item()
                else:
                    # Single-label: argmax
                    predicted = torch.argmax(logits, dim=1)
                    correct += (predicted == labels).sum().item()

                total += labels.size(0)
                num_batches += 1

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                if (not disable_pbar) and (num_batches % 10 == 0):
                    pbar.set_postfix(
                        {
                            "loss": f"{loss / num_batches:.4f}",
                            "acc": f"{correct / total:.4f}",
                            "samples": total,
                        }
                    )

    # =========================
    # Text-only
    # =========================
    elif feature_key in ["text", "content", "sentence"]:
        with torch.no_grad():
            for batch in testloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = net(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss += outputs.loss.item()
                logits = outputs.logits
                predicted = torch.argmax(logits, dim=1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

    # =========================
    # Image-only
    # =========================
    else:
        with torch.no_grad():
            for data in testloader:
                keys = list(data.keys())
                x_label, y_label = keys[0], keys[1]

                images = data[x_label].to(device)
                labels = data[y_label].to(device)

                outputs = net(images)
                loss += criterion(outputs, labels).item()

                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()

                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

    accuracy = correct / total if total > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average="weighted")
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
            elif any(key.startswith("vision_model") for key in model_state.keys()) or any(key.startswith("text_model") for key in model_state.keys()): #CustomCLIP
                if bias:
                    lora_keys = [k for k in model_state.keys() if ("self_attn" in k or "mlp" in k)]
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
