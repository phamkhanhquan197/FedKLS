import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from mak.utils.dataset_info import dataset_info
from transformers import AutoTokenizer, CLIPProcessor

class TextTransformationPipeline:
    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad_token to eos_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.feature_key = dataset_info[self.dataset_name]["feature_key"]
        self.max_sequence_length = dataset_info[self.dataset_name]["max_sequence_length"]

    def apply_transform(self, batch):
        """Apply transformations to the partition from FederatedDataset."""
        # Tokenize the text data
        encodings = self.tokenizer(
            batch[self.feature_key],
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        # Convert labels to tensor and rename key
        encodings["labels"] = torch.tensor(batch["label"]) # "label" → "labels"
        return encodings

    def get_transformations(self):
        """Return transformation functions for train and test data.
        For text datasets, train and test transforms are the same."""
        return self.apply_transform, self.apply_transform


class TransformationPipeline:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.feature_key = dataset_info[self.dataset_name]["feature_key"]
        self.img_shape = dataset_info[self.dataset_name]["input_shape"]

    def apply_transforms_scaffold(self, batch):
        """Apply transforms to the partition from FederatedDataset.
        Transformations based on scaffold flwr baseline implementation
        """
        pytorch_transforms = Compose(
            [
                ToTensor(),
                Lambda(
                    lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False),
                        (4, 4, 4, 4),
                        mode="reflect",
                    ).data.squeeze()
                ),
                ToPILImage(),
                RandomCrop(32),
                RandomHorizontalFlip(),
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_cifar10(self, batch):
        """Apply transforms to the partition from FederatedDataset."""
        pytorch_transforms = Compose(
            [
                ToTensor(),
                Normalize(
                    mean=[0.49139968, 0.48215827, 0.44653124],
                    std=[0.24703233, 0.24348505, 0.26158768],
                ),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_dogfood(self, batch):
        """Apply transforms to the partition from FederatedDataset sasha/dogfood."""
        pytorch_transforms = Compose(
            [
                CenterCrop(self.img_shape[1]),  # Center crop first
                Resize(self.img_shape[1]),
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_usps(self, batch):
        """Apply transforms to the partition from FederatedDataset flwrlabs/usps."""
        pytorch_transforms = Compose(
            [
                Resize(self.img_shape[1]),
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_tiny_imagenet(self, batch):
        """Apply transforms to the partition from FederatedDataset zh-plus/tiny-imagenet."""
        pytorch_transforms = Compose(
            [
                Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),  # Convert grayscale to RGB
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_tiny_imagenet_test(self, batch):
        """Apply transforms to the partition from FederatedDataset zh-plus/tiny-imagenet."""
        pytorch_transforms = Compose(
            [
                Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),  # Convert grayscale to RGB
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_default(self, batch):
        """Apply transforms to the partition from FederatedDataset."""
        pytorch_transforms = Compose(
            [
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def apply_transforms_test(self, batch):
        """Apply transforms to the partition from FederatedDataset."""
        pytorch_transforms = Compose(
            [
                ToTensor(),
            ]
        )
        batch[self.feature_key] = [
            pytorch_transforms(img) for img in batch[self.feature_key]
        ]
        return batch

    def get_transformations(self):
        if self.dataset_name == "cifar10" or self.dataset_name == "cifar100":
            return self.apply_transforms_cifar10, self.apply_transforms_test
        elif self.dataset_name == "sasha/dog-food":
            return self.apply_transforms_dogfood, self.apply_transforms_dogfood
        elif self.dataset_name == "flwrlabs/usps":
            return self.apply_transforms_usps, self.apply_transforms_usps
        elif self.dataset_name == "zh-plus/tiny-imagenet":
            return (
                self.apply_transforms_tiny_imagenet,
                self.apply_transforms_tiny_imagenet_test,
            )
        else:
            return self.apply_transforms_default, self.apply_transforms_test

@dataclass
class CLIPCollator:
    """
    Module-level collator for CLIP multimodal datasets.
    Picklable, supports num_workers > 0.
    """
    processor: Any  # CLIPProcessor from transformers
    label_key: str  # output_column from dataset_info
    multi_label: bool = False
    num_classes: Optional[int] = None
    label_to_idx: Optional[Dict[str, int]] = None

    def _to_onehot(self, y):
        """
        Convert label(s) to one-hot tensor.
        y: list[str|int] or str|int
        """
        t = torch.zeros(self.num_classes, dtype=torch.float32)
        if not isinstance(y, list):
            y = [y]
        for item in y:
            if isinstance(item, int) and 0 <= item < self.num_classes:
                t[item] = 1.0
            elif isinstance(item, str) and self.label_to_idx and item in self.label_to_idx:
                t[self.label_to_idx[item]] = 1.0
        return t

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples into tensors.
        
        Input: List of dicts with keys ["image", "text", label_key]
        Output: Dict with keys ["pixel_values", "input_ids", "attention_mask", "labels"]
        """
        if not batch:
            raise ValueError("Empty batch provided to CLIPCollator")
        
        images = [ex["image"] for ex in batch]
        
        # Use "text" key as defined in dataset (from helper.py: add_text_test/add_empty_text)
        # Fallback to empty string if "text" key doesn't exist
        texts = [ex.get("text", "") for ex in batch]

        # Process images and text with CLIPProcessor (batch processing)
        enc = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,  # Dynamic padding (faster + less waste)
            truncation=True
        )

        # Process labels
        labels = [ex[self.label_key] for ex in batch]
        if self.multi_label:
            # Multi-label: convert each label list to one-hot
            y = torch.stack([
                self._to_onehot(l) if isinstance(l, list) else self._to_onehot([l]) 
                for l in labels
            ])
        else:
            # Single-label: direct tensor conversion
            y = torch.tensor(labels, dtype=torch.long)

        enc["labels"] = y
        return enc
    
class CLIPTransformationPipeline:
    def __init__(self, dataset_name, img_size=224):
        self.dataset_name = dataset_name
        self.feature_key = dataset_info[self.dataset_name]["feature_key"]
        self.img_size = img_size

    def apply_transform(self, batch):
        # CLIP standard normalization
        pytorch_transforms = Compose(
            [
                Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                Resize(self.img_size, interpolation=3),
                CenterCrop(self.img_size),
                ToTensor(),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        batch[self.feature_key] = [pytorch_transforms(img) for img in batch[self.feature_key]]
        return batch

    def get_transformations(self):
        return self.apply_transform, self.apply_transform
