from torchvision.transforms import Compose, ToTensor, Normalize

def apply_transforms_cifar10(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [
            # Resize(256),
            # CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
        ]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def apply_transforms_default(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [
            # Resize(256),
            # CenterCrop(224),
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

def get_transformations(dataset_name):
    if dataset_name == 'cifar10':
        return apply_transforms_cifar10
    else:
        return  apply_transforms_default
