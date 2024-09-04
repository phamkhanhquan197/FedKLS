import torch.nn as nn
from torchvision.models.resnet import (
    resnet18,
    resnet34
)

def ResNet18Pretrained(num_classes, *args, **kwargs):
    model = resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def ResNet34Pretrained(num_classes, *args, **kwargs):
    model = resnet34(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
