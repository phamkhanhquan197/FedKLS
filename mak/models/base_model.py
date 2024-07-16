from typing import Tuple
import torch.nn as nn

class Model(nn.Module):
    """Base model for all the custom models implemented using pytorch."""
    def __init__(self, num_classes: int, input_shape: Tuple = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.pretrained = False
    
    def model_details(self):
        print(f"classes : {self.num_classes} pretrained : {self.pretrained}")
