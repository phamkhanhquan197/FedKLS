from mak.models.base_model import Model
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn, Tensor

class EfficientNetB0(Model):
    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        """
        EfficientNetB0 model with customizable classifier head.

        Args:
            num_classes (int): Number of output classes.
            weights (str or None): Weights to initialize the model ('DEFAULT' or None).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(num_classes, *args, **kwargs)

        if weights == 'DEFAULT':
            self._model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            # change the classifier head for num_classes
            self._model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=1280, out_features=self.num_classes),
            )
            self.pretrained = True
        else:
            self._model = efficientnet_b0(num_classes=num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the EfficientNetB0 model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._model(x)