from typing import Tuple

from torch import nn, Tensor, load, flatten
import torch.nn.functional as F
import torchvision.models as torch_models

from mak.models.base_model import Model
from mak.models.resnet import BaseResNet, BasicBlock

### models implemented so far:
# 1. base model
# 2. SimpleCNN
# 3. KerasExpCNN
# 4. MNISTCNN
# 5. SimpleDNN
# 6. EfficientNetB0
# 7. FMCNNModel
# 8. FedAVGCNN
# 9. resnet18

class CifarNet(Model):
    """
    Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz for three channel input').

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
    """
    def __init__(self, num_classes: int, weights = None, *args, **kwargs):
        """
        Initialize the CifarNet model.

        Args:
            num_classes (int): The number of classes in the classification task.
            weights (str, optional): Path to the pre-trained weights file. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super().__init__(num_classes, *args, **kwargs)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the CifarNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(Model):
    """
    Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz for one channel input').

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
    """

    def __init__(self, num_classes: int, weights = None, *args, **kwargs):
        """
        Initialize the Net model.

        Args:
            num_classes (int): The number of classes in the classification task.
            weights (str, optional): Path to the pre-trained weights file. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super().__init__(num_classes, *args, **kwargs)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MobileNetV2(Model):
    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        """
        MobileNetV2 model with customizable classifier head.

        Args:
            num_classes (int): Number of output classes.
            weights (str or None): Weights to initialize the model ('DEFAULT' or None).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(num_classes, *args, **kwargs)

        if weights == 'DEFAULT':
            self._model = torch_models.mobilenet_v2(weights=torch_models.MobileNet_V2_Weights.DEFAULT)
            # change the classifier head for num_classes
            self._model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=1280, out_features=self.num_classes),
            )
            self.pretrained = True
        else:
            self._model = torch_models.mobilenet_v2(num_classes=num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MobileNetV2 model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._model(x)

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
            self._model = torch_models.efficientnet_b0(weights=torch_models.EfficientNet_B0_Weights.DEFAULT)
            # change the classifier head for num_classes
            self._model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=1280, out_features=self.num_classes),
            )
            self.pretrained = True
        else:
            self._model = torch_models.efficientnet_b0(num_classes=num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the EfficientNetB0 model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._model(x)
    
class SimpleCNN(Model):
    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        """
        Simple CNN model with customizable classifier head.

        Args:
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(num_classes, *args, **kwargs)

        self._model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(in_features=64 * 6 * 6, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=self.num_classes),
        )

        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SimpleCNN model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._model(x)

class KerasExpCNN(Model):
    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        """
        Keras-style experimental CNN model with customizable classifier head.

        Args:
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(num_classes, *args, **kwargs)

        self._model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_features=64 * 6 * 6, out_features=self.num_classes),
        )

        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the KerasExpCNN model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._model(x)
    
class MNISTCNN(Model):
    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        """
        CNN model for MNIST dataset with customizable classifier head.

        Args:
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(num_classes, *args, **kwargs)

        self._model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 15 * 15, out_features=28),
            nn.ReLU(),
            nn.Linear(in_features=28, out_features=self.num_classes),
        )

        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MNISTCNN model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._model(x)
    
class SimpleDNN(Model):
    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        """
        Simple deep neural network (DNN) model with customizable classifier head.

        Args:
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(num_classes, *args, **kwargs)

        self._model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3 * 32 * 32, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.num_classes)
        )

        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SimpleDNN model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._model(x)

class FMCNNModel(Model):
    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        """
        Fully convolutional CNN model with customizable classifier head.

        Args:
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(num_classes, *args, **kwargs)

        self._model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.num_classes)
        )

        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the FMCNNModel model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._model(x)

class FedAVGCNN(Model):
    """Architecture of CNN model used in original FedAVG paper with Cifar-10 dataset.
    Paper: https://doi.org/10.48550/arXiv.1602.05629
    """
    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        """
        FedAVG CNN model with customizable classifier head.

        Args:
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(num_classes, *args, **kwargs)

        self._model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.num_classes)
        )

        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the FedAVGCNN model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._model(x)

class LSTMModel(Model):
    """Create an LSTM model for next character task."""
    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        """
        LSTM model for next character prediction task.

        Args:
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(num_classes, *args, **kwargs)

        self.embedding = nn.Embedding(self.num_classes, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.fc = nn.Linear(256, self.num_classes)

        if weights is not None:
            self.load_state_dict(load(weights), strict=True)
            self.pretrained = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the LSTMModel model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.embedding(x)
        h1, (h1_T,c1_T) = self.lstm(x)
        out = self.fc(h1_T.squeeze(0))
        return out

class Resnet18(BaseResNet):
    """Resnet 18 Model Class."""
    def __init__(self, num_classes: int=10, *args, **kwargs):
        super().__init__(block=BasicBlock, num_blocks=[2, 2, 2, 2], activation=F.relu, num_classes = num_classes, *args, **kwargs)

        self.__class__.__name__ = 'Resnet18 Custom'

class Resnet34(BaseResNet):
    """Resnet 34 Model Class."""
    def __init__(self, num_classes: int=10, *args, **kwargs):
        super().__init__(block=BasicBlock, num_blocks=[3, 4, 6, 3], activation=F.relu, num_classes = num_classes, *args, **kwargs)

        self.__class__.__name__ = 'Resnet34 Custom'
