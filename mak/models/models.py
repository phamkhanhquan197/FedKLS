from typing import Tuple

from torch import nn, Tensor, load, flatten
import torch.nn.functional as F
import torchvision.models as torch_models

from mak.models.base_model import Model
from mak.models.resnet import BaseResNet, BasicBlock

import torch


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
    def __init__(self, input_shape: tuple, num_classes: int, weights=None, *args, **kwargs):
        """
        Initialize the CifarNet model.

        Args:
            input_shape (tuple): The shape of the input tensor (channels, height, width).
            num_classes (int): The number of classes in the classification task.
            weights (str, optional): Path to the pre-trained weights file. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super(CifarNet, self).__init__(num_classes, *args, **kwargs)

        self.conv1 = nn.Conv2d(input_shape[0], 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Calculate the size of the feature map after the convolutional and pooling layers
        self.feature_size = self._get_conv_output(shape= input_shape)

        self.fc1 = nn.Linear(self.feature_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes) #use num_classes instead of 10

        if weights is not None:
            self.load_state_dict(torch.load(weights), strict=True)
            self.pretrained = True

    

    def forward(self, x):
        """
        Forward pass of the CifarNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _get_conv_output(self,shape):
        # Create a dummy tensor with the input shape
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()


class Net(Model):
    """
    Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz ').

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
    """

    def __init__(self, input_shape: tuple, num_classes: int, weights=None, *args, **kwargs):
        """
        Initialize the Net model.

        Args:
            input_shape (tuple): The shape of the input tensor (channels, height, width).
            num_classes (int): The number of classes in the classification task.
            weights (str, optional): Path to the pre-trained weights file. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super(Net, self).__init__(num_classes, *args, **kwargs)

        self.conv1 = nn.Conv2d(input_shape[0], 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Calculate the size of the feature map after the convolutional and pooling layers
        self.feature_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(self.feature_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        if weights is not None:
            self.load_state_dict(torch.load(weights), strict=True)
            self.pretrained = True

    def _get_conv_output(self, shape):
        # Create a dummy tensor with the input shape
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(Model):
    def __init__(self, input_shape: tuple, num_classes: int, weights=None, *args, **kwargs) -> None:
        """
        Simple CNN model with customizable classifier head.

        Args:
            input_shape (tuple): The shape of the input tensor (channels, height, width).
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(SimpleCNN, self).__init__(num_classes, *args, **kwargs)

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        # Calculate the size of the flattened feature map
        self.feature_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)

        if weights is not None:
            self.load_state_dict(torch.load(weights), strict=True)
            self.pretrained = True

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.dropout3(F.relu(self.fc1(x)))
        return self.fc2(x)


class KerasExpCNN(Model):
    def __init__(self, input_shape: tuple, num_classes: int, weights=None, *args, **kwargs) -> None:
        """
        Keras-style experimental CNN model with customizable classifier head.

        Args:
            input_shape (tuple): The shape of the input tensor (channels, height, width).
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(KerasExpCNN, self).__init__(num_classes, *args, **kwargs)

        # Define the convolutional and pooling layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.4)

        # Calculate the size of the feature map after the convolutional and pooling layers
        self.feature_size = self._get_conv_output(input_shape)

        # Define the fully connected layer
        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=num_classes)

        if weights is not None:
            self.load_state_dict(torch.load(weights), strict=True)
            self.pretrained = True

    def _get_conv_output(self, shape):
        # Create a dummy tensor with the input shape
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.dropout1(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the KerasExpCNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


class MNISTCNN(Model):
    def __init__(self, input_shape: tuple, num_classes: int, weights=None, *args, **kwargs) -> None:
        """
        CNN model for MNIST dataset with customizable classifier head.

        Args:
            input_shape (tuple): The shape of the input tensor (channels, height, width).
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(MNISTCNN, self).__init__(num_classes, *args, **kwargs)

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size of the feature map after the convolutional and pooling layers
        self.feature_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=28)
        self.fc2 = nn.Linear(in_features=28, out_features=num_classes)

        if weights is not None:
            self.load_state_dict(torch.load(weights), strict=True)
            self.pretrained = True

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.pool(F.relu(self.conv1(x)))
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleDNN(Model):
    def __init__(self, input_shape: tuple, num_classes: int, weights=None, *args, **kwargs) -> None:
        """
        Simple deep neural network (DNN) model with customizable classifier head.

        Args:
            input_shape (tuple): The shape of the input tensor (channels, height, width).
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(SimpleDNN, self).__init__(num_classes, *args, **kwargs)

        self.input_shape = input_shape
        self.flatten = nn.Flatten()

        # Calculate the size of the flattened input
        self.input_size = self._get_flattened_size(input_shape)

        self.fc1 = nn.Linear(in_features=self.input_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

        if weights is not None:
            self.load_state_dict(torch.load(weights), strict=True)
            self.pretrained = True

    def _get_flattened_size(self, shape):
        # Compute the size after flattening the input
        return torch.zeros(1, *shape).numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class FMCNNModel(Model):
    def __init__(self, input_shape: tuple, num_classes: int, weights=None, *args, **kwargs) -> None:
        """
        Fully convolutional CNN model with customizable classifier head.

        Args:
            input_shape (tuple): The shape of the input tensor (channels, height, width).
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(FMCNNModel, self).__init__(num_classes, *args, **kwargs)

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=5, stride=1, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same')
        
        # Calculate the size of the flattened feature map
        self.feature_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

        if weights is not None:
            self.load_state_dict(torch.load(weights), strict=True)
            self.pretrained = True

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FedAVGCNN(Model):
    """Architecture of CNN model used in original FedAVG paper with Cifar-10 dataset.
    Paper: https://doi.org/10.48550/arXiv.1602.05629
    """
    def __init__(self, input_shape: tuple, num_classes: int, weights=None, *args, **kwargs) -> None:
        """
        FedAVG CNN model with customizable classifier head.

        Args:
            input_shape (tuple): The shape of the input tensor (channels, height, width).
            num_classes (int): Number of output classes.
            weights (str or None): Path to pre-trained weights (if available).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(FedAVGCNN, self).__init__(num_classes, *args, **kwargs)

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=5, padding='same')
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        
        # Calculate the size of the flattened feature map
        self.feature_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(in_features=self.feature_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

        if weights is not None:
            self.load_state_dict(torch.load(weights), strict=True)
            self.pretrained = True

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



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
