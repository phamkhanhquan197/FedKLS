from torch import Tensor, load, nn

from mak.models.base_model import Model


class LSTMModel(Model):
    """Create an LSTM model for next character task."""

    def __init__(self, num_classes: int, weights=None, *args, **kwargs) -> None:
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
        h1, (h1_T, c1_T) = self.lstm(x)
        out = self.fc(h1_T.squeeze(0))
        return out
