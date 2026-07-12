"""
Baseline 1D CNN for ECG Beat Classification
=============================================
A straightforward 1D convolutional network that treats each beat window
as a 1D signal and applies Conv1D layers followed by dense layers.

Architecture:
    Input: (batch_size, 1, 288) - single channel ECG beat
    Conv1D(32, 7) -> ReLU -> MaxPool
    Conv1D(64, 5) -> ReLU -> MaxPool
    Conv1D(128, 3) -> ReLU -> MaxPool
    Flatten -> FC(128) -> ReLU -> Dropout -> FC(64) -> ReLU -> Dropout -> FC(5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    """
    Baseline 1D CNN for ECG arrhythmia classification.

    This model processes fixed-length beat windows (288 samples at 360 Hz)
    and classifies them into 5 AAMI superclasses.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        window_size: int = 288,
        conv_channels: list[int] | None = None,
        fc_layers: list[int] | None = None,
        dropout: float = 0.3,
    ):
        """
        Parameters
        ----------
        num_classes : int
            Number of output classes (5 for AAMI).
        in_channels : int
            Number of input channels (1 for single-lead ECG).
        window_size : int
            Length of the input beat window in samples.
        conv_channels : list of int, optional
            Output channels for each conv layer. Default: [32, 64, 128].
        fc_layers : list of int, optional
            Hidden units for fully connected layers. Default: [128, 64].
        dropout : float
            Dropout rate between FC layers.
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if fc_layers is None:
            fc_layers = [128, 64]

        self.conv_channels = conv_channels
        self.fc_layers = fc_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Build convolutional layers
        conv_layers = []
        in_ch = in_channels
        for out_ch in conv_channels:
            conv_layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )
            in_ch = out_ch
        self.conv_block = nn.Sequential(*conv_layers)

        # Calculate the size after conv layers
        self._conv_out_size = self._get_conv_output_size(in_channels, window_size)

        # Build fully connected layers
        fc_layers_list = []
        in_features = self._conv_out_size
        for hidden in fc_layers:
            fc_layers_list.extend(
                [
                    nn.Linear(in_features, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            in_features = hidden
        fc_layers_list.append(nn.Linear(in_features, num_classes))
        self.fc_block = nn.Sequential(*fc_layers_list)

        # Initialize weights
        self._init_weights()

    def _get_conv_output_size(self, in_channels: int, window_size: int) -> int:
        """Calculate the output size after convolution layers."""
        dummy = torch.zeros(1, in_channels, window_size)
        with torch.no_grad():
            out = self.conv_block(dummy)
        return out.view(1, -1).size(1)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, window_size).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_block(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, window_size).

        Returns
        -------
        torch.Tensor
            Predicted class indices.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, window_size).

        Returns
        -------
        torch.Tensor
            Class probabilities of shape (batch_size, num_classes).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_model(
    model_path: str | None = None,
    num_classes: int = 5,
    device: str = "cpu",
) -> CNNBaseline:
    """
    Load a trained CNN baseline model.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved model checkpoint.
    num_classes : int
        Number of output classes.
    device : str
        Device to load the model onto.

    Returns
    -------
    CNNBaseline
        The loaded model.
    """
    model = CNNBaseline(num_classes=num_classes)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
