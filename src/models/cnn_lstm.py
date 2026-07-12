"""
CNN + LSTM Model for ECG Arrhythmia Detection
================================================
Combines CNN for local waveform feature extraction with LSTM for
rhythm context modeling across consecutive beats.

Architecture:
    Input: (batch_size, 1, 288) - single channel ECG beat
    CNN: Conv1D layers extract local shape features
    LSTM: Captures temporal dependencies across the feature sequence
    FC: Classification head

Key insight: Some arrhythmias (like AFib) are defined by irregular timing
between beats, not just single-beat shape. LSTM captures this rhythm context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    """
    CNN + LSTM model for ECG arrhythmia classification.

    Uses CNN to extract per-beat features, then LSTM to model
    temporal patterns across the feature sequence.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        window_size: int = 288,
        cnn_channels: list[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
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
        cnn_channels : list of int, optional
            Output channels for each CNN layer. Default: [32, 64].
        lstm_hidden : int
            Hidden size for LSTM.
        lstm_layers : int
            Number of LSTM layers.
        bidirectional : bool
            If True, use bidirectional LSTM.
        fc_layers : list of int, optional
            Hidden units for fully connected layers. Default: [64].
        dropout : float
            Dropout rate.
        """
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64]
        if fc_layers is None:
            fc_layers = [64]

        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout

        # CNN feature extractor
        conv_layers = []
        in_ch = in_channels
        for out_ch in cnn_channels:
            conv_layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*conv_layers)

        # Calculate CNN output size
        self._cnn_out_channels = cnn_channels[-1]
        self._cnn_out_length = self._get_cnn_output_length(window_size)

        # LSTM
        lstm_input_size = self._cnn_out_channels
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Calculate LSTM output size
        lstm_output_size = lstm_hidden * 2 if bidirectional else lstm_hidden

        # FC classification head
        fc_layers_list = []
        in_features = lstm_output_size
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
        self.fc = nn.Sequential(*fc_layers_list)

        # Initialize weights
        self._init_weights()

    def _get_cnn_output_length(self, window_size: int) -> int:
        """Calculate the sequence length after CNN layers."""
        dummy = torch.zeros(1, 1, window_size)
        with torch.no_grad():
            out = self.cnn(dummy)
        return out.size(2)

    def _init_weights(self):
        """Initialize weights."""
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
        # CNN feature extraction
        # Input: (batch_size, 1, window_size)
        cnn_out = self.cnn(x)
        # Output: (batch_size, cnn_channels, cnn_seq_length)

        # Reshape for LSTM: (batch_size, seq_length, features)
        # We treat each time step of the CNN output as a sequence element
        cnn_out = cnn_out.permute(0, 2, 1)
        # Output: (batch_size, cnn_seq_length, cnn_channels)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_out)
        # lstm_out: (batch_size, seq_length, lstm_hidden * num_directions)

        # Take the last time step
        lstm_out = lstm_out[:, -1, :]
        # Output: (batch_size, lstm_hidden * num_directions)

        # FC classification
        logits = self.fc(lstm_out)
        # Output: (batch_size, num_classes)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Extract intermediate feature maps for visualization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, window_size).

        Returns
        -------
        dict
            Dictionary with CNN and LSTM outputs.
        """
        self.eval()
        with torch.no_grad():
            cnn_out = self.cnn(x)
            cnn_out_permuted = cnn_out.permute(0, 2, 1)
            lstm_out, (hidden, cell) = self.lstm(cnn_out_permuted)

            return {
                "cnn_output": cnn_out,
                "lstm_output": lstm_out,
                "lstm_hidden": hidden,
                "lstm_cell": cell,
            }


def load_model(
    model_path: str | None = None,
    num_classes: int = 5,
    device: str = "cpu",
) -> CNNLSTM:
    """
    Load a trained CNN+LSTM model.

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
    CNNLSTM
        The loaded model.
    """
    model = CNNLSTM(num_classes=num_classes)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
