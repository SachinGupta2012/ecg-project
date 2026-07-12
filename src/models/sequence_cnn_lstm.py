"""
Sequence-Aware CNN+LSTM for ECG Arrhythmia Detection
======================================================
Properly processes sequences of consecutive beats to capture rhythm context.

Architecture:
    Input: (batch_size, seq_length, 1, window_size)
           - A sequence of consecutive beats
    CNN: Extracts features from each beat independently
    LSTM: Models temporal dependencies across the beat sequence
    FC: Classifies each beat (center beat or all beats)

Key insight: Arrhythmias like AFib are defined by irregular timing
between beats, not just single-beat shape. This model captures that.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceCNNLSTM(nn.Module):
    """
    CNN+LSTM that processes sequences of consecutive ECG beats.
    
    The CNN extracts features from each beat independently,
    then the LSTM models the temporal pattern across the sequence.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        window_size: int = 288,
        seq_length: int = 10,
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
            Length of each beat window in samples.
        seq_length : int
            Number of consecutive beats in each sequence.
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
        super(SequenceCNNLSTM, self).__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64]
        if fc_layers is None:
            fc_layers = [64]

        self.num_classes = num_classes
        self.seq_length = seq_length
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        # CNN feature extractor (applied to each beat independently)
        conv_layers = []
        in_ch = in_channels
        for out_ch in cnn_channels:
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*conv_layers)

        # Calculate CNN output size
        self._cnn_out_channels = cnn_channels[-1]
        self._cnn_out_length = self._get_cnn_output_length(window_size)
        cnn_feature_dim = self._cnn_out_channels * self._cnn_out_length

        # LSTM for temporal modeling across beat sequence
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # LSTM output size
        lstm_output_size = lstm_hidden * 2 if bidirectional else lstm_hidden

        # FC classification head
        fc_layers_list = []
        in_features = lstm_output_size
        for hidden in fc_layers:
            fc_layers_list.extend([
                nn.Linear(in_features, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
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
            Input tensor of shape (batch_size, seq_length, 1, window_size).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Reshape to process all beats through CNN at once
        # (batch_size * seq_length, 1, window_size)
        x_reshaped = x.view(batch_size * seq_len, 1, x.size(3))

        # CNN feature extraction
        cnn_out = self.cnn(x_reshaped)
        # (batch_size * seq_length, cnn_channels, cnn_out_length)

        # Flatten CNN features
        cnn_out = cnn_out.view(batch_size * seq_len, -1)
        # (batch_size * seq_length, cnn_feature_dim)

        # Reshape back to sequence: (batch_size, seq_length, cnn_feature_dim)
        cnn_features = cnn_out.view(batch_size, seq_len, -1)

        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        # lstm_out: (batch_size, seq_length, lstm_hidden * num_directions)

        # Take the last time step (or center beat)
        # For sequence classification, we use the last time step
        lstm_out = lstm_out[:, -1, :]
        # (batch_size, lstm_hidden * num_directions)

        # FC classification
        logits = self.fc(lstm_out)
        # (batch_size, num_classes)

        return logits

    def forward_per_beat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns predictions for each beat in the sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, 1, window_size).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, seq_length, num_classes).
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Process all beats through CNN
        x_reshaped = x.view(batch_size * seq_len, 1, x.size(3))
        cnn_out = self.cnn(x_reshaped)
        cnn_out = cnn_out.view(batch_size * seq_len, -1)
        cnn_features = cnn_out.view(batch_size, seq_len, -1)

        # LSTM
        lstm_out, _ = self.lstm(cnn_features)
        # (batch_size, seq_length, lstm_hidden * num_directions)

        # Process each time step through FC
        lstm_out = lstm_out.view(batch_size * seq_len, -1)
        logits = self.fc(lstm_out)
        logits = logits.view(batch_size, seq_len, -1)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels for the sequence."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities for the sequence."""
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
) -> SequenceCNNLSTM:
    """
    Load a trained Sequence CNN+LSTM model.

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
    SequenceCNNLSTM
        The loaded model.
    """
    model = SequenceCNNLSTM(num_classes=num_classes)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
