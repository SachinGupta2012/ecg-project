"""
Model Inference for ECG Arrhythmia Detection
===============================================
Loads trained models and runs inference on ECG data.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# AAMI class names
AAMI_CLASSES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
AAMI_CLASS_NAMES = {v: k for k, v in AAMI_CLASSES.items()}


class ECGClassifier:
    """
    ECG arrhythmia classifier that wraps a trained model.
    
    Handles model loading, preprocessing, and inference.
    """

    def __init__(
        self,
        model_name: str = "cnn_baseline",
        model_path: str | Path | None = None,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        model_name : str
            Name of the model architecture (cnn_baseline or cnn_lstm).
        model_path : str or Path, optional
            Path to the model checkpoint. If None, uses default path.
        device : str
            Device to run inference on.
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = None
        self.is_loaded = False

        if model_path is None:
            model_path = self._get_default_model_path(model_name)

        self.model_path = Path(model_path)
        self.load_model()

    def _get_default_model_path(self, model_name: str) -> Path:
        """Get the default model path for a given model name."""
        if model_name == "cnn_baseline":
            return PROJECT_ROOT / "models" / "best_model.pt"
        elif model_name in ("cnn_lstm", "cnn_lstm_light"):
            return PROJECT_ROOT / "models" / "cnn_lstm" / "best_model.pt"
        elif model_name == "sequence_cnn_lstm":
            return PROJECT_ROOT / "models" / "sequence_cnn_lstm" / "best_model.pt"
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            logger.warning("Model not found at %s", self.model_path)
            return

        try:
            if self.model_name == "cnn_baseline":
                from src.models.cnn_baseline import CNNBaseline
                self.model = CNNBaseline(num_classes=5, in_channels=1, window_size=288)
            elif self.model_name in ("cnn_lstm", "cnn_lstm_light"):
                from src.models.cnn_lstm import CNNLSTM
                self.model = CNNLSTM(
                    num_classes=5, in_channels=1, window_size=288,
                    cnn_channels=[16, 32], lstm_hidden=64, lstm_layers=1,
                    bidirectional=False, fc_layers=[32], dropout=0.3,
                )
            elif self.model_name == "sequence_cnn_lstm":
                from src.models.sequence_cnn_lstm import SequenceCNNLSTM
                self.model = SequenceCNNLSTM(num_classes=5, in_channels=1, window_size=288)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("Model loaded: %s (%s)", self.model_name, self.model_path)

        except Exception as e:
            logger.error("Failed to load model: %s", e)
            self.is_loaded = False

    def predict(self, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict classes for a batch of beats.

        Parameters
        ----------
        beats : np.ndarray
            Array of shape (N, 1, window_size) containing preprocessed beats.

        Returns
        -------
        tuple
            (predictions, probabilities, confidences)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Convert to tensor
        x = torch.FloatTensor(beats).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confidences = probs.max(dim=1).values

        return (
            preds.cpu().numpy(),
            probs.cpu().numpy(),
            confidences.cpu().numpy(),
        )

    def predict_single(self, beat: np.ndarray) -> dict:
        """
        Predict class for a single beat.

        Parameters
        ----------
        beat : np.ndarray
            Array of shape (1, window_size) containing a single beat.

        Returns
        -------
        dict
            Prediction result with class, confidence, and probabilities.
        """
        if len(beat.shape) == 1:
            beat = beat.reshape(1, -1)
        
        # Add channel dimension if needed
        if len(beat.shape) == 2:
            beat = beat.reshape(1, 1, -1)

        preds, probs, confidences = self.predict(beat)

        pred_class = AAMI_CLASSES[int(preds[0])]
        confidence = float(confidences[0])
        probabilities = {
            AAMI_CLASSES[i]: float(probs[0, i])
            for i in range(5)
        }

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }


# Singleton instance for the API
_classifier: ECGClassifier | None = None


def get_classifier(model_name: str = "cnn_baseline") -> ECGClassifier:
    """
    Get or create the classifier instance.

    Parameters
    ----------
    model_name : str
        Model name to load.

    Returns
    -------
    ECGClassifier
        The classifier instance.
    """
    global _classifier
    if _classifier is None or _classifier.model_name != model_name:
        _classifier = ECGClassifier(model_name=model_name)
    return _classifier
