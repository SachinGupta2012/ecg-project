"""
Train Sequence CNN+LSTM for ECG Arrhythmia Detection
======================================================
Trains the sequence-aware CNN+LSTM model that processes consecutive beats.
"""

import sys
import logging
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(r"D:\data_science\ecg-project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sequence_dataset import load_split_and_create_sequence_dataloaders
from src.models.sequence_cnn_lstm import SequenceCNNLSTM
from src.training.train import Trainer, TrainingConfig
from src.training.evaluate import AAMI_CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SequenceEvaluator:
    """Evaluator for sequence-based models."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, data_loader):
        """Evaluate on a data loader."""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            roc_auc_score, confusion_matrix, classification_report
        )

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for beats, labels in data_loader:
                beats = beats.to(self.device)
                outputs = self.model(beats)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "all_preds": all_preds,
            "all_labels": all_labels,
        }


def main():
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models" / "sequence_cnn_lstm"
    EVAL_DIR = PROJECT_ROOT / "models" / "sequence_cnn_lstm" / "evaluation"

    SEQ_LENGTH = 10  # Use 10 consecutive beats

    logger.info("=" * 60)
    logger.info("SEQUENCE CNN+LSTM TRAINING")
    logger.info("=" * 60)
    logger.info("Sequence length: %d beats", SEQ_LENGTH)

    # Load data with sequences
    logger.info("Loading data with sequences...")
    train_loader, val_loader, test_loader = load_split_and_create_sequence_dataloaders(
        split_dir=PROCESSED_DIR,
        seq_length=SEQ_LENGTH,
        batch_size=128,
        num_workers=0,
        balanced_sampling=True,
    )

    # Create model
    logger.info("Creating Sequence CNN+LSTM model...")
    model = SequenceCNNLSTM(
        num_classes=5,
        in_channels=1,
        window_size=288,
        seq_length=SEQ_LENGTH,
        cnn_channels=[32, 64],
        lstm_hidden=128,
        lstm_layers=2,
        bidirectional=True,
        fc_layers=[64],
        dropout=0.3,
    )

    logger.info("Model parameters: %d", model.count_parameters())

    # Class weights
    train_data = np.load(PROCESSED_DIR / "train.npz")
    unique, counts = np.unique(train_data["labels"], return_counts=True)
    class_weights = torch.FloatTensor(len(train_data["labels"]) / (len(unique) * counts))

    # Training config
    config = TrainingConfig(
        epochs=30,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        early_stopping_patience=10,
        scheduler="cosine",
        scheduler_params={"T_max": 30, "eta_min": 0.00001},
        class_weights="balanced",
        seed=42,
        device="cpu",
        save_dir=str(MODELS_DIR),
        experiment_name="sequence_cnn_lstm",
    )

    # Train
    logger.info("Starting training...")
    trainer = Trainer(model=model, config=config, class_weights=class_weights)
    history = trainer.train(train_loader, val_loader)
    trainer.save_final_model("sequence_cnn_lstm_final.pt")

    # Evaluate
    logger.info("=" * 60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 60)

    evaluator = SequenceEvaluator(model, device="cpu")
    result = evaluator.evaluate(test_loader)

    logger.info("=" * 60)
    logger.info("SEQUENCE CNN+LSTM RESULTS")
    logger.info("=" * 60)
    logger.info("Test Accuracy: %.4f", result["accuracy"])
    logger.info("Test Precision (macro): %.4f", result["precision"])
    logger.info("Test Recall (macro): %.4f", result["recall"])
    logger.info("Test F1 (macro): %.4f", result["f1"])
    logger.info("Test AUC (macro): %.4f", result["auc"])

    return result


if __name__ == "__main__":
    main()
