"""
Train CNN+LSTM Model for ECG Arrhythmia Detection
===================================================
Trains the CNN+LSTM model and compares against the CNN baseline.
"""

import sys
import logging
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(r"D:\data_science\ecg-project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_split_and_create_dataloaders
from src.models.cnn_lstm import CNNLSTM
from src.training.train import Trainer, TrainingConfig
from src.training.evaluate import evaluate_model, AAMI_CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models" / "cnn_lstm"
    EVAL_DIR = PROJECT_ROOT / "models" / "cnn_lstm" / "evaluation"

    logger.info("=" * 60)
    logger.info("CNN+LSTM TRAINING")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = load_split_and_create_dataloaders(
        split_dir=PROCESSED_DIR, batch_size=128, num_workers=0, balanced_sampling=True,
    )

    # Class weights
    train_data = np.load(PROCESSED_DIR / "train.npz")
    unique, counts = np.unique(train_data["labels"], return_counts=True)
    class_weights = torch.FloatTensor(len(train_data["labels"]) / (len(unique) * counts))
    logger.info("Class weights: %s", class_weights)

    # Create model
    logger.info("Creating CNN+LSTM model...")
    model = CNNLSTM(
        num_classes=5,
        in_channels=1,
        window_size=288,
        cnn_channels=[32, 64],
        lstm_hidden=128,
        lstm_layers=2,
        bidirectional=True,
        fc_layers=[64],
        dropout=0.3,
    )

    logger.info("Model parameters: %s", model.count_parameters())

    # Training config
    config = TrainingConfig(
        epochs=50,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        early_stopping_patience=12,
        scheduler="cosine",
        scheduler_params={"T_max": 50, "eta_min": 0.00001},
        class_weights="balanced",
        seed=42,
        device="cpu",
        save_dir=str(MODELS_DIR),
        experiment_name="cnn_lstm",
    )

    # Train
    logger.info("Starting training...")
    trainer = Trainer(model=model, config=config, class_weights=class_weights)
    history = trainer.train(train_loader, val_loader)
    trainer.save_final_model("cnn_lstm_final.pt")

    # Evaluate
    logger.info("=" * 60)
    logger.info("EVALUATING CNN+LSTM ON TEST SET")
    logger.info("=" * 60)

    result = evaluate_model(
        model=model, test_loader=test_loader, device="cpu", save_dir=str(EVAL_DIR),
    )

    # Print results
    logger.info("=" * 60)
    logger.info("CNN+LSTM RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Test Accuracy: %.4f", result.accuracy)
    logger.info("Test Precision (macro): %.4f", result.precision)
    logger.info("Test Recall (macro): %.4f", result.recall)
    logger.info("Test F1 (macro): %.4f", result.f1)
    logger.info("Test AUC (macro): %.4f", result.auc)
    logger.info("")
    logger.info("Per-class results:")
    for i in range(5):
        name = AAMI_CLASS_NAMES[i]
        logger.info(
            "  %s: P=%.3f R=%.3f F1=%.3f AUC=%.3f",
            name,
            result.per_class_precision[i],
            result.per_class_recall[i],
            result.per_class_f1[i],
            result.per_class_auc[i],
        )

    return result


if __name__ == "__main__":
    main()
