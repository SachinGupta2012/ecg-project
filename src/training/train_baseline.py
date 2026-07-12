"""
Train Baseline 1D CNN for ECG Arrhythmia Detection
====================================================
Main script to train and evaluate the baseline CNN model.

Usage:
    python -m src.training.train_baseline
"""

import sys
import logging
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_split_and_create_dataloaders, ECGBeatDataset
from src.models.cnn_baseline import CNNBaseline
from src.training.train import Trainer, TrainingConfig
from src.training.evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Train and evaluate the baseline CNN model."""

    # Configuration
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    EVAL_DIR = PROJECT_ROOT / "models" / "evaluation"

    logger.info("=" * 60)
    logger.info("BASELINE 1D CNN TRAINING")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = load_split_and_create_dataloaders(
        split_dir=PROCESSED_DIR,
        batch_size=64,
        num_workers=0,
        balanced_sampling=True,
    )

    # Compute class weights from training data
    train_data = np.load(PROCESSED_DIR / "train.npz")
    train_labels = train_data["labels"]
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    class_weights = total / (len(unique) * counts)
    class_weights_tensor = torch.FloatTensor(class_weights)

    logger.info(f"Class weights: {class_weights_tensor}")

    # Create model
    logger.info("Creating CNN baseline model...")
    model = CNNBaseline(
        num_classes=5,
        in_channels=1,
        window_size=288,
        conv_channels=[32, 64, 128],
        fc_layers=[128, 64],
        dropout=0.3,
    )

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Model architecture:\n{model}")

    # Training config
    config = TrainingConfig(
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        weight_decay=0.0001,
        early_stopping_patience=15,
        scheduler="cosine",
        scheduler_params={"T_max": 100, "eta_min": 0.00001},
        class_weights="balanced",
        seed=42,
        device="cpu",
        save_dir=str(MODELS_DIR),
        experiment_name="cnn_baseline",
    )

    # Train
    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        config=config,
        class_weights=class_weights_tensor,
    )

    history = trainer.train(train_loader, val_loader)

    # Save final model
    trainer.save_final_model("cnn_baseline_final.pt")

    # Evaluate on test set
    logger.info("=" * 60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 60)

    result = evaluate_model(
        model=model,
        test_loader=test_loader,
        device="cpu",
        save_dir=str(EVAL_DIR),
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Test accuracy: {result.accuracy:.4f}")
    logger.info(f"Test F1 (macro): {result.f1:.4f}")
    logger.info(f"Test AUC (macro): {result.auc:.4f}")
    logger.info(f"Model saved to: {MODELS_DIR}")
    logger.info(f"Evaluation plots saved to: {EVAL_DIR}")

    return result


if __name__ == "__main__":
    main()
