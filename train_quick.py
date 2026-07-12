"""
Quick Training Script - Reduced epochs for faster results
==========================================================
"""
import sys
import logging
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(r"D:\data_science\ecg-project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_split_and_create_dataloaders
from src.models.cnn_baseline import CNNBaseline
from src.training.train import Trainer, TrainingConfig
from src.training.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    EVAL_DIR = PROJECT_ROOT / "models" / "evaluation"

    logger.info("Loading data...")
    train_loader, val_loader, test_loader = load_split_and_create_dataloaders(
        split_dir=PROCESSED_DIR, batch_size=128, num_workers=0, balanced_sampling=True,
    )

    # Class weights
    train_data = np.load(PROCESSED_DIR / "train.npz")
    unique, counts = np.unique(train_data["labels"], return_counts=True)
    class_weights = torch.FloatTensor(len(train_data["labels"]) / (len(unique) * counts))

    # Model
    model = CNNBaseline(num_classes=5, in_channels=1, window_size=288, dropout=0.3)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    config = TrainingConfig(
        epochs=30, batch_size=128, learning_rate=0.001, weight_decay=0.0001,
        early_stopping_patience=10, scheduler="cosine",
        scheduler_params={"T_max": 30, "eta_min": 0.00001},
        class_weights="balanced", seed=42, device="cpu",
        save_dir=str(MODELS_DIR), experiment_name="cnn_baseline",
    )

    trainer = Trainer(model=model, config=config, class_weights=class_weights)
    history = trainer.train(train_loader, val_loader)
    trainer.save_final_model("cnn_baseline_final.pt")

    # Evaluate
    logger.info("Evaluating on test set...")
    result = evaluate_model(model=model, test_loader=test_loader, device="cpu", save_dir=str(EVAL_DIR))

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {result.accuracy:.4f}")
    logger.info(f"Test F1 (macro): {result.f1:.4f}")
    logger.info(f"Test AUC (macro): {result.auc:.4f}")
    logger.info(f"Model: {MODELS_DIR}")
    logger.info(f"Plots: {EVAL_DIR}")

if __name__ == "__main__":
    main()
