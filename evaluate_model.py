"""Evaluate the trained model."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\data_science\ecg-project")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from src.data.dataset import load_split_and_create_dataloaders
from src.models.cnn_baseline import CNNBaseline
from src.training.evaluate import evaluate_model, AAMI_CLASS_NAMES

# Load data
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "models" / "evaluation"

print("Loading data...")
train_loader, val_loader, test_loader = load_split_and_create_dataloaders(
    split_dir=PROCESSED_DIR, batch_size=128, num_workers=0, balanced_sampling=False,
)

# Load model
model = CNNBaseline(num_classes=5, in_channels=1, window_size=288, dropout=0.3)
model_path = PROJECT_ROOT / "models" / "best_model.pt"

if model_path.exists():
    print("Loading model from", model_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    print("Model loaded (epoch", epoch, ", val_loss:", round(val_loss, 4), ")")
else:
    print("No saved model found!")
    sys.exit(1)

# Evaluate
print("Evaluating on test set...")
result = evaluate_model(model=model, test_loader=test_loader, device="cpu", save_dir=str(EVAL_DIR))

print()
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print("Test Accuracy:", round(result.accuracy, 4))
print("Test Precision (macro):", round(result.precision, 4))
print("Test Recall (macro):", round(result.recall, 4))
print("Test F1 (macro):", round(result.f1, 4))
print("Test AUC (macro):", round(result.auc, 4))
print()
print("Per-class results:")
for i in range(5):
    name = AAMI_CLASS_NAMES[i]
    p = round(result.per_class_precision[i], 3)
    r = round(result.per_class_recall[i], 3)
    f = round(result.per_class_f1[i], 3)
    a = round(result.per_class_auc[i], 3)
    print("  " + name + ": P=" + str(p) + " R=" + str(r) + " F1=" + str(f) + " AUC=" + str(a))
print()
print("Confusion matrix saved to:", EVAL_DIR / "confusion_matrix.png")
print("Per-class metrics saved to:", EVAL_DIR / "per_class_metrics.png")
