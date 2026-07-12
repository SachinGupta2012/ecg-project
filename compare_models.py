"""Evaluate CNN+LSTM and compare with baseline."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\data_science\ecg-project")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from src.data.dataset import load_split_and_create_dataloaders
from src.models.cnn_baseline import CNNBaseline
from src.models.cnn_lstm import CNNLSTM
from src.training.evaluate import Evaluator, AAMI_CLASS_NAMES

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "models" / "cnn_lstm" / "evaluation"

print("Loading data...")
_, _, test_loader = load_split_and_create_dataloaders(
    split_dir=PROCESSED_DIR, batch_size=128, num_workers=0, balanced_sampling=False,
)

# Load CNN Baseline
print("Loading CNN Baseline...")
baseline = CNNBaseline(num_classes=5, in_channels=1, window_size=288, dropout=0.3)
baseline_path = PROJECT_ROOT / "models" / "best_model.pt"
if baseline_path.exists():
    ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)
    baseline.load_state_dict(ckpt["model_state_dict"])
    print("  CNN Baseline loaded (epoch", ckpt["epoch"], ")")

# Load CNN+LSTM
print("Loading CNN+LSTM...")
lstm_model = CNNLSTM(
    num_classes=5, in_channels=1, window_size=288,
    cnn_channels=[16, 32], lstm_hidden=64, lstm_layers=1,
    bidirectional=False, fc_layers=[32], dropout=0.3,
)
lstm_path = PROJECT_ROOT / "models" / "cnn_lstm" / "best_model.pt"
if lstm_path.exists():
    ckpt = torch.load(lstm_path, map_location="cpu", weights_only=False)
    lstm_model.load_state_dict(ckpt["model_state_dict"])
    print("  CNN+LSTM loaded (epoch", ckpt["epoch"], ")")

# Evaluate both
print("\nEvaluating CNN Baseline...")
baseline_eval = Evaluator(baseline, device="cpu")
baseline_result = baseline_eval.evaluate(test_loader)

print("\nEvaluating CNN+LSTM...")
lstm_eval = Evaluator(lstm_model, device="cpu")
lstm_result = lstm_eval.evaluate(test_loader)

# Comparison
print()
print("=" * 70)
print("MODEL COMPARISON: CNN Baseline vs CNN+LSTM")
print("=" * 70)
print("%-25s %-15s %-15s" % ("Metric", "CNN Baseline", "CNN+LSTM"))
print("-" * 70)
print("%-25s %-15.4f %-15.4f" % ("Accuracy", baseline_result.accuracy, lstm_result.accuracy))
print("%-25s %-15.4f %-15.4f" % ("Precision (macro)", baseline_result.precision, lstm_result.precision))
print("%-25s %-15.4f %-15.4f" % ("Recall (macro)", baseline_result.recall, lstm_result.recall))
print("%-25s %-15.4f %-15.4f" % ("F1 Score (macro)", baseline_result.f1, lstm_result.f1))
print("%-25s %-15.4f %-15.4f" % ("AUC (macro)", baseline_result.auc, lstm_result.auc))

print()
print("Per-Class F1 Comparison:")
print("%-25s %-15s %-15s" % ("Class", "CNN Baseline", "CNN+LSTM"))
print("-" * 70)
for i in range(5):
    name = AAMI_CLASS_NAMES[i]
    b_f1 = baseline_result.per_class_f1[i]
    l_f1 = lstm_result.per_class_f1[i]
    print("%-25s %-15.3f %-15.3f" % (name, b_f1, l_f1))

# Parameter count
baseline_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
lstm_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print()
print("Model Size:")
print("  CNN Baseline: %s parameters" % f"{baseline_params:,}")
print("  CNN+LSTM:     %s parameters" % f"{lstm_params:,}")

# Save comparison plot
try:
    import matplotlib.pyplot as plt

    classes = [AAMI_CLASS_NAMES[i] for i in range(5)]
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_result.per_class_f1, width, label="CNN Baseline", color="#3498db")
    bars2 = ax.bar(x + width/2, lstm_result.per_class_f1, width, label="CNN+LSTM", color="#e74c3c")

    ax.set_xlabel("AAMI Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("CNN Baseline vs CNN+LSTM: Per-Class F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate("%.2f" % height,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_dir = PROJECT_ROOT / "models" / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nComparison plot saved to", save_dir / "f1_comparison.png")

except Exception as e:
    print("\nCould not save comparison plot:", e)
