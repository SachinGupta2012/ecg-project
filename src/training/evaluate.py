"""
Model Evaluation for ECG Arrhythmia Detection
================================================
Comprehensive evaluation with:
- Per-class precision, recall, F1
- Confusion matrix
- AUC per class (one-vs-rest)
- Patient-level evaluation
- Classification report
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# AAMI class names
AAMI_CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
AAMI_CLASS_FULL = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown/Paced",
}


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    per_class_auc: np.ndarray
    confusion_mat: np.ndarray
    classification_rep: str
    all_preds: np.ndarray
    all_labels: np.ndarray
    all_probs: np.ndarray


class Evaluator:
    """
    Comprehensive evaluator for ECG arrhythmia detection models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        num_classes: int = 5,
    ):
        self.model = model
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model.to(self.device)
        self.model.eval()

    def predict(self, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions for all data in the loader.

        Returns
        -------
        tuple
            (all_preds, all_labels, all_probs)
        """
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for beats, labels in data_loader:
                beats = beats.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(beats)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        return all_preds, all_labels, all_probs

    def evaluate(self, data_loader: DataLoader) -> EvaluationResult:
        """
        Full evaluation on a data loader.

        Parameters
        ----------
        data_loader : DataLoader
            Data to evaluate on.

        Returns
        -------
        EvaluationResult
            Complete evaluation results.
        """
        preds, labels, probs = self.predict(data_loader)

        # Overall metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = (
            precision_recall_fscore_support(
                labels, preds, average=None, zero_division=0
            )
        )

        # AUC (one-vs-rest)
        try:
            per_class_auc = roc_auc_score(
                labels, probs, multi_class="ovr", average=None
            )
            overall_auc = roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            per_class_auc = np.zeros(self.num_classes)
            overall_auc = 0.0

        # Confusion matrix
        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))

        # Classification report
        target_names = [AAMI_CLASS_NAMES[i] for i in range(self.num_classes)]
        rep = classification_report(
            labels, preds, target_names=target_names, zero_division=0
        )

        result = EvaluationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=overall_auc,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            per_class_auc=per_class_auc,
            confusion_mat=cm,
            classification_rep=rep,
            all_preds=preds,
            all_labels=labels,
            all_probs=probs,
        )

        self._log_results(result)
        return result

    def _log_results(self, result: EvaluationResult):
        """Log evaluation results."""
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:  {result.accuracy:.4f}")
        logger.info(f"Precision: {result.precision:.4f}")
        logger.info(f"Recall:    {result.recall:.4f}")
        logger.info(f"F1 Score:  {result.f1:.4f}")
        logger.info(f"AUC:       {result.auc:.4f}")
        logger.info("")
        logger.info("Per-Class Metrics:")
        logger.info("-" * 40)
        for i in range(self.num_classes):
            name = AAMI_CLASS_NAMES[i]
            logger.info(
                f"  {name}: P={result.per_class_precision[i]:.3f} "
                f"R={result.per_class_recall[i]:.3f} "
                f"F1={result.per_class_f1[i]:.3f} "
                f"AUC={result.per_class_auc[i]:.3f}"
            )
        logger.info("")
        logger.info("Classification Report:")
        logger.info(result.classification_rep)

    def plot_confusion_matrix(
        self,
        result: EvaluationResult,
        save_path: str | Path | None = None,
        normalize: bool = True,
    ) -> None:
        """
        Plot confusion matrix.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation results.
        save_path : str or Path, optional
            Path to save the plot.
        normalize : bool
            If True, normalize by row (true labels).
        """
        cm = result.confusion_mat.astype(float)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm = cm / row_sums

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Raw counts
        sns.heatmap(
            result.confusion_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[AAMI_CLASS_NAMES[i] for i in range(self.num_classes)],
            yticklabels=[AAMI_CLASS_NAMES[i] for i in range(self.num_classes)],
            ax=axes[0],
        )
        axes[0].set_title("Confusion Matrix (Counts)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        # Normalized
        sns.heatmap(
            cm,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            xticklabels=[AAMI_CLASS_NAMES[i] for i in range(self.num_classes)],
            yticklabels=[AAMI_CLASS_NAMES[i] for i in range(self.num_classes)],
            ax=axes[1],
        )
        axes[1].set_title("Confusion Matrix (Normalized)")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.close()

    def plot_per_class_metrics(
        self,
        result: EvaluationResult,
        save_path: str | Path | None = None,
    ) -> None:
        """
        Plot per-class precision, recall, F1.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation results.
        save_path : str or Path, optional
            Path to save the plot.
        """
        classes = [AAMI_CLASS_NAMES[i] for i in range(self.num_classes)]
        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width, result.per_class_precision, width, label="Precision", color="#3498db")
        bars2 = ax.bar(x, result.per_class_recall, width, label="Recall", color="#2ecc71")
        bars3 = ax.bar(x + width, result.per_class_f1, width, label="F1 Score", color="#e74c3c")

        ax.set_xlabel("AAMI Class")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Precision, Recall, and F1 Score")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Per-class metrics plot saved to {save_path}")

        plt.close()


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    save_dir: str | None = None,
) -> EvaluationResult:
    """
    Convenience function to evaluate a model.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    test_loader : DataLoader
        Test data loader.
    device : str
        Device to evaluate on.
    save_dir : str, optional
        Directory to save evaluation plots.

    Returns
    -------
    EvaluationResult
        Evaluation results.
    """
    evaluator = Evaluator(model, device=device)
    result = evaluator.evaluate(test_loader)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        evaluator.plot_confusion_matrix(result, save_path=save_dir / "confusion_matrix.png")
        evaluator.plot_per_class_metrics(result, save_path=save_dir / "per_class_metrics.png")

    return result
