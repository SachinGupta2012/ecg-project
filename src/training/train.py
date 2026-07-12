"""
Training Loop for ECG Arrhythmia Detection
=============================================
Complete training pipeline with:
- Early stopping
- Learning rate scheduling
- Class-weighted loss for imbalanced data
- Model checkpointing
- MLflow logging integration
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 15
    scheduler: str = "cosine"  # cosine, step, reduce_on_plateau
    scheduler_params: dict = field(default_factory=lambda: {"T_max": 100, "eta_min": 0.00001})
    class_weights: str = "balanced"  # balanced, none
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "models"
    experiment_name: str = "cnn_baseline"


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Training manager for ECG arrhythmia detection models.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        if config is None:
            config = TrainingConfig()

        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)

        # Loss function with class weights
        if class_weights is not None and config.class_weights == "balanced":
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logger.info("Using standard CrossEntropyLoss")

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode="min",
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_model_state = None

        # Save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler_params.get("T_max", self.config.epochs),
                eta_min=self.config.scheduler_params.get("eta_min", 0.00001),
            )
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_params.get("step_size", 30),
                gamma=self.config.scheduler_params.get("gamma", 0.1),
            )
        elif self.config.scheduler == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=10,
            )
        return None

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _batch_idx, (beats, labels) in enumerate(train_loader):
            beats = beats.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(beats)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * beats.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for beats, labels in val_loader:
                beats = beats.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(beats)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * beats.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mlflow_callback=None,
    ) -> dict:
        """
        Full training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        mlflow_callback : callable, optional
            Function to call after each epoch for MLflow logging.

        Returns
        -------
        dict
            Training history.
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=True)

            epoch_time = time.time() - epoch_start

            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{self.config.epochs}] "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
                )

            # MLflow callback
            if mlflow_callback is not None:
                mlflow_callback(
                    epoch,
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "learning_rate": current_lr,
                    },
                )

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": {
                "num_classes": self.model.num_classes if hasattr(self.model, "num_classes") else 5,
                "conv_channels": self.model.conv_channels
                if hasattr(self.model, "conv_channels")
                else None,
                "fc_layers": self.model.fc_layers if hasattr(self.model, "fc_layers") else None,
            },
        }

        if is_best:
            path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved best model to {path}")

        # Save latest
        path = self.save_dir / "latest_model.pt"
        torch.save(checkpoint, path)

    def save_final_model(self, filename: str = "final_model.pt"):
        """Save the final trained model."""
        path = self.save_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved final model to {path}")
        return path
