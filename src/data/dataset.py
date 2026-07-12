"""
PyTorch Dataset for ECG Arrhythmia Detection
===============================================
Provides Dataset and DataLoader classes for loading preprocessed ECG beats.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


class ECGBeatDataset(Dataset):
    """
    PyTorch Dataset for preprocessed ECG beats.

    Each sample is a single beat segment (fixed-length window around an R-peak)
    with its corresponding AAMI class label.
    """

    def __init__(
        self,
        beats: np.ndarray,
        labels: np.ndarray,
        records: np.ndarray | None = None,
        r_peaks: np.ndarray | None = None,
        transform=None,
    ):
        """
        Parameters
        ----------
        beats : np.ndarray
            Array of shape (N, window_samples) containing beat segments.
        labels : np.ndarray
            Array of shape (N,) containing AAMI class indices (0-4).
        records : np.ndarray, optional
            Array of shape (N,) containing record names.
        r_peaks : np.ndarray, optional
            Array of shape (N,) containing R-peak sample indices.
        transform : callable, optional
            Optional transform to apply to each beat.
        """
        self.beats = torch.FloatTensor(beats).unsqueeze(1)  # Add channel dim: (N, 1, window)
        self.labels = torch.LongTensor(labels)
        self.records = records
        self.r_peaks = r_peaks
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        beat = self.beats[idx]
        label = self.labels[idx]

        if self.transform:
            beat = self.transform(beat)

        return beat, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced classification.

        Returns
        -------
        torch.Tensor
            Class weights inversely proportional to class frequency.
        """
        class_counts = torch.bincount(self.labels, minlength=5).float()
        total = class_counts.sum()
        weights = total / (len(class_counts) * class_counts)
        weights[weights == float("inf")] = 0.0
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.

        Returns
        -------
        torch.Tensor
            Per-sample weights for balanced sampling.
        """
        class_weights = self.get_class_weights()
        sample_weights = class_weights[self.labels]
        return sample_weights

    def get_label_distribution(self) -> dict:
        """
        Get the distribution of labels in the dataset.

        Returns
        -------
        dict
            Mapping of class index to count.
        """
        unique, counts = torch.unique(self.labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}


def create_dataloaders(
    train_beats: np.ndarray,
    train_labels: np.ndarray,
    val_beats: np.ndarray,
    val_labels: np.ndarray,
    test_beats: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
    balanced_sampling: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Parameters
    ----------
    train_beats, val_beats, test_beats : np.ndarray
        Beat arrays for each split.
    train_labels, val_labels, test_labels : np.ndarray
        Label arrays for each split.
    batch_size : int
        Batch size for DataLoaders.
    num_workers : int
        Number of workers for data loading.
    balanced_sampling : bool
        If True, use WeightedRandomSampler for training to handle class imbalance.
    seed : int
        Random seed for shuffling.

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ECGBeatDataset(train_beats, train_labels)
    val_dataset = ECGBeatDataset(val_beats, val_labels)
    test_dataset = ECGBeatDataset(test_beats, test_labels)

    logger.info(f"Train: {len(train_dataset)} beats, distribution: {train_dataset.get_label_distribution()}")
    logger.info(f"Val: {len(val_dataset)} beats, distribution: {val_dataset.get_label_distribution()}")
    logger.info(f"Test: {len(test_dataset)} beats, distribution: {test_dataset.get_label_distribution()}")

    # Create samplers/loaders
    if balanced_sampling and len(train_dataset) > 0:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def load_split_and_create_dataloaders(
    split_dir: Path | str,
    batch_size: int = 64,
    num_workers: int = 0,
    balanced_sampling: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load saved splits and create DataLoaders.

    Parameters
    ----------
    split_dir : Path or str
        Directory containing train.npz, val.npz, test.npz.
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers.
    balanced_sampling : bool
        Use balanced sampling for training.

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    split_dir = Path(split_dir)

    train_data = np.load(split_dir / "train.npz")
    val_data = np.load(split_dir / "val.npz")
    test_data = np.load(split_dir / "test.npz")

    return create_dataloaders(
        train_beats=train_data["beats"],
        train_labels=train_data["labels"],
        val_beats=val_data["beats"],
        val_labels=val_data["labels"],
        test_beats=test_data["beats"],
        test_labels=test_data["labels"],
        batch_size=batch_size,
        num_workers=num_workers,
        balanced_sampling=balanced_sampling,
    )
