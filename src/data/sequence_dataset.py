"""
Sequence Dataset for CNN+LSTM
==============================
Groups consecutive beats into sequences so LSTM can model rhythm context.

Instead of classifying single beats, we feed a sequence of N consecutive beats
and classify each beat in the sequence. This lets the LSTM learn:
- Irregular spacing between beats (arrhythmia timing)
- Beat-to-beat morphology changes
- Rhythm patterns that span multiple beats
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


class ECGSequenceDataset(Dataset):
    """
    Dataset that returns sequences of consecutive ECG beats.

    Each sample is a window of `seq_length` consecutive beats,
    and the label is the class of the center beat.
    """

    def __init__(
        self,
        beats: np.ndarray,
        labels: np.ndarray,
        records: np.ndarray,
        seq_length: int = 10,
        center_only: bool = True,
    ):
        """
        Parameters
        ----------
        beats : np.ndarray
            Array of shape (N, window_samples) containing all beats.
        labels : np.ndarray
            Array of shape (N,) containing AAMI class labels.
        records : np.ndarray
            Array of shape (N,) containing record names.
        seq_length : int
            Number of consecutive beats in each sequence.
        center_only : bool
            If True, only return label for the center beat.
            If False, return labels for all beats in sequence.
        """
        self.beats = beats
        self.labels = labels
        self.records = records
        self.seq_length = seq_length
        self.center_only = center_only
        self.half_seq = seq_length // 2

        # Find boundaries between different records
        # (can't create sequences across record boundaries)
        self.record_changes = [0]
        for i in range(1, len(records)):
            if records[i] != records[i-1]:
                self.record_changes.append(i)
        self.record_changes.append(len(records))

        # Create valid indices (where a full sequence fits within a record)
        self.valid_indices = []
        for boundary_idx in range(len(self.record_changes) - 1):
            start = self.record_changes[boundary_idx]
            end = self.record_changes[boundary_idx + 1]
            for i in range(start + self.half_seq, end - self.half_seq):
                self.valid_indices.append(i)

        logger.info(
            f"SequenceDataset: {len(self.valid_indices)} valid sequences "
            f"(seq_length={seq_length}, from {len(beats)} beats)"
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        center_idx = self.valid_indices[idx]
        start = center_idx - self.half_seq
        end = center_idx + self.half_seq + 1

        # Extract sequence of beats: (seq_length, 1, window_samples)
        seq_beats = self.beats[start:end]
        seq_beats = torch.FloatTensor(seq_beats).unsqueeze(1)  # Add channel dim

        if self.center_only:
            # Return only the center beat's label
            label = torch.LongTensor([self.labels[center_idx]])
            return seq_beats, label.squeeze()
        else:
            # Return labels for all beats in sequence
            seq_labels = torch.LongTensor(self.labels[start:end])
            return seq_beats, seq_labels


class SingleBeatWithNeighbors:
    """
    Extends ECGBeatDataset to include neighboring beat features.
    
    Instead of raw sequences, this extracts handcrafted features from
    neighboring beats (like RR intervals) to augment the single-beat input.
    """

    def __init__(
        self,
        beats: np.ndarray,
        labels: np.ndarray,
        r_peaks: np.ndarray | None = None,
        seq_context: int = 5,
        sampling_rate: int = 360,
    ):
        """
        Parameters
        ----------
        beats : np.ndarray
            Array of shape (N, window_samples).
        labels : np.ndarray
            Array of shape (N,).
        r_peaks : np.ndarray, optional
            Array of shape (N,) with R-peak sample indices.
        seq_context : int
            Number of neighboring beats to use for context features.
        sampling_rate : int
            ECG sampling rate in Hz.
        """
        super().__init__(beats, labels)
        self.r_peaks = r_peaks
        self.seq_context = seq_context
        self.sampling_rate = sampling_rate

        # Precompute RR intervals
        if r_peaks is not None:
            self.rr_intervals = np.zeros(len(r_peaks))
            for i in range(1, len(r_peaks)):
                self.rr_intervals[i] = (r_peaks[i] - r_peaks[i-1]) / sampling_rate
        else:
            self.rr_intervals = None

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        beat = self.beats[idx]
        label = self.labels[idx]

        # Add RR interval features if available
        if self.rr_intervals is not None and self.r_peaks is not None:
            # Get context RR intervals
            context_start = max(0, idx - self.seq_context)
            context_end = min(len(self.rr_intervals), idx + self.seq_context + 1)
            
            # Pad if at boundary
            rr_context = self.rr_intervals[context_start:context_end]
            if len(rr_context) < 2 * self.seq_context + 1:
                rr_context = np.pad(rr_context, (0, 2 * self.seq_context + 1 - len(rr_context)))
            
            # Compute statistics
            mean_rr = np.mean(rr_context)
            std_rr = np.std(rr_context)
            rr_ratio = self.rr_intervals[idx] / mean_rr if mean_rr > 0 else 1.0
            
            # Append features to beat
            extra_features = np.array([mean_rr, std_rr, rr_ratio], dtype=np.float32)
            beat = np.concatenate([beat, extra_features])

        return torch.FloatTensor(beat).unsqueeze(0), torch.LongTensor([label]).squeeze()


# Import ECGBeatDataset from dataset.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import ECGBeatDataset


def create_sequence_dataloaders(
    train_beats: np.ndarray,
    train_labels: np.ndarray,
    train_records: np.ndarray,
    val_beats: np.ndarray,
    val_labels: np.ndarray,
    val_records: np.ndarray,
    test_beats: np.ndarray,
    test_labels: np.ndarray,
    test_records: np.ndarray,
    seq_length: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
    balanced_sampling: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create sequence-based DataLoaders for CNN+LSTM training.

    Parameters
    ----------
    train_beats, val_beats, test_beats : np.ndarray
        Beat arrays for each split.
    train_labels, val_labels, test_labels : np.ndarray
        Label arrays for each split.
    train_records, val_records, test_records : np.ndarray
        Record name arrays for each split.
    seq_length : int
        Number of consecutive beats in each sequence.
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
    # Create datasets
    train_dataset = ECGSequenceDataset(train_beats, train_labels, train_records, seq_length)
    val_dataset = ECGSequenceDataset(val_beats, val_labels, val_records, seq_length)
    test_dataset = ECGSequenceDataset(test_beats, test_labels, test_records, seq_length)

    logger.info(f"Train sequences: {len(train_dataset)}")
    logger.info(f"Val sequences: {len(val_dataset)}")
    logger.info(f"Test sequences: {len(test_dataset)}")

    # Create samplers/loaders
    if balanced_sampling and len(train_dataset) > 0:
        # Get labels for sampling (center beat labels)
        center_labels = np.array([
            train_dataset.labels[train_dataset.valid_indices[i]]
            for i in range(len(train_dataset))
        ])
        class_counts = np.bincount(center_labels, minlength=5).astype(float)
        class_weights = len(center_labels) / (5 * class_counts + 1e-8)
        sample_weights = class_weights[center_labels]
        
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


def load_split_and_create_sequence_dataloaders(
    split_dir: Path | str,
    seq_length: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
    balanced_sampling: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load saved splits and create sequence DataLoaders.

    Parameters
    ----------
    split_dir : Path or str
        Directory containing train.npz, val.npz, test.npz.
    seq_length : int
        Sequence length.
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

    return create_sequence_dataloaders(
        train_beats=train_data["beats"],
        train_labels=train_data["labels"],
        train_records=train_data["records"],
        val_beats=val_data["beats"],
        val_labels=val_data["labels"],
        val_records=val_data["records"],
        test_beats=test_data["beats"],
        test_labels=test_data["labels"],
        test_records=test_data["records"],
        seq_length=seq_length,
        batch_size=batch_size,
        num_workers=num_workers,
        balanced_sampling=balanced_sampling,
    )
