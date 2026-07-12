"""Run the full data pipeline on available records."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from src.data.preprocessing import ECGPreprocessor, load_config, AAMI_CLASS_NAMES
from src.data.split import patient_wise_split, save_split_data
from src.data.dataset import create_dataloaders

DATA_DIR = Path(r"D:\data_science\ecg-project\data\raw\mitdb")
PROCESSED_DIR = Path(r"D:\data_science\ecg-project\data\processed")

# Get available records
available_records = sorted([f.stem for f in DATA_DIR.glob("*.hea")])
print(f"Available records: {len(available_records)}")
print(f"Records: {available_records}")

# Load config and create preprocessor
config = load_config()
preprocessor = ECGPreprocessor(config)

# Process all available records
print()
print("=" * 60)
print("Processing all records...")
print("=" * 60)

all_data = preprocessor.process_all_records(DATA_DIR, available_records)

print()
print(f"Total beats: {all_data['all_beats'].shape[0]}")
print(f"Beat window size: {all_data['all_beats'].shape[1]} samples")
print(f"Total distribution: {all_data['total_distribution']}")

# Patient-wise split
print()
print("=" * 60)
print("Performing patient-wise train/val/test split...")
print("=" * 60)

split_result = patient_wise_split(
    all_beats=all_data["all_beats"],
    all_labels=all_data["all_labels"],
    all_records=all_data["all_records"],
    all_r_peaks=all_data["all_r_peaks"],
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
)

# Save splits
print()
print("=" * 60)
print("Saving split data...")
print("=" * 60)

save_split_data(split_result, PROCESSED_DIR)

# Create dataloaders
print()
print("=" * 60)
print("Creating PyTorch DataLoaders...")
print("=" * 60)

train_loader, val_loader, test_loader = create_dataloaders(
    train_beats=split_result["splits"]["train"]["beats"],
    train_labels=split_result["splits"]["train"]["labels"],
    val_beats=split_result["splits"]["val"]["beats"],
    val_labels=split_result["splits"]["val"]["labels"],
    test_beats=split_result["splits"]["test"]["beats"],
    test_labels=split_result["splits"]["test"]["labels"],
    batch_size=64,
    balanced_sampling=True,
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Test a batch
batch_beats, batch_labels = next(iter(train_loader))
print(f"Batch shape: {batch_beats.shape}")
print(f"Batch labels shape: {batch_labels.shape}")

print()
print("=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)
print(f"Processed data saved to: {PROCESSED_DIR}")
print(f"Files: {[f.name for f in PROCESSED_DIR.iterdir() if f.is_file()]}")
