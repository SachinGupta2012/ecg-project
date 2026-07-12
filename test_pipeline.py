"""Quick test script for the data pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wfdb
from src.data.download import download_mitdb, load_record, get_record_names
from src.data.preprocessing import ECGPreprocessor, load_config, AAMI_CLASS_NAMES

DATA_DIR = Path(r"D:\data_science\ecg-project\data\raw\mitdb")

print("=" * 60)
print("STEP 1: Download single record (100)")
print("=" * 60)

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download record 100 using wfdb.dl_record
try:
    wfdb.dl_record("100", pn_dir="mitdb", write_dir=str(DATA_DIR))
    print("Record 100 downloaded.")
except Exception as e:
    print(f"dl_record failed: {e}")
    print("Trying alternative download...")

    # Download the entire database
    try:
        download_mitdb(DATA_DIR)
        print("Full database downloaded.")
    except Exception as e2:
        print(f"Full download also failed: {e2}")
        print("Trying to load from PhysioNet directly...")
        record = wfdb.rdrecord("100", pn_dir="mitdb")
        ann = wfdb.rdann("100", "atr", pn_dir="mitdb")
        print(f"Record loaded from PhysioNet: {record.sig_len} samples")

# Check what files exist
print()
print("Files in data directory:")
for f in sorted(DATA_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size} bytes)")

print()
print("=" * 60)
print("STEP 2: Test preprocessing on record 100")
print("=" * 60)

# Try to load the record
try:
    record, annotation = load_record("100", DATA_DIR)
    print(f"Loaded record 100: {record.sig_len} samples, {record.n_sig} channels")
    print(f"Annotations: {len(annotation.sample)} beats")
    print(f"Sample annotation symbols: {annotation.symbol[:20]}")
except Exception as e:
    print(f"Could not load from local files: {e}")
    print("Loading directly from PhysioNet...")
    record = wfdb.rdrecord("100", pn_dir="mitdb")
    annotation = wfdb.rdann("100", "atr", pn_dir="mitdb")
    print(f"Loaded from PhysioNet: {record.sig_len} samples")

# Run preprocessing
config = load_config()
preprocessor = ECGPreprocessor(config)

signal = record.p_signal[:, 0]
filtered = preprocessor.bandpass_filter(signal)

print(f"Raw signal shape: {signal.shape}")
print(f"Filtered signal shape: {filtered.shape}")

# Extract beats
result = preprocessor.process_record("100", DATA_DIR)
print(f"Extracted {result['num_beats']} beats")
print(f"Beat window size: {result['beats'].shape[1]} samples")
print(f"Label distribution: {result['labels']}")

# Count labels
import numpy as np
unique, counts = np.unique(result['labels'], return_counts=True)
label_dist = {AAMI_CLASS_NAMES.get(k, f"Class {k}"): int(v) for k, v in zip(unique, counts)}
print(f"Label distribution: {label_dist}")

print()
print("=" * 60)
print("STEP 3: Test Pan-Tompkins on record 100")
print("=" * 60)

from src.features.pan_tompkins import PanTompkins

pan_tompkins = PanTompkins()
detected_peaks = pan_tompkins.detect_r_peaks(signal)
metrics = pan_tompkins.compare_with_annotations(signal, annotation.sample)

print(f"Detected R-peaks: {metrics['detected_count']}")
print(f"Annotated R-peaks: {metrics['annotation_count']}")
print(f"True Positives: {metrics['true_positives']}")
print(f"False Positives: {metrics['false_positives']}")
print(f"False Negatives: {metrics['false_negatives']}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")

print()
print("=" * 60)
print("STEP 4: Test PyTorch Dataset")
print("=" * 60)

from src.data.dataset import ECGBeatDataset

dataset = ECGBeatDataset(result['beats'], result['labels'])
print(f"Dataset size: {len(dataset)}")
print(f"Sample shape: {dataset[0][0].shape}")
print(f"Class weights: {dataset.get_class_weights()}")
print(f"Label distribution: {dataset.get_label_distribution()}")

print()
print("=" * 60)
print("ALL TESTS PASSED - Data Pipeline Working!")
print("=" * 60)
