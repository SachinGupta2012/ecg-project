"""Quick test of the data pipeline on record 100."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import wfdb
from src.data.preprocessing import ECGPreprocessor, load_config, AAMI_CLASS_NAMES
from src.features.pan_tompkins import PanTompkins
from src.data.dataset import ECGBeatDataset

DATA_DIR = Path(r"D:\data_science\ecg-project\data\raw\mitdb")

print("=" * 60)
print("STEP 1: Load and preprocess record 100")
print("=" * 60)

config = load_config()
preprocessor = ECGPreprocessor(config)

result = preprocessor.process_record("100", DATA_DIR)

print("Beats extracted:", result["num_beats"])
print("Beat window size:", result["beats"].shape[1], "samples")
print("Beat array shape:", result["beats"].shape)

# Label distribution
unique, counts = np.unique(result["labels"], return_counts=True)
label_dist = {AAMI_CLASS_NAMES.get(k, "Class " + str(k)): int(v) for k, v in zip(unique, counts)}
print("Label distribution:", label_dist)

print()
print("=" * 60)
print("STEP 2: Test Pan-Tompkins on record 100")
print("=" * 60)

record = wfdb.rdrecord(str(DATA_DIR / "100"))
signal = record.p_signal[:, 0]

pan_tompkins = PanTompkins()
detected_peaks = pan_tompkins.detect_r_peaks(signal)
annotation = wfdb.rdann(str(DATA_DIR / "100"), "atr")
metrics = pan_tompkins.compare_with_annotations(signal, annotation.sample)

print("Detected R-peaks:", metrics["detected_count"])
print("Annotated R-peaks:", metrics["annotation_count"])
print("Precision:", round(metrics["precision"], 4))
print("Recall:", round(metrics["recall"], 4))
print("F1 Score:", round(metrics["f1"], 4))

print()
print("=" * 60)
print("STEP 3: Test PyTorch Dataset")
print("=" * 60)

dataset = ECGBeatDataset(result["beats"], result["labels"])
print("Dataset size:", len(dataset))
print("Sample shape:", dataset[0][0].shape)
print("Class weights:", dataset.get_class_weights())
print("Label distribution:", dataset.get_label_distribution())

print()
print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
