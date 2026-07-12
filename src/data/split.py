"""
Patient-Wise Train/Val/Test Split
==================================
Splits the MIT-BIH database by patient/recording, NOT by individual beats.
This prevents data leakage where the model sees beats from the same patient
in both training and test sets.

Critical: Randomly splitting beats inflates reported accuracy because
the model learns patient-specific patterns rather than generalizable
arrhythmia features.
"""

import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# MIT-BIH patient IDs for each record
# Records 100-124: patient IDs 100-124 (some patients have multiple records)
# Records 200-234: patient IDs 200-234
RECORD_TO_PATIENT = {
    "100": "101", "101": "101",
    "102": "102", "103": "103", "104": "104", "105": "105",
    "106": "106", "107": "107", "108": "108", "109": "109",
    "111": "111", "112": "112", "113": "113", "114": "114",
    "115": "115", "116": "116", "117": "117", "118": "118",
    "119": "119", "121": "121", "122": "122", "123": "123",
    "124": "124",
    "200": "200", "201": "201", "202": "202", "203": "203",
    "205": "205", "207": "207", "208": "208", "209": "209",
    "210": "210", "212": "212", "213": "213", "214": "214",
    "215": "215", "217": "217", "219": "219", "220": "220",
    "221": "221", "222": "222", "223": "223", "228": "228",
    "230": "230", "231": "231", "232": "232", "233": "233",
    "234": "234",
}


def get_patient_id(record_name: str) -> str:
    """Get the patient ID for a given record name."""
    return RECORD_TO_PATIENT.get(record_name, record_name)


def patient_wise_split(
    all_beats: np.ndarray,
    all_labels: np.ndarray,
    all_records: np.ndarray,
    all_r_peaks: np.ndarray | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by_patient: bool = True,
) -> dict:
    """
    Split data by patient/recording to prevent data leakage.

    Parameters
    ----------
    all_beats : np.ndarray
        Array of shape (N, window_samples) containing all beats.
    all_labels : np.ndarray
        Array of shape (N,) containing AAMI class labels.
    all_records : np.ndarray
        Array of shape (N,) containing record names for each beat.
    all_r_peaks : np.ndarray, optional
        Array of shape (N,) containing R-peak sample indices.
    train_ratio : float
        Fraction of patients for training.
    val_ratio : float
        Fraction of patients for validation.
    test_ratio : float
        Fraction of patients for testing.
    seed : int
        Random seed for reproducibility.
    stratify_by_patient : bool
        If True, ensures each patient's beats stay together.

    Returns
    -------
    dict
        {
            "train": {"beats", "labels", "records", "r_peaks", "patient_ids"},
            "val": {...},
            "test": {...},
            "patient_split": dict mapping patient_id to split name,
        }
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    rng = np.random.RandomState(seed)

    # Get unique patients and their record mappings
    unique_records = np.unique(all_records)
    patient_records = {}
    for record in unique_records:
        patient_id = get_patient_id(record)
        if patient_id not in patient_records:
            patient_records[patient_id] = []
        patient_records[patient_id].append(record)

    unique_patients = list(patient_records.keys())
    num_patients = len(unique_patients)

    logger.info(f"Total unique patients: {num_patients}")
    logger.info(f"Total records: {len(unique_records)}")
    logger.info(f"Total beats: {len(all_labels)}")

    # Shuffle patients
    rng.shuffle(unique_patients)

    # Split patients
    n_train = int(num_patients * train_ratio)
    n_val = int(num_patients * val_ratio)

    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:n_train + n_val]
    test_patients = unique_patients[n_train + n_val:]

    logger.info(f"Train patients: {len(train_patients)}")
    logger.info(f"Val patients: {len(val_patients)}")
    logger.info(f"Test patients: {len(test_patients)}")

    # Map each record to its split
    record_to_split = {}
    for patient_id in train_patients:
        for record in patient_records[patient_id]:
            record_to_split[record] = "train"
    for patient_id in val_patients:
        for record in patient_records[patient_id]:
            record_to_split[record] = "val"
    for patient_id in test_patients:
        for record in patient_records[patient_id]:
            record_to_split[record] = "test"

    # Create masks for each split
    train_mask = np.array([record_to_split.get(r, "train") == "train" for r in all_records])
    val_mask = np.array([record_to_split.get(r, "val") == "val" for r in all_records])
    test_mask = np.array([record_to_split.get(r, "test") == "test" for r in all_records])

    # Build split data
    splits = {}
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        split_beats = all_beats[mask]
        split_labels = all_labels[mask]
        split_records = all_records[mask]

        if all_r_peaks is not None:
            split_r_peaks = all_r_peaks[mask]
        else:
            split_r_peaks = None

        # Get unique patients in this split
        split_patient_ids = list(set(
            get_patient_id(r) for r in split_records
        ))

        # Label distribution for this split
        unique, counts = np.unique(split_labels, return_counts=True)
        label_dist = {int(k): int(v) for k, v in zip(unique, counts)}

        splits[split_name] = {
            "beats": split_beats,
            "labels": split_labels,
            "records": split_records,
            "r_peaks": split_r_peaks,
            "patient_ids": split_patient_ids,
            "num_beats": len(split_labels),
            "num_patients": len(split_patient_ids),
            "label_distribution": label_dist,
        }

        logger.info(
            f"{split_name}: {len(split_labels)} beats, "
            f"{len(split_patient_ids)} patients, "
            f"distribution: {label_dist}"
        )

    # Patient-to-split mapping
    patient_split = {}
    for patient_id in train_patients:
        patient_split[patient_id] = "train"
    for patient_id in val_patients:
        patient_split[patient_id] = "val"
    for patient_id in test_patients:
        patient_split[patient_id] = "test"

    return {
        "splits": splits,
        "patient_split": patient_split,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "test_patients": test_patients,
    }


def save_split_data(
    split_result: dict,
    output_dir: Path | str,
) -> None:
    """
    Save split data to disk as .npz files.

    Parameters
    ----------
    split_result : dict
        Output from patient_wise_split().
    output_dir : Path or str
        Directory to save the splits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_data = split_result["splits"][split_name]

        save_dict = {
            "beats": split_data["beats"],
            "labels": split_data["labels"],
            "records": split_data["records"],
        }

        if split_data["r_peaks"] is not None:
            save_dict["r_peaks"] = split_data["r_peaks"]

        np.savez_compressed(
            output_dir / f"{split_name}.npz",
            **save_dict,
        )

        logger.info(f"Saved {split_name} split to {output_dir / f'{split_name}.npz'}")

    # Save patient split mapping
    import json
    with open(output_dir / "patient_split.json", "w") as f:
        json.dump(split_result["patient_split"], f, indent=2)

    logger.info(f"Saved patient_split.json to {output_dir}")


def load_split_data(split_dir: Path | str) -> dict:
    """
    Load split data from disk.

    Parameters
    ----------
    split_dir : Path or str
        Directory containing the split .npz files.

    Returns
    -------
    dict
        Loaded split data.
    """
    split_dir = Path(split_dir)
    splits = {}

    for split_name in ["train", "val", "test"]:
        npz_path = split_dir / f"{split_name}.npz"
        if not npz_path.exists():
            logger.warning(f"Split file not found: {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True)
        splits[split_name] = {
            "beats": data["beats"],
            "labels": data["labels"],
            "records": data["records"],
            "r_peaks": data["r_peaks"] if "r_peaks" in data else None,
            "num_beats": len(data["labels"]),
        }

    return splits


def load_config() -> dict:
    """Load split configuration from config.yaml."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}
