"""
Full Data Pipeline Runner
==========================
Downloads, preprocesses, splits, and saves the MIT-BIH dataset.

Usage:
    python -m src.data.pipeline
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_mitdb, get_record_names, load_record
from src.data.preprocessing import ECGPreprocessor, load_config
from src.data.split import patient_wise_split, save_split_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Execute the full data pipeline."""
    config = load_config()
    preprocessor = ECGPreprocessor(config)

    data_dir = PROJECT_ROOT / "data" / "raw" / "mitdb"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # Step 1: Download
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading MIT-BIH database")
    logger.info("=" * 60)
    download_mitdb(data_dir)

    # Step 2: Preprocess all records
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing all records")
    logger.info("=" * 60)
    all_data = preprocessor.process_all_records(data_dir)

    logger.info(f"Total beats extracted: {all_data['all_beats'].shape[0]}")
    logger.info(f"Beat window size: {all_data['all_beats'].shape[1]} samples")
    logger.info(f"Label distribution: {all_data['total_distribution']}")

    # Step 3: Patient-wise split
    logger.info("=" * 60)
    logger.info("STEP 3: Patient-wise train/val/test split")
    logger.info("=" * 60)
    split_result = patient_wise_split(
        all_beats=all_data["all_beats"],
        all_labels=all_data["all_labels"],
        all_records=all_data["all_records"],
        all_r_peaks=all_data["all_r_peaks"],
        train_ratio=config.get("dataset", {}).get("train_ratio", 0.7),
        val_ratio=config.get("dataset", {}).get("val_ratio", 0.15),
        test_ratio=config.get("dataset", {}).get("test_ratio", 0.15),
        seed=config.get("training", {}).get("seed", 42),
    )

    # Step 4: Save splits
    logger.info("=" * 60)
    logger.info("STEP 4: Saving split data")
    logger.info("=" * 60)
    save_split_data(split_result, processed_dir)

    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Processed directory: {processed_dir}")

    for split_name in ["train", "val", "test"]:
        split = split_result["splits"][split_name]
        logger.info(
            f"{split_name.upper()}: {split['num_beats']} beats, "
            f"{split['num_patients']} patients"
        )


if __name__ == "__main__":
    run_pipeline()
