"""
MIT-BIH Arrhythmia Database Download Script
=============================================
Downloads the full MIT-BIH database (48 records) from PhysioNet
using the wfdb Python package.

Usage:
    python -m src.data.download
"""

import logging
from pathlib import Path

import wfdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# MIT-BIH record names (48 records)
MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def download_mitdb(dl_dir: Path | str | None = None) -> Path:
    """
    Download the MIT-BIH Arrhythmia Database.

    Parameters
    ----------
    dl_dir : Path or str, optional
        Directory to download data into. Defaults to data/raw/mitdb.

    Returns
    -------
    Path
        Path to the downloaded data directory.
    """
    if dl_dir is None:
        dl_dir = DATA_RAW_DIR / "mitdb"
    else:
        dl_dir = Path(dl_dir)

    dl_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    sample_file = dl_dir / "100.hea"
    if sample_file.exists():
        logger.info(f"MIT-BIH database already exists at {dl_dir}. Skipping download.")
        return dl_dir

    logger.info(f"Downloading MIT-BIH Arrhythmia Database to {dl_dir}...")
    logger.info("This may take a few minutes depending on your internet connection.")

    try:
        wfdb.dl_database("mitdb", dl_dir=str(dl_dir))
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Error downloading database: {e}")
        raise

    return dl_dir


def download_single_record(record_name: str, dl_dir: Path | str | None = None) -> Path:
    """
    Download a single record from MIT-BIH.

    Parameters
    ----------
    record_name : str
        Record number (e.g., "100").
    dl_dir : Path or str, optional
        Directory to download into.

    Returns
    -------
    Path
        Path to the downloaded record.
    """
    if dl_dir is None:
        dl_dir = DATA_RAW_DIR / "mitdb"
    else:
        dl_dir = Path(dl_dir)

    dl_dir.mkdir(parents=True, exist_ok=True)

    record_path = dl_dir / record_name
    if record_path.with_suffix(".hea").exists():
        logger.info(f"Record {record_name} already exists. Skipping.")
        return record_path

    logger.info(f"Downloading record {record_name}...")
    try:
        record = wfdb.rdrecord(record_name, pn_dir="mitdb")
        wfdb.plot.record(record, write_dir=str(dl_dir), ann_sym=["atr"])
        # Also save the record to disk
        wfdb.writedatas(record, filesigs=[str(dl_dir / record_name)])
        logger.info(f"Record {record_name} downloaded.")
    except Exception as e:
        logger.error(f"Error downloading record {record_name}: {e}")
        raise

    return record_path


def load_record(record_name: str, data_dir: Path | str | None = None) -> tuple:
    """
    Load a single record and its annotations.

    Parameters
    ----------
    record_name : str
        Record number (e.g., "100").
    data_dir : Path or str, optional
        Directory containing the data.

    Returns
    -------
    tuple
        (wfdb.Record, wfdb.Annotation)
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR / "mitdb"
    else:
        data_dir = Path(data_dir)

    record_path = data_dir / record_name
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    return record, annotation


def get_record_names(data_dir: Path | str | None = None) -> list[str]:
    """
    Get all available record names from the data directory.

    Parameters
    ----------
    data_dir : Path or str, optional
        Directory containing the data.

    Returns
    -------
    list of str
        List of record names.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR / "mitdb"
    else:
        data_dir = Path(data_dir)

    records = sorted([
        f.stem for f in data_dir.glob("*.hea")
    ])
    return records


def list_available_records(data_dir: Path | str | None = None) -> list[str]:
    """
    List all available records and their annotation counts.

    Parameters
    ----------
    data_dir : Path or str, optional
        Directory containing the data.

    Returns
    -------
    list of dict
        List of dicts with record info.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR / "mitdb"
    else:
        data_dir = Path(data_dir)

    records_info = []
    for record_name in get_record_names(data_dir):
        try:
            record, annotation = load_record(record_name, data_dir)
            records_info.append({
                "record": record_name,
                "num_beats": len(annotation.sample),
                "duration_sec": record.sig_len / record.fs,
                "sampling_rate": record.fs,
                "channels": record.n_sig,
            })
        except Exception as e:
            logger.warning(f"Could not load record {record_name}: {e}")
            continue

    return records_info


if __name__ == "__main__":
    # Download the full database
    data_dir = download_mitdb()
    logger.info(f"Data saved to: {data_dir}")

    # List available records
    records = get_record_names(data_dir)
    logger.info(f"Total records available: {len(records)}")
    logger.info(f"Records: {records}")
