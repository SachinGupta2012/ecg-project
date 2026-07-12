"""
ECG Signal Preprocessing Pipeline
===================================
Handles noise removal, beat segmentation, normalization, and label mapping
for the MIT-BIH Arrhythmia Database.

This module is the single source of truth for preprocessing — used by both
training and inference to prevent preprocessing mismatch.

Usage:
    from src.data.preprocessing import ECGPreprocessor
    preprocessor = ECGPreprocessor()
    segments, labels = preprocessor.process_record(record_name)
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, iirnotch

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration loaded from config.yaml."""
    sampling_rate: int = 360
    window_samples: int = 288          # ~0.8 seconds at 360 Hz
    low_cutoff: float = 0.5            # Hz - bandpass low cutoff
    high_cutoff: float = 40.0          # Hz - bandpass high cutoff
    filter_order: int = 4              # Butterworth filter order
    normalization: str = "z-score"     # z-score per beat


# AAMI annotation mapping
# Maps MIT-BIH annotation symbols to 5 AAMI superclasses
AAMI_MAPPING = {
    "N": 0, "L": 0, "R": 0, "B": 0,    # Normal
    "A": 1, "a": 1, "J": 1, "S": 1,    # Supraventricular ectopic
    "V": 2, "r": 2, "E": 2,             # Ventricular ectopic
    "F": 3,                              # Fusion
    "Q": 4, "/": 4, "f": 4, "x": 4, "?": 4,  # Unknown/paced
}

AAMI_CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
AAMI_CLASS_FULL_NAMES = {
    0: "Normal",
    1: "Supraventricular Ectopic",
    2: "Ventricular Ectopic",
    3: "Fusion",
    4: "Unknown/Paced",
}


class ECGPreprocessor:
    """
    End-to-end ECG preprocessing pipeline.

    Steps:
    1. Bandpass filter the raw signal (remove baseline wander + high-freq noise)
    2. Extract fixed-size windows around each annotated R-peak
    3. Normalize each beat segment (z-score)
    4. Map annotation symbols to AAMI superclass labels
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        if config is None:
            config = PreprocessingConfig()
        self.config = config
        self._filter_coeffs = None

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter to the ECG signal.

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal (1D array).

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        fs = self.config.sampling_rate
        low = self.config.low_cutoff
        high = self.config.high_cutoff
        order = self.config.filter_order

        nyquist = fs / 2.0
        low_norm = low / nyquist
        high_norm = high / nyquist

        b, a = butter(order, [low_norm, high_norm], btype="band")
        filtered = filtfilt(b, a, signal)

        return filtered

    def normalize_beat(self, beat: np.ndarray) -> np.ndarray:
        """
        Z-score normalize a single beat segment.

        Parameters
        ----------
        beat : np.ndarray
            Single beat segment.

        Returns
        -------
        np.ndarray
            Normalized beat.
        """
        if self.config.normalization == "z-score":
            mean = np.mean(beat)
            std = np.std(beat)
            if std < 1e-8:
                return beat - mean
            return (beat - mean) / std
        elif self.config.normalization == "min-max":
            min_val = np.min(beat)
            max_val = np.max(beat)
            if max_val - min_val < 1e-8:
                return np.zeros_like(beat)
            return (beat - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization: {self.config.normalization}")

    def extract_beat(
        self,
        signal: np.ndarray,
        r_peak_idx: int,
        window_samples: int | None = None,
    ) -> np.ndarray | None:
        """
        Extract a fixed-size window centered on an R-peak.

        Parameters
        ----------
        signal : np.ndarray
            Filtered ECG signal.
        r_peak_idx : int
            Index of the R-peak in the signal.
        window_samples : int, optional
            Number of samples per beat window.

        Returns
        -------
        np.ndarray or None
            Beat segment, or None if window goes out of bounds.
        """
        if window_samples is None:
            window_samples = self.config.window_samples

        half_window = window_samples // 2
        start = r_peak_idx - half_window
        end = r_peak_idx + half_window

        if start < 0 or end > len(signal):
            return None

        beat = signal[start:end]
        return beat

    def map_annotation(self, symbol: str) -> int:
        """
        Map a MIT-BIH annotation symbol to an AAMI superclass label.

        Parameters
        ----------
        symbol : str
            Single annotation symbol.

        Returns
        -------
        int
            AAMI class index (0-4).
        """
        return AAMI_MAPPING.get(symbol, 4)  # Default to unknown

    def process_record(
        self,
        record_name: str,
        data_dir: Path | str,
        return_raw_signal: bool = False,
    ) -> dict:
        """
        Full preprocessing pipeline for a single record.

        Parameters
        ----------
        record_name : str
            Record number (e.g., "100").
        data_dir : Path or str
            Directory containing the .dat/.hea/.atr files.
        return_raw_signal : bool
            If True, also return the raw filtered signal.

        Returns
        -------
        dict
            {
                "record_name": str,
                "beats": np.ndarray,        # (N, window_samples) array of beats
                "labels": np.ndarray,        # (N,) array of AAMI class indices
                "r_peak_indices": np.ndarray, # (N,) array of R-peak sample indices
                "symbols": list[str],         # Original annotation symbols
                "sampling_rate": int,
                "num_beats": int,
                "signal": np.ndarray,         # Full filtered signal (optional)
            }
        """
        data_dir = Path(data_dir)
        record_path = data_dir / record_name

        # Load record and annotations
        record = wfdb.rdrecord(str(record_path))
        annotation = wfdb.rdann(str(record_path), "atr")

        # Get the signal (use first channel - Lead II)
        signal = record.p_signal[:, 0]

        # Apply bandpass filter
        filtered_signal = self.bandpass_filter(signal)

        # Extract beats and labels
        beats = []
        labels = []
        r_peak_indices = []
        symbols = []

        for i, (sample_idx, symbol) in enumerate(
            zip(annotation.sample, annotation.symbol)
        ):
            beat = self.extract_beat(filtered_signal, sample_idx)
            if beat is None:
                continue

            beat = self.normalize_beat(beat)
            label = self.map_annotation(symbol)

            beats.append(beat)
            labels.append(label)
            r_peak_indices.append(sample_idx)
            symbols.append(symbol)

        beats = np.array(beats)
        labels = np.array(labels)
        r_peak_indices = np.array(r_peak_indices)

        result = {
            "record_name": record_name,
            "beats": beats,
            "labels": labels,
            "r_peak_indices": r_peak_indices,
            "symbols": symbols,
            "sampling_rate": self.config.sampling_rate,
            "num_beats": len(beats),
        }

        if return_raw_signal:
            result["signal"] = filtered_signal

        logger.info(
            f"Record {record_name}: {len(beats)} beats extracted, "
            f"labels: {dict(zip(*np.unique(labels, return_counts=True)))}"
        )

        return result

    def process_all_records(
        self,
        data_dir: Path | str,
        record_names: list[str] | None = None,
    ) -> dict:
        """
        Process all records in the database.

        Parameters
        ----------
        data_dir : Path or str
            Directory containing the MIT-BIH data.
        record_names : list of str, optional
            Specific records to process. If None, processes all.

        Returns
        -------
        dict
            {
                "all_beats": np.ndarray,      # (Total, window_samples)
                "all_labels": np.ndarray,      # (Total,)
                "all_records": list[str],      # Record name per beat
                "all_r_peaks": np.ndarray,     # R-peak indices per beat
                "record_info": dict,           # Per-record metadata
            }
        """
        data_dir = Path(data_dir)

        if record_names is None:
            record_names = sorted([f.stem for f in data_dir.glob("*.hea")])

        logger.info(f"Processing {len(record_names)} records...")

        all_beats = []
        all_labels = []
        all_records = []
        all_r_peaks = []
        record_info = {}

        for record_name in record_names:
            try:
                result = self.process_record(record_name, data_dir)
            except Exception as e:
                logger.warning(f"Skipping record {record_name}: {e}")
                continue

            all_beats.append(result["beats"])
            all_labels.append(result["labels"])
            all_records.extend([record_name] * result["num_beats"])
            all_r_peaks.append(result["r_peak_indices"])

            record_info[record_name] = {
                "num_beats": result["num_beats"],
                "label_distribution": {
                    AAMI_CLASS_NAMES[k]: int(v)
                    for k, v in zip(*np.unique(result["labels"], return_counts=True))
                },
            }

        all_beats = np.concatenate(all_beats, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_r_peaks = np.concatenate(all_r_peaks, axis=0)

        # Final label distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        total_dist = {AAMI_CLASS_NAMES[k]: int(v) for k, v in zip(unique, counts)}

        logger.info(f"Total beats: {len(all_labels)}")
        logger.info(f"Label distribution: {total_dist}")

        return {
            "all_beats": all_beats,
            "all_labels": all_labels,
            "all_records": np.array(all_records),
            "all_r_peaks": all_r_peaks,
            "record_info": record_info,
            "total_distribution": total_dist,
        }


def load_config() -> PreprocessingConfig:
    """Load preprocessing config from config.yaml."""
    import yaml

    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        preprocessing = cfg.get("preprocessing", {})
        dataset = cfg.get("dataset", {})

        return PreprocessingConfig(
            sampling_rate=dataset.get("sampling_rate", 360),
            window_samples=dataset.get("window_samples", 288),
            low_cutoff=preprocessing.get("low_cutoff", 0.5),
            high_cutoff=preprocessing.get("high_cutoff", 40.0),
            filter_order=preprocessing.get("filter_order", 4),
            normalization=preprocessing.get("normalization", "z-score"),
        )

    return PreprocessingConfig()
