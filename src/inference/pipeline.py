"""
Full Inference Pipeline for ECG Arrhythmia Detection
=====================================================
Complete pipeline from raw ECG signal to beat-by-beat predictions.
"""

import logging
import time
from pathlib import Path

import numpy as np
import wfdb

from src.data.preprocessing import ECGPreprocessor, load_config
from src.inference.predict import AAMI_CLASSES, get_classifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ECGAnalysisPipeline:
    """
    Complete ECG analysis pipeline.

    Steps:
    1. Load ECG signal (from file or MIT-BIH record)
    2. Preprocess (filter, segment, normalize)
    3. Run model inference
    4. Generate summary and flagged segments
    """

    def __init__(self, model_name: str = "cnn_baseline"):
        """
        Parameters
        ----------
        model_name : str
            Model to use for inference.
        """
        self.model_name = model_name
        self.config = load_config()
        self.preprocessor = ECGPreprocessor(self.config)
        self.classifier = get_classifier(model_name)

    def load_mitdb_record(
        self, record_name: str, data_dir: Path | str | None = None
    ) -> tuple[np.ndarray, int, wfdb.Annotation]:
        """
        Load a record from the MIT-BIH database.

        Parameters
        ----------
        record_name : str
            Record name (e.g., "100").
        data_dir : Path or str, optional
            Directory containing the data.

        Returns
        -------
        tuple
            (signal, sampling_rate, annotation)
        """
        if data_dir is None:
            data_dir = PROJECT_ROOT / "data" / "raw" / "mitdb"
        else:
            data_dir = Path(data_dir)

        record_path = data_dir / record_name

        # Try local first, then remote
        if record_path.with_suffix(".hea").exists():
            record = wfdb.rdrecord(str(record_path))
            annotation = wfdb.rdann(str(record_path), "atr")
        else:
            logger.info("Loading record %s from PhysioNet...", record_name)
            record = wfdb.rdrecord(record_name, pn_dir="mitdb")
            annotation = wfdb.rdann(record_name, "atr", pn_dir="mitdb")

        signal = record.p_signal[:, 0]  # Lead II
        fs = record.fs

        return signal, fs, annotation

    def analyze(
        self,
        signal: np.ndarray,
        sampling_rate: int = 360,
        record_name: str | None = None,
        original_annotations: wfdb.Annotation | None = None,
    ) -> dict:
        """
        Run full analysis on an ECG signal.

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal (1D array).
        sampling_rate : int
            Sampling rate in Hz.
        record_name : str, optional
            Record name for reference.
        original_annotations : wfdb.Annotation, optional
            Original annotations for comparison.

        Returns
        -------
        dict
            Complete analysis results.
        """
        start_time = time.time()

        # Step 1: Preprocess
        logger.info("Preprocessing signal...")
        filtered = self.preprocessor.bandpass_filter(signal)

        # Step 2: Extract beats using annotations or Pan-Tompkins
        if original_annotations is not None:
            r_peaks = original_annotations.sample
            symbols = original_annotations.symbol
        else:
            # Use Pan-Tompkins for R-peak detection
            from src.features.pan_tompkins import PanTompkins
            pan_tompkins = PanTompkins()
            r_peaks = pan_tompkins.detect_r_peaks(signal)
            symbols = ["N"] * len(r_peaks)  # Unknown annotations

        # Step 3: Segment and normalize beats
        beats = []
        valid_indices = []
        valid_symbols = []

        for i, (r_peak, symbol) in enumerate(zip(r_peaks, symbols)):
            beat = self.preprocessor.extract_beat(filtered, r_peak)
            if beat is not None:
                beat = self.preprocessor.normalize_beat(beat)
                beats.append(beat)
                valid_indices.append(r_peak)
                valid_symbols.append(symbol)

        if len(beats) == 0:
            logger.warning("No valid beats extracted")
            return {"error": "No valid beats found"}

        beats_array = np.array(beats)
        beats_tensor = beats_array.reshape(len(beats), 1, -1)

        # Step 4: Run inference
        logger.info("Running inference on %d beats...", len(beats))
        predictions, probabilities, confidences = self.classifier.predict(beats_tensor)

        # Step 5: Generate beat-level results
        beat_results = []
        for i in range(len(beats)):
            timestamp = valid_indices[i] / sampling_rate
            beat_results.append({
                "beat_index": i,
                "sample_index": int(valid_indices[i]),
                "timestamp_sec": round(timestamp, 3),
                "predicted_class": AAMI_CLASSES[int(predictions[i])],
                "confidence": round(float(confidences[i]), 4),
                "probabilities": {
                    AAMI_CLASSES[j]: round(float(probabilities[i, j]), 4)
                    for j in range(5)
                },
                "original_symbol": valid_symbols[i] if i < len(valid_symbols) else None,
            })

        # Step 6: Generate summary
        predictions_array = predictions
        unique, counts = np.unique(predictions_array, return_counts=True)
        total = len(predictions_array)

        class_dist = []
        normal_count = 0
        for cls_idx, count in zip(unique, counts):
            class_name = AAMI_CLASSES[int(cls_idx)]
            percentage = round(count / total * 100, 1)
            class_dist.append({
                "class_name": class_name,
                "count": int(count),
                "percentage": percentage,
            })
            if class_name == "N":
                normal_count = int(count)

        # Step 7: Find abnormal segments
        abnormal_segments = self._find_abnormal_segments(beat_results, sampling_rate)

        # Step 8: Overall confidence and flag
        avg_confidence = float(np.mean(confidences))
        abnormal_count = total - normal_count
        flagged = abnormal_count > 10 or any(
            seg["avg_confidence"] > 0.9 for seg in abnormal_segments
        )

        processing_time = time.time() - start_time

        return {
            "record_name": record_name,
            "total_beats": total,
            "normal_beats": normal_count,
            "abnormal_beats": abnormal_count,
            "class_distribution": class_dist,
            "abnormal_segments": abnormal_segments,
            "overall_confidence": round(avg_confidence, 4),
            "flagged_for_review": flagged,
            "beat_predictions": beat_results,
            "processing_time_sec": round(processing_time, 2),
            "model_used": self.model_name,
        }

    def _find_abnormal_segments(
        self,
        beat_results: list[dict],
        sampling_rate: int,
        min_abnormal_beats: int = 3,
        time_window_sec: float = 10.0,
    ) -> list[dict]:
        """
        Find segments with clustered abnormal beats.

        Parameters
        ----------
        beat_results : list of dict
            Beat-level predictions.
        sampling_rate : int
            Sampling rate.
        min_abnormal_beats : int
            Minimum abnormal beats to form a segment.
        time_window_sec : float
            Time window for clustering.

        Returns
        -------
        list of dict
            Abnormal segments.
        """
        abnormal_beats = [
            (i, b) for i, b in enumerate(beat_results)
            if b["predicted_class"] != "N"
        ]

        if len(abnormal_beats) < min_abnormal_beats:
            return []

        # Cluster by time
        segments = []
        current_segment = [abnormal_beats[0]]

        for i in range(1, len(abnormal_beats)):
            prev_time = abnormal_beats[i-1][1]["timestamp_sec"]
            curr_time = abnormal_beats[i][1]["timestamp_sec"]

            if curr_time - prev_time <= time_window_sec:
                current_segment.append(abnormal_beats[i])
            else:
                if len(current_segment) >= min_abnormal_beats:
                    segments.append(current_segment)
                current_segment = [abnormal_beats[i]]

        if len(current_segment) >= min_abnormal_beats:
            segments.append(current_segment)

        # Format segments
        result_segments = []
        for segment in segments:
            beat_indices = [b[0] for b in segment]
            classes = [b[1]["predicted_class"] for b in segment]
            confidences = [b[1]["confidence"] for b in segment]

            # Dominant class
            from collections import Counter
            dominant_class = Counter(classes).most_common(1)[0][0]

            result_segments.append({
                "start_time_sec": round(segment[0][1]["timestamp_sec"], 3),
                "end_time_sec": round(segment[-1][1]["timestamp_sec"], 3),
                "duration_sec": round(
                    segment[-1][1]["timestamp_sec"] - segment[0][1]["timestamp_sec"], 3
                ),
                "abnormal_beat_indices": beat_indices,
                "dominant_class": dominant_class,
                "avg_confidence": round(float(np.mean(confidences)), 4),
                "num_beats": len(beat_indices),
            })

        return result_segments

    def analyze_mitdb_record(
        self, record_name: str, data_dir: Path | str | None = None
    ) -> dict:
        """
        Analyze a record from the MIT-BIH database.

        Parameters
        ----------
        record_name : str
            Record name (e.g., "100").
        data_dir : Path or str, optional
            Directory containing the data.

        Returns
        -------
        dict
            Complete analysis results.
        """
        signal, fs, annotation = self.load_mitdb_record(record_name, data_dir)
        return self.analyze(signal, fs, record_name, annotation)
