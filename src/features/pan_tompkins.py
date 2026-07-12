"""
Pan-Tompkins QRS Detection Algorithm
======================================
Implementation of the Pan-Tompkins algorithm for R-peak detection in ECG signals.
This is a classic real-time QRS detection algorithm (Pan & Tompkins, 1985).

Steps:
1. Bandpass filter (5-11 Hz)
2. Derivative
3. Squaring
4. Moving window integration
5. Adaptive thresholding

Note: For MIT-BIH, we use the provided annotations for R-peak locations during
training. This implementation is useful for:
- Inference on unlabeled data
- Validating annotation quality
- Future real-time streaming
"""

import numpy as np
from scipy.signal import butter, filtfilt, windows
from dataclasses import dataclass


@dataclass
class PanTompkinsConfig:
    """Configuration for the Pan-Tompkins algorithm."""
    sampling_rate: int = 360
    low_pass_cutoff: float = 5.0       # Hz
    high_pass_cutoff: float = 11.0     # Hz
    filter_order: int = 1
    integration_window_ms: int = 150   # ms
    refractory_period_ms: int = 200    # ms
    threshold_adaptive: bool = True


class PanTompkins:
    """
    Pan-Tompkins QRS detection algorithm.

    Detects R-peaks in ECG signals using a pipeline of:
    bandpass filter → derivative → squaring → integration → adaptive thresholding.
    """

    def __init__(self, config: PanTompkinsConfig | None = None):
        if config is None:
            config = PanTompkinsConfig()
        self.config = config
        self.fs = config.sampling_rate

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Step 1: Bandpass filter (5-11 Hz) to isolate QRS complex energy.

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal.

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        nyquist = self.fs / 2.0
        low = self.config.low_pass_cutoff / nyquist
        high = self.config.high_pass_cutoff / nyquist

        b, a = butter(self.config.filter_order, [low, high], btype="band")
        return filtfilt(b, a, signal)

    def derivative(self, signal: np.ndarray) -> np.ndarray:
        """
        Step 2: Five-point derivative to highlight steep slopes (QRS complex).

        Parameters
        ----------
        signal : np.ndarray
            Bandpass-filtered signal.

        Returns
        -------
        np.ndarray
            Derivative of the signal.
        """
        # Five-point derivative: (1/8T) * [-x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2)]
        # Simplified to numpy gradient
        h = np.array([-1, -2, 0, 2, 1]) / 8.0
        return np.convolve(signal, h, mode="same")

    def squaring(self, signal: np.ndarray) -> np.ndarray:
        """
        Step 3: Square the signal to make all values positive and
        emphasize large deviations.

        Parameters
        ----------
        signal : np.ndarray
            Derivative signal.

        Returns
        -------
        np.ndarray
            Squared signal.
        """
        return signal ** 2

    def moving_window_integration(self, signal: np.ndarray) -> np.ndarray:
        """
        Step 4: Moving window integration to obtain waveform feature info.

        Parameters
        ----------
        signal : np.ndarray
            Squared signal.

        Returns
        -------
        np.ndarray
            Integrated signal.
        """
        window_size = int(
            self.config.integration_window_ms * self.fs / 1000.0
        )
        if window_size < 1:
            window_size = 1

        # Use a rectangular window (moving average)
        window = np.ones(window_size) / window_size
        integrated = np.convolve(signal, window, mode="same")
        return integrated

    def adaptive_threshold(
        self,
        signal: np.ndarray,
        initial_signal_peak: float | None = None,
        initial_noise_peak: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Step 5: Adaptive thresholding to detect QRS complexes.

        Parameters
        ----------
        signal : np.ndarray
            Integrated signal.
        initial_signal_peak : float, optional
            Initial threshold for signal peaks.
        initial_noise_peak : float, optional
            Initial threshold for noise peaks.

        Returns
        -------
        tuple
            (r_peak_indices, thresholds_used)
        """
        # Initialize thresholds
        if initial_signal_peak is None:
            # Use the first 2 seconds to estimate initial thresholds
            init_samples = int(2 * self.fs)
            init_segment = signal[:init_samples]
            signal_peak = np.max(init_segment) * 0.25
        else:
            signal_peak = initial_signal_peak

        if initial_noise_peak is None:
            noise_peak = signal_peak * 0.1
        else:
            noise_peak = initial_noise_peak

        # Refractory period (200ms)
        refractory_samples = int(
            self.config.refractory_period_ms * self.fs / 1000.0
        )

        # Thresholds
        threshold = signal_peak
        signal_peak_rolling = signal_peak
        noise_peak_rolling = noise_peak

        r_peaks = []
        thresholds = []

        i = 0
        while i < len(signal):
            # Find local maximum within a search window
            search_window = int(0.15 * self.fs)  # 150ms search back
            start = max(0, i - search_window)
            peak_idx = start + np.argmax(signal[start:i + 1])
            peak_val = signal[peak_idx]

            if peak_val > threshold:
                # Check refractory period
                if len(r_peaks) == 0 or (peak_idx - r_peaks[-1]) > refractory_samples:
                    r_peaks.append(peak_idx)

                    # Update thresholds
                    if self.config.threshold_adaptive:
                        signal_peak_rolling = 0.125 * peak_val + 0.875 * signal_peak_rolling
                        threshold = noise_peak_rolling + 0.25 * (
                            signal_peak_rolling - noise_peak_rolling
                        )

                    thresholds.append(threshold)
                    i = peak_idx + refractory_samples
                    continue
            else:
                # Update noise peak
                if self.config.threshold_adaptive:
                    noise_peak_rolling = 0.125 * peak_val + 0.875 * noise_peak_rolling
                    threshold = noise_peak_rolling + 0.25 * (
                        signal_peak_rolling - noise_peak_rolling
                    )
                    thresholds.append(threshold)

            i += 1

        return np.array(r_peaks), np.array(thresholds)

    def detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Full Pan-Tompkins pipeline: detect R-peaks in an ECG signal.

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal (1D).

        Returns
        -------
        np.ndarray
            Array of R-peak sample indices.
        """
        # Step 1: Bandpass filter
        filtered = self.bandpass_filter(signal)

        # Step 2: Derivative
        differentiated = self.derivative(filtered)

        # Step 3: Squaring
        squared = self.squaring(differentiated)

        # Step 4: Moving window integration
        integrated = self.moving_window_integration(squared)

        # Step 5: Adaptive thresholding
        r_peaks, thresholds = self.adaptive_threshold(integrated)

        return r_peaks

    def detect_and_filter(
        self, signal: np.ndarray, min_distance_ms: float = 200.0
    ) -> np.ndarray:
        """
        Detect R-peaks and apply additional filtering.

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal.
        min_distance_ms : float
            Minimum distance between consecutive R-peaks in ms.

        Returns
        -------
        np.ndarray
            Filtered R-peak indices.
        """
        r_peaks = self.detect_r_peaks(signal)

        if len(r_peaks) < 2:
            return r_peaks

        # Apply minimum distance filter
        min_distance_samples = int(min_distance_ms * self.fs / 1000.0)
        filtered_peaks = [r_peaks[0]]

        for peak in r_peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance_samples:
                filtered_peaks.append(peak)

        return np.array(filtered_peaks)

    def compare_with_annotations(
        self,
        signal: np.ndarray,
        annotation_samples: np.ndarray,
        tolerance_ms: float = 75.0,
    ) -> dict:
        """
        Compare detected R-peaks with annotated R-peaks.

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal.
        annotation_samples : np.ndarray
            Annotated R-peak sample indices.
        tolerance_ms : float
            Tolerance for matching in ms.

        Returns
        -------
        dict
            Comparison metrics.
        """
        detected = self.detect_r_peaks(signal)
        tolerance_samples = int(tolerance_ms * self.fs / 1000.0)

        # Match detected peaks to annotations
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        matched_annotations = set()
        matched_detections = set()

        for det_idx, det_peak in enumerate(detected):
            for ann_idx, ann_peak in enumerate(annotation_samples):
                if ann_idx in matched_annotations:
                    continue
                if abs(det_peak - ann_peak) <= tolerance_samples:
                    true_positives += 1
                    matched_annotations.add(ann_idx)
                    matched_detections.add(det_idx)
                    break

        false_positives = len(detected) - len(matched_detections)
        false_negatives = len(annotation_samples) - len(matched_annotations)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "detected_count": len(detected),
            "annotation_count": len(annotation_samples),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
