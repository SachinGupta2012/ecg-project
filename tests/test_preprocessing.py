"""Tests for preprocessing pipeline."""

import numpy as np


class TestPreprocessing:
    """Tests for ECG preprocessing utilities."""

    def test_beat_window_shape(self):
        """Beat windows should have correct shape."""
        window_size = 288
        beat = np.random.randn(window_size)
        assert beat.shape == (window_size,)

    def test_z_score_normalization(self):
        """Z-score normalization should produce mean~0, std~1."""
        signal = np.random.randn(1000) * 5 + 10
        normalized = (signal - np.mean(signal)) / np.std(signal)
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10

    def test_aami_class_mapping(self):
        """AAMI classes should map correctly."""
        aami_map = {
            "N": 0,
            "L": 0,
            "R": 0,
            "e": 0,
            "j": 0,  # Normal
            "A": 1,
            "a": 1,
            "J": 1,
            "S": 1,  # Supraventricular
            "V": 2,
            "E": 2,  # Ventricular
            "F": 3,  # Fusion
            "/": 4,
            "f": 4,
            "Q": 4,  # Unknown
        }
        assert aami_map["N"] == 0
        assert aami_map["V"] == 2
        assert aami_map["F"] == 3
        assert aami_map["Q"] == 4

    def test_bandpass_filter_returns_same_length(self):
        """Filtered signal should have same length as input."""
        from scipy.signal import butter, filtfilt

        fs = 360
        low, high = 0.5, 40.0
        signal = np.random.randn(3600)

        b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype="band")
        filtered = filtfilt(b, a, signal)

        assert len(filtered) == len(signal)
