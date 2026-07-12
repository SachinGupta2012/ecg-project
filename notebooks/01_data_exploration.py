"""
ECG Data Exploration & Visualization
======================================
Exploratory Data Analysis for the MIT-BIH Arrhythmia Database.

Run this script to:
1. Download and explore the raw MIT-BIH data
2. Visualize raw vs filtered ECG signals
3. Show beat segmentation examples
4. Display class distribution
5. Compare annotation types

Usage:
    python -m notebooks.01_data_exploration
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import butter, filtfilt

from src.data.download import download_mitdb, load_record, get_record_names
from src.data.preprocessing import (
    ECGPreprocessor,
    load_config,
    AAMI_CLASS_NAMES,
    AAMI_CLASS_FULL_NAMES,
)
from src.features.pan_tompkins import PanTompkins

# Style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mitdb"


def plot_raw_vs_filtered() -> None:
    """Compare raw and bandpass-filtered ECG signals."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Load record 100
    record, annotation = load_record("100", DATA_DIR)
    signal = record.p_signal[:, 0]  # Lead II
    fs = record.fs
    time = np.arange(len(signal)) / fs

    # Raw signal
    axes[0].plot(time, signal, linewidth=0.5, color="steelblue")
    axes[0].set_title("Raw ECG Signal (Record 100, Lead II)", fontsize=12)
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_xlim(0, 10)  # Show first 10 seconds

    # Filtered signal
    preprocessor = ECGPreprocessor()
    filtered = preprocessor.bandpass_filter(signal)

    axes[1].plot(time, filtered, linewidth=0.5, color="darkorange")
    axes[1].set_title("After Bandpass Filter (0.5-40 Hz)", fontsize=12)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "notebooks" / "raw_vs_filtered.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: notebooks/raw_vs_filtered.png")


def plot_beat_segmentation() -> None:
    """Show how individual beats are segmented around R-peaks."""
    record, annotation = load_record("100", DATA_DIR)
    signal = record.p_signal[:, 0]
    fs = record.fs

    preprocessor = ECGPreprocessor()
    filtered = preprocessor.bandpass_filter(signal)

    # Get first 10 R-peaks
    r_peaks = annotation.sample[:10]
    window = preprocessor.config.window_samples
    half_window = window // 2

    fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharey=True)
    axes = axes.flatten()

    for i, r_peak in enumerate(r_peaks):
        start = r_peak - half_window
        end = r_peak + half_window

        if start < 0 or end > len(filtered):
            continue

        beat = filtered[start:end]
        beat_normalized = preprocessor.normalize_beat(beat)
        time_ms = np.arange(-half_window, half_window) / fs * 1000

        axes[i].plot(time_ms, beat_normalized, linewidth=1.5, color="steelblue")
        axes[i].axvline(x=0, color="red", linestyle="--", alpha=0.7, label="R-peak")
        axes[i].set_title(f"Beat {i+1}\n({annotation.symbol[i]})", fontsize=10)
        axes[i].set_xlabel("Time (ms)")
        if i % 5 == 0:
            axes[i].set_ylabel("Normalized Amplitude")
        axes[i].legend(fontsize=8)

    plt.suptitle("Segmented Beat Windows (Centered on R-peaks)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "notebooks" / "beat_segmentation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: notebooks/beat_segmentation.png")


def plot_beat_classes() -> None:
    """Show example beats from each AAMI class."""
    # Load multiple records to find examples of each class
    class_examples = {0: [], 1: [], 2: [], 3: [], 4: []}

    for record_name in ["100", "105", "106", "109", "114", "119", "200", "201", "207", "208", "213", "214", "217", "220", "222", "228", "233"]:
        try:
            record, annotation = load_record(record_name, DATA_DIR)
        except Exception:
            continue

        signal = record.p_signal[:, 0]
        preprocessor = ECGPreprocessor()
        filtered = preprocessor.bandpass_filter(signal)

        for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
            label = preprocessor.map_annotation(symbol)
            if label in class_examples and len(class_examples[label]) < 3:
                beat = preprocessor.extract_beat(filtered, sample_idx)
                if beat is not None:
                    beat = preprocessor.normalize_beat(beat)
                    class_examples[label].append((beat, symbol, record_name))

        # Check if we have enough examples
        if all(len(v) >= 2 for v in class_examples.values()):
            break

    # Plot
    fig, axes = plt.subplots(5, 3, figsize=(15, 12), sharex=True, sharey=True)

    for class_idx in range(5):
        examples = class_examples[class_idx]
        class_name = AAMI_CLASS_NAMES[class_idx]
        full_name = AAMI_CLASS_FULL_NAMES[class_idx]

        for col in range(3):
            ax = axes[class_idx, col]
            if col < len(examples):
                beat, symbol, record_name = examples[col]
                time_ms = np.arange(len(beat)) / 360 * 1000
                ax.plot(time_ms, beat, linewidth=1.2, color="steelblue")
                ax.set_title(f"Rec: {record_name} ({symbol})", fontsize=9)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10, color="gray")

            if col == 0:
                ax.set_ylabel(f"{class_name}\n{full_name}", fontsize=10, fontweight="bold")
            if class_idx == 4:
                ax.set_xlabel("Time (ms)")

    plt.suptitle("Example Beats from Each AAMI Class", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "notebooks" / "beat_classes.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: notebooks/beat_classes.png")


def plot_class_distribution() -> None:
    """Plot the overall class distribution in the MIT-BIH database."""
    config = load_config()
    preprocessor = ECGPreprocessor(config)

    # Process a few records to get distribution
    record_names = get_record_names(DATA_DIR)[:10]  # Sample first 10

    all_labels = []
    for record_name in record_names:
        result = preprocessor.process_record(record_name, DATA_DIR)
        all_labels.append(result["labels"])

    all_labels = np.concatenate(all_labels)

    # Count classes
    unique, counts = np.unique(all_labels, return_counts=True)
    class_names = [AAMI_CLASS_NAMES.get(i, f"Class {i}") for i in unique]
    full_names = [AAMI_CLASS_FULL_NAMES.get(i, "Unknown") for i in unique]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#95a5a6"]
    bars = axes[0].bar(class_names, counts, color=colors[:len(unique)])
    axes[0].set_title("Beat Class Distribution (Sample)", fontsize=12)
    axes[0].set_xlabel("AAMI Class")
    axes[0].set_ylabel("Number of Beats")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}\n({count/len(all_labels)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Pie chart
    axes[1].pie(
        counts,
        labels=class_names,
        colors=colors[:len(unique)],
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.05] * len(unique),
    )
    axes[1].set_title("Class Distribution (Percentage)", fontsize=12)

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "notebooks" / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: notebooks/class_distribution.png")


def plot_pan_tompkins_comparison() -> None:
    """Compare Pan-Tompkins detected R-peaks vs annotations."""
    record, annotation = load_record("100", DATA_DIR)
    signal = record.p_signal[:, 0]

    pan_tompkins = PanTompkins()
    metrics = pan_tompkins.compare_with_annotations(signal, annotation.sample)

    # Plot the full pipeline
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

    time = np.arange(len(signal)) / record.fs

    # Raw signal
    axes[0].plot(time[:3600], signal[:3600], linewidth=0.5, color="steelblue")
    axes[0].set_title("1. Raw ECG Signal", fontsize=10)
    axes[0].set_ylabel("Amplitude")

    # Bandpass filtered
    filtered = pan_tompkins.bandpass_filter(signal)
    axes[1].plot(time[:3600], filtered[:3600], linewidth=0.5, color="darkorange")
    axes[1].set_title("2. Bandpass Filtered (5-11 Hz)", fontsize=10)
    axes[1].set_ylabel("Amplitude")

    # Derivative
    differentiated = pan_tompkins.derivative(filtered)
    axes[2].plot(time[:3600], differentiated[:3600], linewidth=0.5, color="green")
    axes[2].set_title("3. Five-Point Derivative", fontsize=10)
    axes[2].set_ylabel("Amplitude")

    # Squared
    squared = pan_tompkins.squaring(differentiated)
    axes[3].plot(time[:3600], squared[:3600], linewidth=0.5, color="red")
    axes[3].set_title("4. Squared Signal", fontsize=10)
    axes[3].set_ylabel("Amplitude")

    # Integrated with peaks
    integrated = pan_tompkins.moving_window_integration(squared)
    axes[4].plot(time[:3600], integrated[:3600], linewidth=0.5, color="purple")

    # Mark detected peaks
    detected_peaks = pan_tompkins.detect_r_peaks(signal)
    detected_in_window = detected_peaks[detected_peaks < 3600]
    axes[4].scatter(
        detected_in_window / record.fs,
        integrated[detected_in_window],
        color="red",
        s=30,
        zorder=5,
        label=f"Detected ({len(detected_peaks)} total)",
    )

    # Mark annotation peaks
    annotation_in_window = annotation.sample[annotation.sample < 3600]
    axes[4].scatter(
        annotation_in_window / record.fs,
        integrated[annotation_in_window],
        color="blue",
        s=30,
        zorder=5,
        marker="x",
        label=f"Annotations ({len(annotation.sample)} total)",
    )

    axes[4].set_title("5. Moving Window Integration + R-peak Detection", fontsize=10)
    axes[4].set_ylabel("Amplitude")
    axes[4].set_xlabel("Time (seconds)")
    axes[4].legend(fontsize=9)

    # Add metrics text
    metrics_text = (
        f"Precision: {metrics['precision']:.3f} | "
        f"Recall: {metrics['recall']:.3f} | "
        f"F1: {metrics['f1']:.3f}"
    )
    fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=11, fontweight="bold")

    plt.suptitle("Pan-Tompkins QRS Detection Pipeline", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "notebooks" / "pan_tompkins.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: notebooks/pan_tompkins.png")
    print(f"Pan-Tompkins Metrics: {metrics}")


def run_all_explorations() -> None:
    """Run all exploration plots."""
    # Download data first
    print("=" * 60)
    print("Downloading MIT-BIH data...")
    print("=" * 60)
    download_mitdb(DATA_DIR)

    print("\n" + "=" * 60)
    print("1. Raw vs Filtered Signal")
    print("=" * 60)
    plot_raw_vs_filtered()

    print("\n" + "=" * 60)
    print("2. Beat Segmentation")
    print("=" * 60)
    plot_beat_segmentation()

    print("\n" + "=" * 60)
    print("3. Beat Classes")
    print("=" * 60)
    plot_beat_classes()

    print("\n" + "=" * 60)
    print("4. Class Distribution")
    print("=" * 60)
    plot_class_distribution()

    print("\n" + "=" * 60)
    print("5. Pan-Tompkins Comparison")
    print("=" * 60)
    plot_pan_tompkins_comparison()

    print("\n" + "=" * 60)
    print("All explorations complete!")
    print("Check notebooks/ for saved plots.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_explorations()
