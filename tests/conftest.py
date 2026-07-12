"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest

# Add project root to path so src modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    """Return the project root path."""
    return PROJECT_ROOT


@pytest.fixture
def sample_config():
    """Return a minimal config for testing."""
    return {
        "data": {
            "sampling_rate": 360,
            "window_size": 288,
            "raw_dir": str(PROJECT_ROOT / "data" / "raw" / "mitdb"),
            "processed_dir": str(PROJECT_ROOT / "data" / "processed"),
        },
        "model": {
            "name": "cnn_baseline",
            "num_classes": 5,
        },
        "training": {
            "batch_size": 64,
            "epochs": 1,
        },
    }
