"""
Unit tests for dataset loading, signal preprocessing, and window creation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    sanitize_traffic_series,
    remove_dc_component,
    create_traffic_windows,
    prepare_windowed_dataset,
    split_train_test
)
from src.dataset_loader import load_benchmark_dataset, validate_dataframe


class TestDatasetAndPreprocessing:

    def test_sanitize_traffic_series(self):
        raw = np.array([10.0, -5.0, np.nan, np.inf, 12.0])
        cleaned = sanitize_traffic_series(raw)
        assert np.isfinite(cleaned).all()
        assert np.all(cleaned >= 0.0)
        assert cleaned[1] == 0.0  # clipped negative
        assert cleaned[2] == 0.0  # replaced nan

    def test_remove_dc_component(self):
        sig = np.array([10.0, 12.0, 8.0, 10.0])
        centered = remove_dc_component(sig)
        assert np.isclose(np.mean(centered), 0.0)

    def test_create_traffic_windows(self):
        sig = np.ones(250)
        windows, labels = create_traffic_windows(sig, label=1, window_size=100, stride=100)
        assert len(windows) == 2
        assert len(labels) == 2
        assert len(windows[0]) == 100
        assert labels[0] == 1

    def test_split_train_test_no_leakage(self):
        X = np.arange(100).reshape(50, 2)
        y = np.array([0] * 25 + [1] * 25)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.30, random_state=42)
        assert len(X_train) + len(X_test) == 50
        assert len(y_train) + len(y_test) == 50
        # Verify no overlapping samples
        train_set = set(map(tuple, X_train))
        test_set = set(map(tuple, X_test))
        assert len(train_set.intersection(test_set)) == 0

    def test_load_benchmark_dataset(self):
        norm_df, ldos_df = load_benchmark_dataset()
        assert validate_dataframe(norm_df, expected_label=0)
        assert validate_dataframe(ldos_df, expected_label=1)
        assert len(norm_df) > 0
        assert len(ldos_df) > 0
