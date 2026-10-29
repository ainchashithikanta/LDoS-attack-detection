"""
Unit tests for all six feature extraction functions.
Tests normal inputs, edge cases (zero matrix, constant matrix, single spike),
and numerical stability.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contrast_feature import calculate_contrast
from src.energy_ratio_feature import calculate_energy_ratio
from src.correlation_feature import calculate_correlation
from src.entropy_feature import calculate_entropy
from src.spectral_centroid_feature import calculate_spectral_centroid
from src.kurtosis_feature import calculate_kurtosis


class TestFeatures:

    def setup_method(self):
        """Create sample test matrices."""
        self.zero_mat = np.zeros((10, 10))
        self.uniform_mat = np.ones((10, 10))
        self.diag_mat = np.eye(10)
        self.spike_mat = np.zeros((10, 10))
        self.spike_mat[2, 2] = 50.0

    def test_contrast(self):
        # Zero matrix returns 0.0
        assert calculate_contrast(self.zero_mat) == 0.0
        # Diagonal matrix (i == j) has 0 contrast
        assert np.isclose(calculate_contrast(self.diag_mat), 0.0)
        # Uniform matrix has positive contrast
        val_unif = calculate_contrast(self.uniform_mat)
        assert val_unif > 0.0
        assert np.isfinite(val_unif)

    def test_energy_ratio(self):
        # Zero matrix returns 0.0
        assert calculate_energy_ratio(self.zero_mat) == 0.0
        # Energy concentrated only at row 0 (low frequency)
        mat_low = np.zeros((10, 10))
        mat_low[0, :] = 10.0
        assert np.isclose(calculate_energy_ratio(mat_low), 1.0)
        # Energy concentrated only at row 9 (high frequency)
        mat_high = np.zeros((10, 10))
        mat_high[-1, :] = 10.0
        assert np.isclose(calculate_energy_ratio(mat_high), 0.0)

    def test_correlation(self):
        # Zero matrix returns 0.0
        assert calculate_correlation(self.zero_mat) == 0.0
        # Diagonal matrix has perfect linear correlation 1.0
        assert np.isclose(calculate_correlation(self.diag_mat), 1.0)
        # Uniform matrix has 0 correlation
        assert np.isclose(calculate_correlation(self.uniform_mat), 0.0, atol=1e-12)

    def test_entropy(self):
        # Zero matrix returns 0.0
        assert calculate_entropy(self.zero_mat) == 0.0
        # Single spike has zero entropy (completely predictable)
        assert np.isclose(calculate_entropy(self.spike_mat), 0.0)
        # Uniform matrix across N elements has maximum entropy ln(N)
        expected_max = np.log(100)
        assert np.isclose(calculate_entropy(self.uniform_mat), expected_max)

    def test_spectral_centroid(self):
        # Zero matrix returns 0.0
        assert calculate_spectral_centroid(self.zero_mat) == 0.0
        # Energy concentrated at row 0
        mat_dc = np.zeros((10, 10))
        mat_dc[0, :] = 5.0
        assert np.isclose(calculate_spectral_centroid(mat_dc), 0.0)
        # Energy concentrated at row 9
        mat_top = np.zeros((10, 10))
        mat_top[-1, :] = 5.0
        assert np.isclose(calculate_spectral_centroid(mat_top), 1.0)

    def test_kurtosis(self):
        # Zero matrix returns 0.0
        assert calculate_kurtosis(self.zero_mat) == 0.0
        # Uniform matrix across time has variance 0 -> kurtosis 0.0
        assert calculate_kurtosis(self.uniform_mat) == 0.0
        # Spike profile has high positive kurtosis
        pulse_mat = np.zeros((10, 50))
        pulse_mat[:, 20:23] = 100.0
        val_kurt = calculate_kurtosis(pulse_mat)
        assert val_kurt > 0.0
        assert np.isfinite(val_kurt)
