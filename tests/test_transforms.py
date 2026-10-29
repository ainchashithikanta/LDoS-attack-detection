"""
Unit tests for the three time-frequency transforms: FSWT, STFT, and CWT.
Tests shape, finite values, and response to known signals.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fswt_transform import compute_fswt
from src.stft_transform import compute_stft
from src.cwt_transform import compute_cwt


class TestTransforms:

    def setup_method(self):
        # 100 samples at 10 Hz (10.0 seconds) with a 1.0 Hz sinusoidal pulse
        self.sampling_rate = 10
        self.t = np.arange(100) / self.sampling_rate
        self.signal = np.sin(2 * np.pi * 1.0 * self.t)

    def test_fswt_transform(self):
        energy, freqs = compute_fswt(self.signal, sampling_rate=self.sampling_rate)
        assert energy.shape == (100, 100)
        assert len(freqs) == 100
        assert np.isfinite(energy).all()
        assert np.all(energy >= 0.0)
        assert np.sum(energy) > 0.0

    def test_stft_transform(self):
        energy, freqs, times = compute_stft(self.signal, sampling_rate=self.sampling_rate)
        assert energy.ndim == 2
        assert len(freqs) == energy.shape[0]
        assert len(times) == energy.shape[1]
        assert np.isfinite(energy).all()
        assert np.all(energy >= 0.0)
        assert np.sum(energy) > 0.0

    def test_cwt_transform(self):
        energy, freqs, scales = compute_cwt(self.signal, sampling_rate=self.sampling_rate, num_scales=20)
        assert energy.shape == (20, 100)
        assert len(freqs) == 20
        assert len(scales) == 20
        assert np.isfinite(energy).all()
        assert np.all(energy >= 0.0)
        assert np.sum(energy) > 0.0

    def test_zero_signal_handling(self):
        zero_sig = np.zeros(100)
        # All transforms should safely produce non-negative finite arrays
        e_fswt, _ = compute_fswt(zero_sig, sampling_rate=self.sampling_rate)
        assert np.isfinite(e_fswt).all()
        assert np.isclose(np.sum(e_fswt), 0.0)

        e_stft, _, _ = compute_stft(zero_sig, sampling_rate=self.sampling_rate)
        assert np.isfinite(e_stft).all()
        assert np.isclose(np.sum(e_stft), 0.0)

        e_cwt, _, _ = compute_cwt(zero_sig, sampling_rate=self.sampling_rate)
        assert np.isfinite(e_cwt).all()
        assert np.isclose(np.sum(e_cwt), 0.0)
