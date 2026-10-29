"""
Module: stft_transform.py
Purpose: Implements the Short-Time Fourier Transform (STFT) for time-frequency analysis
         of network traffic signals.

Mathematical Background for Students:
-------------------------------------
1. While standard FFT calculates frequency content over the whole signal without time
   localization, the Short-Time Fourier Transform (STFT) slides a short window w[n]
   across the signal to capture time-localized frequency variations.
2. Formula:
       STFT{x[n]}(m, omega) = sum_{n} x[n] * w[n - m] * exp(-j * omega * n)
   where:
       - x[n] is the traffic signal
       - w[n] is a Hann window function
       - m is the discrete time shift
       - omega is the angular frequency
3. The time-frequency Spectrogram / Energy Distribution is:
       E(f, t) = |STFT{x[n]}(m, omega)|^2
"""

from typing import Tuple
import numpy as np
from scipy import signal as sp_signal

import config


def compute_stft(
    signal: np.ndarray,
    sampling_rate: int = config.SAMPLING_RATE,
    window_type: str = "hann",
    nperseg: int = 32,
    noverlap: int = 24
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Short-Time Fourier Transform (STFT) energy distribution
    for a 1D input traffic signal.

    Parameters:
    -----------
    signal : np.ndarray
        1D input signal array (e.g. length 100).
    sampling_rate : int
        Sampling frequency in Hz (default: 10 Hz).
    window_type : str
        Window function to apply (default: 'hann').
    nperseg : int
        Length of each segment for STFT (default: 32).
    noverlap : int
        Number of points of overlap between segments (default: 24).

    Returns:
    --------
    energy_matrix : np.ndarray
        2D matrix of shape (n_freqs, n_time_steps) representing squared magnitudes |STFT|^2.
    frequencies : np.ndarray
        1D array containing discrete frequency bins (Hz).
    time_bins : np.ndarray
        1D array containing time centers for each window slice (seconds).
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)

    if N == 0:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)

    # Adjust segment length if signal is smaller than requested nperseg
    actual_nperseg = min(nperseg, N)
    actual_noverlap = min(noverlap, actual_nperseg - 1)
    if actual_noverlap < 0:
        actual_noverlap = 0

    frequencies, time_bins, Zxx = sp_signal.stft(
        signal,
        fs=sampling_rate,
        window=window_type,
        nperseg=actual_nperseg,
        noverlap=actual_noverlap,
        boundary=None,
        padded=False
    )

    # Energy matrix is squared magnitude of complex STFT coefficients
    energy_matrix = np.abs(Zxx) ** 2

    # Clean non-finite values safely
    if not np.isfinite(energy_matrix).all():
        energy_matrix = np.nan_to_num(energy_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return energy_matrix, frequencies, time_bins


if __name__ == "__main__":
    print("Testing stft_transform.py...")
    test_signal = np.sin(2 * np.pi * 1.0 * np.arange(100) / 10.0)
    energy, freqs, times = compute_stft(test_signal, sampling_rate=10)
    print(f"STFT Energy Matrix Shape: {energy.shape}")
    print(f"Frequencies range: [{freqs[0]:.2f}, {freqs[-1]:.2f}] Hz")
    print(f"Total Energy: {np.sum(energy):.4f}")
    assert energy.ndim == 2
    assert np.isfinite(energy).all()
    print("STFT Transform successfully verified!")
