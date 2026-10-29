"""
Module: cwt_transform.py
Purpose: Implements the Continuous Wavelet Transform (CWT) for multiresolution
         time-scale / time-frequency analysis of network traffic signals.

Mathematical Background for Students:
-------------------------------------
1. Unlike the Short-Time Fourier Transform (which has fixed time-frequency resolution
   due to the Heisenberg-Gabor uncertainty principle), the Continuous Wavelet
   Transform (CWT) provides variable resolution: high time resolution for fast transients
   (high frequencies) and high frequency resolution for long-duration trends (low frequencies).
2. Formula:
       CWT{x(t)}(a, b) = (1 / sqrt(|a|)) * integral x(t) * psi*((t - b) / a) dt
   where:
       - a is the scale factor (inversely related to frequency)
       - b is the temporal translation / shift
       - psi(t) is the mother wavelet (e.g. Morlet wavelet: exp(j*omega_0*t) * exp(-t^2 / 2))
3. The resulting Scalogram / Energy Distribution is:
       E(scale, t) = |CWT(a, b)|^2
"""

from typing import Tuple
import numpy as np
import pywt

import config


def compute_cwt(
    signal: np.ndarray,
    sampling_rate: int = config.SAMPLING_RATE,
    wavelet_name: str = "morl",
    num_scales: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Continuous Wavelet Transform (CWT) energy scalogram
    for a 1D input traffic signal.

    Parameters:
    -----------
    signal : np.ndarray
        1D input signal array (length N).
    sampling_rate : int
        Sampling frequency in Hz (default: 10 Hz).
    wavelet_name : str
        Wavelet family name (default: 'morl' for Morlet wavelet).
    num_scales : int
        Number of continuous scales to compute (default: 32).

    Returns:
    --------
    energy_matrix : np.ndarray
        2D matrix of shape (num_scales, N) representing squared magnitudes |CWT|^2.
    frequencies : np.ndarray
        1D array containing pseudo-frequencies (Hz) corresponding to each scale.
    scales : np.ndarray
        1D array containing scale values.
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)

    if N == 0:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)

    # Define logarithmically or linearly spaced scales covering the signal frequencies
    scales = np.arange(1, num_scales + 1, dtype=float)

    # Compute CWT using PyWavelets
    coefficients, frequencies = pywt.cwt(
        signal,
        scales=scales,
        wavelet=wavelet_name,
        sampling_period=1.0 / sampling_rate
    )

    # Scalogram energy matrix is squared magnitude of coefficients
    energy_matrix = np.abs(coefficients) ** 2

    # Clean non-finite values safely
    if not np.isfinite(energy_matrix).all():
        energy_matrix = np.nan_to_num(energy_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return energy_matrix, frequencies, scales


if __name__ == "__main__":
    print("Testing cwt_transform.py...")
    test_signal = np.sin(2 * np.pi * 1.0 * np.arange(100) / 10.0)
    energy, freqs, scales = compute_cwt(test_signal, sampling_rate=10)
    print(f"CWT Energy Scalogram Shape: {energy.shape}")
    print(f"Frequencies range: [{freqs[-1]:.2f}, {freqs[0]:.2f}] Hz")
    print(f"Total Energy: {np.sum(energy):.4f}")
    assert energy.ndim == 2
    assert np.isfinite(energy).all()
    print("CWT Transform successfully verified!")
