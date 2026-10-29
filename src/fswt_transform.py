"""
Module: fswt_transform.py
Purpose: Implements the Frequency Slice Wavelet Transform (FSWT) as defined in the
         reference paper (Xiaocai Wang et al., 2024, Expert Systems with Applications)
         and the original research repository.

Mathematical Background for Students:
-------------------------------------
1. The Frequency Slice Wavelet Transform (FSWT) is a time-frequency analysis tool
   that slices the signal's Fourier spectrum using a frequency slice function (FSF).
2. The standard Frequency Slice Function is a Gaussian window:
       p_hat(omega) = exp(-0.5 * omega^2)
3. For each discrete frequency index i with center frequency f_i:
       sigma_i = |f_i| / kappa    (where kappa = 23.5482 is the resolution parameter)
       omega = (frequencies - f_i) / sigma_i
       P_i = p_hat(omega)
4. The FSWT coefficient slice is calculated in the frequency domain and inverted:
       W(i, t) = lambda * IFFT{ FFT{x} * P_i^* }
   where lambda = 1.0, and P_i^* is the complex conjugate of the slice filter.
5. The resulting 2D energy distribution matrix is:
       E(i, t) = |W(i, t)|^2
"""

from typing import Tuple
import numpy as np

import config


def gaussian_fsf(omega: np.ndarray) -> np.ndarray:
    """
    Gaussian Frequency Slice Function (FSF):
        p_hat(omega) = exp(-0.5 * omega^2)

    Parameters:
    -----------
    omega : np.ndarray
        Normalized frequency deviation array.

    Returns:
    --------
    p : np.ndarray
        Gaussian filter response in frequency domain.
    """
    return np.exp(-0.5 * (omega ** 2))


def compute_fswt(
    signal: np.ndarray,
    sampling_rate: int = config.SAMPLING_RATE,
    lam: float = config.FSWT_LAMBDA,
    kappa: float = config.FSWT_KAPPA
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 2D Frequency Slice Wavelet Transform (FSWT) energy distribution
    for a 1D input traffic signal.

    Parameters:
    -----------
    signal : np.ndarray
        1D input signal array (length N).
    sampling_rate : int
        Sampling frequency in Hz (default: 10 Hz).
    lam : float
        Scaling parameter lambda (default: 1.0).
    kappa : float
        Frequency scaling parameter kappa (default: 23.5482).

    Returns:
    --------
    energy_matrix : np.ndarray
        2D matrix of shape (N, N) where rows represent frequency slices
        and columns represent time samples. Values are squared magnitudes (|W|^2).
    frequencies : np.ndarray
        1D array of length N containing the corresponding frequency bins.
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)

    if N == 0:
        return np.zeros((0, 0)), np.zeros(0)

    # 1. Compute Fourier transform of the signal
    X = np.fft.fft(signal)

    # 2. Compute FFT frequency bins
    frequencies = np.fft.fftfreq(N, d=1.0 / sampling_rate)

    # 3. Pre-allocate FSWT complex matrix of size (N, N)
    W = np.zeros((N, N), dtype=complex)

    # 4. Generate frequency slices
    for i in range(N):
        center_frequency = frequencies[i]

        if np.isclose(center_frequency, 0.0):
            # DC component: isolate the DC bin
            P = np.zeros(N, dtype=float)
            P[i] = 1.0
        else:
            # Bandwidth scale proportional to center frequency
            sigma = abs(center_frequency) / kappa
            if sigma < 1e-12:
                sigma = 1e-12

            # Frequency difference from the slice center
            frequency_diff = frequencies - center_frequency
            omega = frequency_diff / sigma

            # Gaussian frequency slice function
            P = gaussian_fsf(omega)

        # Slice equation: W(i, t) = lambda * IFFT(X * conj(P))
        W[i, :] = lam * np.fft.ifft(X * np.conjugate(P))

    # 5. Energy distribution: squared magnitude of the complex coefficients
    energy_matrix = np.abs(W) ** 2

    # Replace any potential NaNs or Infs safely
    if not np.isfinite(energy_matrix).all():
        energy_matrix = np.nan_to_num(energy_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return energy_matrix, frequencies


if __name__ == "__main__":
    print("Testing fswt_transform.py...")
    test_signal = np.sin(2 * np.pi * 1.0 * np.arange(100) / 10.0)
    energy, freqs = compute_fswt(test_signal, sampling_rate=10)
    print(f"FSWT Energy Matrix Shape: {energy.shape}")
    print(f"Total Energy: {np.sum(energy):.4f}")
    assert energy.shape == (100, 100)
    assert np.isfinite(energy).all()
    print("FSWT Transform successfully verified!")
