"""
Module: spectral_centroid_feature.py
Purpose: Implements the Spectral Centroid feature, representing the 'center of mass'
         of the spectral energy distribution.

What it measures in plain English:
----------------------------------
The Spectral Centroid indicates the weighted average frequency where the signal's energy
is concentrated:
- High Spectral Centroid: Energy is located mainly at higher frequencies (fast fluctuations,
  wideband random noise).
- Low Spectral Centroid: Energy is concentrated at low frequencies (slow trends, low-rate
  periodic pulses).
- Under LDoS attacks, the periodic rectangular bursts concentrate power in low-frequency
  fundamental and harmonic bands (f <= 1.0 Hz), pulling the spectral centroid sharply downwards.

Mathematical Formula:
---------------------
Given a 2D time-frequency energy distribution E(f_k, t) where f_k is the frequency of row k:
1. Aggregate energy over time for each frequency bin:
       E(f_k) = sum_{t} E(f_k, t)
2. Total energy:
       E_total = sum_{k} E(f_k)
3. The Spectral Centroid is the energy-weighted frequency mean:
       Spectral_Centroid = [ sum_{k} ( |f_k| * E(f_k) ) ] / E_total
   (If frequencies array is omitted, normalized row indices 0, 1, ..., rows-1 are used).

Representation Used:
--------------------
Calculated from the frequency-energy profile of the 2D time-frequency distribution.
"""

from typing import Optional
import numpy as np


def calculate_spectral_centroid(
    energy_matrix: np.ndarray,
    frequencies: Optional[np.ndarray] = None
) -> float:
    """
    Calculates the spectral centroid feature.

    Parameters:
    -----------
    energy_matrix : np.ndarray
        2D time-frequency energy distribution matrix of shape (rows, cols).
    frequencies : np.ndarray, optional
        1D array of frequency values for each row. If None, normalized row
        indices in [0, 1] are used.

    Returns:
    --------
    centroid : float
        Scalar spectral centroid value (always non-negative and finite).
    """
    energy = np.asarray(energy_matrix, dtype=float)

    if energy.ndim != 2 or energy.size == 0:
        return 0.0

    # Total energy across all frequency and time bins
    total_energy = float(np.sum(energy))

    # Safe handling of zero, negative, or invalid energy
    if total_energy <= 0.0 or not np.isfinite(total_energy):
        return 0.0

    rows, cols = energy.shape

    # Sum energy across time for each frequency row
    freq_energy_profile = np.sum(energy, axis=1)  # shape (rows,)

    if frequencies is not None and len(frequencies) == rows:
        freq_weights = np.abs(np.asarray(frequencies, dtype=float))
    else:
        # Default: normalized frequency axis from 0.0 to 1.0
        freq_weights = np.linspace(0.0, 1.0, rows)

    centroid = float(np.sum(freq_weights * freq_energy_profile) / total_energy)

    return centroid if np.isfinite(centroid) else 0.0


if __name__ == "__main__":
    print("Testing spectral_centroid_feature.py...")
    # Test 1: Energy concentrated at row 0 (frequency = 0.0)
    mat_dc = np.zeros((10, 10))
    mat_dc[0, :] = 10.0
    val_dc = calculate_spectral_centroid(mat_dc)
    print("DC-concentrated centroid (expected 0.0):", val_dc)
    assert np.isclose(val_dc, 0.0)

    # Test 2: Energy concentrated at top row (frequency = 1.0)
    mat_high = np.zeros((10, 10))
    mat_high[-1, :] = 10.0
    val_high = calculate_spectral_centroid(mat_high)
    print("High-frequency centroid (expected 1.0):", val_high)
    assert np.isclose(val_high, 1.0)

    # Test 3: Safe zero handling
    assert calculate_spectral_centroid(np.zeros((10, 10))) == 0.0
    print("Spectral centroid feature successfully verified!")
