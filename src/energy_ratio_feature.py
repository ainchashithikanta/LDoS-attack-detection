"""
Module: energy_ratio_feature.py
Purpose: Implements the Low-Frequency Energy Ratio feature from the reference research paper.

What it measures in plain English:
----------------------------------
The Energy Ratio measures the fraction of total signal energy that resides in the
low-frequency band compared to the total energy across all frequencies.
- An LDoS attack injects periodic bursts with a cycle time T (e.g. T = 15s). The fundamental
  frequency of the attack is f_attack = 1 / T = 0.067 Hz, with low-order harmonics.
  This causes a major concentration of energy in the low-frequency region (<= fs / 10).
- Normal background traffic fluctuations are spread more uniformly across the spectrum.
- Consequently, LDoS attack windows display a significantly higher low-frequency energy ratio.

Mathematical Formula:
---------------------
Given a 2D time-frequency energy matrix E where rows correspond to discrete frequencies f:
1. Identify the low-frequency sub-band mask:
       Low-Frequency Band: |f| <= f_low_limit  (where f_low_limit = sampling_rate / 10)
2. Sum the energy within the low-frequency band:
       E_low = sum_{f in Low-Band} sum_{t} E(f, t)
3. Total energy:
       E_total = sum_{all f} sum_{t} E(f, t)
4. Energy Ratio:
       Energy_Ratio = E_low / E_total

Representation Used:
--------------------
Calculated from the 2D time-frequency energy distribution and corresponding frequency coordinates.
"""

from typing import Optional
import numpy as np

import config


def calculate_energy_ratio(
    energy_matrix: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    sampling_rate: int = config.SAMPLING_RATE,
    low_freq_fraction: float = config.LOW_FREQ_FRACTION
) -> float:
    """
    Calculates the low-frequency energy ratio feature.

    Parameters:
    -----------
    energy_matrix : np.ndarray
        2D time-frequency energy distribution matrix of shape (rows, cols).
    frequencies : np.ndarray, optional
        1D array of frequency values for each row. If None, the lowest 1/5th
        of rows is used as the low-frequency boundary (matching repo convention).
    sampling_rate : int
        Sampling frequency in Hz (default: 10 Hz).
    low_freq_fraction : float
        Fraction of sampling rate defining low frequency (default: 0.10, i.e., 1.0 Hz).

    Returns:
    --------
    ratio : float
        Energy ratio value between 0.0 and 1.0.
    """
    energy = np.asarray(energy_matrix, dtype=float)

    if energy.ndim != 2 or energy.size == 0:
        return 0.0

    total_energy = float(np.sum(energy))

    # Safe handling of zero, negative, or invalid energy
    if total_energy <= 0.0 or not np.isfinite(total_energy):
        return 0.0

    rows, cols = energy.shape

    # Determine which rows belong to the low-frequency band
    if frequencies is not None and len(frequencies) == rows:
        freq_limit = sampling_rate * low_freq_fraction
        low_mask = np.abs(frequencies) <= freq_limit
        low_energy = float(np.sum(energy[low_mask, :]))
    else:
        # Default boundary: lowest rows (first row//5 rows)
        low_limit_idx = max(1, rows // 5)
        low_energy = float(np.sum(energy[:low_limit_idx, :]))

    ratio = low_energy / total_energy
    ratio = float(np.clip(ratio, 0.0, 1.0))

    return ratio if np.isfinite(ratio) else 0.0


if __name__ == "__main__":
    print("Testing energy_ratio_feature.py...")
    # Test 1: Energy concentrated entirely in top row
    mat_top = np.zeros((10, 10))
    mat_top[0, :] = 5.0
    val_top = calculate_energy_ratio(mat_top)
    print("Top row concentrated energy ratio (expected 1.0):", val_top)
    assert np.isclose(val_top, 1.0)

    # Test 2: Energy concentrated in bottom row (high frequency)
    mat_bottom = np.zeros((10, 10))
    mat_bottom[-1, :] = 5.0
    val_bottom = calculate_energy_ratio(mat_bottom)
    print("Bottom row energy ratio (expected 0.0):", val_bottom)
    assert np.isclose(val_bottom, 0.0)

    # Test 3: Safe zero handling
    assert calculate_energy_ratio(np.zeros((10, 10))) == 0.0
    print("Energy ratio feature successfully verified!")
