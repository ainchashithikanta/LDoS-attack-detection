"""
Module: kurtosis_feature.py
Purpose: Implements the Kurtosis feature, measuring the 'peakedness' and heavy-tailed
         burstiness of the time-frequency energy distribution.

What it measures in plain English:
----------------------------------
Kurtosis measures how outlier-prone or spike-heavy a distribution is:
- A standard Gaussian normal distribution has an excess kurtosis of 0.
- Normal background traffic fluctuations are smoothly distributed around the mean,
  yielding low kurtosis near zero.
- LDoS attack traffic introduces sudden, high-intensity pulses separated by quiet intervals.
  This creates an extreme 'peaked' energy profile with heavy outliers, leading to
  distinctively high positive kurtosis (leptokurtic distribution).

Mathematical Formula:
---------------------
Given the temporal energy profile z(t) = sum_{f} E(f, t) across the window (or 2D energy values):
1. Mean:
       mu = (1 / N) * sum_{i=1}^{N} z_i
2. Variance:
       sigma^2 = (1 / N) * sum_{i=1}^{N} (z_i - mu)^2
3. Fourth central moment:
       m_4 = (1 / N) * sum_{i=1}^{N} (z_i - mu)^4
4. Fisher's Excess Kurtosis (standard normal = 0):
       Kurtosis = (m_4 / sigma^4) - 3.0
   If sigma^2 < 1e-12 (constant or zero signal), returns 0.0.

Representation Used:
--------------------
Calculated from the temporal projection profile z(t) = sum_{f} E(f, t) of the
2D time-frequency energy matrix.
"""

import numpy as np


def calculate_kurtosis(energy_matrix: np.ndarray, excess: bool = True) -> float:
    """
    Calculates the kurtosis feature from the time-frequency energy distribution.

    Parameters:
    -----------
    energy_matrix : np.ndarray
        2D time-frequency energy distribution matrix of shape (rows, cols).
    excess : bool
        If True (default), computes Fisher's excess kurtosis (subtracts 3 so Gaussian = 0).
        If False, computes Pearson's kurtosis (m_4 / sigma^4).

    Returns:
    --------
    kurt : float
        Scalar kurtosis value (always finite).
    """
    energy = np.asarray(energy_matrix, dtype=float)

    if energy.ndim != 2 or energy.size == 0:
        return 0.0

    # Project 2D energy onto time axis to capture burstiness over time
    temporal_profile = np.sum(energy, axis=0)  # shape (cols,)
    N = len(temporal_profile)

    if N < 4:
        return 0.0

    mean_val = float(np.mean(temporal_profile))
    var_val = float(np.var(temporal_profile))

    # Safe handling: if variance is zero (constant signal), peakedness is 0
    if var_val <= 1e-12 or not np.isfinite(var_val):
        return 0.0

    fourth_moment = float(np.mean((temporal_profile - mean_val) ** 4))
    kurt = fourth_moment / (var_val ** 2)

    if excess:
        kurt -= 3.0

    return float(kurt) if np.isfinite(kurt) else 0.0


if __name__ == "__main__":
    print("Testing kurtosis_feature.py...")
    # Test 1: Gaussian-like distribution (excess kurtosis ~ 0)
    rng = np.random.default_rng(42)
    gaussian_profile = rng.normal(0, 1, (10, 1000)) ** 2
    val_gauss = calculate_kurtosis(gaussian_profile)
    print(f"Gaussian energy profile excess kurtosis: {val_gauss:.4f}")

    # Test 2: Sharp pulse burst (high positive kurtosis)
    pulse_profile = np.zeros((10, 100))
    pulse_profile[:, 10:15] = 100.0  # short burst
    val_pulse = calculate_kurtosis(pulse_profile)
    print(f"Sharp pulse burst excess kurtosis: {val_pulse:.4f}")
    assert val_pulse > val_gauss

    # Test 3: Constant signal (variance = 0 -> kurtosis = 0)
    assert calculate_kurtosis(np.ones((10, 10))) == 0.0
    print("Kurtosis feature successfully verified!")
