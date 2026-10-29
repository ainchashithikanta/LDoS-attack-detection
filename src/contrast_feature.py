"""
Module: contrast_feature.py
Purpose: Implements the Time-Frequency Contrast feature from the reference research paper.

What it measures in plain English:
----------------------------------
Contrast measures the degree of local variation and intensity difference across different
coordinates in the 2D time-frequency energy distribution matrix.
- Normal background network traffic fluctuates smoothly, yielding low contrast values.
- LDoS attack traffic injects sharp, periodic rectangular pulses that create abrupt transitions
  and localized high-energy concentrations, producing distinctively higher contrast.

Mathematical Formula:
---------------------
Given a 2D time-frequency energy matrix E of shape (rows, cols):
1. Compute the normalized energy probability distribution:
       P(i, j) = E(i, j) / sum_{i, j} E(i, j)
2. Contrast is the second-order moment of coordinate differences:
       Contrast = sum_{i=0}^{rows-1} sum_{j=0}^{cols-1} (i - j)^2 * P(i, j)

Representation Used:
--------------------
Calculated directly from the normalized 2D time-frequency energy matrix E(i, j)
(e.g., from FSWT, STFT, or CWT).
"""

import numpy as np


def calculate_contrast(energy_matrix: np.ndarray) -> float:
    """
    Calculates the time-frequency contrast feature.

    Parameters:
    -----------
    energy_matrix : np.ndarray
        2D time-frequency energy distribution matrix.

    Returns:
    --------
    contrast : float
        Scalar contrast value (always non-negative and finite).
    """
    energy = np.asarray(energy_matrix, dtype=float)

    if energy.ndim != 2 or energy.size == 0:
        return 0.0

    total_energy = float(np.sum(energy))

    # Safe handling of zero, negative, or invalid energy
    if total_energy <= 0.0 or not np.isfinite(total_energy):
        return 0.0

    # Normalized energy distribution (probability mass function)
    probability = energy / total_energy

    rows, cols = energy.shape

    # Vectorized computation of (i - j)^2 * P(i, j)
    row_indices = np.arange(rows, dtype=float)[:, None]
    col_indices = np.arange(cols, dtype=float)[None, :]

    diff_squared = (row_indices - col_indices) ** 2
    contrast = float(np.sum(diff_squared * probability))

    return contrast if np.isfinite(contrast) else 0.0


if __name__ == "__main__":
    print("Testing contrast_feature.py...")
    # Test 1: Uniform energy
    mat_uniform = np.ones((10, 10))
    val_uniform = calculate_contrast(mat_uniform)
    print("Uniform contrast:", val_uniform)

    # Test 2: Diagonal energy (i == j -> contrast should be 0)
    mat_diagonal = np.eye(10)
    val_diagonal = calculate_contrast(mat_diagonal)
    print("Diagonal contrast (expected 0.0):", val_diagonal)
    assert np.isclose(val_diagonal, 0.0)

    # Test 3: Safe zero handling
    mat_zero = np.zeros((10, 10))
    val_zero = calculate_contrast(mat_zero)
    assert val_zero == 0.0
    print("Contrast feature successfully verified!")
