"""
Module: correlation_feature.py
Purpose: Implements the Time-Frequency Correlation feature from the reference research paper.

What it measures in plain English:
----------------------------------
Correlation measures the joint linear dependency and structural alignment between
temporal and spectral coordinates in the 2D time-frequency energy distribution.
- High correlation indicates that spectral energy changes predictably and synchronously
  over time, which is characteristic of the rhythmic, repeating pulses of an LDoS attack.
- Random noise and benign fluctuating traffic display uncorrelated, scattered energy spreads.

Mathematical Formula:
---------------------
Given a normalized 2D probability distribution P(i, j) = E(i, j) / sum(E):
1. Compute the coordinate expectations (means):
       mu_1 = sum_{i, j} i * P(i, j)
       mu_2 = sum_{i, j} j * P(i, j)
2. Compute the coordinate variances:
       sigma_1^2 = sum_{i, j} (i - mu_1)^2 * P(i, j)
       sigma_2^2 = sum_{i, j} (j - mu_2)^2 * P(i, j)
3. Compute the joint covariance:
       Cov = sum_{i, j} (i - mu_1) * (j - mu_2) * P(i, j)
           = [ sum_{i, j} i * j * P(i, j) ] - mu_1 * mu_2
4. Normalized Haralick Correlation:
       Correlation = Cov / ( sqrt(sigma_1^2 * sigma_2^2) + eps )
   where eps = 1e-12 prevents division by zero for uniform or zero matrices.

Representation Used:
--------------------
Calculated from the normalized 2D time-frequency distribution matrix P(i, j).
"""

import numpy as np


def calculate_correlation(energy_matrix: np.ndarray, use_haralick: bool = True) -> float:
    """
    Calculates the time-frequency correlation feature.

    Parameters:
    -----------
    energy_matrix : np.ndarray
        2D time-frequency energy distribution matrix.
    use_haralick : bool
        If True (default), uses the standard normalized Haralick correlation (sigma1 * sigma2),
        bounded in [-1.0, 1.0]. If False, uses the repo's unrooted variant (sigma1^2 * sigma2^2).

    Returns:
    --------
    correlation : float
        Scalar correlation value (always finite).
    """
    energy = np.asarray(energy_matrix, dtype=float)

    if energy.ndim != 2 or energy.size == 0:
        return 0.0

    total_energy = float(np.sum(energy))

    # Safe handling of zero or non-finite energy
    if total_energy <= 0.0 or not np.isfinite(total_energy):
        return 0.0

    probability = energy / total_energy
    rows, cols = energy.shape

    row_indices = np.arange(rows, dtype=float)[:, None]
    col_indices = np.arange(cols, dtype=float)[None, :]

    # Marginal means
    mu1 = float(np.sum(row_indices * probability))
    mu2 = float(np.sum(col_indices * probability))

    # Marginal variances
    sigma1_squared = float(np.sum(((row_indices - mu1) ** 2) * probability))
    sigma2_squared = float(np.sum(((col_indices - mu2) ** 2) * probability))

    # If variance is zero along either axis, correlation is zero
    if sigma1_squared <= 1e-12 or sigma2_squared <= 1e-12:
        return 0.0

    # Covariance numerator
    numerator = float(np.sum(row_indices * col_indices * probability)) - (mu1 * mu2)

    # Denominator
    if use_haralick:
        denominator = np.sqrt(sigma1_squared * sigma2_squared)
        corr = numerator / denominator
        # Clip to mathematical range [-1, 1]
        corr = float(np.clip(corr, -1.0, 1.0))
    else:
        denominator = sigma1_squared * sigma2_squared
        corr = float(numerator / (denominator + 1e-12))

    # Snap floating-point precision noise to exact zero
    if abs(corr) < 1e-12:
        corr = 0.0

    return corr if np.isfinite(corr) else 0.0


if __name__ == "__main__":
    print("Testing correlation_feature.py...")
    # Test 1: Perfectly correlated diagonal matrix (should give 1.0)
    mat_diag = np.eye(10)
    val_diag = calculate_correlation(mat_diag)
    print("Diagonal correlation (expected 1.0):", val_diag)
    assert np.isclose(val_diag, 1.0)

    # Test 2: Uniform matrix (zero variance / zero correlation)
    mat_unif = np.ones((10, 10))
    val_unif = calculate_correlation(mat_unif)
    print("Uniform correlation (expected 0.0):", val_unif)
    assert val_unif == 0.0

    # Test 3: Zero matrix
    assert calculate_correlation(np.zeros((10, 10))) == 0.0
    print("Correlation feature successfully verified!")
