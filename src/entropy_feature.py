"""
Module: entropy_feature.py
Purpose: Implements the Time-Frequency Shannon Entropy feature from the reference research paper.

What it measures in plain English:
----------------------------------
Entropy measures the degree of disorder, uncertainty, or dispersion of energy in the
2D time-frequency spectrum.
- Normal background traffic has random, uncorrelated noise fluctuations spread broadly
  across all frequency bins, yielding high entropy (high randomness).
- LDoS attack traffic introduces highly structured, periodic rectangular pulses, causing
  the energy to collapse into strong, concentrated spectral peaks and harmonic spikes.
  This structural order reduces the entropy.

Mathematical Formula:
---------------------
Given a 2D time-frequency energy distribution E(i, j):
1. Compute the normalized energy probability distribution:
       P(i, j) = E(i, j) / sum_{i, j} E(i, j)
2. Shannon Entropy:
       Entropy = - sum_{P(i, j) > 0} P(i, j) * ln( P(i, j) )
   Note: Zero probabilities are excluded because lim_{p -> 0} [ p * ln(p) ] = 0.

Representation Used:
--------------------
Calculated from the normalized 2D time-frequency energy distribution P(i, j).
"""

import numpy as np


def calculate_entropy(energy_matrix: np.ndarray) -> float:
    """
    Calculates the time-frequency Shannon entropy feature.

    Parameters:
    -----------
    energy_matrix : np.ndarray
        2D time-frequency energy distribution matrix.

    Returns:
    --------
    entropy : float
        Scalar Shannon entropy value (always non-negative and finite).
    """
    energy = np.asarray(energy_matrix, dtype=float)

    if energy.ndim != 2 or energy.size == 0:
        return 0.0

    total_energy = float(np.sum(energy))

    # Safe handling of zero, negative, or invalid energy
    if total_energy <= 0.0 or not np.isfinite(total_energy):
        return 0.0

    probability = energy / total_energy

    # Only include non-zero probabilities
    nonzero = probability > 0.0
    if not np.any(nonzero):
        return 0.0

    p_nonzero = probability[nonzero]
    entropy = -float(np.sum(p_nonzero * np.log(p_nonzero)))

    return entropy if np.isfinite(entropy) else 0.0


if __name__ == "__main__":
    print("Testing entropy_feature.py...")
    # Test 1: Completely concentrated energy (single point -> entropy = 0)
    mat_spike = np.zeros((10, 10))
    mat_spike[2, 3] = 100.0
    val_spike = calculate_entropy(mat_spike)
    print("Single spike entropy (expected 0.0):", val_spike)
    assert np.isclose(val_spike, 0.0)

    # Test 2: Uniform energy across N elements (entropy = ln(N))
    mat_unif = np.ones((10, 10))
    val_unif = calculate_entropy(mat_unif)
    expected_unif = np.log(100)
    print(f"Uniform entropy (expected ln(100) = {expected_unif:.4f}):", val_unif)
    assert np.isclose(val_unif, expected_unif)

    # Test 3: Safe zero handling
    assert calculate_entropy(np.zeros((10, 10))) == 0.0
    print("Entropy feature successfully verified!")
