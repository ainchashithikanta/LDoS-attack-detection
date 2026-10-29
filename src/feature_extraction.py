"""
Module: feature_extraction.py
Purpose: Orchestrates time-frequency transformation (FSWT, STFT, or CWT) and extracts
         the six features from traffic windows into a structured Pandas DataFrame.
"""

from typing import List, Dict, Union, Optional
import time
import numpy as np
import pandas as pd

import config
from src.preprocessing import remove_dc_component
from src.fswt_transform import compute_fswt
from src.stft_transform import compute_stft
from src.cwt_transform import compute_cwt
from src.contrast_feature import calculate_contrast
from src.energy_ratio_feature import calculate_energy_ratio
from src.correlation_feature import calculate_correlation
from src.entropy_feature import calculate_entropy
from src.spectral_centroid_feature import calculate_spectral_centroid
from src.kurtosis_feature import calculate_kurtosis


def extract_features_from_window(
    window: np.ndarray,
    transform_type: str = "fswt",
    feature_set: str = "all",
    sampling_rate: int = config.SAMPLING_RATE
) -> Dict[str, float]:
    """
    Extracts features from a single 1D traffic window using the specified transform.

    Parameters:
    -----------
    window : np.ndarray
        1D signal array representing a single window.
    transform_type : str
        Transform to apply: 'fswt', 'stft', or 'cwt'.
    feature_set : str
        'original' (4 features) or 'all' (6 features).
    sampling_rate : int
        Sampling frequency in Hz (default: 10).

    Returns:
    --------
    features : Dict[str, float]
        Dictionary mapping feature names to extracted scalar values.
    """
    # 1. Remove DC component to center fluctuations around zero
    centered_window = remove_dc_component(window)

    # 2. Compute the chosen time-frequency transform
    trans_lower = transform_type.lower()
    if trans_lower == "fswt":
        energy_matrix, frequencies = compute_fswt(centered_window, sampling_rate=sampling_rate)
    elif trans_lower == "stft":
        energy_matrix, frequencies, _ = compute_stft(centered_window, sampling_rate=sampling_rate)
    elif trans_lower == "cwt":
        energy_matrix, frequencies, _ = compute_cwt(centered_window, sampling_rate=sampling_rate)
    else:
        raise ValueError(f"Unsupported transform_type: '{transform_type}'. Choose 'fswt', 'stft', or 'cwt'.")

    # 3. Extract original 4 features
    contrast = calculate_contrast(energy_matrix)
    energy_ratio = calculate_energy_ratio(energy_matrix, frequencies=frequencies, sampling_rate=sampling_rate)
    correlation = calculate_correlation(energy_matrix)
    entropy = calculate_entropy(energy_matrix)

    feature_dict = {
        "contrast": float(contrast),
        "energy_ratio": float(energy_ratio),
        "correlation": float(correlation),
        "entropy": float(entropy),
    }

    # 4. Extract 2 additional features if requested
    if feature_set.lower() == "all":
        centroid = calculate_spectral_centroid(energy_matrix, frequencies=frequencies)
        kurt = calculate_kurtosis(energy_matrix)
        feature_dict["spectral_centroid"] = float(centroid)
        feature_dict["kurtosis"] = float(kurt)

    return feature_dict


def build_feature_dataset(
    windows: List[np.ndarray],
    labels: List[int],
    transform_type: str = "fswt",
    feature_set: str = "all",
    sampling_rate: int = config.SAMPLING_RATE
) -> pd.DataFrame:
    """
    Extracts features for an entire list of windows and formats them into
    a clean Pandas DataFrame with the target label column.

    Parameters:
    -----------
    windows : List[np.ndarray]
        List of 1D signal window arrays.
    labels : List[int]
        Corresponding class labels (0 for Normal, 1 for LDoS).
    transform_type : str
        'fswt', 'stft', or 'cwt'.
    feature_set : str
        'original' (4 features) or 'all' (6 features).
    sampling_rate : int
        Sampling frequency in Hz.

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with feature columns and 'label'.
    """
    records = []
    start_time = time.time()

    for idx, (win, lbl) in enumerate(zip(windows, labels)):
        feat = extract_features_from_window(
            win,
            transform_type=transform_type,
            feature_set=feature_set,
            sampling_rate=sampling_rate
        )
        feat["label"] = int(lbl)
        records.append(feat)

    elapsed = time.time() - start_time
    df = pd.DataFrame(records)
    print(f"[Feature Extraction] Transformed {len(windows)} windows using '{transform_type.upper()}' in {elapsed:.2f}s.")
    return df


if __name__ == "__main__":
    print("Testing feature_extraction.py...")
    test_wins = [np.sin(np.arange(100)), np.cos(np.arange(100))]
    test_lbls = [0, 1]
    df_fswt = build_feature_dataset(test_wins, test_lbls, transform_type="fswt", feature_set="all")
    print(df_fswt.head())
    assert df_fswt.shape == (2, 7)  # 6 features + 1 label
    print("Feature extraction pipeline successfully verified!")
