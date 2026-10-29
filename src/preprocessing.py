"""
Module: preprocessing.py
Purpose: Performs principled signal and tabular preprocessing for LDoS detection:
         1. Validates and sanitizes missing/invalid values.
         2. Removes the DC component (centering the signal so the 0 Hz baseline does not dominate spectra).
         3. Segments continuous 1D traffic signals into fixed-length analysis windows.
         4. Provides train/test splitting that prevents data leakage across windows.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def sanitize_traffic_series(series: np.ndarray) -> np.ndarray:
    """
    Checks for NaN, Inf, or negative rates, replaces NaNs with 0, and clips negatives.

    Parameters:
    -----------
    series : np.ndarray
        Raw 1D traffic signal array.

    Returns:
    --------
    cleaned : np.ndarray
        Cleaned, finite 1D float array.
    """
    arr = np.asarray(series, dtype=float)

    # Replace non-finite with 0
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Traffic rates cannot be negative
    arr = np.maximum(arr, 0.0)
    return arr


def remove_dc_component(signal: np.ndarray) -> np.ndarray:
    """
    Removes the DC (Direct Current) component by subtracting the mean of the signal.

    In signal processing and time-frequency analysis, network traffic has a constant baseline
    (e.g., ~10 Mbps average rate). Subtracting the mean centers the fluctuations around zero,
    ensuring that the DC component (0 Hz) does not swamp the Fourier or Wavelet transform spectra.

    Parameters:
    -----------
    signal : np.ndarray
        1D signal array.

    Returns:
    --------
    centered_signal : np.ndarray
        Zero-mean 1D signal.
    """
    signal = np.asarray(signal, dtype=float)
    return signal - np.mean(signal)


def create_traffic_windows(
    traffic_signal: np.ndarray,
    label: int,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.WINDOW_STRIDE
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Slices a continuous 1D traffic signal into consecutive fixed-size windows.

    Parameters:
    -----------
    traffic_signal : np.ndarray
        1D array of traffic measurements (e.g., Mbps).
    label : int
        Ground truth label for this traffic stream (0 for Normal, 1 for LDoS).
    window_size : int
        Number of samples per window (default: 100 samples = 10s at 10 Hz).
    stride : int
        Step size between windows (default: window_size for non-overlapping windows).

    Returns:
    --------
    windows : List[np.ndarray]
        List of 1D numpy arrays, each of shape (window_size,).
    labels : List[int]
        List of integer labels corresponding to each window.
    """
    traffic_signal = sanitize_traffic_series(traffic_signal)
    windows = []
    labels = []

    num_samples = len(traffic_signal)
    for start in range(0, num_samples - window_size + 1, stride):
        win = traffic_signal[start : start + window_size]
        windows.append(win)
        labels.append(label)

    return windows, labels


def prepare_windowed_dataset(
    normal_df: pd.DataFrame,
    ldos_df: pd.DataFrame,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.WINDOW_STRIDE
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Takes normal and LDoS DataFrames, segments each into windows,
    and returns combined windows and labels.

    Parameters:
    -----------
    normal_df : pd.DataFrame
        DataFrame with normal traffic.
    ldos_df : pd.DataFrame
        DataFrame with LDoS attack traffic.
    window_size : int
        Samples per window.
    stride : int
        Step between windows.

    Returns:
    --------
    all_windows : List[np.ndarray]
        List of all traffic window arrays.
    all_labels : List[int]
        List of class labels (0 or 1).
    """
    normal_signal = normal_df["Traffic (Mbps)"].to_numpy()
    ldos_signal = ldos_df["Traffic (Mbps)"].to_numpy()

    norm_windows, norm_labels = create_traffic_windows(
        normal_signal, label=config.LABEL_NORMAL, window_size=window_size, stride=stride
    )
    ldos_windows, ldos_labels = create_traffic_windows(
        ldos_signal, label=config.LABEL_LDOS, window_size=window_size, stride=stride
    )

    all_windows = norm_windows + ldos_windows
    all_labels = norm_labels + ldos_labels

    return all_windows, all_labels


def temporal_split_by_class(
    total_normal_windows: int,
    total_ldos_windows: int,
    test_size: float = config.TEST_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partitions windows into train and test sets by preserving chronological order
    within each class (first 70% of Normal + first 70% of LDoS for training;
    last 30% of each for testing).

    This mirrors the methodology of the reference research repository (Experiment E),
    eliminating temporal data leakage and reflecting realistic chronological deployment.

    Parameters:
    -----------
    total_normal_windows : int
        Number of normal traffic windows.
    total_ldos_windows : int
        Number of LDoS attack traffic windows.
    test_size : float
        Fraction of trailing windows reserved for testing (default: 0.30).

    Returns:
    --------
    train_indices : np.ndarray
        Array of indices for training.
    test_indices : np.ndarray
        Array of indices for testing.
    """
    n_norm_train = int(np.round(total_normal_windows * (1.0 - test_size)))
    n_ldos_train = int(np.round(total_ldos_windows * (1.0 - test_size)))

    norm_train = np.arange(0, n_norm_train)
    norm_test = np.arange(n_norm_train, total_normal_windows)

    ldos_offset = total_normal_windows
    ldos_train = np.arange(ldos_offset, ldos_offset + n_ldos_train)
    ldos_test = np.arange(ldos_offset + n_ldos_train, ldos_offset + total_ldos_windows)

    train_indices = np.concatenate([norm_train, ldos_train])
    test_indices = np.concatenate([norm_test, ldos_test])

    return train_indices, test_indices


def split_train_test(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits features and labels into training and held-out test sets.
    Uses stratification by default to ensure both Normal and LDoS classes
    are represented proportionally in train and test sets.

    Parameters:
    -----------
    features : np.ndarray
        2D matrix of shape (n_samples, n_features).
    labels : np.ndarray
        1D array of class labels.
    test_size : float
        Fraction of data reserved for testing (default: 0.30).
    random_state : int
        Seed for reproducibility (default: 42).
    stratify : bool
        Whether to stratify split by class label.

    Returns:
    --------
    X_train, X_test, y_train, y_test : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    strat = labels if stratify else None
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )


if __name__ == "__main__":
    print("Testing preprocessing.py...")
    test_sig = np.array([10.0, 11.0, 9.0, 10.5, 12.0, 8.5] * 20)
    wins, lbls = create_traffic_windows(test_sig, label=0, window_size=10, stride=10)
    print(f"Created {len(wins)} windows of length {len(wins[0])}.")
    assert len(wins[0]) == 10
    print("Preprocessing module successfully verified!")
