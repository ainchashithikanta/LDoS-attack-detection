"""
Configuration settings for the LDoS Attack Detection Project.

This module centralizes all paths, signal-processing parameters,
machine learning settings, and dataset URLs so that no hardcoded
magic numbers or paths are scattered throughout the codebase.
"""

from pathlib import Path

# Base project paths (using pathlib for Windows compatibility)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"

# Ensure output directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Signal processing & windowing parameters
SAMPLING_RATE = 10         # Samples per second (10 Hz)
WINDOW_SIZE = 100          # Number of samples per window (100 samples = 10 seconds)
WINDOW_STRIDE = 100        # Non-overlapping windows (stride = window size)

# FSWT (Frequency Slice Wavelet Transform) parameters (from original research paper)
FSWT_LAMBDA = 1.0          # Scaling parameter
FSWT_KAPPA = 23.5482       # Frequency scaling factor
LOW_FREQ_FRACTION = 0.10   # Fraction of sampling rate considered "low frequency" (fs / 10 = 1.0 Hz)

# Machine Learning parameters
RANDOM_STATE = 42          # Reproducible random seed
TEST_SIZE = 0.30           # 30% held-out test data
DT_MAX_DEPTH = 3           # Decision Tree maximum depth (prevents overfitting on small features)
RF_N_ESTIMATORS = 100      # Random Forest number of trees
RF_MAX_DEPTH = 5           # Random Forest maximum tree depth

# Feature sets
ORIGINAL_FEATURES = [
    "contrast",
    "energy_ratio",
    "correlation",
    "entropy",
]

ALL_SIX_FEATURES = [
    "contrast",
    "energy_ratio",
    "correlation",
    "entropy",
    "spectral_centroid",
    "kurtosis",
]

# Ground truth class mapping
LABEL_NORMAL = 0
LABEL_LDOS = 1
CLASS_NAMES = ["Normal Traffic (0)", "LDoS Attack (1)"]

# Dataset configuration
# 1. Benchmark LDoS Dataset (from original paper model / synthetic traffic generator)
BENCHMARK_NORMAL_CSV = RAW_DATA_DIR / "benchmark_normal_traffic.csv"
BENCHMARK_LDOS_CSV = RAW_DATA_DIR / "benchmark_ldos_traffic.csv"

# 2. UTSA-2021 Real SDN LDoS Dataset (University of Texas at San Antonio)
# Repository: https://github.com/utsanetsys/UTSA-2021-Low-rate-DoS-Attack
UTSA_REPO_URL = "https://raw.githubusercontent.com/utsanetsys/UTSA-2021-Low-rate-DoS-Attack/main"
UTSA_ATTACK_ZIP_URL = f"{UTSA_REPO_URL}/1Client_PCAPs.zip"
UTSA_BENIGN_ZIP_URL = f"{UTSA_REPO_URL}/Benign_Only.zip"

UTSA_RAW_DIR = RAW_DATA_DIR / "utsa_2021"
PROCESSED_DATASET_CSV = PROCESSED_DATA_DIR / "ldos_dataset_windows.csv"
