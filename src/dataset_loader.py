"""
Module: dataset_loader.py
Purpose: Handles downloading, loading, and validating datasets for Low-rate
         Denial-of-Service (LDoS) attack detection.

Supported Datasets:
1. Benchmark LDoS Dataset:
   - Parametric pulse-train LDoS traffic matching the research model by
     Xiaocai Wang et al. (2024, Expert Systems with Applications).
   - Contains normal traffic (10 Mbps baseline) and LDoS attack traffic
     (bursts of +7 Mbps every 15s with pulse duration 5s).
2. UTSA-2021 Real SDN LDoS Attack Dataset:
   - Hosted publicly by the University of Texas at San Antonio (UTSA)
     Systems and Networks Lab:
     https://github.com/utsanetsys/UTSA-2021-Low-rate-DoS-Attack
   - Features real testbed traffic: SYN flood bursts at the first 0.1s
     of every second for 100s, followed by 100s quiet intervals, alongside
     benign web client downloads.
"""

import sys
import zipfile
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

import config


def download_utsa_dataset(target_dir: Path = config.UTSA_RAW_DIR) -> dict:
    """
    Downloads and extracts the UTSA-2021 Low-rate DoS dataset archives
    directly from GitHub into the specified directory.

    Parameters:
    -----------
    target_dir : Path
        Directory where raw zip files and extracted PCAPs will be saved.

    Returns:
    --------
    status : dict
        Dictionary reporting downloaded files and extraction paths.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "attack_pcap": config.UTSA_ATTACK_ZIP_URL,
        "benign_pcap": config.UTSA_BENIGN_ZIP_URL,
    }

    downloaded = {}
    for key, url in urls.items():
        zip_path = target_dir / f"{key}.zip"
        if not zip_path.exists():
            print(f"[Dataset Loader] Downloading {key} from {url}...")
            try:
                urllib.request.urlretrieve(url, zip_path)
                print(f"[Dataset Loader] Downloaded: {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
            except Exception as e:
                print(f"[Dataset Loader] Download failed for {url}: {e}")
                downloaded[key] = None
                continue
        else:
            print(f"[Dataset Loader] Existing archive found: {zip_path.name}")

        # Extract archive
        extract_folder = target_dir / key
        if not extract_folder.exists():
            print(f"[Dataset Loader] Extracting {zip_path.name}...")
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(extract_folder)
                print(f"[Dataset Loader] Extracted to {extract_folder}")
            except Exception as e:
                print(f"[Dataset Loader] Extraction failed for {zip_path.name}: {e}")
                downloaded[key] = None
                continue

        downloaded[key] = extract_folder

    return downloaded


def load_benchmark_dataset(
    normal_csv_path: Path = config.BENCHMARK_NORMAL_CSV,
    ldos_csv_path: Path = config.BENCHMARK_LDOS_CSV
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and validates the baseline benchmark normal and LDoS CSV files.
    If the files do not exist, they are generated using the research paper parameters.

    Parameters:
    -----------
    normal_csv_path : Path
        Path to normal traffic CSV file.
    ldos_csv_path : Path
        Path to LDoS attack traffic CSV file.

    Returns:
    --------
    normal_df : pd.DataFrame
        Validated normal traffic DataFrame.
    ldos_df : pd.DataFrame
        Validated LDoS attack traffic DataFrame.
    """
    from src.signal_generation import generate_benchmark_ldos_signals

    normal_csv_path = Path(normal_csv_path)
    ldos_csv_path = Path(ldos_csv_path)

    if not normal_csv_path.exists() or not ldos_csv_path.exists():
        print("[Dataset Loader] Benchmark files missing. Generating from paper model...")
        normal_df, ldos_df = generate_benchmark_ldos_signals()
        normal_csv_path.parent.mkdir(parents=True, exist_ok=True)
        normal_df.to_csv(normal_csv_path, index=False)
        ldos_df.to_csv(ldos_csv_path, index=False)
        print(f"[Dataset Loader] Saved benchmark data to {normal_csv_path.parent}")
    else:
        normal_df = pd.read_csv(normal_csv_path)
        ldos_df = pd.read_csv(ldos_csv_path)

    validate_dataframe(normal_df, expected_label=config.LABEL_NORMAL, dataset_name="Benchmark Normal")
    validate_dataframe(ldos_df, expected_label=config.LABEL_LDOS, dataset_name="Benchmark LDoS")

    return normal_df, ldos_df


def load_utsa_dataset(
    target_dir: Path = config.UTSA_RAW_DIR,
    max_duration: float = 200.0,
    time_bin: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and parses the real UTSA SDN LDoS dataset into continuous
    numerical time-series DataFrames.

    Parameters:
    -----------
    target_dir : Path
        Directory where UTSA dataset was extracted.
    max_duration : float
        Duration in seconds to extract from each trace (default 200s).
    time_bin : float
        Binning interval in seconds (0.1s for 10 Hz sampling rate).

    Returns:
    --------
    normal_df : pd.DataFrame
        Parsed benign traffic DataFrame with label 0.
    ldos_df : pd.DataFrame
        Parsed LDoS attack traffic DataFrame with label 1.
    """
    from src.signal_generation import pcap_to_time_series

    target_dir = Path(target_dir)
    # Ensure downloaded
    download_utsa_dataset(target_dir)

    # Locate benign PCAP
    benign_pcaps = list(target_dir.glob("benign_pcap/**/*.pcap"))
    # Locate attack PCAP
    attack_pcaps = list(target_dir.glob("attack_pcap/**/*.pcap"))

    if not benign_pcaps or not attack_pcaps:
        print("[Dataset Loader] Warning: UTSA PCAPs not found on disk. Falling back to benchmark dataset.")
        return load_benchmark_dataset()

    print(f"[Dataset Loader] Processing real benign PCAP: {benign_pcaps[0].name}")
    normal_df = pcap_to_time_series(benign_pcaps[0], time_bin=time_bin, max_duration=max_duration)
    normal_df["label"] = config.LABEL_NORMAL

    print(f"[Dataset Loader] Processing real LDoS attack PCAP: {attack_pcaps[0].name}")
    ldos_df = pcap_to_time_series(attack_pcaps[0], time_bin=time_bin, max_duration=max_duration)
    ldos_df["label"] = config.LABEL_LDOS

    validate_dataframe(normal_df, expected_label=config.LABEL_NORMAL, dataset_name="UTSA Real Benign")
    validate_dataframe(ldos_df, expected_label=config.LABEL_LDOS, dataset_name="UTSA Real LDoS")

    return normal_df, ldos_df


def validate_dataframe(df: pd.DataFrame, expected_label: int = None, dataset_name: str = "Dataset") -> bool:
    """
    Validates that a traffic DataFrame contains required columns, valid numerical values,
    finite data points, and matches the expected label.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate.
    expected_label : int, optional
        Expected ground truth label (0 for Normal, 1 for LDoS).
    dataset_name : str
        Human-readable name of dataset for error messages.

    Returns:
    --------
    valid : bool
        True if valid, raises ValueError otherwise.
    """
    required_columns = ["Time (s)", "Traffic (Mbps)", "label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"[{dataset_name}] Missing required column '{col}'. Present: {df.columns.tolist()}")

    if len(df) == 0:
        raise ValueError(f"[{dataset_name}] DataFrame is empty (0 rows).")

    # Check for NaN / null
    nan_count = df[["Time (s)", "Traffic (Mbps)"]].isna().sum().sum()
    if nan_count > 0:
        raise ValueError(f"[{dataset_name}] Contains {nan_count} missing (NaN) values.")

    # Check for Inf
    traffic_vals = df["Traffic (Mbps)"].to_numpy()
    if not np.isfinite(traffic_vals).all():
        raise ValueError(f"[{dataset_name}] Contains non-finite (Inf/-Inf) values.")

    # Check label integrity
    labels = df["label"].unique()
    if expected_label is not None and not np.all(labels == expected_label):
        raise ValueError(f"[{dataset_name}] Expected label {expected_label}, but found labels: {labels}")

    return True


if __name__ == "__main__":
    print("Testing dataset_loader.py...")
    norm_df, att_df = load_benchmark_dataset()
    print("Benchmark Normal shape:", norm_df.shape)
    print("Benchmark LDoS shape:", att_df.shape)
    print("Dataset loader successfully verified!")
