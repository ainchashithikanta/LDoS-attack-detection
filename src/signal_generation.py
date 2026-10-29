"""
Module: signal_generation.py
Purpose: Converts packet-level / timestamped network captures into a numerical
         time-series signal (e.g., Traffic in Mbps or Packets/sec), and provides
         controlled synthetic traffic signal generation based on the LDoS pulse model
         from the original research paper.

Background for students:
A Low-rate Denial-of-Service (LDoS) attack periodically injects short, high-rate bursts
of traffic designed to throttle TCP's Retransmission Timeout (RTO) mechanism while maintaining
a low long-term average rate:
    R_avg = (L / T) * R_burst
where:
    - T: Attack period (seconds)
    - L: Burst duration (seconds), where L < T
    - R_burst: Peak attack burst rate (Mbps)
"""

import numpy as np
import pandas as pd


def generate_benchmark_ldos_signals(
    duration: float = 200.0,
    sampling_rate: int = 10,
    attack_period: float = 15.0,
    attack_duration: float = 5.0,
    burst_rate: float = 7.0,
    baseline_traffic: float = 10.0,
    noise_std: float = 0.5,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic normal and LDoS attack traffic time-series signals
    matching the original research paper and repository benchmark.

    Parameters:
    -----------
    duration : float
        Total duration in seconds (default: 200.0s).
    sampling_rate : int
        Number of samples per second (default: 10 Hz).
    attack_period : float
        Period T of the LDoS pulse train in seconds (default: 15.0s).
    attack_duration : float
        Duration L of the active attack pulse in seconds (default: 5.0s).
    burst_rate : float
        Magnitude R of the attack burst rate in Mbps (default: 7.0 Mbps).
    baseline_traffic : float
        Normal background traffic mean rate in Mbps (default: 10.0 Mbps).
    noise_std : float
        Standard deviation of background Gaussian noise (default: 0.5 Mbps).
    random_seed : int
        Seed for reproducibility (default: 42).

    Returns:
    --------
    normal_df : pd.DataFrame
        DataFrame with columns ['Time (s)', 'Traffic (Mbps)', 'label'] where label=0.
    ldos_df : pd.DataFrame
        DataFrame with columns ['Time (s)', 'Traffic (Mbps)', 'label'] where label=1.
    """
    num_samples = int(duration * sampling_rate)
    time = np.arange(num_samples) / sampling_rate

    # Generate normal background traffic with Gaussian fluctuations
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(0, noise_std, size=num_samples)
    normal_traffic = np.maximum(0.0, baseline_traffic + noise)

    # Generate LDoS attack traffic: baseline + periodic square pulses
    ldos_traffic = normal_traffic.copy()
    for i in range(num_samples):
        current_time = time[i]
        position_in_period = current_time % attack_period
        if position_in_period < attack_duration:
            ldos_traffic[i] += burst_rate

    normal_df = pd.DataFrame({
        "Time (s)": time,
        "Traffic (Mbps)": normal_traffic,
        "label": 0
    })

    ldos_df = pd.DataFrame({
        "Time (s)": time,
        "Traffic (Mbps)": ldos_traffic,
        "label": 1
    })

    return normal_df, ldos_df


def pcap_to_time_series(
    pcap_path: str,
    time_bin: float = 0.1,
    max_duration: float = 300.0
) -> pd.DataFrame:
    """
    Converts a raw network packet capture (.pcap) file into a discrete
    time-series signal of traffic rate in Mbps.

    Methodology:
    1. Read packet timestamps and packet byte lengths.
    2. Bin packets into discrete time intervals of length `time_bin` (default 0.1s for 10 Hz).
    3. Sum bytes in each bin and convert to Megabits per second (Mbps):
           Rate (Mbps) = (bytes * 8) / (time_bin * 1,000,000)

    Parameters:
    -----------
    pcap_path : str
        Path to the .pcap file.
    time_bin : float
        Bin width in seconds (0.1s corresponds to 10 Hz sampling rate).
    max_duration : float
        Maximum seconds of traffic to process (caps processing time on large files).

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with ['Time (s)', 'Traffic (Mbps)'].
    """
    from scapy.utils import RawPcapReader
    from scapy.layers.l2 import Ether

    timestamps = []
    sizes = []

    start_time = None
    for pkt_data, pkt_metadata in RawPcapReader(str(pcap_path)):
        # Support both standard PCAP (sec, usec) and PCAP-NG (tshigh, tslow, tsresol)
        if hasattr(pkt_metadata, "sec"):
            ts = float(pkt_metadata.sec) + float(pkt_metadata.usec) / 1e6
        elif hasattr(pkt_metadata, "tshigh") and hasattr(pkt_metadata, "tslow"):
            resol = float(getattr(pkt_metadata, "tsresol", 1_000_000))
            ts = ((pkt_metadata.tshigh << 32) | pkt_metadata.tslow) / resol
        else:
            ts = float(pkt_metadata[0])

        if start_time is None:
            start_time = ts

        rel_time = ts - start_time
        if rel_time > max_duration:
            break

        timestamps.append(rel_time)
        sizes.append(pkt_metadata.wirelen)

    if not timestamps:
        # Fallback empty dataframe
        return pd.DataFrame(columns=["Time (s)", "Traffic (Mbps)"])

    max_t = min(max(timestamps), max_duration)
    num_bins = int(np.ceil(max_t / time_bin))
    time_bins = np.arange(num_bins + 1) * time_bin
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2.0

    # Aggregate packet sizes into bins
    byte_counts, _ = np.histogram(timestamps, bins=time_bins, weights=sizes)
    # Convert bytes to Mbps: (bytes * 8) / (time_bin * 1,000,000)
    traffic_mbps = (byte_counts * 8.0) / (time_bin * 1_000_000.0)

    return pd.DataFrame({
        "Time (s)": bin_centers,
        "Traffic (Mbps)": traffic_mbps
    })
