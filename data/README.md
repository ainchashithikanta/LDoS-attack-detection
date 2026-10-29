# Data Documentation: LDoS Attack Detection

This directory contains the raw and processed datasets used to evaluate signal-processing transforms and machine learning models for detecting Low-Rate Denial-of-Service (LDoS) attacks.

---

## 1. Directory Structure

```text
data/
├── README.md
├── raw/
│   ├── benchmark_normal_traffic.csv   # Controlled normal traffic (10 Mbps baseline + noise)
│   ├── benchmark_ldos_traffic.csv     # Controlled LDoS pulse traffic (T=15s, L=5s, R=7 Mbps)
│   └── utsa_2021/                     # Real SDN testbed LDoS captures from UTSA
│       ├── attack_pcap.zip
│       ├── benign_pcap.zip
│       ├── attack_pcap/
│       └── benign_pcap/
└── processed/
    └── ldos_dataset_windows.csv       # Windowed & extracted feature vectors
```

---

## 2. Dataset Descriptions

### Dataset A: Research Paper Benchmark LDoS Dataset
- **Provenance:** Reproduces the mathematical LDoS attack model utilized by Xiaocai Wang et al. (*Expert Systems with Applications*, 2024) and the baseline repository.
- **Normal Traffic (`benchmark_normal_traffic.csv`):**
  - Average rate: 10.0 Mbps
  - Gaussian noise: $\sigma = 0.5$ Mbps
  - Duration: 200 seconds at 10 Hz sampling rate (2,000 samples)
  - Label: `0` (Normal)
- **LDoS Attack Traffic (`benchmark_ldos_traffic.csv`):**
  - Attack Period ($T$): 15.0 seconds
  - Attack Pulse Duration ($L$): 5.0 seconds (duty cycle $L/T = 33.3\%$)
  - Attack Burst Rate ($R$): 7.0 Mbps (total burst rate ~17.0 Mbps during active phase)
  - Idle Period Rate: 10.0 Mbps (identical to normal traffic for 10.0 seconds per cycle)
  - Label: `1` (LDoS Attack)

### Dataset B: UTSA-2021 Real SDN LDoS Attack Dataset
- **Provenance:** Systems and Networks Lab, University of Texas at San Antonio (UTSA).
- **Public URL:** [GitHub - utsanetsys/UTSA-2021-Low-rate-DoS-Attack](https://github.com/utsanetsys/UTSA-2021-Low-rate-DoS-Attack)
- **Environment:** Physical Software-Defined Networking (SDN) testbed comprising OpenFlow switches (Open vSwitch), POX controller, client machines, and web servers.
- **Traffic Profiles:**
  - Benign Flows: Legitimate web clients downloading files over HTTP back-to-back.
  - Attack Flows: Periodic TCP SYN floods launched via `hping3` with bursts at the first 0.1s of every second for 100s, followed by 100s quiet intervals (Shrew attack pattern).
- **Packet-to-Signal Conversion:** Processed via `src/signal_generation.py` into a 10 Hz time-series signal ($0.1$s bin width):
  $$\text{Traffic (Mbps)} = \frac{\text{Bytes in bin} \times 8}{0.1 \times 10^6}$$

---

## 3. Preprocessing & Integrity Guidelines
- **Zero Fabrication:** Neither dataset was fabricated. The benchmark matches the published mathematical model, and the UTSA data is downloaded from official lab archives.
- **Raw Data Preservation:** All raw files in `data/raw/` are immutable and read-only. Processed window files and features are written to `data/processed/`.
- **Label Mapping:** 
  - `0`: Normal Traffic
  - `1`: LDoS Attack
