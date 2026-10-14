# Experiment A: Normal vs LDoS in Time Domain
**Member 4 — Experimental Design & Evaluation**  
*Project: Low-rate Denial of Service (LDoS) Attack Detection*

---

## 1. Experimental Roadmap Context

As designed in the research project evaluation matrix, the study compares 5 distinct detection paradigms:

| Experiment | Method | Domain | Status |
| :--- | :--- | :--- | :--- |
| **Experiment A** | **Normal vs LDoS in Time Domain** | **Time Domain** | **Completed** |
| Experiment B | FFT-Based Detection | Frequency Domain | Next Milestone |
| Experiment C | STFT-Based Detection | Time-Frequency Domain | Planned |
| Experiment D | Wavelet-Based Detection (DWT / CWT) | Multiresolution Domain | Planned |
| Experiment E | FSWT + 4 Features + Decision Tree | Adaptive Time-Frequency + ML | Benchmark Pipeline |

---

## 2. Theoretical Motivation for Experiment A

A **Low-rate Denial of Service (LDoS)** attack periodically injects short, high-rate packet bursts (pulse duration $L$) at a periodic interval $T$, throttling TCP congestion control mechanisms while keeping the long-term average rate low:

$$\bar{R}_{\text{attack}} = \frac{L}{T} \cdot R$$

In this benchmark dataset:
- Background Normal Traffic: $\mu \approx 10\text{ Mbps}$, $\sigma \approx 0.5\text{ Mbps}$
- Attack Parameters: Period $T = 15.0\text{ s}$, Pulse Duration $L = 5.0\text{ s}$, Burst Rate $R = 7.0\text{ Mbps}$
- Active Pulse Traffic: $\sim 17\text{ Mbps}$ (during $0 \le t \pmod T < L$)
- Idle Period Traffic: $\sim 10\text{ Mbps}$ (during $L \le t \pmod T < T$)

### Why Time-Domain Detection is Challenging:
1. **Idle Phase Invisibility**: During $T - L = 10\text{ s}$ of each cycle (66.7% of the time), the attack traffic is mathematically identical to normal background traffic.
2. **False Negatives in Instantaneous Detectors**: Any point-by-point detector fails during the idle intervals unless sliding window aggregation is employed.
3. **Stealthy Attack Degradation**: If an adversary reduces $R$ closer to $3\sigma_{\text{noise}}$, time-domain thresholding completely fails, necessitating spectral and wavelet methods (Experiments B–E).

---

## 3. Evaluated Time-Domain Detectors

| Detector | Time Scope | Decision Rule | Description |
| :--- | :--- | :--- | :--- |
| **Instantaneous 3-Sigma** | Point-by-point ($0.1\text{s}$) | $x(t) > \mu_{\text{norm}} + 3\sigma_{\text{norm}}$ ($> 11.50\text{ Mbps}$) | Classical statistical anomaly detector |
| **Window Mean Threshold** | 5.0s window ($50$ samples) | $\bar{x}_w > \theta_{\text{mean}}$ ($> 11.20\text{ Mbps}$) | Average rate over sliding window |
| **Window Std Dev Threshold** | 5.0s window ($50$ samples) | $\sigma_w > \theta_{\sigma}$ ($> 1.19\text{ Mbps}$) | Dispersion capturing burst transitions |
| **Window Peak Rate Threshold** | 5.0s window ($50$ samples) | $\max(x_w) > \theta_{\text{peak}}$ ($> 13.59\text{ Mbps}$) | Peak traffic rate within the window |
| **Window Peak-to-Average (PAR)** | 5.0s window ($50$ samples) | $\frac{\max(x_w)}{\bar{x}_w} > \theta_{\text{PAR}}$ ($> 1.17$) | Crest factor / Burstiness index |
| **Window Autocorr Subpeak** | 5.0s window ($50$ samples) | $\max_{\tau \ge 2} R_{xx}(\tau) > \theta_{\text{acf}}$ ($> 0.37$) | Windowed temporal correlation |

---

## 4. Evaluation Metrics (Strict Definitions)

In accordance with the evaluation requirements:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$F1 = \frac{2 \cdot P \cdot R}{P + R}$$

Where:
- **TP (True Positive)**: LDoS attack correctly flagged as LDoS.
- **TN (True Negative)**: Normal traffic correctly identified as Normal.
- **FP (False Positive)**: Normal traffic falsely alarmed as an attack.
- **FN (False Negative)**: LDoS attack traffic misclassified as Normal.

---

## 5. Experimental Results Summary

| Detector | Domain | TP | TN | FP | FN | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Instantaneous 3-Sigma** | Point-by-Point ($0.1\text{s}$) | 702 | 1996 | 4 | 1298 | **67.45%** | 99.43% | 35.10% | 0.5188 |
| **Window Mean Threshold** | Windowed ($5\text{s}$) | 118 | 196 | 0 | 78 | **80.10%** | 100.00% | 60.20% | 0.7516 |
| **Window Std Dev Threshold** | Windowed ($5\text{s}$) | 104 | 196 | 0 | 92 | **76.53%** | 100.00% | 53.06% | 0.6933 |
| **Window Peak Threshold** | Windowed ($5\text{s}$) | 118 | 196 | 0 | 78 | **80.10%** | 100.00% | 60.20% | 0.7516 |
| **Window PAR Threshold** | Windowed ($5\text{s}$) | 84 | 191 | 5 | 112 | **70.15%** | 94.38% | 42.86% | 0.5895 |
| **Window Autocorr Subpeak** | Windowed ($5\text{s}$) | 104 | 194 | 2 | 92 | **76.02%** | 98.11% | 53.06% | 0.6887 |

### Key Takeaways:
1. **Instantaneous Detection Flaw**: While Precision is 99.43%, Recall is only **35.10%** because during the 10-second idle period between bursts, attack traffic drops back to $\sim 10\text{ Mbps}$, causing 1,298 False Negatives.
2. **Windowing Advantage**: Windowed statistical features boost Accuracy to **80.10%** and F1-score to **0.7516** by capturing the aggregate impact of bursts.
3. **Upper Bound of Time Domain**: Even with windowing, time-domain features cannot easily differentiate quiescent attack phases without broad sliding windows that introduce significant detection latency.

---

## 6. Generated Visualizations

All generated visualization figures are stored under [`results/`](file:///C:/Users/Pavan/OneDrive/Desktop/project/results/):

1. **[`experiment_a_time_domain_signals.png`](file:///C:/Users/Pavan/OneDrive/Desktop/project/results/experiment_a_time_domain_signals.png)**:
   - Full 200-second time series of Normal vs LDoS traffic.
   - Zoomed-in 45-second window highlighting the 3 periodic square-wave attack pulses ($T=15\text{s}, L=5\text{s}$).
2. **[`experiment_a_statistical_distributions.png`](file:///C:/Users/Pavan/OneDrive/Desktop/project/results/experiment_a_statistical_distributions.png)**:
   - Probability density histogram showing unimodal normal distribution vs bimodal LDoS distribution (idle state at 10 Mbps and burst state at 17 Mbps).
   - Boxplots demonstrating dispersion and variance explosion.
3. **[`experiment_a_sliding_window_features.png`](file:///C:/Users/Pavan/OneDrive/Desktop/project/results/experiment_a_sliding_window_features.png)**:
   - Time-varying trajectories of Window Mean, Window Standard Deviation, and Peak-to-Average Ratio.
4. **[`experiment_a_autocorrelation.png`](file:///C:/Users/Pavan/OneDrive/Desktop/project/results/experiment_a_autocorrelation.png)**:
   - Autocorrelation Function ($R_{xx}(\tau)$) showing zero-correlation noise for Normal traffic vs prominent periodic peaks at lag $\tau = 15.0\text{s}$ and $2\tau = 30.0\text{s}$ for LDoS traffic.
5. **[`experiment_a_confusion_matrices.png`](file:///C:/Users/Pavan/OneDrive/Desktop/project/results/experiment_a_confusion_matrices.png)**:
   - 2x3 panel of labeled confusion matrices (TP, TN, FP, FN) with accuracy and F1 annotations.
6. **[`experiment_a_roc_curves.png`](file:///C:/Users/Pavan/OneDrive/Desktop/project/results/experiment_a_roc_curves.png)**:
   - ROC curves (True Positive Rate vs False Positive Rate) and Precision-Recall curves with Area Under Curve (AUC) for each time-domain feature.
7. **[`experiment_a_sensitivity_analysis.png`](file:///C:/Users/Pavan/OneDrive/Desktop/project/results/experiment_a_sensitivity_analysis.png)**:
   - Demonstrates the collapse of time-domain detection accuracy and F1 when burst rate $R$ drops below $2.0\text{ Mbps}$, proving why frequency and wavelet transforms (Experiments B–E) are indispensable.

---

## 7. Deliverable Files

As requested in the Member 4 deliverables:
- [`results/experiment_a_metrics.csv`](file:///C:/Users/Pavan/OneDrive/Desktop/project/results/experiment_a_metrics.csv): Complete metrics table in CSV format.
- [`results/final_results.xlsx`](file:///C:/Users/Pavan/OneDrive/Desktop/project/results/final_results.xlsx): Multi-tab Excel workbook containing:
  - Tab 1: `Experiment A` (evaluation metrics for each detector)
  - Tab 2: `Summary Statistics` (mean, std, min, max, PAR, IQR)

---

## 8. How to Run

To run Experiment A:
```powershell
python experiment_a.py
```
or
```powershell
python experiments/experiment_a.py
```
