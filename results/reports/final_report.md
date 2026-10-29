# LDoS Attack Detection Project: Final Experimental Report

**Project Title:** Low-Rate Denial-of-Service (LDoS) Attack Detection Using Signal Transforms and Machine Learning  
**Author:** Student Implementation & Extended Framework  
**Reference Paper:** Xiaocai Wang, Dan Tang, Ye Feng, Zheng Qin, Bing Xiong, and Yufeng Liu, *"An LDoS attack detection method based on FSWT time–frequency distribution,"* *Expert Systems with Applications*, Vol. 256, 2024.

---

## 1. Executive Summary & Problem Formulation

A **Low-rate Denial of Service (LDoS)** attack periodically transmits short, high-rate bursts of packets at a constant interval $T$ with burst duration $L$, throttling TCP's Retransmission Timeout (RTO) congestion control mechanism:
$$\bar{R}_{\text{attack}} = \frac{L}{T} \cdot R_{\text{burst}}$$
Because the overall average rate $\bar{R}_{\text{attack}}$ is low and identical to normal background traffic during the $(T - L)$ idle period, traditional volume-based threshold anomaly detectors completely fail.

This project extends the state-of-the-art FSWT + Decision Tree baseline by systematically incorporating:
1. **Two Extra Transforms:** Short-Time Fourier Transform (STFT) and Continuous Wavelet Transform (CWT).
2. **Two Extra Features:** Spectral Centroid and Kurtosis.
3. **One Extra Classifier:** Random Forest ensemble.

---

## 2. Dataset Specifications

- **Dataset Source:** Benchmark LDoS Pulse Dataset (Wang et al. 2024 model)
- **Sampling Frequency ($f_s$):** 10 Hz (1 sample = 0.1s)
- **Window Length ($N$):** 100 samples per window (10.0 seconds per window)
- **Window Stride:** 100 samples (non-overlapping windows to avoid temporal leakage)
- **Total Windows:** 40
- **Class Distribution:**
  - Normal Traffic (Class 0): 20 windows
  - LDoS Attack (Class 1): 20 windows
- **Train/Test Split:** 70% Training / 30% Held-Out Testing (stratified, reproducible seed = 42)

---

## 3. Mathematical Feature Definitions

1. **Contrast (Original):**
   $$\text{Contrast} = \sum_{i, j} (i - j)^2 P(i, j)$$
   Measures local intensity gradients in the normalized 2D time-frequency energy distribution $P(i, j) = E(i, j) / \sum E$.
2. **Energy Ratio (Original):**
   $$\text{Energy Ratio} = \frac{\sum_{|f| \le f_s/10} E(f, t)}{\sum_{all} E(f, t)}$$
   Quantifies the proportion of energy concentrated in the low-frequency band (<= 1.0 Hz) characteristic of periodic attack repetitions.
3. **Correlation (Original):**
   $$\text{Correlation} = \frac{\sum_{i, j} i \cdot j \cdot P(i, j) - \mu_1 \mu_2}{\sqrt{\sigma_1^2 \sigma_2^2} + \epsilon}$$
   Evaluates linear dependency and temporal-spectral alignment across time-frequency bins.
4. **Entropy (Original):**
   $$\text{Entropy} = -\sum_{P(i, j) > 0} P(i, j) \ln(P(i, j))$$
   Measures dispersion and uncertainty; LDoS harmonic concentration causes entropy to drop significantly.
5. **Spectral Centroid (Added):**
   $$\text{Spectral Centroid} = \frac{\sum_k |f_k| \cdot \sum_t E(f_k, t)}{\sum_{k, t} E(f_k, t)}$$
   Represents the frequency center of mass; drops toward low frequencies during attack pulses.
6. **Kurtosis (Added):**
   $$\text{Kurtosis} = \frac{\frac{1}{N} \sum (z_t - \mu)^4}{\sigma^4} - 3.0$$
   Quantifies peakedness and heavy-tailed impulsive burst spikes in the temporal energy profile.

---

## 4. Transform Configurations

- **FSWT (Frequency Slice Wavelet Transform):** Gaussian FSF $\hat{p}(\omega) = \exp(-0.5 \omega^2)$, $\sigma = |f| / \kappa$ with $\kappa = 23.5482$, $\lambda = 1.0$.
- **STFT (Short-Time Fourier Transform):** Segment length 32, overlap 24, Hann window, 10 Hz sampling rate.
- **CWT (Continuous Wavelet Transform):** Morlet mother wavelet (`morl`), 32 scales mapped to pseudo-frequencies.

---

## 5. Experimental Results Comparison

| Experiment | Accuracy | Precision | Recall | F1-Score | FPR | TP | TN | FP | FN | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: FSWT + 4 Feats + Decision Tree** | 83.33% | 100.00% | 66.67% | 0.8000 | 0.00% | 4 | 6 | 0 | 2 | 24.70 ms |
| **Exp 2: FSWT + 6 Feats + Decision Tree** | 83.33% | 100.00% | 66.67% | 0.8000 | 0.00% | 4 | 6 | 0 | 2 | 1.02 ms |
| **Exp 3: STFT + 6 Feats + Decision Tree** | 83.33% | 100.00% | 66.67% | 0.8000 | 0.00% | 4 | 6 | 0 | 2 | 0.98 ms |
| **Exp 4: CWT + 6 Feats + Decision Tree** | 83.33% | 100.00% | 66.67% | 0.8000 | 0.00% | 4 | 6 | 0 | 2 | 0.91 ms |
| **Exp 5: FSWT + 6 Feats + Random Forest** | 83.33% | 100.00% | 66.67% | 0.8000 | 0.00% | 4 | 6 | 0 | 2 | 88.16 ms |


---

## 6. Key Analysis & Educational Takeaways

1. **Baseline Reproduction:** Experiment 1 reproduces the research paper's original FSWT + 4 features + Decision Tree baseline.
2. **Impact of Added Features:** Experiment 2 examines whether incorporating Spectral Centroid and Kurtosis improves discrimination of pulse bursts without increasing false alarms.
3. **Transform Comparison:** Experiments 2, 3, and 4 compare FSWT against STFT and CWT under the identical six-feature set and held-out test split, isolating the effect of time-frequency resolution.
4. **Classifier Comparison:** Experiment 5 evaluates whether an ensemble Random Forest further enhances robustness, precision, and stability over a single Decision Tree.
5. **No Data Leakage:** All scalers and feature transformers are fitted strictly on training data; all reported metrics represent genuine unseen test-set evaluations.

---

*Report automatically generated by `evaluation.py` on 2026-10-29 17:53:21.*
