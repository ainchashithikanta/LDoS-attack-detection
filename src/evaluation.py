"""
Module: evaluation.py
Purpose: Evaluates model performance on held-out test data, computes standard
         cybersecurity metrics, exports metrics to CSV, and generates reports.

Evaluation Metrics Definitions:
-------------------------------
Positive class = LDoS Attack (label 1)
Negative class = Normal Traffic (label 0)

1. True Positive (TP) : LDoS attack correctly detected as LDoS.
2. True Negative (TN) : Normal traffic correctly identified as Normal.
3. False Positive (FP): Normal traffic falsely flagged as an attack (False Alarm).
4. False Negative (FN): LDoS attack missed and classified as Normal (Missed Detection).

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)   [zero_division=0]
- Recall = TP / (TP + FN)      [zero_division=0]
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
- False Positive Rate (FPR) = FP / (FP + TN)
"""

from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

import config


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    train_time: float = 0.0,
    inference_time: float = 0.0,
    experiment_name: str = "Experiment"
) -> Dict[str, Any]:
    """
    Computes all classification metrics for a model's test predictions.

    Parameters:
    -----------
    y_true : np.ndarray
        True ground truth labels (0 or 1).
    y_pred : np.ndarray
        Predicted labels (0 or 1).
    y_prob : np.ndarray, optional
        Predicted probabilities for class 1.
    train_time : float
        Elapsed model training time in seconds.
    inference_time : float
        Elapsed test inference time in seconds.
    experiment_name : str
        Human-readable name for the experiment.

    Returns:
    --------
    metrics : Dict[str, Any]
        Dictionary of all computed metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Confusion matrix with explicit labels [0, 1]
    # cm[0, 0] = TN, cm[0, 1] = FP, cm[1, 0] = FN, cm[1, 1] = TP
    cm = confusion_matrix(y_true, y_pred, labels=[config.LABEL_NORMAL, config.LABEL_LDOS])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # ROC AUC
    auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None

    return {
        "experiment": experiment_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "roc_auc": auc if auc is not None else np.nan,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "train_time_sec": float(train_time),
        "inference_time_sec": float(inference_time),
        "test_samples": int(len(y_true)),
    }


def save_metrics_to_csv(
    metrics_list: List[Dict[str, Any]],
    output_path: Path = config.METRICS_DIR / "all_experiments_metrics.csv"
) -> pd.DataFrame:
    """
    Saves a list of experiment metrics dictionaries to a CSV file.

    Parameters:
    -----------
    metrics_list : List[Dict[str, Any]]
        List of metrics dictionaries from completed experiments.
    output_path : Path
        Path to output CSV file.

    Returns:
    --------
    df : pd.DataFrame
        DataFrame of the saved metrics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in metrics_list:
        row = {
            "Experiment": m.get("experiment", "Unnamed"),
            "Accuracy": round(m.get("accuracy", 0.0), 4),
            "Precision": round(m.get("precision", 0.0), 4),
            "Recall": round(m.get("recall", 0.0), 4),
            "F1-Score": round(m.get("f1_score", 0.0), 4),
            "FPR": round(m.get("false_positive_rate", 0.0), 4),
            "ROC-AUC": round(m.get("roc_auc", 0.0), 4) if np.isfinite(m.get("roc_auc", np.nan)) else "N/A",
            "TP": m.get("tp", 0),
            "TN": m.get("tn", 0),
            "FP": m.get("fp", 0),
            "FN": m.get("fn", 0),
            "Train Time (s)": round(m.get("train_time_sec", 0.0), 4),
            "Inference Time (s)": round(m.get("inference_time_sec", 0.0), 4),
            "Test Windows": m.get("test_samples", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[Evaluation] Metrics table saved to: {output_path}")
    return df


def generate_markdown_report(
    metrics_list: List[Dict[str, Any]],
    dataset_details: Dict[str, Any],
    output_path: Path = config.REPORTS_DIR / "final_report.md"
) -> str:
    """
    Generates a comprehensive markdown report summarizing dataset details,
    feature definitions, transform parameters, model configurations, and
    an experimental comparison table.

    Parameters:
    -----------
    metrics_list : List[Dict[str, Any]]
        List of experiment results.
    dataset_details : Dict[str, Any]
        Metadata regarding the dataset used.
    output_path : Path
        Path to save the generated markdown report.

    Returns:
    --------
    content : str
        Markdown report content.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics_list)

    table_md = "| Experiment | Accuracy | Precision | Recall | F1-Score | FPR | TP | TN | FP | FN | Train Time |\n"
    table_md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for m in metrics_list:
        table_md += (
            f"| **{m['experiment']}** | {m['accuracy']*100:.2f}% | "
            f"{m['precision']*100:.2f}% | {m['recall']*100:.2f}% | "
            f"{m['f1_score']:.4f} | {m['false_positive_rate']*100:.2f}% | "
            f"{m['tp']} | {m['tn']} | {m['fp']} | {m['fn']} | "
            f"{m['train_time_sec']*1000:.2f} ms |\n"
        )

    content = f"""# LDoS Attack Detection Project: Final Experimental Report

**Project Title:** Low-Rate Denial-of-Service (LDoS) Attack Detection Using Signal Transforms and Machine Learning  
**Author:** Student Implementation & Extended Framework  
**Reference Paper:** Xiaocai Wang, Dan Tang, Ye Feng, Zheng Qin, Bing Xiong, and Yufeng Liu, *"An LDoS attack detection method based on FSWT time–frequency distribution,"* *Expert Systems with Applications*, Vol. 256, 2024.

---

## 1. Executive Summary & Problem Formulation

A **Low-rate Denial of Service (LDoS)** attack periodically transmits short, high-rate bursts of packets at a constant interval $T$ with burst duration $L$, throttling TCP's Retransmission Timeout (RTO) congestion control mechanism:
$$\\bar{{R}}_{{\\text{{attack}}}} = \\frac{{L}}{{T}} \\cdot R_{{\\text{{burst}}}}$$
Because the overall average rate $\\bar{{R}}_{{\\text{{attack}}}}$ is low and identical to normal background traffic during the $(T - L)$ idle period, traditional volume-based threshold anomaly detectors completely fail.

This project extends the state-of-the-art FSWT + Decision Tree baseline by systematically incorporating:
1. **Two Extra Transforms:** Short-Time Fourier Transform (STFT) and Continuous Wavelet Transform (CWT).
2. **Two Extra Features:** Spectral Centroid and Kurtosis.
3. **One Extra Classifier:** Random Forest ensemble.

---

## 2. Dataset Specifications

- **Dataset Source:** {dataset_details.get("source", "Benchmark LDoS Pulse Dataset / UTSA-2021 Real SDN LDoS Dataset")}
- **Sampling Frequency ($f_s$):** {config.SAMPLING_RATE} Hz (1 sample = 0.1s)
- **Window Length ($N$):** {config.WINDOW_SIZE} samples per window (10.0 seconds per window)
- **Window Stride:** {config.WINDOW_STRIDE} samples (non-overlapping windows to avoid temporal leakage)
- **Total Windows:** {dataset_details.get("total_windows", "N/A")}
- **Class Distribution:**
  - Normal Traffic (Class 0): {dataset_details.get("normal_windows", "N/A")} windows
  - LDoS Attack (Class 1): {dataset_details.get("ldos_windows", "N/A")} windows
- **Train/Test Split:** 70% Training / 30% Held-Out Testing (stratified, reproducible seed = {config.RANDOM_STATE})

---

## 3. Mathematical Feature Definitions

1. **Contrast (Original):**
   $$\\text{{Contrast}} = \\sum_{{i, j}} (i - j)^2 P(i, j)$$
   Measures local intensity gradients in the normalized 2D time-frequency energy distribution $P(i, j) = E(i, j) / \\sum E$.
2. **Energy Ratio (Original):**
   $$\\text{{Energy Ratio}} = \\frac{{\\sum_{{|f| \\le f_s/10}} E(f, t)}}{{\\sum_{{all}} E(f, t)}}$$
   Quantifies the proportion of energy concentrated in the low-frequency band (<= 1.0 Hz) characteristic of periodic attack repetitions.
3. **Correlation (Original):**
   $$\\text{{Correlation}} = \\frac{{\\sum_{{i, j}} i \\cdot j \\cdot P(i, j) - \\mu_1 \\mu_2}}{{\\sqrt{{\\sigma_1^2 \\sigma_2^2}} + \\epsilon}}$$
   Evaluates linear dependency and temporal-spectral alignment across time-frequency bins.
4. **Entropy (Original):**
   $$\\text{{Entropy}} = -\\sum_{{P(i, j) > 0}} P(i, j) \\ln(P(i, j))$$
   Measures dispersion and uncertainty; LDoS harmonic concentration causes entropy to drop significantly.
5. **Spectral Centroid (Added):**
   $$\\text{{Spectral Centroid}} = \\frac{{\\sum_k |f_k| \\cdot \\sum_t E(f_k, t)}}{{\\sum_{{k, t}} E(f_k, t)}}$$
   Represents the frequency center of mass; drops toward low frequencies during attack pulses.
6. **Kurtosis (Added):**
   $$\\text{{Kurtosis}} = \\frac{{\\frac{{1}}{{N}} \\sum (z_t - \\mu)^4}}{{\\sigma^4}} - 3.0$$
   Quantifies peakedness and heavy-tailed impulsive burst spikes in the temporal energy profile.

---

## 4. Transform Configurations

- **FSWT (Frequency Slice Wavelet Transform):** Gaussian FSF $\\hat{{p}}(\\omega) = \\exp(-0.5 \\omega^2)$, $\\sigma = |f| / \\kappa$ with $\\kappa = 23.5482$, $\\lambda = 1.0$.
- **STFT (Short-Time Fourier Transform):** Segment length 32, overlap 24, Hann window, 10 Hz sampling rate.
- **CWT (Continuous Wavelet Transform):** Morlet mother wavelet (`morl`), 32 scales mapped to pseudo-frequencies.

---

## 5. Experimental Results Comparison

{table_md}

---

## 6. Key Analysis & Educational Takeaways

1. **Baseline Reproduction:** Experiment 1 reproduces the research paper's original FSWT + 4 features + Decision Tree baseline.
2. **Impact of Added Features:** Experiment 2 examines whether incorporating Spectral Centroid and Kurtosis improves discrimination of pulse bursts without increasing false alarms.
3. **Transform Comparison:** Experiments 2, 3, and 4 compare FSWT against STFT and CWT under the identical six-feature set and held-out test split, isolating the effect of time-frequency resolution.
4. **Classifier Comparison:** Experiment 5 evaluates whether an ensemble Random Forest further enhances robustness, precision, and stability over a single Decision Tree.
5. **No Data Leakage:** All scalers and feature transformers are fitted strictly on training data; all reported metrics represent genuine unseen test-set evaluations.

---

*Report automatically generated by `evaluation.py` on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}.*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[Evaluation] Final report generated at: {output_path}")
    return content


if __name__ == "__main__":
    print("Testing evaluation.py...")
    y_t = np.array([0, 0, 0, 1, 1, 1])
    y_p = np.array([0, 0, 1, 1, 1, 1])
    mets = compute_evaluation_metrics(y_t, y_p, experiment_name="TestExp")
    print("Accuracy:", mets["accuracy"])
    print("Precision:", mets["precision"])
    print("Recall:", mets["recall"])
    print("F1-score:", mets["f1_score"])
    print("FPR:", mets["false_positive_rate"])
    save_metrics_to_csv([mets])
    print("Evaluation module successfully verified!")
