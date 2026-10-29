

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial', 'DejaVu Sans', 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def ensure_directories():
    """Ensure required output directories exist."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return base_dir, results_dir


def load_or_generate_data(base_dir):
    """
    Load traffic data from data/ folder or generate synthetic data if not found.
    """
    normal_path = os.path.join(base_dir, "data", "normal_traffic.csv")
    ldos_path = os.path.join(base_dir, "data", "ldos_traffic.csv")

    if os.path.exists(normal_path) and os.path.exists(ldos_path):
        normal_df = pd.read_csv(normal_path)
        ldos_df = pd.read_csv(ldos_path)
        print(f"[+] Loaded existing traffic datasets:")
        print(f"    - Normal traffic: {len(normal_df)} samples ({normal_path})")
        print(f"    - LDoS traffic:   {len(ldos_df)} samples ({ldos_path})")
    else:
        print("[!] Dataset files not found. Synthesizing standard benchmark traffic...")
        duration = 200.0  # seconds
        sampling_rate = 10  # Hz
        n_samples = int(duration * sampling_rate)
        time = np.arange(n_samples) / sampling_rate

        np.random.seed(42)
        normal_traffic = 10.0 + np.random.normal(0, 0.5, size=n_samples)
        ldos_traffic = normal_traffic.copy()

        # Attack params: T=15s period, L=5s pulse duration, R=7Mbps burst rate
        T, L, R = 15.0, 5.0, 7.0
        for i in range(n_samples):
            if (time[i] % T) < L:
                ldos_traffic[i] += R

        normal_df = pd.DataFrame({'Time (s)': time, 'Traffic (Mbps)': normal_traffic, 'label': 0})
        ldos_df = pd.DataFrame({'Time (s)': time, 'Traffic (Mbps)': ldos_traffic, 'label': 1})

        os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
        normal_df.to_csv(normal_path, index=False)
        ldos_df.to_csv(ldos_path, index=False)
        print(f"[+] Generated and saved synthetic data to {normal_path} and {ldos_path}")

    return normal_df, ldos_df


def compute_metrics(y_true, y_pred):
    """
    Compute TP, TN, FP, FN, Accuracy, Precision, Recall, and F1-score.
    Formulas strictly adhere to experimental guidelines:
        Accuracy  = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 * P * R / (P + R)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1)
    }


def extract_window_features(traffic_series, window_size=50, step_size=10):
    """
    Segment a traffic time series into sliding windows and extract time-domain features.
    Features:
      - mean: Mean rate in window
      - std: Standard deviation in window
      - peak: Maximum rate in window
      - min: Minimum rate in window
      - par: Peak-to-Average Ratio (max / mean)
      - iqr: Interquartile range (75th percentile - 25th percentile)
      - acf_peak: Peak autocorrelation value at non-zero lag
    """
    n = len(traffic_series)
    features = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = traffic_series[start:end]

        mean_val = np.mean(window)
        std_val = np.std(window)
        max_val = np.max(window)
        min_val = np.min(window)
        par_val = max_val / mean_val if mean_val != 0 else 0.0
        iqr_val = np.percentile(window, 75) - np.percentile(window, 25)

        # Autocorrelation of the window
        centered = window - mean_val
        var = np.var(window)
        if var > 1e-9:
            autocorr = np.correlate(centered, centered, mode='full')
            autocorr = autocorr[len(window) - 1:] / (var * len(window))
            # Peak lag > 2 samples to avoid zero-lag trivial peak
            acf_subpeak = np.max(autocorr[2:]) if len(autocorr) > 2 else 0.0
        else:
            acf_subpeak = 0.0

        features.append({
            'start_idx': start,
            'end_idx': end,
            'mean': mean_val,
            'std': std_val,
            'peak': max_val,
            'min': min_val,
            'par': par_val,
            'iqr': iqr_val,
            'acf_subpeak': acf_subpeak
        })

    return pd.DataFrame(features)


def compute_time_autocorrelation(signal, max_lag_samples=400):
    """
    Compute normalized autocorrelation of a time-domain signal.
    """
    centered = signal - np.mean(signal)
    var = np.var(signal)
    if var < 1e-9:
        return np.zeros(max_lag_samples)
    full_corr = np.correlate(centered, centered, mode='full')
    lags = np.arange(len(signal))
    mid = len(signal) - 1
    norm_corr = full_corr[mid:mid + max_lag_samples] / (var * len(signal))
    return norm_corr


def run_experiment_a():
    print("=" * 75)
    print(" EXPERIMENT A: NORMAL VS LDoS IN TIME DOMAIN")
    print(" Member 4 — Experimental Design + Evaluation")
    print("=" * 75)

    base_dir, results_dir = ensure_directories()
    normal_df, ldos_df = load_or_generate_data(base_dir)

    sampling_rate = 10  # 10 samples per second
    time = normal_df['Time (s)'].values
    normal_sig = normal_df['Traffic (Mbps)'].values
    ldos_sig = ldos_df['Traffic (Mbps)'].values

    # 1. Summary Statistics in Time Domain
    print("\n--- 1. Time-Domain Summary Statistics ---")
    stats = {
        'Trace': ['Normal Traffic', 'LDoS Traffic'],
        'Mean (Mbps)': [np.mean(normal_sig), np.mean(ldos_sig)],
        'Std Dev (Mbps)': [np.std(normal_sig), np.std(ldos_sig)],
        'Min (Mbps)': [np.min(normal_sig), np.min(ldos_sig)],
        'Max / Peak (Mbps)': [np.max(normal_sig), np.max(ldos_sig)],
        'Peak-to-Average Ratio': [np.max(normal_sig) / np.mean(normal_sig), np.max(ldos_sig) / np.mean(ldos_sig)],
        'Median (Mbps)': [np.median(normal_sig), np.median(ldos_sig)],
        'IQR (Mbps)': [np.percentile(normal_sig, 75) - np.percentile(normal_sig, 25),
                       np.percentile(ldos_sig, 75) - np.percentile(ldos_sig, 25)]
    }
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

    # 2. Instantaneous Point-by-Point Detection
    # Baseline instantaneous detector: flag as attack if traffic > normal_mean + 3 * normal_std
    print("\n--- 2. Instantaneous Sample-Level Thresholding (3-Sigma Rule) ---")
    normal_mean = np.mean(normal_sig)
    normal_std = np.std(normal_sig)
    instant_threshold = normal_mean + 3.0 * normal_std

    y_true_instant = np.concatenate([np.zeros(len(normal_sig)), np.ones(len(ldos_sig))])
    y_pred_instant = np.concatenate([
        (normal_sig > instant_threshold).astype(int),
        (ldos_sig > instant_threshold).astype(int)
    ])
    metrics_instant = compute_metrics(y_true_instant, y_pred_instant)
    metrics_instant['Detector'] = f'Instantaneous 3-Sigma (> {instant_threshold:.2f} Mbps)'
    metrics_instant['Domain'] = 'Point-by-Point Time Domain'

    # 3. Sliding Window Time-Domain Feature Detection
    # Window of 5 seconds (50 samples), step of 1 second (10 samples)
    print("\n--- 3. Sliding Window Time-Domain Feature Extraction ---")
    win_size = 50  # 5.0 seconds
    step_size = 10  # 1.0 second

    normal_feats = extract_window_features(normal_sig, window_size=win_size, step_size=step_size)
    normal_feats['label'] = 0

    ldos_feats = extract_window_features(ldos_sig, window_size=win_size, step_size=step_size)
    ldos_feats['label'] = 1

    all_windows = pd.concat([normal_feats, ldos_feats], ignore_index=True)
    y_true_win = all_windows['label'].values

    # Detector A1: Window Mean Threshold (threshold = midpoint between normal mean and ldos mean)
    mean_thresh = (np.mean(normal_feats['mean']) + np.mean(ldos_feats['mean'])) / 2.0
    y_pred_mean = (all_windows['mean'] > mean_thresh).astype(int)
    metrics_mean = compute_metrics(y_true_win, y_pred_mean)
    metrics_mean['Detector'] = f'Window Mean Threshold (> {mean_thresh:.2f} Mbps)'
    metrics_mean['Domain'] = 'Windowed Time Domain (5s)'

    # Detector A2: Window Standard Deviation Threshold
    std_thresh = (np.mean(normal_feats['std']) + np.mean(ldos_feats['std'])) / 2.0
    y_pred_std = (all_windows['std'] > std_thresh).astype(int)
    metrics_std = compute_metrics(y_true_win, y_pred_std)
    metrics_std['Detector'] = f'Window Std Dev Threshold (> {std_thresh:.2f} Mbps)'
    metrics_std['Domain'] = 'Windowed Time Domain (5s)'

    # Detector A3: Window Peak Rate Threshold
    peak_thresh = (np.max(normal_feats['peak']) + np.mean(ldos_feats['peak'])) / 2.0
    y_pred_peak = (all_windows['peak'] > peak_thresh).astype(int)
    metrics_peak = compute_metrics(y_true_win, y_pred_peak)
    metrics_peak['Detector'] = f'Window Peak Threshold (> {peak_thresh:.2f} Mbps)'
    metrics_peak['Domain'] = 'Windowed Time Domain (5s)'

    # Detector A4: Window Peak-to-Average Ratio (PAR) Threshold
    par_thresh = (np.mean(normal_feats['par']) + np.mean(ldos_feats['par'])) / 2.0
    y_pred_par = (all_windows['par'] > par_thresh).astype(int)
    metrics_par = compute_metrics(y_true_win, y_pred_par)
    metrics_par['Detector'] = f'Window PAR Threshold (> {par_thresh:.2f})'
    metrics_par['Domain'] = 'Windowed Time Domain (5s)'

    # Detector A5: Window Autocorrelation Subpeak Threshold
    acf_thresh = (np.mean(normal_feats['acf_subpeak']) + np.mean(ldos_feats['acf_subpeak'])) / 2.0
    y_pred_acf = (all_windows['acf_subpeak'] > acf_thresh).astype(int)
    metrics_acf = compute_metrics(y_true_win, y_pred_acf)
    metrics_acf['Detector'] = f'Window Autocorr Subpeak (> {acf_thresh:.2f})'
    metrics_acf['Domain'] = 'Windowed Time Domain (5s)'

    # Compile all detector metrics
    all_metrics = [
        metrics_instant,
        metrics_mean,
        metrics_std,
        metrics_peak,
        metrics_par,
        metrics_acf
    ]
    metrics_df = pd.DataFrame(all_metrics)
    cols_order = ['Detector', 'Domain', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1']
    metrics_df = metrics_df[cols_order]

    print("\n" + "=" * 95)
    print(" EXPERIMENT A — EVALUATION METRICS TABLE")
    print("=" * 95)
    print(metrics_df.to_string(index=False))

    # Save to CSV and Excel deliverables
    csv_path = os.path.join(results_dir, "experiment_a_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved metrics CSV: {csv_path}")

    excel_path = os.path.join(results_dir, "final_results.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
            metrics_df.to_excel(writer, sheet_name='Experiment A', index=False)
            stats_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
        print(f"[+] Saved deliverables Excel: {excel_path}")
    except Exception as e:
        print(f"[-] Excel export note: {e}")

    # 4. Generate Comprehensive Visualizations
    print("\n--- 4. Generating Publication-Quality Visualizations ---")
    generate_visualizations(time, normal_sig, ldos_sig, normal_feats, ldos_feats,
                            all_windows, y_true_win, metrics_df, results_dir)

    # 5. Sensitivity Analysis: Degradation of Time-Domain Detection with Low-Rate Attacks
    print("\n--- 5. Evaluating Stealthy Low-Rate Attacks (Sensitivity Analysis) ---")
    run_stealthiness_sensitivity(sampling_rate, results_dir)

    print("\n" + "=" * 75)
    print(" EXPERIMENT A COMPLETED SUCCESSFULLY!")
    print(f" All outputs and figures saved in: {results_dir}")
    print("=" * 75)


def generate_visualizations(time, normal_sig, ldos_sig, normal_feats, ldos_feats,
                            all_windows, y_true_win, metrics_df, results_dir):
    """
    Create comprehensive multi-panel visualizations for Experiment A.
    """

    # -------------------------------------------------------------
    # Plot 1: Normal vs LDoS Time Domain Signals with Pulse Zoom-in
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    axes[0].plot(time, normal_sig, color='#1f77b4', lw=1.2, label='Normal Traffic')
    axes[0].axhline(np.mean(normal_sig), color='navy', linestyle='--', label=f'Normal Mean ({np.mean(normal_sig):.2f} Mbps)')
    axes[0].set_title("A1: Normal Network Traffic (Time Domain)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Traffic (Mbps)", fontsize=11)
    axes[0].set_ylim(0, 22)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, ldos_sig, color='#d62728', lw=1.2, label='LDoS Attack Traffic')
    axes[1].axhline(np.mean(ldos_sig), color='darkred', linestyle='--', label=f'LDoS Mean ({np.mean(ldos_sig):.2f} Mbps)')
    axes[1].set_title("A2: LDoS Network Traffic with Periodic Pulsing Bursts (T=15s, L=5s, R=7Mbps)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Traffic (Mbps)", fontsize=11)
    axes[1].set_ylim(0, 22)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Zoomed comparison of 0-45 seconds (3 complete cycles)
    zoom_mask = time <= 45.0
    axes[2].plot(time[zoom_mask], normal_sig[zoom_mask], color='#1f77b4', lw=1.5, alpha=0.8, label='Normal Traffic (Zoom)')
    axes[2].plot(time[zoom_mask], ldos_sig[zoom_mask], color='#d62728', lw=1.8, label='LDoS Traffic (Zoom)')
    axes[2].axvspan(0, 5, color='red', alpha=0.15, label='Attack Pulse Active (L=5s)')
    axes[2].axvspan(15, 20, color='red', alpha=0.15)
    axes[2].axvspan(30, 35, color='red', alpha=0.15)
    axes[2].set_title("A3: Direct Comparison (Zoomed First 45 Seconds: Periodic Burst Pattern)", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Time (seconds)", fontsize=11)
    axes[2].set_ylabel("Traffic (Mbps)", fontsize=11)
    axes[2].set_ylim(0, 22)
    axes[2].legend(loc='upper right', ncol=4)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot1_path = os.path.join(results_dir, "experiment_a_time_domain_signals.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[+] Saved visualization: {plot1_path}")

    # -------------------------------------------------------------
    # Plot 2: Statistical Distributions & Probability Density
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Histogram & Probability Density
    axes[0].hist(normal_sig, bins=35, density=True, alpha=0.6, color='#1f77b4', edgecolor='black', label='Normal Traffic')
    axes[0].hist(ldos_sig, bins=45, density=True, alpha=0.6, color='#d62728', edgecolor='black', label='LDoS Traffic (Bimodal)')
    axes[0].axvline(np.mean(normal_sig), color='#1f77b4', linestyle='--', lw=2, label='Normal Mean')
    axes[0].axvline(np.mean(ldos_sig), color='#d62728', linestyle='--', lw=2, label='LDoS Mean')
    axes[0].set_title("Traffic Rate Probability Distribution (Time Domain)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Traffic Rate (Mbps)", fontsize=11)
    axes[0].set_ylabel("Probability Density", fontsize=11)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Boxplots
    box_data = [normal_sig, ldos_sig]
    bplot = axes[1].boxplot(box_data, patch_artist=True, tick_labels=['Normal Traffic', 'LDoS Traffic'])
    colors = ['#1f77b4', '#d62728']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bplot['medians']:
        median.set(color='black', linewidth=2)
    axes[1].set_title("Time-Domain Variance & Outlier Dispersion", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Traffic Rate (Mbps)", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot2_path = os.path.join(results_dir, "experiment_a_statistical_distributions.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[+] Saved visualization: {plot2_path}")

    # -------------------------------------------------------------
    # Plot 3: Sliding Window Feature Dynamics (Mean, Std, PAR)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    win_centers = (normal_feats['start_idx'] + normal_feats['end_idx']) / (2 * 10.0)

    # Window Mean
    axes[0].plot(win_centers, normal_feats['mean'], color='#1f77b4', lw=1.8, label='Normal Window Mean')
    axes[0].plot(win_centers, ldos_feats['mean'], color='#d62728', lw=1.8, label='LDoS Window Mean')
    axes[0].set_title("Sliding Window Feature Dynamics (Window = 5s)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Window Mean (Mbps)", fontsize=11)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Window Standard Deviation
    axes[1].plot(win_centers, normal_feats['std'], color='#1f77b4', lw=1.8, label='Normal Window Std Dev')
    axes[1].plot(win_centers, ldos_feats['std'], color='#d62728', lw=1.8, label='LDoS Window Std Dev')
    axes[1].set_ylabel("Window Std Dev (Mbps)", fontsize=11)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Window Peak-to-Average Ratio
    axes[2].plot(win_centers, normal_feats['par'], color='#1f77b4', lw=1.8, label='Normal Window PAR')
    axes[2].plot(win_centers, ldos_feats['par'], color='#d62728', lw=1.8, label='LDoS Window PAR')
    axes[2].set_xlabel("Time (seconds)", fontsize=11)
    axes[2].set_ylabel("Peak-to-Average Ratio", fontsize=11)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot3_path = os.path.join(results_dir, "experiment_a_sliding_window_features.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[+] Saved visualization: {plot3_path}")

    # -------------------------------------------------------------
    # Plot 4: Time-Domain Autocorrelation Function (ACF)
    # -------------------------------------------------------------
    max_lags = 350  # 35 seconds of lag (sampling rate = 10Hz)
    lags_sec = np.arange(max_lags) / 10.0

    acf_normal = compute_time_autocorrelation(normal_sig, max_lag_samples=max_lags)
    acf_ldos = compute_time_autocorrelation(ldos_sig, max_lag_samples=max_lags)

    plt.figure(figsize=(12, 6))
    plt.plot(lags_sec, acf_normal, color='#1f77b4', lw=1.5, label='Normal Traffic ACF (Uncorrelated Noise)')
    plt.plot(lags_sec, acf_ldos, color='#d62728', lw=2.0, label='LDoS Traffic ACF (Strong Periodic Spikes)')

    # Annotate periodic peaks at T = 15s and 2T = 30s
    plt.axvline(15.0, color='darkred', linestyle='--', alpha=0.7, label='Attack Period T = 15.0s')
    plt.axvline(30.0, color='darkorange', linestyle='--', alpha=0.7, label='Harmonic Period 2T = 30.0s')

    plt.title("Time-Domain Autocorrelation Analysis: Revealing LDoS Periodicity", fontsize=13, fontweight='bold')
    plt.xlabel("Lag Time (seconds)", fontsize=11)
    plt.ylabel("Normalized Autocorrelation", fontsize=11)
    plt.ylim(-0.6, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot4_path = os.path.join(results_dir, "experiment_a_autocorrelation.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"[+] Saved visualization: {plot4_path}")

    # -------------------------------------------------------------
    # Plot 5: Confusion Matrices for All Detectors
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, row in metrics_df.iterrows():
        ax = axes[idx]
        cm = np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]])

        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(f"{row['Detector']}\nAcc: {row['Accuracy']*100:.1f}% | F1: {row['F1']:.3f}", fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Norm (0)', 'Pred LDoS (1)'])
        ax.set_yticklabels(['True Norm (0)', 'True LDoS (1)'])

        # Print cell counts
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > np.max(cm) / 2 else 'black'
                cell_label = 'TN' if (i == 0 and j == 0) else ('FP' if (i == 0 and j == 1) else ('FN' if (i == 1 and j == 0) else 'TP'))
                ax.text(j, i, f"{cell_label}\n{cm[i, j]}", ha='center', va='center', color=color, fontsize=11, fontweight='bold')

    plt.suptitle("Experiment A: Confusion Matrices across Time-Domain Detectors", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot5_path = os.path.join(results_dir, "experiment_a_confusion_matrices.png")
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"[+] Saved visualization: {plot5_path}")

    # -------------------------------------------------------------
    # Plot 6: ROC & PR Curves for Time-Domain Sliding Window Features
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    features_to_plot = [
        ('mean', 'Window Mean Rate', '#1f77b4'),
        ('std', 'Window Std Dev', '#2ca02c'),
        ('peak', 'Window Peak Rate', '#d62728'),
        ('par', 'Peak-to-Average Ratio', '#9467bd'),
        ('acf_subpeak', 'Autocorr Subpeak', '#ff7f0e')
    ]

    for feat_col, label, color in features_to_plot:
        scores = all_windows[feat_col].values
        fpr, tpr, _ = roc_curve(y_true_win, scores)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true_win, scores)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, color=color, lw=2, label=f"{label} (AUC = {pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.7)
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Experiment A: Discrimination Power of Time-Domain Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot6_path = os.path.join(results_dir, "experiment_a_roc_curves.png")
    plt.savefig(plot6_path, dpi=300)
    plt.close()
    print(f"[+] Saved visualization: {plot6_path}")


def run_stealthiness_sensitivity(sampling_rate, results_dir):
    """
    Sensitivity Analysis: How time-domain detection degrades as the attack
    burst rate R decreases (becoming stealthier).
    This highlights the motivation for frequency and wavelet analysis (Experiments B-E).
    """
    attack_rates = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    duration = 200.0
    n_samples = int(duration * sampling_rate)
    time = np.arange(n_samples) / sampling_rate

    np.random.seed(42)
    normal_traffic = 10.0 + np.random.normal(0, 0.5, size=n_samples)
    normal_mean = np.mean(normal_traffic)
    normal_std = np.std(normal_traffic)

    # 5-second window features on normal traffic
    win_size, step_size = 50, 10
    normal_feats = extract_window_features(normal_traffic, window_size=win_size, step_size=step_size)
    normal_feats['label'] = 0

    sensitivity_records = []

    for R in attack_rates:
        ldos = normal_traffic.copy()
        T, L = 15.0, 5.0
        for i in range(n_samples):
            if (time[i] % T) < L:
                ldos[i] += R

        ldos_feats = extract_window_features(ldos, window_size=win_size, step_size=step_size)
        ldos_feats['label'] = 1

        all_df = pd.concat([normal_feats, ldos_feats], ignore_index=True)
        y_true = all_df['label'].values

        # Time-domain standard deviation threshold detector
        thresh = (np.mean(normal_feats['std']) + np.mean(ldos_feats['std'])) / 2.0
        y_pred = (all_df['std'] > thresh).astype(int)
        m = compute_metrics(y_true, y_pred)
        m['Attack_Rate_R'] = R
        sensitivity_records.append(m)

    sens_df = pd.DataFrame(sensitivity_records)

    # Plot Sensitivity
    plt.figure(figsize=(10, 5.5))
    plt.plot(sens_df['Attack_Rate_R'], sens_df['Accuracy'] * 100, marker='o', lw=2, color='#1f77b4', label='Accuracy (%)')
    plt.plot(sens_df['Attack_Rate_R'], sens_df['Precision'] * 100, marker='s', lw=2, color='#2ca02c', label='Precision (%)')
    plt.plot(sens_df['Attack_Rate_R'], sens_df['Recall'] * 100, marker='^', lw=2, color='#ff7f0e', label='Recall (%)')
    plt.plot(sens_df['Attack_Rate_R'], sens_df['F1'] * 100, marker='d', lw=2, color='#d62728', label='F1-Score (%)')

    plt.axvline(1.5, color='gray', linestyle=':', label='Stealthy Threshold (R ~ 3*noise)')
    plt.title("Time-Domain Detection Breakdown Under Stealthy LDoS Attacks", fontsize=13, fontweight='bold')
    plt.xlabel("Attack Burst Rate R (Mbps)", fontsize=11)
    plt.ylabel("Score (%)", fontsize=11)
    plt.ylim(30, 105)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot7_path = os.path.join(results_dir, "experiment_a_sensitivity_analysis.png")
    plt.savefig(plot7_path, dpi=300)
    plt.close()
    print(f"[+] Saved sensitivity plot: {plot7_path}")


if __name__ == "__main__":
    run_experiment_a()
