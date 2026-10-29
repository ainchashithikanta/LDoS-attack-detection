"""
Module: visualization.py
Purpose: Generates clear, student-friendly figures for time-series signals,
         transform energy distributions (FSWT, STFT, CWT), feature distributions,
         and model confusion matrices.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import config


def plot_time_domain_signals(
    normal_df,
    ldos_df,
    duration: float = 45.0,
    save_path: Optional[Path] = config.FIGURES_DIR / "time_domain_signals.png"
) -> Path:
    """
    Plots the time-domain signal traces for Normal Traffic and LDoS Attack Traffic,
    highlighting the periodic pulse nature of LDoS.
    """
    time_norm = normal_df["Time (s)"].to_numpy()
    traffic_norm = normal_df["Traffic (Mbps)"].to_numpy()

    time_ldos = ldos_df["Time (s)"].to_numpy()
    traffic_ldos = ldos_df["Traffic (Mbps)"].to_numpy()

    # Limit to specified duration for clarity
    mask_norm = time_norm <= duration
    mask_ldos = time_ldos <= duration

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(time_norm[mask_norm], traffic_norm[mask_norm], color="#1f77b4", lw=1.5, label="Normal Traffic")
    ax1.set_title("Normal Network Traffic (Background Fluctuations)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Traffic Rate (Mbps)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    ax2.plot(time_ldos[mask_ldos], traffic_ldos[mask_ldos], color="#d62728", lw=1.5, label="LDoS Attack Traffic")
    ax2.set_title("LDoS Attack Traffic (Periodic Bursts + Quiet Intervals)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Traffic Rate (Mbps)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[Visualization] Time-domain plot saved to: {save_path}")
    plt.close()
    return save_path


def plot_transform_comparisons(
    fswt_norm: np.ndarray,
    fswt_ldos: np.ndarray,
    stft_norm: np.ndarray,
    stft_ldos: np.ndarray,
    cwt_norm: np.ndarray,
    cwt_ldos: np.ndarray,
    save_path: Optional[Path] = config.FIGURES_DIR / "transform_comparisons.png"
) -> Path:
    """
    Generates a 3x2 comparison grid of FSWT, STFT, and CWT energy matrices
    for Normal Traffic (left column) and LDoS Traffic (right column).
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    transforms = [
        ("FSWT Energy Matrix", fswt_norm, fswt_ldos, "plasma"),
        ("STFT Spectrogram", stft_norm, stft_ldos, "viridis"),
        ("CWT Scalogram", cwt_norm, cwt_ldos, "inferno")
    ]

    for row_idx, (title, norm_mat, ldos_mat, cmap) in enumerate(transforms):
        # Normal Traffic
        im0 = axes[row_idx, 0].imshow(norm_mat, aspect="auto", origin="lower", cmap=cmap)
        axes[row_idx, 0].set_title(f"{title} - Normal Traffic", fontsize=11, fontweight="bold")
        axes[row_idx, 0].set_ylabel("Frequency / Scale Index", fontsize=10)
        fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # LDoS Traffic
        im1 = axes[row_idx, 1].imshow(ldos_mat, aspect="auto", origin="lower", cmap=cmap)
        axes[row_idx, 1].set_title(f"{title} - LDoS Attack Traffic", fontsize=11, fontweight="bold")
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        if row_idx == 2:
            axes[row_idx, 0].set_xlabel("Time Samples", fontsize=10)
            axes[row_idx, 1].set_xlabel("Time Samples", fontsize=10)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[Visualization] Transform comparison plot saved to: {save_path}")
    plt.close()
    return save_path


def plot_confusion_matrices(
    metrics_list: List[Dict[str, Any]],
    save_path: Optional[Path] = config.FIGURES_DIR / "all_confusion_matrices.png"
) -> Path:
    """
    Plots side-by-side annotated confusion matrices for all completed experiments.
    """
    n_exp = len(metrics_list)
    if n_exp == 0:
        return save_path

    cols = min(3, n_exp)
    rows = int(np.ceil(n_exp / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_exp == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, m in enumerate(metrics_list):
        ax = axes[idx]
        cm = np.array(m["confusion_matrix"])

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(
            f"{m['experiment']}\nAcc: {m['accuracy']*100:.1f}% | F1: {m['f1_score']:.3f}",
            fontsize=10,
            fontweight="bold"
        )

        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Normal", "LDoS"], fontsize=9)
        ax.set_yticklabels(["Normal", "LDoS"], fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

        # Annotate cells
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold"
                )

    # Hide extra subplots
    for j in range(n_exp, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[Visualization] Confusion matrices plot saved to: {save_path}")
    plt.close()
    return save_path


def plot_feature_distributions(
    features_df,
    save_path: Optional[Path] = config.FIGURES_DIR / "feature_distributions.png"
) -> Path:
    """
    Plots boxplots for all features grouped by class label (0: Normal vs 1: LDoS).
    """
    feature_cols = [c for c in features_df.columns if c != "label"]
    n_feats = len(feature_cols)

    fig, axes = plt.subplots(2, int(np.ceil(n_feats / 2)), figsize=(14, 8))
    axes = axes.flatten()

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        norm_vals = features_df[features_df["label"] == 0][col].dropna()
        ldos_vals = features_df[features_df["label"] == 1][col].dropna()

        bp = ax.boxplot([norm_vals, ldos_vals], patch_artist=True, tick_labels=["Normal", "LDoS"])
        bp['boxes'][0].set_facecolor('#6baed6')
        bp['boxes'][1].set_facecolor('#fc9272')

        ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Hide unused axes
    for j in range(n_feats, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[Visualization] Feature distributions plot saved to: {save_path}")
    plt.close()
    return save_path
