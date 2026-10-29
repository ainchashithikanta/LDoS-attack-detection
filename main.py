"""
Main Orchestration Pipeline for LDoS Attack Detection Project.

Runs the five comparative experiments strictly on held-out test data:
  Experiment 1: Original 4 Features + FSWT + Decision Tree
  Experiment 2: All 6 Features + FSWT + Decision Tree
  Experiment 3: All 6 Features + STFT + Decision Tree
  Experiment 4: All 6 Features + CWT + Decision Tree
  Experiment 5: Best Justified Configuration + Random Forest

Saves figures, CSV metrics, and generates a comprehensive final report.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.dataset_loader import load_benchmark_dataset, load_utsa_dataset
from src.preprocessing import (
    prepare_windowed_dataset,
    split_train_test,
    temporal_split_by_class,
    remove_dc_component
)
from src.fswt_transform import compute_fswt
from src.stft_transform import compute_stft
from src.cwt_transform import compute_cwt
from src.feature_extraction import build_feature_dataset
from src.decision_tree_model import train_decision_tree, predict_decision_tree
from src.random_forest_model import train_random_forest, predict_random_forest
from src.evaluation import compute_evaluation_metrics, save_metrics_to_csv, generate_markdown_report
from src.visualization import (
    plot_time_domain_signals,
    plot_transform_comparisons,
    plot_confusion_matrices,
    plot_feature_distributions
)


def run_pipeline(dataset_type: str = "benchmark", split_type: str = "temporal"):
    print("=" * 80)
    print("   LOW-RATE DENIAL-OF-SERVICE (LDOS) ATTACK DETECTION PIPELINE")
    print("   Signal Transforms & Machine Learning Comparative Study")
    print("=" * 80)

    # 1. Load Dataset
    print(f"\n[Step 1/6] Loading '{dataset_type}' dataset...")
    if dataset_type.lower() == "utsa":
        normal_df, ldos_df = load_utsa_dataset()
        dataset_name = "UTSA-2021 Real SDN LDoS Dataset"
    else:
        normal_df, ldos_df = load_benchmark_dataset()
        dataset_name = "Benchmark LDoS Pulse Dataset (Wang et al. 2024 model)"

    print(f" -> Normal Traffic shape: {normal_df.shape}")
    print(f" -> LDoS Attack shape:   {ldos_df.shape}")

    # 2. Windowing
    print("\n[Step 2/6] Slicing traffic streams into non-overlapping windows...")
    windows, labels = prepare_windowed_dataset(
        normal_df,
        ldos_df,
        window_size=config.WINDOW_SIZE,
        stride=config.WINDOW_STRIDE
    )
    y = np.array(labels, dtype=int)
    n_normal = int(np.sum(y == config.LABEL_NORMAL))
    n_ldos = int(np.sum(y == config.LABEL_LDOS))
    print(f" -> Extracted {len(windows)} total windows (Normal: {n_normal}, LDoS: {n_ldos})")
    print(f" -> Window duration: {config.WINDOW_SIZE / config.SAMPLING_RATE:.1f}s ({config.WINDOW_SIZE} samples @ {config.SAMPLING_RATE} Hz)")

    # 3. Generate Visualizations (Time-Domain & Transforms)
    print("\n[Step 3/6] Generating initial visualizations...")
    plot_time_domain_signals(normal_df, ldos_df)

    # Sample one normal window and one LDoS window for transform spectrograms
    sample_norm = remove_dc_component(windows[0])
    sample_ldos = remove_dc_component(windows[n_normal])  # first LDoS window

    fswt_norm, _ = compute_fswt(sample_norm, config.SAMPLING_RATE)
    fswt_ldos, _ = compute_fswt(sample_ldos, config.SAMPLING_RATE)
    stft_norm, _, _ = compute_stft(sample_norm, config.SAMPLING_RATE)
    stft_ldos, _, _ = compute_stft(sample_ldos, config.SAMPLING_RATE)
    cwt_norm, _, _ = compute_cwt(sample_norm, config.SAMPLING_RATE)
    cwt_ldos, _, _ = compute_cwt(sample_ldos, config.SAMPLING_RATE)

    plot_transform_comparisons(
        fswt_norm, fswt_ldos,
        stft_norm, stft_ldos,
        cwt_norm, cwt_ldos
    )

    # 4. Feature Extraction across Transforms
    print("\n[Step 4/6] Extracting features across transforms (FSWT, STFT, CWT)...")

    # A. FSWT All 6 features
    print(" -> Computing FSWT features (all 6 features)...")
    df_fswt_6 = build_feature_dataset(windows, labels, transform_type="fswt", feature_set="all")
    plot_feature_distributions(df_fswt_6)

    # B. STFT All 6 features
    print(" -> Computing STFT features (all 6 features)...")
    df_stft_6 = build_feature_dataset(windows, labels, transform_type="stft", feature_set="all")

    # C. CWT All 6 features
    print(" -> Computing CWT features (all 6 features)...")
    df_cwt_6 = build_feature_dataset(windows, labels, transform_type="cwt", feature_set="all")

    # 5. Execute 5 Comparative Experiments on Same Train/Test Split
    print("\n[Step 5/6] Executing 5 Comparative Experiments...")
    if split_type.lower() == "temporal":
        print(" -> Using Chronological Temporal Split (Preserves temporal causality, 0% data leakage)...")
        idx_train, idx_test = temporal_split_by_class(n_normal, n_ldos, test_size=config.TEST_SIZE)
        y_train, y_test = y[idx_train], y[idx_test]
    else:
        print(" -> Using Stratified Random Split...")
        indices = np.arange(len(windows))
        idx_train, idx_test, y_train, y_test = split_train_test(
            indices, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
    print(f" -> Train set size: {len(y_train)} windows | Test set size: {len(y_test)} windows")

    metrics_results = []

    # -------------------------------------------------------------
    # Experiment 1: Original 4 Features + FSWT + Decision Tree
    # -------------------------------------------------------------
    print("\n--- Running Experiment 1: FSWT + 4 Original Features + Decision Tree ---")
    X_fswt_4 = df_fswt_6[config.ORIGINAL_FEATURES].to_numpy()
    X_train_exp1, X_test_exp1 = X_fswt_4[idx_train], X_fswt_4[idx_test]

    pipe_dt_1, t_train_1 = train_decision_tree(X_train_exp1, y_train)
    preds_1, probs_1, t_inf_1 = predict_decision_tree(pipe_dt_1, X_test_exp1)
    m1 = compute_evaluation_metrics(
        y_test, preds_1, probs_1,
        train_time=t_train_1,
        inference_time=t_inf_1,
        experiment_name="Exp 1: FSWT + 4 Feats + Decision Tree"
    )
    metrics_results.append(m1)
    print(f"    Accuracy: {m1['accuracy']*100:.2f}% | Precision: {m1['precision']*100:.2f}% | Recall: {m1['recall']*100:.2f}% | F1: {m1['f1_score']:.4f}")

    # -------------------------------------------------------------
    # Experiment 2: All 6 Features + FSWT + Decision Tree
    # -------------------------------------------------------------
    print("\n--- Running Experiment 2: FSWT + All 6 Features + Decision Tree ---")
    X_fswt_6 = df_fswt_6[config.ALL_SIX_FEATURES].to_numpy()
    X_train_exp2, X_test_exp2 = X_fswt_6[idx_train], X_fswt_6[idx_test]

    pipe_dt_2, t_train_2 = train_decision_tree(X_train_exp2, y_train)
    preds_2, probs_2, t_inf_2 = predict_decision_tree(pipe_dt_2, X_test_exp2)
    m2 = compute_evaluation_metrics(
        y_test, preds_2, probs_2,
        train_time=t_train_2,
        inference_time=t_inf_2,
        experiment_name="Exp 2: FSWT + 6 Feats + Decision Tree"
    )
    metrics_results.append(m2)
    print(f"    Accuracy: {m2['accuracy']*100:.2f}% | Precision: {m2['precision']*100:.2f}% | Recall: {m2['recall']*100:.2f}% | F1: {m2['f1_score']:.4f}")

    # -------------------------------------------------------------
    # Experiment 3: All 6 Features + STFT + Decision Tree
    # -------------------------------------------------------------
    print("\n--- Running Experiment 3: STFT + All 6 Features + Decision Tree ---")
    X_stft_6 = df_stft_6[config.ALL_SIX_FEATURES].to_numpy()
    X_train_exp3, X_test_exp3 = X_stft_6[idx_train], X_stft_6[idx_test]

    pipe_dt_3, t_train_3 = train_decision_tree(X_train_exp3, y_train)
    preds_3, probs_3, t_inf_3 = predict_decision_tree(pipe_dt_3, X_test_exp3)
    m3 = compute_evaluation_metrics(
        y_test, preds_3, probs_3,
        train_time=t_train_3,
        inference_time=t_inf_3,
        experiment_name="Exp 3: STFT + 6 Feats + Decision Tree"
    )
    metrics_results.append(m3)
    print(f"    Accuracy: {m3['accuracy']*100:.2f}% | Precision: {m3['precision']*100:.2f}% | Recall: {m3['recall']*100:.2f}% | F1: {m3['f1_score']:.4f}")

    # -------------------------------------------------------------
    # Experiment 4: All 6 Features + CWT + Decision Tree
    # -------------------------------------------------------------
    print("\n--- Running Experiment 4: CWT + All 6 Features + Decision Tree ---")
    X_cwt_6 = df_cwt_6[config.ALL_SIX_FEATURES].to_numpy()
    X_train_exp4, X_test_exp4 = X_cwt_6[idx_train], X_cwt_6[idx_test]

    pipe_dt_4, t_train_4 = train_decision_tree(X_train_exp4, y_train)
    preds_4, probs_4, t_inf_4 = predict_decision_tree(pipe_dt_4, X_test_exp4)
    m4 = compute_evaluation_metrics(
        y_test, preds_4, probs_4,
        train_time=t_train_4,
        inference_time=t_inf_4,
        experiment_name="Exp 4: CWT + 6 Feats + Decision Tree"
    )
    metrics_results.append(m4)
    print(f"    Accuracy: {m4['accuracy']*100:.2f}% | Precision: {m4['precision']*100:.2f}% | Recall: {m4['recall']*100:.2f}% | F1: {m4['f1_score']:.4f}")

    # -------------------------------------------------------------
    # Determine Best Configuration among Exp 2, 3, 4 for Experiment 5
    # -------------------------------------------------------------
    tree_experiments = [m2, m3, m4]
    best_exp = max(tree_experiments, key=lambda x: (x["f1_score"], x["accuracy"]))
    if best_exp["experiment"].startswith("Exp 2"):
        best_transform_name = "FSWT"
        X_train_exp5, X_test_exp5 = X_train_exp2, X_test_exp2
    elif best_exp["experiment"].startswith("Exp 3"):
        best_transform_name = "STFT"
        X_train_exp5, X_test_exp5 = X_train_exp3, X_test_exp3
    else:
        best_transform_name = "CWT"
        X_train_exp5, X_test_exp5 = X_train_exp4, X_test_exp4

    # -------------------------------------------------------------
    # Experiment 5: Best Justified Configuration + Random Forest
    # -------------------------------------------------------------
    print(f"\n--- Running Experiment 5: Best Config ({best_transform_name} + 6 Feats) + Random Forest ---")
    pipe_rf_5, t_train_5 = train_random_forest(X_train_exp5, y_train)
    preds_5, probs_5, t_inf_5 = predict_random_forest(pipe_rf_5, X_test_exp5)
    m5 = compute_evaluation_metrics(
        y_test, preds_5, probs_5,
        train_time=t_train_5,
        inference_time=t_inf_5,
        experiment_name=f"Exp 5: {best_transform_name} + 6 Feats + Random Forest"
    )
    metrics_results.append(m5)
    print(f"    Accuracy: {m5['accuracy']*100:.2f}% | Precision: {m5['precision']*100:.2f}% | Recall: {m5['recall']*100:.2f}% | F1: {m5['f1_score']:.4f}")

    # 6. Evaluation, Exports & Terminal Report
    print("\n[Step 6/6] Generating reports and export deliverables...")
    plot_confusion_matrices(metrics_results)
    metrics_df = save_metrics_to_csv(metrics_results)

    dataset_details = {
        "source": dataset_name,
        "total_windows": len(windows),
        "normal_windows": n_normal,
        "ldos_windows": n_ldos,
    }
    generate_markdown_report(metrics_results, dataset_details)

    # Print nicely formatted terminal table
    print("\n" + "=" * 95)
    print("                           FINAL EXPERIMENTAL COMPARISON TABLE")
    print("=" * 95)
    header = f"{'Experiment':<42} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'FPR':<7}"
    print(header)
    print("-" * 95)
    for m in metrics_results:
        row_str = (
            f"{m['experiment']:<42} | "
            f"{m['accuracy']*100:>8.2f}% | "
            f"{m['precision']*100:>8.2f}% | "
            f"{m['recall']*100:>8.2f}% | "
            f"{m['f1_score']:>9.4f} | "
            f"{m['false_positive_rate']*100:>6.2f}%"
        )
        print(row_str)
    print("=" * 95)
    print("\nAll deliverables, metrics, graphs, and reports successfully generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDoS Attack Detection Main Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark",
        choices=["benchmark", "utsa"],
        help="Dataset to evaluate: 'benchmark' (default) or 'utsa'."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="temporal",
        choices=["temporal", "random"],
        help="Train/test split strategy: 'temporal' (default, chronological order) or 'random'."
    )
    args = parser.parse_args()
    run_pipeline(dataset_type=args.dataset, split_type=args.split)
