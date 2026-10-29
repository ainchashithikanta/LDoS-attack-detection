import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "new_data"
RESULTS_DIR = PROJECT_ROOT / "results"

WINDOW_SIZE = 100
RANDOM_STATE = 42

# These labels are inferred from filenames and must be verified.
NORMAL_FILES = sorted(DATA_DIR.glob("normal_*"))
ATTACK_FILES = sorted(DATA_DIR.glob("ldos_6_*")) + sorted(
    DATA_DIR.glob("ldos_8_*")
)


def load_trace(path):
    """Load a whitespace-separated trace with date, time, and 10 numeric columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None)

    if df.shape[1] != 12:
        raise ValueError(
            f"{path.name}: expected 12 columns, found {df.shape[1]}"
        )

    numeric = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

    if numeric.isna().any().any():
        raise ValueError(f"{path.name}: non-numeric or missing measurements")

    # Column 3 (zero-based index 3) is a numeric measurement whose meaning
    # has not yet been verified. This experiment tests it as a provisional signal.
    signal = numeric.iloc[:, 1].to_numpy(dtype=float)

    if not np.isfinite(signal).all():
        raise ValueError(f"{path.name}: non-finite values")

    return signal


def extract_features(signal):
    """Extract time-domain and FFT-based features from a fixed-size window."""
    signal = np.asarray(signal, dtype=float)

    if len(signal) != WINDOW_SIZE:
        raise ValueError("Unexpected window size")

    # Center the signal to reduce sensitivity to its absolute baseline.
    centered = signal - np.mean(signal)

    std = float(np.std(centered))
    rms = float(np.sqrt(np.mean(centered ** 2)))
    mean_abs_diff = float(np.mean(np.abs(np.diff(centered))))
    peak_to_peak = float(np.ptp(centered))

    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    total_energy = float(np.sum(spectrum))

    if total_energy > 0:
        probabilities = spectrum / total_energy
        probabilities = probabilities[probabilities > 0]
        entropy = float(-np.sum(probabilities * np.log(probabilities)))

        frequencies = np.fft.rfftfreq(len(centered), d=1.0)
        low_mask = frequencies <= 0.1
        energy_ratio = float(
            np.sum(spectrum[low_mask]) / total_energy
        )
    else:
        entropy = 0.0
        energy_ratio = 0.0

    return [
        std,
        rms,
        mean_abs_diff,
        peak_to_peak,
        entropy,
        energy_ratio,
    ]


def make_windows(path, label):
    """Convert one trace into non-overlapping windows and keep its file ID."""
    signal = load_trace(path)
    records = []

    for start in range(0, len(signal) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = signal[start:start + WINDOW_SIZE]
        features = extract_features(window)

        records.append({
            "trace": path.name,
            "label": label,
            "features": features,
        })

    return records


def main():
    if not NORMAL_FILES or not ATTACK_FILES:
        raise FileNotFoundError(
            "No normal_* or ldos_6_*/ldos_8_* files found in new_data."
        )

    traces = []

    for path in NORMAL_FILES:
        traces.extend(make_windows(path, 0))

    for path in ATTACK_FILES:
        traces.extend(make_windows(path, 1))

    data = pd.DataFrame(traces)

    if data.empty:
        raise ValueError("No complete windows were extracted.")

    feature_names = [
        "std",
        "rms",
        "mean_abs_diff",
        "peak_to_peak",
        "fft_entropy",
        "low_frequency_energy_ratio",
    ]

    X = np.vstack(data["features"].to_numpy())
    y = data["label"].to_numpy()
    groups = data["trace"].to_numpy()

    # Hold out entire traces so no windows from a test trace appear in training.
    rng = np.random.default_rng(RANDOM_STATE)
    normal_groups = np.array(sorted(set(groups[y == 0])))
    attack_groups = np.array(sorted(set(groups[y == 1])))

    rng.shuffle(normal_groups)
    rng.shuffle(attack_groups)

    test_groups = set(normal_groups[:max(1, len(normal_groups) // 4)])
    test_groups.update(attack_groups[:max(1, len(attack_groups) // 4)])

    test_mask = np.array([group in test_groups for group in groups])
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    model = DecisionTreeClassifier(
        max_depth=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n--- Experiment F: New-trace evaluation ---")
    print("Normal traces:", len(NORMAL_FILES))
    print("Attack-labelled traces:", len(ATTACK_FILES))
    print("Total complete windows:", len(data))
    print("Training windows:", int(train_mask.sum()))
    print("Testing windows:", int(test_mask.sum()))
    print("Held-out traces:", ", ".join(sorted(test_groups)))
    print("Features:", ", ".join(feature_names))
    print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Precision:",
        round(precision_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(y_test, predictions, zero_division=0), 4),
    )
    print("F1-score:", round(f1_score(y_test, predictions, zero_division=0), 4))
    print("\nConfusion matrix (rows=true, columns=predicted; labels 0,1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "LDoS-labelled"],
            zero_division=0,
        )
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame([{
        "Experiment": "F",
        "Model": "FFT statistical features + Decision Tree",
        "Window Size": WINDOW_SIZE,
        "Training Windows": int(train_mask.sum()),
        "Testing Windows": int(test_mask.sum()),
        "Training Traces": len(set(groups[train_mask])),
        "Testing Traces": len(set(groups[test_mask])),
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
    }])

    output_file = RESULTS_DIR / "experiment_f_results.csv"
    metrics.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print(
        "\nImportant: this is a provisional baseline. The measurement column "
        "and filename-derived labels have not yet been independently verified."
    )


if __name__ == "__main__":
    main()