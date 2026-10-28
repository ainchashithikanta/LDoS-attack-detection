import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.fswt import fswt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

NORMAL_FILE = DATA_DIR / "normal_traffic.csv"
LDOS_FILE = DATA_DIR / "ldos_traffic.csv"


def load_data():
    normal = pd.read_csv(NORMAL_FILE)
    ldos = pd.read_csv(LDOS_FILE)

    print("Normal traffic shape:", normal.shape)
    print("LDoS traffic shape:", ldos.shape)

    print("\nNormal traffic columns:")
    print(normal.columns.tolist())

    print("\nLDoS traffic columns:")
    print(ldos.columns.tolist())

    return normal, ldos


def compare_signals(normal, ldos):
    normal_traffic = normal["Traffic (Mbps)"].to_numpy()
    ldos_traffic = ldos["Traffic (Mbps)"].to_numpy()

    difference = ldos_traffic - normal_traffic
    changed = np.where(np.abs(difference) > 1e-6)[0]

    print("\n--- Attack Signal Analysis ---")
    print("Minimum difference:", difference.min())
    print("Maximum difference:", difference.max())
    print("Number of samples with a difference:", len(changed))
    print("Number of samples with no difference:", len(difference) - len(changed))

    if len(changed) > 0:
        print("First changed sample index:", changed[0])
        print("Last changed sample index:", changed[-1])

        print("\nFirst 10 changed samples (index, difference):")
        for i in changed[:10]:
            print(i, difference[i])

    print("\nDifference counts:")
    print(pd.Series(np.round(difference, 4)).value_counts().head(10))

def extract_fswt_features(signal, sampling_rate=10):
    W = fswt(signal, sampling_rate)
    energy = np.abs(W) ** 2

    total_energy = np.sum(energy)

    if total_energy <= 0 or not np.isfinite(total_energy):
        return [0.0, 0.0, 0.0]

    # Normalize energy into a probability distribution
    probability = energy / total_energy

    # 1. Time-frequency entropy; ignore zero probabilities
    nonzero = probability > 0
    entropy = -np.sum(
        probability[nonzero] * np.log(probability[nonzero])
    )

    # 2. Low-frequency energy ratio
    frequencies = np.fft.fftfreq(len(signal), d=1 / sampling_rate)
    low_frequency_mask = np.abs(frequencies) <= sampling_rate / 10
    low_frequency_energy = np.sum(energy[low_frequency_mask, :])
    energy_ratio = low_frequency_energy / total_energy

    # 3. Time-frequency contrast
    row, column = energy.shape
    time_indices = np.arange(row)[:, None]
    frequency_indices = np.arange(column)[None, :]
    contrast = np.sum(
        (time_indices - frequency_indices) ** 2 * probability
    )

    return [float(entropy), float(energy_ratio), float(contrast)]

if __name__ == "__main__":
    normal, ldos = load_data()

    WINDOW_SIZE = 100
    SAMPLING_RATE = 10

    X = []
    y = []

    for dataset, label in [(normal, 0), (ldos, 1)]:
        traffic = dataset["Traffic (Mbps)"].to_numpy()

        for start in range(0, len(traffic) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = traffic[start:start + WINDOW_SIZE]

            features = extract_fswt_features(
                window,
                sampling_rate=SAMPLING_RATE
            )

            X.append(features)
            y.append(label)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    # Keep the first 70% of each class for training
    # and the last 30% for testing.
    # This preserves temporal order within each class.
    train_indices = np.concatenate([
        np.arange(0, 14),
        np.arange(20, 34)
    ])
    test_indices = np.concatenate([
        np.arange(14, 20),
        np.arange(34, 40)
    ])

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n--- FSWT Decision Tree Results ---")
    print("Training examples:", len(X_train))
    print("Testing examples:", len(X_test))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1-score:", f1_score(y_test, predictions, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # Save experiment results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results = pd.DataFrame([{
    "Model": "FSWT + Decision Tree",
    "Window Size": WINDOW_SIZE,
    "Training Examples": len(X_train),
    "Testing Examples": len(X_test),
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1-score": f1_score(y_test, predictions, zero_division=0)
}])

output_file = RESULTS_DIR / "experiment_e_results.csv"
results.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")