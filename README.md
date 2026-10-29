# Low-Rate Denial of Service (LDoS) Attack Detection

This project investigates the detection of Low-Rate Denial of Service (LDoS) attacks using traffic analysis, signal-processing features, and machine-learning classification.

All collaborators should update the documentation as experiments are implemented and evaluated.

## Ground Truth Labels

- Normal traffic: `0`
- LDoS traffic: `1`

---

## Experimental Framework

### Experiment A — Time-Domain Detection

**Status:** Implemented

- Analyzes normal and LDoS traffic using time-domain statistical characteristics.
- Evaluates threshold-based detection methods.
- Reports accuracy, precision, recall, and F1-score.
- Generates experimental results and visualizations.

Run from the project root:

```bash
python experiment_a.py
```

See [Experiment A documentation](experiments/README.md).

### Experiment B — FFT-Based Detection

**Status:** Not yet completed.

Planned investigation of frequency-domain characteristics using the Fast Fourier Transform (FFT).

### Experiment C — STFT-Based Detection

**Status:** Not yet completed.

Planned investigation of time-frequency characteristics using the Short-Time Fourier Transform (STFT).

### Experiment D — Wavelet-Based Detection

**Status:** Not yet completed.

Planned investigation of wavelet-based traffic analysis.

### Experiment E — FSWT + Decision Tree

**Status:** Basic pipeline implemented; preliminary evaluation completed.

The current implementation:

1. Divides traffic into non-overlapping windows of 100 samples.
2. Uses a sampling rate of 10 Hz, corresponding to 10-second windows.
3. Extracts three features: entropy, low-frequency energy ratio, and contrast.
4. Trains a Decision Tree classifier with `max_depth=3` and `random_state=42`.
5. Evaluates predictions using accuracy, precision, recall, F1-score, and a confusion matrix.

The current implementation uses three features, not all four features described in the research paper. Its feature calculations are provisional and should not be considered a verified reproduction of the paper's complete method.

Run from the project root:

```bash
python experiments/experiment_e.py
```

The evaluation metrics are saved to:

`results/experiment_e_results.csv`

#### Preliminary Results

| Metric | Result |
|---|---:|
| Accuracy | 83.33% |
| Precision | 100.00% |
| Recall | 66.67% |
| F1-score | 80.00% |
| Training windows | 28 |
| Testing windows | 12 |

Confusion matrix, with actual classes as rows and predicted classes as columns:

```text
                 Predicted Normal  Predicted LDoS
Actual Normal             6               0
Actual LDoS               2               4
```

#### Interpretation

The model correctly classified 10 of the 12 test windows. It detected four of the six LDoS windows and missed two.

These results are preliminary because the test set is small. They do not establish that Experiment E outperforms Experiment A or generalizes to real-world network traffic.

---

## Implemented Components

The repository contains the following feature modules:

- `features/fswt.py` — FSWT implementation.
- `features/entropy.py` — Entropy feature calculation.
- `features/energyRatio.py` — Energy-ratio feature calculation.
- `features/contrast.py` — Contrast feature calculation.
- `features/correlation.py` — Correlation feature calculation.

The current Experiment E classifier uses entropy, low-frequency energy ratio, and contrast. The correlation module is present in the repository but is not used by the current Experiment E classifier.

### Current Implementation Checklist

- [x] Experiment A: Time-domain threshold-based evaluation.
- [x] FSWT implementation available.
- [x] Feature modules available.
- [x] Experiment E: Basic FSWT feature extraction and Decision Tree training.
- [x] Experiment E: Preliminary evaluation metrics saved.
- [ ] Verify feature definitions against the research paper.
- [ ] Incorporate and evaluate the fourth feature in Experiment E.
- [ ] Conduct a fair comparison between Experiments A and E.
- [ ] Evaluate on additional data and a larger test set.
- [ ] Complete the FFT, STFT, and wavelet experiments if required.

A checked item indicates that the corresponding basic implementation or output exists; it


## Experiment F: New-Trace Evaluation Using FFT Features

### Objective
Evaluate whether statistical and frequency-domain features can distinguish normal traffic from traces labelled as Low-Rate Denial-of-Service (LDoS) traffic.

### Methodology
- **Dataset:** Selected `normal_*`, `ldos_6_*`, and `ldos_8_*` files from `new_data`.
- **Window size:** 100 samples per window.
- **Features extracted:**
  - Standard deviation
  - Root mean square (RMS)
  - Mean absolute difference
  - Peak-to-peak amplitude
  - FFT spectral entropy
  - Low-frequency energy ratio
- **Classifier:** Decision Tree with maximum depth 3 and balanced class weights.
- **Evaluation:** Entire traces were held out for testing to prevent windows from the same trace appearing in both training and testing sets.

### Results

| Metric | Result |
|---|---:|
| Accuracy | 92.27% |
| Precision | 96.19% |
| Recall | 90.99% |
| F1-score | 93.52% |
| Training windows | 496 |
| Testing windows | 181 |

**Confusion matrix** (rows = actual class; columns = predicted class):

| | Predicted Normal | Predicted LDoS-labelled |
|---|---:|---:|
| Actual Normal | 66 | 4 |
| Actual LDoS-labelled | 10 | 101 |

### Limitations
- This experiment uses FFT-based features; it does not apply FSWT.
- The meaning of the selected measurement column has not been independently verified.
- Dataset provenance and filename-derived labels require confirmation.
- These results are preliminary and do not establish generalization to real-world LDoS traffic.

### Reproducibility
Run the experiment from the repository root:

```bash
python experiments/experiment_f.py
```

Results are saved to `results/experiment_f_results.csv`.