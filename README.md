# Low-rate Denial of Service (LDoS) Attack Detection

all collaborators are requested to read and update this readme.md file regularly so we can keep track of project updates

### Ground Truth Labels:
- All LDoS traffic should have label as `1`
- All normal traffic should have label as `0`

---

## Experimental Framework (Member 4 — Experimental Design + Evaluation)

- [x] **Experiment A**: Normal vs LDoS in time domain (`python experiment_a.py`) — [Documentation](experiments/README.md)
- [ ] **Experiment B**: FFT-based detection
- [ ] **Experiment C**: STFT-based detection
- [ ] **Experiment D**: Wavelet-based detection
- [ ] **Experiment E**: FSWT + four features + Decision Tree

---

## Status of Implemented Features & Pipeline
- [x] Implemented FSWT ([features/fswt.py](features/fswt.py))
- [x] Implemented feature extraction of:
  - Entropy ([features/entropy.py](features/entropy.py))
  - Energy ratio ([features/energyRatio.py](features/energyRatio.py))
  - Contrast ([features/contrast.py](features/contrast.py))
  - Correlation ([features/correlation.py](features/correlation.py))
- [x] Implemented Experiment A (Time Domain Evaluation & Visualizations)
- [ ] Decision Trees / Classifier training for Experiment E

---

## Evaluation Metrics
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$F1 = \frac{2 \cdot P \cdot R}{P + R}$$

Deliverables output directory: `results/` (`final_results.xlsx`, CSV metrics, and generated figures).