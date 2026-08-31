# Selective ECG Classification Under Signal Degradation

Code and reproduction pipeline for the manuscript:

> **Selective ECG Classification Under Signal Degradation: Evaluating Auxiliary Reliability Signals Against Output Confidence**
> Muhammad Idrees, Adnan Amin, Salma Azizi
> Institute of Management Sciences (IMSciences), Peshawar, Pakistan
> *Manuscript in preparation / under review — not yet published.*

---

## Overview

Selective prediction lets a classifier defer unreliable cases for manual review instead of forcing a decision on every input. This only works if the score used to rank cases actually identifies which predictions are wrong — not just which cases *look* harder on average.

This project asks one question: **do annotation support, waveform quality, and model uncertainty rank ECG prediction errors better than plain output confidence, especially when the signal is degraded?**

We test this on PTB-XL (21,388 recordings, 5 diagnostic superclasses, patient-disjoint folds) using a frozen ResNet1D classifier, four reliability signals, and 12 controlled corruption conditions (baseline wander, white noise, amplitude clipping, lead masking — each at 3 severities).

**Bottom line:** output confidence won. Across three prespecified statistical comparisons and 13 conditions (clean + 12 corrupted), 38 of 39 bootstrap intervals favored confidence over the alternatives. Model uncertainty rose consistently with corruption severity — but that cohort-level response did not translate into better case-by-case error ranking.

---

## Repository structure


The raw PTB-XL dataset is **not** included in this repository. Download it separately (instructions below).

---

## Method summary

**Data.** PTB-XL v1.0.3, 21,388 recordings from 18,617 patients after excluding records with no diagnostic superclass. Patient-disjoint split following PTB-XL's recommended folds: 1–8 for training (17,084 ECGs), 9 for validation (2,146), 10 for final test (2,158 recordings from 1,877 patients).

**Classifier.** ResNet1DWang (473,349 parameters), trained with AdamW and a One-Cycle learning-rate schedule for 50 epochs. Class-specific decision thresholds were tuned on the validation fold and frozen before any test-set evaluation.

**Four reliability signals, evaluated per ECG record:**

| Signal | What it measures | Needs reference labels? |
|---|---|---|
| **C** — Output confidence | Probability extremeness around 0.5, averaged across 5 labels | No |
| **L** — Annotation support | Strength of the reference diagnostic annotation | Yes (retrospective only) |
| **Q** — Waveform quality | Artifact burden: low/high-frequency noise, lead dropout, clipping | No |
| **U** — Model uncertainty | MC-dropout disagreement across 30 stochastic forward passes | No |

Eight rankings were compared in total: C, L, Q, U, and four equal-weight combinations (L+Q, L+U, Q+U, L+Q+U). Confidence was never included inside a combination — it is the baseline the others are tested against.

**Endpoint.** Area under the risk–coverage curve (AURC), using five-label Hamming error. Lower AURC indicates better selective ranking.

**Corruption.** Four synthetic degradation families at three severities each, applied to the raw waveform before normalization: baseline wander, additive white noise, amplitude clipping, and lead masking.

**Statistics.** For three prespecified contrasts (L+U, U, and L+Q+U, each versus confidence) across 13 conditions (clean plus 12 corrupted), we ran 5,000 paired patient-clustered bootstrap replicates to obtain condition-specific 95% intervals. These 39 intervals are dependent and were not adjusted for multiplicity — they are reported as condition-specific evidence, not as a family-wise test.

---

## Key results

| Ranking | Clean-test AURC (lower is better) |
|---|---|
| **Confidence (C)** | **0.055157** |
| L + U | 0.060360 |
| U | 0.064734 |
| L + Q + U | 0.079144 |
| Q + U | 0.090890 |
| L | 0.101339 |
| L + Q | 0.117802 |
| Q | 0.132054 |

Confidence had the lowest AURC on both validation and the held-out test fold. Under corruption, 38 of 39 bootstrap intervals favored confidence; the single exception (L+U under moderate amplitude clipping) crossed zero. Mean model uncertainty (U) rose monotonically with corruption severity across all four families — evidence that the signal responds to degradation, but not evidence that it ranks individual errors better than confidence.

Held-out classifier performance (fold 10, 2,158 ECGs): macro-AUROC 0.9265, macro-F1 0.7448, full-coverage Hamming risk 0.1190.

Full classwise performance, all 39 bootstrap contrasts, and per-condition breakdowns are reproducible via the scripts in this repository.

---

## Reproducing the results

### 1. Get the data

Download PTB-XL v1.0.3 from PhysioNet:


Place it outside this repository (e.g. `~/datasets/ptbxl/`) and point the scripts to that path. Do not copy raw waveform files into this repo.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline in order

```bash
# Load, filter, and split PTB-XL into patient-disjoint folds
python preprocessing/load_ptbxl.py --data-dir /path/to/ptbxl

# Train the classifier (seed 20260826 reproduces the reported checkpoint)
python model/train.py --seed 20260826

# Compute the four reliability signals on the trained checkpoint
python reliability_signals/compute_all.py

# Generate the 12 synthetic corruption conditions
python corruption/generate_all.py

# Evaluate AURC and risk-coverage for all 8 rankings across all 13 conditions
python evaluation/aurc.py

# Run the patient-clustered bootstrap comparison (5,000 replicates)
python statistics/bootstrap_patient_clustered.py --seed 20260826
```

Each script writes its outputs to `results/` and `figures/`, matching the tables and figures reported in the manuscript.

---

## Reproducibility notes

- All preprocessing statistics (normalization mean/SD) were computed from the training split only.
- Decision thresholds were frozen on the validation fold before any test-set inspection.
- Three training seeds (20260826, 20260827, 20260828) were run to confirm validation stability; only 20260826 was evaluated on the final test fold.
- A full integrity audit (299 checks) and automated test suite (118 tests) passed on the reported run.

---

## Citation

This work is not yet published. If you use this code ahead of publication, please cite the repository directly:

```bibtex
@software{idrees_selective_ecg_code,
  title  = {Code and Analysis Pipeline for "Selective ECG Classification Under Signal Degradation"},
  author = {Idrees, Muhammad and Amin, Adnan and Azizi, Salma},
  year   = {2026},
  url    = {https://github.com/idrees118/selective-ecg-signal-degradation}
}
```

A full paper citation and DOI will be added here once the manuscript is accepted.

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

Muhammad Idrees — [github.com/idrees118](https://github.com/idrees118)
Dr. Adnan Amin (corresponding author) — adnan.amin@imsciences.edu.pk
