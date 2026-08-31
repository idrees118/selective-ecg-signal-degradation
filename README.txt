<div align="center">

# 🫀 Selective ECG Classification Under Signal Degradation

### Evaluating auxiliary reliability signals against output confidence

**Reproducible code and analysis pipeline for selective ECG classification on PTB-XL**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PTB--XL-2F80ED)](https://physionet.org/content/ptb-xl/1.0.3/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/Manuscript-Under%20Review-orange)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Tested-success)

**Muhammad Idrees · Adnan Amin · Salma Azizi**  
Institute of Management Sciences (IMSciences), Peshawar, Pakistan

</div>

---

## Overview

Selective classification allows a model to **defer uncertain predictions** instead of forcing a decision on every ECG.

This repository examines a simple but important question:

> **Do auxiliary reliability signals rank ECG prediction errors better than ordinary output confidence, especially when the signal is degraded?**

We evaluate four reliability signals and their combinations on **PTB-XL**, using:

- **21,388 ECG recordings**
- **5 diagnostic superclasses**
- **Patient-disjoint train/validation/test folds**
- **12 controlled signal corruption conditions**

---

## Main finding

> **Output confidence was the best selective ranking signal in our experiments.**

Across **13 conditions** (clean + 12 corrupted) and **3 prespecified comparisons**,  
**38 of 39 patient-clustered bootstrap intervals favored confidence** over the compared alternatives.

The only exception was:

- **L + U under moderate amplitude clipping**, where the interval crossed zero.

A second important finding is that **model uncertainty increased consistently with corruption severity**, but this did **not** translate into better case-by-case error ranking than confidence.

---

## Study summary

| Component | Setting |
|---|---|
| Dataset | PTB-XL v1.0.3 |
| Total ECGs used | 21,388 |
| Training ECGs | 17,084 |
| Validation ECGs | 2,146 |
| Test ECGs | 2,158 |
| Test patients | 1,877 |
| Model | ResNet1DWang |
| Trainable parameters | 473,349 |
| Reliability signals | C, L, Q, U |
| Rankings evaluated | 8 |
| Corruption families | 4 |
| Corruption conditions | 12 |
| Primary endpoint | AURC |
| Bootstrap replicates | 5,000 |

---

## Dataset split

We use **PTB-XL v1.0.3** after excluding records with no diagnostic superclass.

The split follows the recommended PTB-XL patient-disjoint folds:

| Split | PTB-XL folds | ECG recordings |
|---|---:|---:|
| Training | 1–8 | 17,084 |
| Validation | 9 | 2,146 |
| Test | 10 | 2,158 |
| **Total** | **1–10** | **21,388** |

> Raw PTB-XL waveform files are **not included** in this repository.

---

## Model

The classifier is **ResNet1DWang**, trained using:

- **AdamW**
- **One-Cycle learning-rate scheduling**
- **50 epochs**
- **Training-only normalization statistics**

Class-specific thresholds were tuned on the **validation fold only** and frozen before final test evaluation.

---

## Reliability signals

Each ECG is assigned four reliability scores:

| Symbol | Signal | What it measures | Requires reference labels? |
|:---:|---|---|:---:|
| **C** | Output confidence | Probability extremeness around 0.5, averaged across 5 labels | No |
| **L** | Annotation support | Strength of the reference diagnostic annotation | **Yes** |
| **Q** | Waveform quality | Artifact burden, including noise, clipping, and lead failure | No |
| **U** | Model uncertainty | MC-dropout disagreement across 30 stochastic forward passes | No |

> **Important:** Annotation support (**L**) is retrospective and depends on reference information, so it is not a deployable test-time signal.

---

## Rankings evaluated

Eight ranking strategies were compared:

```text
C
L
Q
U
L + Q
L + U
Q + U
L + Q + U
```

All combinations use equal weighting.

**Confidence is not included in the combined scores**, because it is the baseline against which the auxiliary signals are tested.

---

## Evaluation metric

The main endpoint is **AURC** (**Area Under the Risk–Coverage Curve**), using **five-label Hamming error**.

```text
Lower AURC = better selective ranking
```

A lower AURC means the ranking removes likely errors earlier as coverage decreases.

---

## Signal corruption setup

To test robustness under degraded signal quality, four corruption families were applied to the raw ECG waveform before normalization:

| Corruption family | Description |
|---|---|
| Baseline wander | Low-frequency baseline drift |
| White noise | Additive broadband noise |
| Amplitude clipping | Saturation / limited dynamic range |
| Lead masking | Missing lead information |

Each corruption was tested at **3 severity levels**, producing:

```text
1 clean condition + 12 corrupted conditions = 13 total conditions
```

---

## Key clean-test results

| Rank | Reliability ranking | AURC ↓ |
|---:|---|---:|
| **1** | **Confidence (C)** | **0.055157** |
| 2 | L + U | 0.060360 |
| 3 | U | 0.064734 |
| 4 | L + Q + U | 0.079144 |
| 5 | Q + U | 0.090890 |
| 6 | L | 0.101339 |
| 7 | L + Q | 0.117802 |
| 8 | Q | 0.132054 |

Confidence achieved the best AURC on both the validation and held-out test folds.

---

## Held-out classifier performance

Performance on PTB-XL fold 10:

| Metric | Value |
|---|---:|
| Macro-AUROC | **0.9265** |
| Macro-F1 | **0.7448** |
| Full-coverage Hamming risk | **0.1190** |

These metrics describe classification quality at full coverage, while the selective prediction analysis focuses on **ranking likely errors**.

---

## Statistical analysis

Three prespecified comparisons were tested against confidence:

```text
L + U     vs Confidence
U         vs Confidence
L + Q + U vs Confidence
```

Across **13 conditions**, this yields:

```text
3 × 13 = 39 comparisons
```

For each comparison, we ran:

- **5,000 paired bootstrap replicates**
- **Patient-clustered resampling**
- **Condition-specific 95% intervals**

### Result

- **38 of 39 intervals favored confidence**
- The remaining interval (**L + U under moderate amplitude clipping**) crossed zero

These intervals are reported as **condition-specific evidence** and were **not adjusted for multiplicity**.

---

## Repository structure

```text
selective-ecg-signal-degradation/
│
├── preprocessing/
│   └── load_ptbxl.py
├── model/
│   └── train.py
├── reliability_signals/
│   └── compute_all.py
├── corruption/
│   └── generate_all.py
├── evaluation/
│   └── aurc.py
├── statistics/
│   └── bootstrap_patient_clustered.py
├── results/
├── figures/
├── tests/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Reproducing the results

### 1. Clone the repository

```bash
git clone https://github.com/idrees118/selective-ecg-signal-degradation.git
cd selective-ecg-signal-degradation
```

### 2. Download PTB-XL

Download **PTB-XL v1.0.3** from PhysioNet:

https://physionet.org/content/ptb-xl/1.0.3/

Store it outside the repository, for example:

```text
~/datasets/ptbxl/
```

### 3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Prepare the dataset

```bash
python preprocessing/load_ptbxl.py --data-dir /path/to/ptbxl
```

### 5. Train the classifier

```bash
python model/train.py --seed 20260826
```

### 6. Compute reliability signals

```bash
python reliability_signals/compute_all.py
```

### 7. Generate corruption conditions

```bash
python corruption/generate_all.py
```

### 8. Evaluate AURC and risk–coverage

```bash
python evaluation/aurc.py
```

### 9. Run the statistical analysis

```bash
python statistics/bootstrap_patient_clustered.py --seed 20260826
```

Outputs are written to:

```text
results/
figures/
```

---

## Reproducibility notes

- Normalization statistics were computed from the **training split only**
- Thresholds were selected on the **validation fold only**
- Test data were not used for model selection
- Three training seeds were checked for validation stability:
  - `20260826`
  - `20260827`
  - `20260828`
- Final held-out analysis used seed:
  - `20260826`
- The reported run passed:
  - **299 integrity checks**
  - **118 automated tests**

---

## Data availability

The **PTB-XL** dataset is not redistributed in this repository.

Dataset source:  
https://physionet.org/content/ptb-xl/1.0.3/

Users should follow PhysioNet’s data usage and citation requirements.

> The **MIT license** in this repository applies to the **code**, not to PTB-XL itself.

---

## Citation

If you use this repository before the manuscript is published, please cite the repository directly:

```bibtex
@software{idrees2026selectiveecg,
  author  = {Idrees, Muhammad and Amin, Adnan and Azizi, Salma},
  title   = {Selective ECG Classification Under Signal Degradation: Code and Reproduction Pipeline},
  year    = {2026},
  url     = {https://github.com/idrees118/selective-ecg-signal-degradation}
}
```

A full article citation and DOI will be added after publication.

---

## License

This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for details.

---

## Contact

**Muhammad Idrees**  
GitHub: [@idrees118](https://github.com/idrees118)

**Dr. Adnan Amin**  
Corresponding author  
Institute of Management Sciences (IMSciences), Peshawar, Pakistan  
Email: **adnan.amin@imsciences.edu.pk**
