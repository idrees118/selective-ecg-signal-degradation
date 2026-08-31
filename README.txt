<div align="center">

# 🫀 Selective ECG Classification Under Signal Degradation

### Evaluating auxiliary reliability signals against output confidence

**Reproducible code and analysis pipeline for selective ECG classification on PTB-XL**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PTB--XL-2F80ED)](https://physionet.org/content/ptb-xl/1.0.3/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/Manuscript-Under%20Review-orange)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Tested-success)

**Muhammad Idrees · Adnan Amin · Salma Azizi**
Institute of Management Sciences (IMSciences), Peshawar, Pakistan

</div>

---

## What is this project about?

A medical classifier does not always need to make a prediction.

In **selective classification**, uncertain or unreliable cases can instead be deferred for manual review. The important question is therefore not only:

> **How accurate is the classifier?**

but also:

> **Can we correctly identify which individual predictions are most likely to be wrong?**

This repository investigates whether several auxiliary reliability signals can rank ECG prediction errors better than the classifier's own **output confidence**, particularly when ECG signals are degraded.

We evaluate:

* **Output confidence**
* **Annotation support**
* **Waveform quality**
* **Model uncertainty**
* Equal-weight combinations of the auxiliary signals

The experiments use **PTB-XL**, five diagnostic superclasses, patient-disjoint evaluation, and **12 controlled signal corruption conditions**.

---

## Main finding

> **Output confidence provided the strongest selective ranking in our experiments.**

Across the three prespecified statistical comparisons and **13 evaluation conditions** — clean ECGs plus 12 corrupted conditions — **38 of 39 patient-clustered bootstrap intervals favored confidence over the compared auxiliary rankings**.

The only exception was **L + U under moderate amplitude clipping**, where the interval crossed zero.

An important secondary finding was that **model uncertainty increased consistently as signal corruption became more severe**.

However, detecting that a cohort has become more degraded is not the same as correctly identifying **which individual predictions are wrong**.

That distinction is central to this study.

---

## Study at a glance

| Component             | Setting          |
| --------------------- | ---------------- |
| Dataset               | PTB-XL v1.0.3    |
| ECG recordings used   | 21,388           |
| Diagnostic targets    | 5 superclasses   |
| Training fold         | PTB-XL folds 1–8 |
| Validation fold       | Fold 9           |
| Final test fold       | Fold 10          |
| Test recordings       | 2,158            |
| Test patients         | 1,877            |
| Classifier            | ResNet1DWang     |
| Reliability signals   | C, L, Q, U       |
| Rankings evaluated    | 8                |
| Corruption families   | 4                |
| Corruption conditions | 12               |
| Primary endpoint      | AURC             |
| Bootstrap replicates  | 5,000            |
| Bootstrap unit        | Patient          |

---

## Experimental design

### Dataset

We use **PTB-XL v1.0.3**, a large publicly available 12-lead ECG dataset.

After applying the study's diagnostic-superclass selection procedure, the data are divided according to the recommended patient-disjoint PTB-XL folds:

| Split      | PTB-XL folds | ECG recordings |
| ---------- | -----------: | -------------: |
| Training   |          1–8 |         17,084 |
| Validation |            9 |          2,146 |
| Final test |           10 |          2,158 |
| **Total**  |     **1–10** |     **21,388** |

The final test set contains **2,158 ECGs from 1,877 patients**.

Raw PTB-XL waveform files are **not included** in this repository.

---

## Classifier

The ECG classifier is a **ResNet1DWang** model with **473,349 trainable parameters**.

Training uses:

* AdamW optimization
* One-Cycle learning-rate scheduling
* 50 training epochs
* Patient-disjoint data partitions
* Training-only normalization statistics

Class-specific decision thresholds are selected using the **validation fold** and frozen before final test evaluation.

This prevents the held-out test fold from influencing model selection or threshold tuning.

---

## Reliability signals

Each ECG receives four reliability scores.

| Symbol | Reliability signal | Interpretation                                         | Requires reference labels? |
| :----: | ------------------ | ------------------------------------------------------ | :------------------------: |
|  **C** | Output confidence  | Extremeness of predicted probabilities relative to 0.5 |             No             |
|  **L** | Annotation support | Strength of the reference diagnostic annotation        |           **Yes**          |
|  **Q** | Waveform quality   | Estimated artifact and signal-quality burden           |             No             |
|  **U** | Model uncertainty  | MC-dropout disagreement across stochastic predictions  |             No             |

> **Important:** Annotation support (**L**) uses reference information and is therefore evaluated as a retrospective research signal rather than a deployable inference-time score.

---

## Rankings compared

Eight ranking strategies are evaluated:

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

All combinations use fixed equal weighting.

**Output confidence is deliberately excluded from the combined scores.**

This keeps confidence as an independent baseline against which the auxiliary reliability signals are evaluated.

---

## Selective prediction metric

The primary endpoint is the **Area Under the Risk–Coverage Curve (AURC)**.

As increasingly unreliable cases are deferred, coverage decreases and the error rate among retained predictions changes.

The evaluation uses **five-label Hamming error** as the prediction risk.

### Interpretation

```text
Lower AURC = better error ranking
```

A ranking is useful when incorrect predictions are removed earlier than correct ones as coverage decreases.

---

## Signal degradation experiments

Reliability rankings are evaluated on the clean test set and under four controlled corruption families.

Each corruption is applied at **three severity levels**.

| Corruption family  | Purpose                                       |
| ------------------ | --------------------------------------------- |
| Baseline wander    | Simulates low-frequency baseline drift        |
| White noise        | Adds broadband measurement noise              |
| Amplitude clipping | Simulates saturation or limited dynamic range |
| Lead masking       | Removes information from ECG leads            |

This produces:

```text
1 clean condition
+
4 corruption families × 3 severities
=
13 evaluation conditions
```

Corruptions are applied to the waveform **before normalization**.

---

## Key results

### Clean-test selective ranking

|  Rank | Reliability ranking |       AURC ↓ |
| ----: | ------------------- | -----------: |
| **1** | **Confidence (C)**  | **0.055157** |
|     2 | L + U               |     0.060360 |
|     3 | U                   |     0.064734 |
|     4 | L + Q + U           |     0.079144 |
|     5 | Q + U               |     0.090890 |
|     6 | L                   |     0.101339 |
|     7 | L + Q               |     0.117802 |
|     8 | Q                   |     0.132054 |

**Confidence achieved the lowest AURC on both the validation and held-out test sets.**

---

## Held-out classifier performance

Performance on PTB-XL fold 10:

| Metric                     |     Result |
| -------------------------- | ---------: |
| Macro-AUROC                | **0.9265** |
| Macro-F1                   | **0.7448** |
| Full-coverage Hamming risk | **0.1190** |
| Test ECGs                  |  **2,158** |
| Test patients              |  **1,877** |

These metrics describe classifier discrimination and classification performance.

The primary research question, however, concerns something different:

> **How well can each reliability score rank individual prediction errors?**

---

## Statistical analysis

Three comparisons were defined before the final analysis:

```text
L + U     vs. Confidence
U         vs. Confidence
L + Q + U vs. Confidence
```

Each comparison is evaluated across all **13 conditions**.

This gives:

```text
3 contrasts × 13 conditions = 39 comparisons
```

For every comparison, we use:

* **5,000 bootstrap replicates**
* paired resampling
* patient-level clustering
* condition-specific 95% bootstrap intervals

### Result

**38 of 39 intervals favored confidence.**

The remaining comparison — **L + U under moderate amplitude clipping** — crossed zero.

Because these 39 intervals are statistically dependent and no multiplicity correction was applied, they are interpreted as **condition-specific evidence**, not as a single family-wise hypothesis test.

---

## An important negative result

One of the most useful findings from this study is that:

> **A reliability signal can respond strongly to corrupted inputs without becoming a better ranking signal for individual prediction errors.**

Model uncertainty (**U**) increased monotonically with corruption severity for all four degradation families.

This shows that the uncertainty measure detects distributional or signal-quality deterioration at the **cohort level**.

But its AURC results show that this does not automatically translate into better **case-level error ranking** than output confidence.

---

## Repository structure

```text
selective-ecg-signal-degradation/
│
├── preprocessing/
│   └── load_ptbxl.py
│
├── model/
│   └── train.py
│
├── reliability_signals/
│   └── compute_all.py
│
├── corruption/
│   └── generate_all.py
│
├── evaluation/
│   └── aurc.py
│
├── statistics/
│   └── bootstrap_patient_clustered.py
│
├── results/
│   └── Generated numerical results
│
├── figures/
│   └── Generated manuscript figures
│
├── tests/
│   └── Automated integrity and pipeline tests
│
├── requirements.txt
├── LICENSE
└── README.md
```

> The raw PTB-XL dataset is intentionally excluded from the repository.

---

# Reproducing the study

## 1. Clone the repository

```bash
git clone https://github.com/idrees118/selective-ecg-signal-degradation.git
cd selective-ecg-signal-degradation
```

---

## 2. Download PTB-XL

Download **PTB-XL v1.0.3** from PhysioNet:

https://physionet.org/content/ptb-xl/1.0.3/

Keep the dataset outside the Git repository, for example:

```text
~/datasets/ptbxl/
```

Do not commit the raw ECG waveform files to this repository.

---

## 3. Install dependencies

Creating an isolated Python environment is recommended.

```bash
python -m venv .venv
```

Activate it.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## 4. Prepare PTB-XL

```bash
python preprocessing/load_ptbxl.py \
    --data-dir /path/to/ptbxl
```

This stage loads the dataset, applies the study inclusion rules, and constructs the patient-disjoint train, validation, and test partitions.

---

## 5. Train the ECG classifier

```bash
python model/train.py --seed 20260826
```

Seed `20260826` corresponds to the checkpoint used for the reported final test analysis.

---

## 6. Compute reliability signals

```bash
python reliability_signals/compute_all.py
```

This computes:

```text
C — output confidence
L — annotation support
Q — waveform quality
U — model uncertainty
```

---

## 7. Generate corrupted ECG conditions

```bash
python corruption/generate_all.py
```

The pipeline generates all 12 prespecified signal-degradation conditions.

---

## 8. Evaluate selective classification

```bash
python evaluation/aurc.py
```

This computes risk–coverage curves and AURC for all eight rankings across the clean and corrupted conditions.

---

## 9. Run the statistical analysis

```bash
python statistics/bootstrap_patient_clustered.py \
    --seed 20260826
```

This performs the **5,000-replicate paired patient-clustered bootstrap analysis** used for the reported statistical comparisons.

---

## Reproducibility safeguards

Several controls were used to reduce information leakage and accidental test-set optimization.

* Normalization statistics are calculated from the **training split only**.
* Training, validation, and testing follow **patient-disjoint PTB-XL folds**.
* Decision thresholds are selected using the **validation fold only**.
* Thresholds are frozen before final test evaluation.
* Synthetic corruptions are applied using predefined procedures.
* Statistical comparisons use **patient-level clustered resampling**.
* Three training seeds were used to evaluate validation stability.
* Only the selected `20260826` checkpoint was evaluated on the final test fold.
* The reported pipeline passed a **299-check integrity audit**.
* The automated test suite passed **118 tests**.

---

## Training seeds

The following training seeds were used during model-development stability checks:

```text
20260826
20260827
20260828
```

The final reported held-out evaluation uses:

```text
20260826
```

---

## Expected outputs

The analysis scripts write generated artifacts to:

```text
results/
figures/
```

These outputs contain the numerical results and visualizations used to reproduce the study analyses, including:

* classifier performance
* reliability scores
* AURC measurements
* risk–coverage results
* corruption experiments
* bootstrap contrasts
* condition-level comparisons
* manuscript figures

---

## Data availability

PTB-XL is distributed separately through **PhysioNet** and is not redistributed in this repository.

Dataset:

**PTB-XL: A Large Publicly Available Electrocardiography Dataset**

https://physionet.org/content/ptb-xl/1.0.3/

Users should follow the dataset's own terms and citation requirements.

The MIT license in this repository applies to the **code provided here**, not to PTB-XL itself.

---

## Manuscript

**Selective ECG Classification Under Signal Degradation: Evaluating Auxiliary Reliability Signals Against Output Confidence**

**Authors:**
Muhammad Idrees
Adnan Amin
Salma Azizi

**Affiliation:**
Institute of Management Sciences (IMSciences), Peshawar, Pakistan

**Status:** Manuscript under review / not yet published.

The final bibliographic citation and DOI will be added after publication.

---

## Citation

Until the associated manuscript is published, please cite this repository if you use the implementation or analysis pipeline:

```bibtex
@software{idrees2026selectiveecg,
  author  = {Idrees, Muhammad and Amin, Adnan and Azizi, Salma},
  title   = {Selective ECG Classification Under Signal Degradation:
             Code and Reproduction Pipeline},
  year    = {2026},
  url     = {https://github.com/idrees118/selective-ecg-signal-degradation}
}
```

After publication, this section will be updated with the manuscript DOI and full citation.

---

## License

This repository is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Contact

### Muhammad Idrees

GitHub: [@idrees118](https://github.com/idrees118)

### Dr. Adnan Amin

Corresponding author
Institute of Management Sciences (IMSciences), Peshawar, Pakistan
Email: **[adnan.amin@imsciences.edu.pk](mailto:adnan.amin@imsciences.edu.pk)**

---

<div align="center">

### Research code for reliable and selective ECG classification

**PTB-XL · Selective Prediction · Reliability · Signal Degradation · AURC · ECG**

</div>
