from pathlib import Path

content = r'''<div align="center">

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

## What is this project about?

A medical classifier does not always need to make a prediction.

In **selective classification**, uncertain or unreliable cases can instead be deferred for manual review. The important question is therefore not only:

> **How accurate is the classifier?**

but also:

> **Can we correctly identify which individual predictions are most likely to be wrong?**

This repository investigates whether several auxiliary reliability signals can rank ECG prediction errors better than the classifier's own **output confidence**, particularly when ECG signals are degraded.

We evaluate:

- **Output confidence**
- **Annotation support**
- **Waveform quality**
- **Model uncertainty**
- Equal-weight combinations of the auxiliary signals

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

| Component | Setting |
|---|---|
| Dataset | PTB-XL v1.0.3 |
| ECG recordings used | 21,388 |
| Diagnostic targets | 5 superclasses |
| Training fold | PTB-XL folds 1–8 |
| Validation fold | Fold 9 |
| Final test fold | Fold 10 |
| Test recordings | 2,158 |
| Test patients | 1,877 |
| Classifier | ResNet1DWang |
| Reliability signals | C, L, Q, U |
| Rankings evaluated | 8 |
| Corruption families | 4 |
| Corruption conditions | 12 |
| Primary endpoint | AURC |
| Bootstrap replicates | 5,000 |
| Bootstrap unit | Patient |

---

## Experimental design

### Dataset

We use **PTB-XL v1.0.3**, a large publicly available 12-lead ECG dataset.

After applying the study's diagnostic-superclass selection procedure, the data are divided according to the recommended patient-disjoint PTB-XL folds:

| Split | PTB-XL folds | ECG recordings |
|---|---:|---:|
| Training | 1–8 | 17,084 |
| Validation | 9 | 2,146 |
| Final test | 10 | 2,158 |
| **Total** | **1–10** | **21,388** |

The final test set contains **2,158 ECGs from 1,877 patients**.

Raw PTB-XL waveform files are **not included** in this repository.

---

## Classifier

The ECG classifier is a **ResNet1DWang** model with **473,349 trainable parameters**.

Training uses:

- AdamW optimization
- One-Cycle learning-rate scheduling
- 50 training epochs
- Patient-disjoint data partitions
- Training-only normalization statistics

Class-specific decision thresholds are selected using the **validation fold** and frozen before final test evaluation.

This prevents the held-out test fold from influencing model selection or threshold tuning.

---

## Reliability signals

Each ECG receives four reliability scores.

| Symbol | Reliability signal | Interpretation | Requires reference labels? |
|:---:|---|---|:---:|
| **C** | Output confidence | Extremeness of predicted probabilities relative to 0.5 | No |
| **L** | Annotation support | Strength of the reference diagnostic annotation | **Yes** |
| **Q** | Waveform quality | Estimated artifact and signal-quality burden | No |
| **U** | Model uncertainty | MC-dropout disagreement across stochastic predictions | No |

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
