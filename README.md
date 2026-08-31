<div align="center">

# 🫀 Selective ECG Classification Under Signal Degradation
### Reliability-Aware ECG Prediction Under Controlled Signal Corruption

[![Domain](https://img.shields.io/badge/Domain-Medical_AI-007EC6?style=for-the-badge&logo=heart)](https://physionet.org/content/ptb-xl/1.0.3/)
[![Dataset](https://img.shields.io/badge/Dataset-PTB--XL-6F42C1?style=for-the-badge)](https://physionet.org/content/ptb-xl/1.0.3/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge)](LICENSE)

<br/>

**A reproducible selective-classification pipeline that tests whether annotation support, waveform quality, and model uncertainty can identify unreliable ECG predictions better than standard output confidence.**

</div>

---

## 📌 Executive Summary

Modern ECG classifiers can achieve strong predictive performance, but in real clinical settings the important question is not only:

> **“Is the model accurate?”**

It is also:

> **“Can the model recognize which individual predictions are most likely to be wrong?”**

This project studies that question using **PTB-XL**, one of the largest publicly available clinical ECG datasets.

A deep residual ECG classifier was trained on **21,388 ECG recordings** using patient-disjoint folds. We then compared four reliability signals:

- **C — Output Confidence**
- **L — Annotation Support**
- **Q — Waveform Quality**
- **U — Model Uncertainty**

These signals were tested individually and in combinations under both **clean ECGs** and **12 controlled signal degradation conditions**.

### Main finding

> **Output confidence remained the strongest error-ranking signal.**

Across **39 prespecified statistical comparisons**, **38 bootstrap intervals favored confidence** over the tested auxiliary alternatives.

Model uncertainty increased consistently as ECG corruption became more severe, showing that it reacted to degraded signals. However, this did **not** make it better than confidence at ranking individual prediction errors.

---

## 🎯 Research Question

Selective prediction allows a classifier to **defer unreliable cases for manual review** instead of forcing a decision on every ECG.

For this to work well, the reliability score must correctly rank cases so that errors are rejected before correct predictions.

This study asks:

> **Do auxiliary reliability signals provide better selective error ranking than ordinary output confidence when ECG signals become degraded?**

The analysis focuses on a key distinction:

**Detecting that signals are becoming worse is not necessarily the same as identifying which individual predictions are wrong.**

---

## 🧪 Study Design

| Component | Setting |
|------|------|
| **Dataset** | PTB-XL v1.0.3 |
| **Total ECGs** | 21,388 |
| **Diagnostic Targets** | 5 superclasses |
| **Training Folds** | 1–8 |
| **Validation Fold** | 9 |
| **Test Fold** | 10 |
| **Test ECGs** | 2,158 |
| **Test Patients** | 1,877 |
| **Classifier** | ResNet1DWang |
| **Trainable Parameters** | 473,349 |
| **Reliability Signals** | C, L, Q, U |
| **Ranking Strategies** | 8 |
| **Corruption Families** | 4 |
| **Corruption Conditions** | 12 |
| **Primary Metric** | AURC |
| **Bootstrap Replicates** | 5,000 |

---

## 🗂️ Dataset & Patient-Disjoint Splitting

The study uses **PTB-XL v1.0.3**, a large 12-lead ECG dataset.

The recommended PTB-XL fold structure was preserved so that recordings from the same patient do not leak across training and evaluation sets.

| Split | PTB-XL Folds | ECG Recordings |
|------|------|------:|
| **Training** | 1–8 | 17,084 |
| **Validation** | 9 | 2,146 |
| **Final Test** | 10 | 2,158 |
| **Total** | 1–10 | **21,388** |

The held-out test set contains:

- **2,158 ECG recordings**
- **1,877 unique patients**

> **Leakage Protection:** normalization statistics were calculated using the training split only, while class-specific decision thresholds were selected on the validation fold and frozen before final test evaluation.

---

## 🧠 ECG Classification Model

The classifier is a **ResNet1DWang** architecture designed for one-dimensional physiological time-series signals.

### Training Configuration

| Setting | Value |
|------|------|
| Architecture | ResNet1DWang |
| Parameters | 473,349 |
| Optimizer | AdamW |
| LR Schedule | One-Cycle |
| Training Epochs | 50 |
| Output Classes | 5 diagnostic superclasses |
| Final Evaluation Seed | 20260826 |

Three training seeds were used to check validation stability:

```text
20260826
20260827
20260828
