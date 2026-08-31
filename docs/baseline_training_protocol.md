# Baseline Training Protocol — v1

## Fixed model

ResNet1D-Wang-style classifier.

Input:
    (batch, 12, 1000)

Output:
    five raw logits

Task:
    multi-label diagnostic superclass prediction

Loss:
    BCEWithLogitsLoss

No sigmoid is included inside the model.

## Training data

PTB-XL folds 1–8 only.

Records:
    17,084

## Validation data

PTB-XL fold 9 only.

Records:
    2,146

Validation may be used for:
- checkpoint selection
- later threshold selection
- later calibration/reliability development

Validation may not alter cohort membership or preprocessing.

## Test data

PTB-XL fold 10.

Records:
    2,158

The test split is forbidden during:
- training
- checkpoint selection
- hyperparameter selection
- threshold selection
- calibration fitting
- reliability-weight optimization

## Optimization

Optimizer:
    AdamW

Maximum learning rate:
    0.01

Weight decay:
    0.01

Batch size:
    128

Epochs:
    50

Scheduler:
    OneCycleLR

pct_start:
    0.30

annealing:
    cosine

div_factor:
    25

final_div_factor:
    10000

## Checkpoint selection

All 50 epochs are executed.

After every epoch, validation probabilities are evaluated.

Primary checkpoint-selection metric:

    macro AUROC

For classes k = 1,...,5:

    AUROC_macro =
        (1/5) * sum_k AUROC(y_k, p_k)

The checkpoint with the largest validation macro-AUROC is retained.

No test result participates in this decision.

## Random seeds

Three pre-specified independent training seeds:

    20260826
    20260827
    20260828

Hyperparameters remain identical for all seeds.

No seed will be discarded because its performance is lower.

## Thresholds

No classification threshold is optimized during baseline training.

AUROC uses continuous sigmoid probabilities and is threshold-independent.

Threshold-dependent metrics such as macro-F1 will be handled later using
validation data only and then frozen before final test evaluation.
