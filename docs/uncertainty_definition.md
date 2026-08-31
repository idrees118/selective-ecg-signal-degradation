# Model Uncertainty Definition — v1

## Purpose

Estimate epistemic uncertainty from the already-trained baseline model
without retraining or modifying its weights.

## Method

Monte Carlo dropout.

The network is first placed in evaluation mode. Batch-normalization layers
therefore use their frozen running statistics.

Only nn.Dropout modules are then switched to training mode so that a new
dropout realization is sampled on each forward pass.

No gradients are calculated.

## Number of stochastic passes

T = 30

This value is fixed before evaluating uncertainty performance.

## Multilabel formulation

The five PTB-XL diagnostic outputs are independent sigmoid/Bernoulli
predictions. They are not treated as one categorical distribution.

For ECG i, Monte Carlo pass t, and label k:

    p_itk = sigmoid(z_itk)

Mean predictive probability:

    pbar_ik = (1/T) sum_t p_itk

Bernoulli entropy:

    H(p) = -p ln(p) - (1-p) ln(1-p)

Predictive entropy:

    PE_ik = H(pbar_ik)

Expected entropy:

    EE_ik = (1/T) sum_t H(p_itk)

Mutual information:

    MI_ik = PE_ik - EE_ik

MI is non-negative theoretically. Tiny negative values caused only by
floating-point roundoff may be clamped to zero.

## Primary ECG-level model uncertainty

    U_i = (1/5) sum_k MI_ik

Higher U_i means greater model uncertainty.

## Secondary quantities retained

For auditing and sensitivity analysis we also retain:

- classwise MI
- classwise predictive entropy
- mean predictive entropy
- maximum classwise MI
- MC mean probabilities
- MC probability standard deviations

These secondary quantities are not allowed to replace the primary
uncertainty definition after results are observed.

## Leakage rule

Uncertainty implementation is developed and audited on validation data.

The test split is not used during implementation, parameter selection,
method selection, or debugging.
