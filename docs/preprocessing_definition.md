# ECG Preprocessing Definition — Version 1

Dataset: PTB-XL v1.0.3
Input: records100
Sampling frequency: 100 Hz
Duration: 10 seconds
Native WFDB shape: (1000 time samples, 12 leads)
Model shape: (12 leads, 1000 time samples)
Physical unit: mV

## Signal processing

No additional:
- resampling
- band-pass filtering
- notch filtering
- detrending
- baseline correction
- clipping
- amplitude normalization per ECG
- normalization per lead

is applied in preprocessing version 1.

The PTB-XL 100-Hz waveform is used as distributed.

This avoids introducing transformations that could interfere with the later
signal-quality and controlled-corruption experiments.

## Standardization

A single global scalar mean and standard deviation are estimated using
training records only (PTB-XL folds 1–8).

Let all scalar ECG values in the training cohort be:

    x_1, ..., x_N

where:

    N = number_of_training_ECGs × 1000 × 12

The training mean is:

    mu = (1/N) * sum_j x_j

The population variance is:

    variance = (1/N) * sum_j (x_j - mu)^2

and:

    sigma = sqrt(variance)

Every training, validation, and test signal is then transformed using:

    z = (x - mu) / sigma

The same frozen mu and sigma are used for all splits.

Validation and test values MUST NOT contribute to mu or sigma.

Variance uses ddof = 0, consistent with sklearn StandardScaler and the
reference PTB-XL benchmarking implementation.

No epsilon is added to sigma. A non-positive or non-finite sigma is treated
as an implementation/data error.

Statistics are accumulated in float64. Model tensors may later be converted
to float32 only after standardization.

## Leakage rule

Only manifest rows where:

    split == "train"

may be used to estimate preprocessing parameters.

Any attempt to fit preprocessing statistics using validation or test records
must fail explicitly.
