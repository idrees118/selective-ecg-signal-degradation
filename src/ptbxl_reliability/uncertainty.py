"""Multilabel Monte Carlo dropout uncertainty mathematics."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


ENTROPY_EPSILON = 1.0e-12


def bernoulli_entropy(
    probabilities: np.ndarray,
    *,
    epsilon: float = ENTROPY_EPSILON,
) -> np.ndarray:
    """Element-wise Bernoulli entropy in nats."""

    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if not np.isfinite(p).all():
        raise ValueError(
            "Probabilities contain NaN or Inf"
        )

    if (p < 0.0).any() or (p > 1.0).any():
        raise ValueError(
            "Probabilities must lie in [0,1]"
        )

    if epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError(
            "epsilon must lie in (0, 0.5)"
        )

    clipped = np.clip(
        p,
        epsilon,
        1.0 - epsilon,
    )

    return -(
        clipped * np.log(clipped)
        + (1.0 - clipped)
        * np.log(1.0 - clipped)
    )


def mc_uncertainty(
    mc_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute multilabel uncertainty from MC probabilities.

    Expected input shape:

        (T, N, K)

    T = MC passes
    N = ECGs
    K = 5 labels
    """

    samples = np.asarray(
        mc_probabilities,
        dtype=np.float64,
    )

    if samples.ndim != 3:
        raise ValueError(
            "Expected MC probabilities with shape "
            "(passes, records, classes)"
        )

    if samples.shape[0] < 2:
        raise ValueError(
            "At least two MC passes are required"
        )

    if samples.shape[2] != 5:
        raise ValueError(
            "Expected exactly five classes"
        )

    if not np.isfinite(samples).all():
        raise ValueError(
            "MC probabilities contain NaN/Inf"
        )

    if (
        (samples < 0.0).any()
        or (samples > 1.0).any()
    ):
        raise ValueError(
            "MC probabilities must lie in [0,1]"
        )

    mean_probability = np.mean(
        samples,
        axis=0,
        dtype=np.float64,
    )

    predictive_entropy = bernoulli_entropy(
        mean_probability
    )

    expected_entropy = np.mean(
        bernoulli_entropy(
            samples
        ),
        axis=0,
        dtype=np.float64,
    )

    mutual_information = (
        predictive_entropy
        - expected_entropy
    )

    # MI is mathematically non-negative.
    # Permit only numerical-roundoff violations.
    minimum_mi = float(
        np.min(
            mutual_information
        )
    )

    if minimum_mi < -1e-10:
        raise ValueError(
            "Mutual information is materially negative: "
            f"{minimum_mi}"
        )

    mutual_information = np.maximum(
        mutual_information,
        0.0,
    )

    probability_std = np.std(
        samples,
        axis=0,
        ddof=0,
        dtype=np.float64,
    )

    mean_mi = np.mean(
        mutual_information,
        axis=1,
        dtype=np.float64,
    )

    max_mi = np.max(
        mutual_information,
        axis=1,
    )

    mean_predictive_entropy = np.mean(
        predictive_entropy,
        axis=1,
        dtype=np.float64,
    )

    return {
        "mean_probability":
            mean_probability,

        "probability_std":
            probability_std,

        "predictive_entropy":
            predictive_entropy,

        "expected_entropy":
            expected_entropy,

        "mutual_information":
            mutual_information,

        "mean_mutual_information":
            mean_mi,

        "max_mutual_information":
            max_mi,

        "mean_predictive_entropy":
            mean_predictive_entropy,
    }


def enable_mc_dropout(
    model: nn.Module,
) -> int:
    """Freeze normal evaluation behavior except Dropout.

    Returns the number of Dropout modules activated.
    """

    model.eval()

    activated = 0

    for module in model.modules():
        if isinstance(
            module,
            nn.Dropout,
        ):
            module.train()
            activated += 1

    if activated == 0:
        raise ValueError(
            "Model contains no nn.Dropout modules"
        )

    # Explicitly verify every BatchNorm remains in eval mode.
    for module in model.modules():
        if isinstance(
            module,
            nn.BatchNorm1d,
        ):
            if module.training:
                raise ValueError(
                    "BatchNorm unexpectedly entered "
                    "training mode during MC dropout"
                )

    return activated
