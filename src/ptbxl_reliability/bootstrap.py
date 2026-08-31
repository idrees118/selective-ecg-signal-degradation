"""Patient-clustered bootstrap utilities for selective prediction."""

from __future__ import annotations

import numpy as np


def weighted_grouped_aurc(
    reliability: np.ndarray,
    losses: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute tie-grouped AURC with non-negative observation weights.

    Integer weights are exactly equivalent to explicitly replicating each
    observation that many times. This is used for the patient-level cluster
    bootstrap, where every ECG from a resampled patient receives the same
    patient multiplicity.
    """

    reliability = np.asarray(
        reliability,
        dtype=np.float64,
    )

    losses = np.asarray(
        losses,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    if (
        reliability.ndim != 1
        or losses.ndim != 1
        or weights.ndim != 1
    ):
        raise ValueError(
            "reliability, losses, and weights must be one-dimensional"
        )

    if not (
        len(reliability)
        == len(losses)
        == len(weights)
    ):
        raise ValueError(
            "reliability, losses, and weights must have equal length"
        )

    if len(reliability) == 0:
        raise ValueError(
            "Inputs must be non-empty"
        )

    for name, values in (
        ("reliability", reliability),
        ("losses", losses),
        ("weights", weights),
    ):
        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} contains NaN/Inf"
            )

    if (weights < 0.0).any():
        raise ValueError(
            "weights must be non-negative"
        )

    total_weight = float(
        np.sum(
            weights,
            dtype=np.float64,
        )
    )

    if total_weight <= 0.0:
        raise ValueError(
            "At least one observation must have positive weight"
        )

    order = np.argsort(
        -reliability,
        kind="mergesort",
    )

    score = reliability[
        order
    ]

    loss = losses[
        order
    ]

    weight = weights[
        order
    ]

    starts = np.concatenate(
        (
            np.asarray(
                [0],
                dtype=np.int64,
            ),
            np.flatnonzero(
                score[1:]
                != score[:-1]
            ).astype(
                np.int64
            )
            + 1,
        )
    )

    group_weight = np.add.reduceat(
        weight,
        starts,
    )

    group_weighted_loss = np.add.reduceat(
        weight * loss,
        starts,
    )

    # A bootstrap replicate can assign zero total weight to an entire
    # reliability tie group. Such a group contributes no coverage.
    keep = (
        group_weight > 0.0
    )

    group_weight = group_weight[
        keep
    ]

    group_weighted_loss = (
        group_weighted_loss[
            keep
        ]
    )

    cumulative_weight = np.cumsum(
        group_weight,
        dtype=np.float64,
    )

    cumulative_loss = np.cumsum(
        group_weighted_loss,
        dtype=np.float64,
    )

    coverage = (
        cumulative_weight
        / total_weight
    )

    risk = (
        cumulative_loss
        / cumulative_weight
    )

    delta_coverage = np.diff(
        np.concatenate(
            (
                np.asarray(
                    [0.0],
                    dtype=np.float64,
                ),
                coverage,
            )
        )
    )

    aurc = float(
        np.sum(
            delta_coverage
            * risk,
            dtype=np.float64,
        )
    )

    full_risk = float(
        np.sum(
            weights * losses,
            dtype=np.float64,
        )
        / total_weight
    )

    if not np.isclose(
        coverage[-1],
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Full coverage is not one"
        )

    if not np.isclose(
        risk[-1],
        full_risk,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Full-coverage weighted risk mismatch"
        )

    return aurc


def patient_record_weights(
    patient_ids: np.ndarray,
    *,
    seed: int,
    replicate: int,
) -> np.ndarray:
    """Return record weights for one deterministic patient bootstrap draw."""

    patient_ids = np.asarray(
        patient_ids,
        dtype=np.int64,
    )

    if patient_ids.ndim != 1:
        raise ValueError(
            "patient_ids must be one-dimensional"
        )

    if len(patient_ids) == 0:
        raise ValueError(
            "patient_ids must be non-empty"
        )

    unique_patients, inverse = np.unique(
        patient_ids,
        return_inverse=True,
    )

    n_patients = len(
        unique_patients
    )

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(replicate),
            ]
        )
    )

    sampled = rng.integers(
        0,
        n_patients,
        size=n_patients,
        endpoint=False,
    )

    multiplicity = np.bincount(
        sampled,
        minlength=n_patients,
    ).astype(
        np.float64
    )

    return multiplicity[
        inverse
    ]
