"""Selective-prediction mathematics for multilabel PTB-XL."""

from __future__ import annotations

import numpy as np


def sample_hamming_error(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Per-record fraction of incorrectly predicted labels."""

    y = np.asarray(
        targets
    )

    pred = np.asarray(
        predictions
    )

    if y.ndim != 2 or pred.ndim != 2:
        raise ValueError(
            "targets and predictions must be 2-D"
        )

    if y.shape != pred.shape:
        raise ValueError(
            "target/prediction shape mismatch"
        )

    if y.shape[1] != 5:
        raise ValueError(
            "Expected exactly five labels"
        )

    if not np.isin(
        y,
        [0, 1],
    ).all():
        raise ValueError(
            "Targets must be binary"
        )

    if not np.isin(
        pred,
        [0, 1],
    ).all():
        raise ValueError(
            "Predictions must be binary"
        )

    return np.mean(
        y != pred,
        axis=1,
        dtype=np.float64,
    )


def mean_bernoulli_confidence(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Mean probability assigned to the more likely binary state."""

    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if p.ndim != 2 or p.shape[1] != 5:
        raise ValueError(
            "Expected probabilities with shape (N,5)"
        )

    if not np.isfinite(p).all():
        raise ValueError(
            "Probabilities contain NaN/Inf"
        )

    if (
        (p < 0.0).any()
        or (p > 1.0).any()
    ):
        raise ValueError(
            "Probabilities must lie in [0,1]"
        )

    certainty = np.maximum(
        p,
        1.0 - p,
    )

    return np.mean(
        certainty,
        axis=1,
        dtype=np.float64,
    )


def midrank_ecdf(
    reference: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Empirical percentile with midpoint treatment of exact ties.

    Larger output means larger value relative to the reference
    distribution.

    For x:

        percentile =
            [#(reference < x)
             + 0.5 * #(reference == x)] / N
    """

    ref = np.asarray(
        reference,
        dtype=np.float64,
    )

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if ref.ndim != 1 or x.ndim != 1:
        raise ValueError(
            "reference and values must be one-dimensional"
        )

    if ref.size == 0:
        raise ValueError(
            "reference cannot be empty"
        )

    if not np.isfinite(ref).all():
        raise ValueError(
            "reference contains NaN/Inf"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "values contain NaN/Inf"
        )

    sorted_ref = np.sort(
        ref
    )

    left = np.searchsorted(
        sorted_ref,
        x,
        side="left",
    )

    right = np.searchsorted(
        sorted_ref,
        x,
        side="right",
    )

    return (
        left
        + 0.5 * (
            right - left
        )
    ) / float(
        sorted_ref.size
    )


def grouped_risk_coverage(
    reliability: np.ndarray,
    losses: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Tie-invariant risk-coverage curve and discrete AURC.

    Higher reliability is retained first.

    Exact reliability ties are added as one group. This prevents
    arbitrary ordering within a tied group from changing AURC.

    For each group g:

        coverage_g = retained_count / N
        risk_g = retained_loss_sum / retained_count

    AURC is the right-continuous discrete integral:

        sum_g delta_coverage_g * risk_g
    """

    score = np.asarray(
        reliability,
        dtype=np.float64,
    )

    loss = np.asarray(
        losses,
        dtype=np.float64,
    )

    if score.ndim != 1:
        raise ValueError(
            "reliability must be one-dimensional"
        )

    if loss.ndim != 1:
        raise ValueError(
            "losses must be one-dimensional"
        )

    if score.shape != loss.shape:
        raise ValueError(
            "reliability/loss shape mismatch"
        )

    if score.size == 0:
        raise ValueError(
            "Inputs cannot be empty"
        )

    if not np.isfinite(score).all():
        raise ValueError(
            "Reliability contains NaN/Inf"
        )

    if not np.isfinite(loss).all():
        raise ValueError(
            "Loss contains NaN/Inf"
        )

    if (
        (loss < 0.0).any()
        or (loss > 1.0).any()
    ):
        raise ValueError(
            "Loss must lie in [0,1]"
        )

    unique_scores = np.unique(
        score
    )[::-1]

    total_records = int(
        score.size
    )

    cumulative_records = 0
    cumulative_loss = 0.0
    previous_coverage = 0.0

    coverages = []
    risks = []
    group_sizes = []
    score_levels = []

    aurc = 0.0

    for value in unique_scores:
        mask = score == value

        group_size = int(
            np.sum(mask)
        )

        group_loss = float(
            np.sum(
                loss[mask],
                dtype=np.float64,
            )
        )

        cumulative_records += (
            group_size
        )

        cumulative_loss += (
            group_loss
        )

        coverage = (
            cumulative_records
            / total_records
        )

        risk = (
            cumulative_loss
            / cumulative_records
        )

        delta_coverage = (
            coverage
            - previous_coverage
        )

        aurc += (
            delta_coverage
            * risk
        )

        coverages.append(
            coverage
        )

        risks.append(
            risk
        )

        group_sizes.append(
            group_size
        )

        score_levels.append(
            float(value)
        )

        previous_coverage = coverage

    if cumulative_records != total_records:
        raise AssertionError(
            "Risk-coverage calculation lost records"
        )

    if not np.isclose(
        coverages[-1],
        1.0,
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError(
            "Final coverage is not one"
        )

    full_risk = float(
        np.mean(
            loss,
            dtype=np.float64,
        )
    )

    if not np.isclose(
        risks[-1],
        full_risk,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Full-coverage risk mismatch"
        )

    return {
        "coverage":
            np.asarray(
                coverages,
                dtype=np.float64,
            ),

        "risk":
            np.asarray(
                risks,
                dtype=np.float64,
            ),

        "group_size":
            np.asarray(
                group_sizes,
                dtype=np.int64,
            ),

        "score_level":
            np.asarray(
                score_levels,
                dtype=np.float64,
            ),

        "aurc":
            float(aurc),

        "full_coverage_risk":
            full_risk,
    }


def risk_at_or_above_coverage(
    coverage: np.ndarray,
    risk: np.ndarray,
    target_coverage: float,
) -> tuple[float, float]:
    """Return first achievable point at or above requested coverage."""

    c = np.asarray(
        coverage,
        dtype=np.float64,
    )

    r = np.asarray(
        risk,
        dtype=np.float64,
    )

    if c.shape != r.shape:
        raise ValueError(
            "coverage/risk shape mismatch"
        )

    if (
        target_coverage <= 0.0
        or target_coverage > 1.0
    ):
        raise ValueError(
            "target_coverage must lie in (0,1]"
        )

    indices = np.flatnonzero(
        c >= target_coverage
    )

    if indices.size == 0:
        raise ValueError(
            "Requested coverage is not reachable"
        )

    index = int(
        indices[0]
    )

    return (
        float(c[index]),
        float(r[index]),
    )
