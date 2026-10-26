"""AURC/AUGRC, risk at coverage, extra selectors and bootstrap helpers.

Same tie-grouped right-endpoint convention as grouped_risk_coverage.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from ptbxl_reliability.selective_prediction import (
    grouped_risk_coverage,
    midrank_ecdf,
    risk_at_or_above_coverage,
)


PROBABILITY_CLIP = 1e-7


def _as_1d(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if x.ndim != 1 or x.size == 0:
        raise ValueError(
            f"{name} must be a non-empty one-dimensional array"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"{name} contains NaN/Inf"
        )

    return x


# ---------------------------------------------------------------------------
# Risk-coverage metrics
# ---------------------------------------------------------------------------

def tie_groups(
    reliability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Descending order and start index of each exact-tie group."""
    score = _as_1d(
        reliability,
        "reliability",
    )

    order = np.argsort(
        -score,
        kind="stable",
    )

    sorted_score = score[order]

    starts = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.flatnonzero(
                sorted_score[1:]
                != sorted_score[:-1]
            ).astype(np.int64)
            + 1,
        )
    )

    return order, starts


def weighted_aurc_augrc(
    reliability: np.ndarray,
    losses: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Tie-grouped AURC and AUGRC; weights (N,) or (B, N).

    selective risk = S_g / W_g (AURC), generalized risk = S_g / W (AUGRC).
    """
    loss = _as_1d(
        losses,
        "losses",
    )

    if (loss < 0.0).any() or (loss > 1.0).any():
        raise ValueError(
            "losses must lie in [0,1]"
        )

    w = np.asarray(
        weights,
        dtype=np.float64,
    )

    single = w.ndim == 1

    w = np.atleast_2d(w)

    if w.shape[1] != loss.size:
        raise ValueError(
            "weights/losses length mismatch"
        )

    if (w < 0.0).any() or not np.isfinite(w).all():
        raise ValueError(
            "weights must be finite and non-negative"
        )

    order, starts = tie_groups(
        reliability
    )

    if order.size != loss.size:
        raise ValueError(
            "reliability/losses length mismatch"
        )

    w_sorted = w[:, order]
    loss_sorted = loss[order]

    group_weight = np.add.reduceat(
        w_sorted,
        starts,
        axis=1,
    )

    group_loss = np.add.reduceat(
        w_sorted
        * loss_sorted[None, :],
        starts,
        axis=1,
    )

    total = group_weight.sum(
        axis=1,
        keepdims=True,
    )

    if (total <= 0.0).any():
        raise ValueError(
            "Total weight must be positive"
        )

    cumulative_weight = np.cumsum(
        group_weight,
        axis=1,
    )

    cumulative_loss = np.cumsum(
        group_loss,
        axis=1,
    )

    delta_coverage = group_weight / total

    # Zero-weight tie groups add no coverage; avoid 0/0.
    selective_risk = np.divide(
        cumulative_loss,
        cumulative_weight,
        out=np.zeros_like(cumulative_loss),
        where=cumulative_weight > 0.0,
    )

    generalized_risk = cumulative_loss / total

    aurc = np.sum(
        delta_coverage
        * selective_risk,
        axis=1,
    )

    augrc = np.sum(
        delta_coverage
        * generalized_risk,
        axis=1,
    )

    if single:
        return (
            np.asarray(aurc[0]),
            np.asarray(augrc[0]),
        )

    return aurc, augrc


def aurc_augrc(
    reliability: np.ndarray,
    losses: np.ndarray,
) -> tuple[float, float]:
    loss = _as_1d(
        losses,
        "losses",
    )

    aurc, augrc = weighted_aurc_augrc(
        reliability,
        loss,
        np.ones_like(loss),
    )

    return float(aurc), float(augrc)


def risk_at_coverages(
    reliability: np.ndarray,
    losses: np.ndarray,
    targets: tuple[float, ...] = (0.7, 0.8, 0.9),
) -> dict[float, tuple[float, float]]:
    """(achieved coverage, risk) at the first tie group reaching each target."""
    curve = grouped_risk_coverage(
        reliability,
        losses,
    )

    return {
        target: risk_at_or_above_coverage(
            curve["coverage"],
            curve["risk"],
            target,
        )
        for target in targets
    }


# ---------------------------------------------------------------------------
# Confidence variants
# ---------------------------------------------------------------------------

def _logit(
    probabilities: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
        PROBABILITY_CLIP,
        1.0 - PROBABILITY_CLIP,
    )

    return np.log(p) - np.log1p(-p)


def threshold_margin(
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Mean |logit(p) - logit(threshold)| over labels. Higher is more reliable."""
    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    tau = np.asarray(
        thresholds,
        dtype=np.float64,
    )

    if p.ndim != 2 or tau.shape != (p.shape[1],):
        raise ValueError(
            "Expected probabilities (N,K) and thresholds (K,)"
        )

    return np.mean(
        np.abs(
            _logit(p)
            - _logit(tau)[None, :]
        ),
        axis=1,
    )


def fit_classwise_temperature(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> np.ndarray:
    """One temperature per class minimizing validation binary cross-entropy."""
    z = _logit(probabilities)

    y = np.asarray(
        targets,
        dtype=np.float64,
    )

    if z.shape != y.shape or z.ndim != 2:
        raise ValueError(
            "probabilities/targets shape mismatch"
        )

    temperatures = np.empty(
        z.shape[1],
        dtype=np.float64,
    )

    for k in range(z.shape[1]):

        def nll(t, zk=z[:, k], yk=y[:, k]):
            s = zk / t
            return float(
                np.mean(
                    yk * np.logaddexp(0.0, -s)
                    + (1.0 - yk) * np.logaddexp(0.0, s)
                )
            )

        result = minimize_scalar(
            nll,
            bounds=bounds,
            method="bounded",
        )

        temperatures[k] = float(result.x)

    return temperatures


def temperature_scaled_confidence(
    probabilities: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    z = _logit(probabilities)

    t = np.asarray(
        temperatures,
        dtype=np.float64,
    )

    if t.shape != (z.shape[1],) or (t <= 0.0).any():
        raise ValueError(
            "temperatures must be positive with one value per class"
        )

    p = 1.0 / (1.0 + np.exp(-z / t[None, :]))

    return np.mean(
        np.maximum(p, 1.0 - p),
        axis=1,
    )


# ---------------------------------------------------------------------------
# Combined selectors
# ---------------------------------------------------------------------------

def percentile_selector(
    badness: list[np.ndarray],
    references: list[np.ndarray],
) -> np.ndarray:
    """Negated equal-weight mean of midrank badness percentiles (v1 rule)."""
    if len(badness) != len(references) or not badness:
        raise ValueError(
            "badness and references must be non-empty and aligned"
        )

    percentiles = [
        midrank_ecdf(
            _as_1d(ref, "reference"),
            _as_1d(values, "badness"),
        )
        for values, ref in zip(badness, references)
    ]

    return -np.mean(
        percentiles,
        axis=0,
    )


class ErrorStacker:
    """Logistic P(any label error) from reliability features; selector = -P."""

    def __init__(
        self,
        feature_names: list[str],
    ) -> None:
        self.feature_names = list(feature_names)
        self.mean_ = None
        self.scale_ = None
        self.model_ = None

    def _matrix(
        self,
        features: dict[str, np.ndarray],
    ) -> np.ndarray:
        return np.column_stack(
            [
                _as_1d(features[name], name)
                for name in self.feature_names
            ]
        )

    def fit(
        self,
        features: dict[str, np.ndarray],
        losses: np.ndarray,
    ) -> "ErrorStacker":
        x = self._matrix(features)

        y = (
            _as_1d(losses, "losses")
            > 0.0
        ).astype(np.int64)

        if np.unique(y).size != 2:
            raise ValueError(
                "Stacker needs records with and without errors"
            )

        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0

        self.model_ = LogisticRegression(
            C=1.0,
            max_iter=1000,
        ).fit(
            (x - self.mean_) / self.scale_,
            y,
        )

        return self

    def selector(
        self,
        features: dict[str, np.ndarray],
    ) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError(
                "Stacker is not fitted"
            )

        x = (self._matrix(features) - self.mean_) / self.scale_

        return -self.model_.predict_proba(x)[:, 1]

    def coefficients(
        self,
    ) -> dict[str, float]:
        return {
            "intercept": float(self.model_.intercept_[0]),
            **{
                name: float(value)
                for name, value in zip(
                    self.feature_names,
                    self.model_.coef_[0],
                )
            },
        }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def cluster_weight_matrix(
    cluster_ids: np.ndarray,
    *,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    """(B, N) record weights; records share their cluster's multiplicity."""
    ids = np.asarray(cluster_ids)

    if ids.ndim != 1 or ids.size == 0:
        raise ValueError(
            "cluster_ids must be a non-empty one-dimensional array"
        )

    _, inverse = np.unique(
        ids,
        return_inverse=True,
    )

    n_clusters = int(inverse.max()) + 1

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), n_clusters]
        )
    )

    weights = np.empty(
        (n_replicates, ids.size),
        dtype=np.float64,
    )

    for b in range(n_replicates):
        multiplicity = np.bincount(
            rng.integers(0, n_clusters, size=n_clusters),
            minlength=n_clusters,
        )
        weights[b] = multiplicity[inverse]

    return weights


def bootstrap_two_sided_p(
    deltas: np.ndarray,
) -> float:
    """Two-sided bootstrap p-value (twice the smaller tail), floored at 1/B."""
    d = _as_1d(
        deltas,
        "deltas",
    )

    below = np.mean(d <= 0.0)
    above = np.mean(d >= 0.0)

    return float(min(1.0, max(2.0 * min(below, above), 1.0 / d.size)))


def holm_adjust(
    p_values: np.ndarray,
) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = _as_1d(
        p_values,
        "p_values",
    )

    m = p.size
    order = np.argsort(p, kind="stable")

    adjusted_sorted = np.maximum.accumulate(
        (m - np.arange(m)) * p[order]
    )

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)

    return adjusted
