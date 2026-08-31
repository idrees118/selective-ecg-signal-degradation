"""Validation-only metrics and threshold selection.

These functions operate on multilabel binary targets.

Threshold selection is performed independently for each class using
validation data only. For each class, every unique predicted probability
is considered as a threshold under the rule:

    prediction = probability >= threshold

The threshold maximizing F1 is selected. If multiple thresholds produce
the same F1 to numerical precision, the highest threshold is retained.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from ptbxl_reliability.metrics import validate_multilabel_arrays


def classwise_average_precision(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    y, p = validate_multilabel_arrays(
        targets,
        probabilities,
    )

    scores = np.empty(
        y.shape[1],
        dtype=np.float64,
    )

    for k in range(y.shape[1]):
        if np.unique(y[:, k]).size != 2:
            raise ValueError(
                f"Class {k} lacks both labels"
            )

        scores[k] = average_precision_score(
            y[:, k],
            p[:, k],
        )

    return scores


def macro_average_precision(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    return float(
        np.mean(
            classwise_average_precision(
                targets,
                probabilities,
            ),
            dtype=np.float64,
        )
    )


def classwise_brier(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Binary Brier score for each label.

    Brier_k = mean_i (p_ik - y_ik)^2
    """

    y, p = validate_multilabel_arrays(
        targets,
        probabilities,
    )

    return np.mean(
        (p - y) ** 2,
        axis=0,
        dtype=np.float64,
    )


def macro_brier(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    return float(
        np.mean(
            classwise_brier(
                targets,
                probabilities,
            ),
            dtype=np.float64,
        )
    )


def binary_counts(
    target: np.ndarray,
    prediction: np.ndarray,
) -> tuple[int, int, int, int]:
    y = np.asarray(
        target,
        dtype=np.int64,
    )

    pred = np.asarray(
        prediction,
        dtype=np.int64,
    )

    if y.shape != pred.shape:
        raise ValueError(
            "Target/prediction shape mismatch"
        )

    if not np.isin(
        y,
        [0, 1],
    ).all():
        raise ValueError(
            "Target must be binary"
        )

    if not np.isin(
        pred,
        [0, 1],
    ).all():
        raise ValueError(
            "Prediction must be binary"
        )

    tp = int(
        np.sum(
            (y == 1)
            & (pred == 1)
        )
    )

    fp = int(
        np.sum(
            (y == 0)
            & (pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y == 1)
            & (pred == 0)
        )
    )

    tn = int(
        np.sum(
            (y == 0)
            & (pred == 0)
        )
    )

    return tp, fp, fn, tn


def precision_recall_f1(
    target: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, float]:
    tp, fp, fn, _ = binary_counts(
        target,
        prediction,
    )

    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        tp / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    return (
        float(precision),
        float(recall),
        float(f1),
    )


def select_f1_threshold(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    """Find validation threshold maximizing binary F1."""

    y = np.asarray(
        target,
        dtype=np.int64,
    )

    p = np.asarray(
        probability,
        dtype=np.float64,
    )

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError(
            "Inputs must be one-dimensional"
        )

    if y.shape != p.shape:
        raise ValueError(
            "Target/probability shape mismatch"
        )

    if not np.isin(
        y,
        [0, 1],
    ).all():
        raise ValueError(
            "Target must be binary"
        )

    if not np.isfinite(p).all():
        raise ValueError(
            "Probability contains NaN/Inf"
        )

    if (p < 0).any() or (p > 1).any():
        raise ValueError(
            "Probabilities must be in [0,1]"
        )

    if np.unique(y).size != 2:
        raise ValueError(
            "Threshold fitting requires "
            "positive and negative examples"
        )

    # Descending order means the first threshold in an
    # exact F1 tie is the highest, which is our frozen
    # conservative tie policy.
    thresholds = np.unique(p)[::-1]

    best_threshold = None
    best_f1 = -1.0
    best_precision = None
    best_recall = None
    best_counts = None

    tolerance = 1e-15

    for threshold in thresholds:
        prediction = (
            p >= threshold
        ).astype(
            np.int64
        )

        precision, recall, f1 = (
            precision_recall_f1(
                y,
                prediction,
            )
        )

        if f1 > best_f1 + tolerance:
            best_threshold = float(
                threshold
            )
            best_f1 = float(f1)
            best_precision = float(
                precision
            )
            best_recall = float(
                recall
            )
            best_counts = binary_counts(
                y,
                prediction,
            )

    if best_threshold is None:
        raise RuntimeError(
            "No threshold selected"
        )

    tp, fp, fn, tn = best_counts

    return {
        "threshold":
            best_threshold,
        "f1":
            best_f1,
        "precision":
            best_precision,
        "recall":
            best_recall,
        "tp":
            tp,
        "fp":
            fp,
        "fn":
            fn,
        "tn":
            tn,
    }


def select_classwise_f1_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, float | int]]:
    y, p = validate_multilabel_arrays(
        targets,
        probabilities,
    )

    return [
        select_f1_threshold(
            y[:, k],
            p[:, k],
        )
        for k in range(
            y.shape[1]
        )
    ]
