"""Validated metrics for PTB-XL baseline evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


NUM_CLASSES = 5


def validate_multilabel_arrays(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    y = np.asarray(
        targets,
        dtype=np.float64,
    )

    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if y.ndim != 2:
        raise ValueError(
            "targets must be two-dimensional"
        )

    if p.ndim != 2:
        raise ValueError(
            "probabilities must be two-dimensional"
        )

    if y.shape != p.shape:
        raise ValueError(
            f"Shape mismatch: {y.shape} vs {p.shape}"
        )

    if y.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes"
        )

    if not np.isin(
        y,
        [0.0, 1.0],
    ).all():
        raise ValueError(
            "Targets must be binary"
        )

    if not np.isfinite(
        p
    ).all():
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

    return y, p


def classwise_auroc(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Compute one ROC AUC per superclass."""

    y, p = validate_multilabel_arrays(
        targets,
        probabilities,
    )

    scores = np.empty(
        NUM_CLASSES,
        dtype=np.float64,
    )

    for class_index in range(
        NUM_CLASSES
    ):
        labels = y[
            :,
            class_index,
        ]

        if np.unique(
            labels
        ).size != 2:
            raise ValueError(
                f"Class {class_index} does not "
                "contain both positive and "
                "negative examples"
            )

        scores[class_index] = (
            roc_auc_score(
                labels,
                p[
                    :,
                    class_index,
                ],
            )
        )

    return scores


def macro_auroc(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Unweighted mean of five classwise ROC AUC values."""

    scores = classwise_auroc(
        targets,
        probabilities,
    )

    result = float(
        np.mean(
            scores,
            dtype=np.float64,
        )
    )

    if (
        not np.isfinite(result)
        or result < 0.0
        or result > 1.0
    ):
        raise ValueError(
            f"Invalid macro AUROC: {result}"
        )

    return result
