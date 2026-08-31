"""PTB-XL annotation-confidence score for the superclass task."""

from __future__ import annotations

import numpy as np


def positive_superclass_confidence(
    likelihoods: list[float],
) -> float:
    """Strongest available likelihood for one positive superclass.

    Zero means unavailable and is therefore ignored.

    Returns NaN if no likelihood is available.
    """

    values = np.asarray(
        likelihoods,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "likelihoods must be one-dimensional"
        )

    if values.size == 0:
        raise ValueError(
            "Positive superclass must have "
            "at least one contributing code"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Likelihood contains NaN/Inf"
        )

    if (
        (values < 0.0).any()
        or (values > 100.0).any()
    ):
        raise ValueError(
            "Likelihood must lie in [0,100]"
        )

    known = values[
        values > 0.0
    ]

    if known.size == 0:
        return float("nan")

    return float(
        np.max(known) / 100.0
    )


def record_annotation_confidence(
    positive_superclass_likelihoods:
        list[list[float]],
) -> dict[str, float | int | bool]:
    """Aggregate positive-superclass annotation confidence.

    Missing superclass likelihoods are not interpreted as zero.
    """

    if len(
        positive_superclass_likelihoods
    ) == 0:
        raise ValueError(
            "Record must have at least one "
            "positive superclass"
        )

    superclass_scores = [
        positive_superclass_confidence(
            values
        )
        for values
        in positive_superclass_likelihoods
    ]

    known = np.asarray(
        [
            value
            for value in superclass_scores
            if np.isfinite(value)
        ],
        dtype=np.float64,
    )

    n_positive = len(
        superclass_scores
    )

    n_known = int(
        known.size
    )

    coverage = (
        n_known / n_positive
    )

    if n_known == 0:
        confidence = float("nan")
        ambiguity = float("nan")
    else:
        confidence = float(
            np.mean(
                known,
                dtype=np.float64,
            )
        )

        ambiguity = float(
            1.0 - confidence
        )

    return {
        "label_confidence":
            confidence,

        "label_ambiguity":
            ambiguity,

        "positive_superclasses":
            n_positive,

        "known_superclass_likelihoods":
            n_known,

        "coverage":
            float(coverage),

        "complete_coverage":
            bool(
                n_known == n_positive
            ),
    }
