"""Metadata-derived PTB-XL signal-quality proxy."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


QUALITY_FIELDS = (
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
)


def metadata_annotation_present(value) -> bool:
    """Return whether a PTB-XL quality field has an annotation.

    The fields are free-text without regular syntax. We therefore use
    presence/absence only and do not infer severity or affected-lead count.
    """

    if value is None:
        return False

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return False

    if isinstance(value, str):
        return value.strip().lower() not in {
            "",
            "nan",
            "none",
        }

    return True


def signal_quality_from_presence(
    field_presence: Mapping[str, bool],
) -> dict[str, float | int | bool]:
    """Compute equal-weight artifact burden and Q."""

    expected = set(QUALITY_FIELDS)
    observed = set(field_presence)

    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)

        raise ValueError(
            f"Quality-field mismatch; "
            f"missing={missing}, extra={extra}"
        )

    for field, value in field_presence.items():
        if not isinstance(
            value,
            (bool, np.bool_),
        ):
            raise TypeError(
                f"{field} presence must be boolean"
            )

    n_present = int(
        sum(
            bool(field_presence[field])
            for field in QUALITY_FIELDS
        )
    )

    artifact_burden = float(
        n_present / len(QUALITY_FIELDS)
    )

    signal_quality = float(
        1.0 - artifact_burden
    )

    return {
        "n_quality_artifact_categories":
            n_present,

        "artifact_burden":
            artifact_burden,

        "signal_quality":
            signal_quality,

        "any_quality_artifact":
            bool(n_present > 0),

        "high_quality_no_recorded_artifact":
            bool(n_present == 0),
    }
