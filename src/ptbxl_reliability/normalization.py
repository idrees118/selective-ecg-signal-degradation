"""Numerically stable global z-score normalization.

Definition
----------
For all scalar ECG values x_i from TRAINING ECGs only:

    mean = (1/N) * sum_i x_i

    variance = (1/N) * sum_i (x_i - mean)^2

    std = sqrt(variance)

    z = (x - mean) / std

Variance uses ddof=0.

Statistics are accumulated in float64 with the parallel/Chan form of
Welford's algorithm to avoid constructing one enormous in-memory array.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass
class RunningPopulationStats:
    """Streaming population mean/variance accumulator."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """Add one finite numeric batch."""

        x = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if x.size == 0:
            raise ValueError(
                "Cannot update statistics with an empty batch"
            )

        if not np.isfinite(x).all():
            raise ValueError(
                "Normalization fitting data contain NaN or Inf"
            )

        batch_count = int(x.size)

        batch_mean = float(
            np.mean(x, dtype=np.float64)
        )

        centered = x - batch_mean

        batch_m2 = float(
            np.sum(
                centered * centered,
                dtype=np.float64,
            )
        )

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        previous_count = self.count
        combined_count = previous_count + batch_count

        delta = batch_mean - self.mean

        combined_mean = (
            self.mean
            + delta
            * batch_count
            / combined_count
        )

        combined_m2 = (
            self.m2
            + batch_m2
            + delta * delta
            * previous_count
            * batch_count
            / combined_count
        )

        self.count = combined_count
        self.mean = float(combined_mean)
        self.m2 = float(combined_m2)

    @property
    def variance(self) -> float:
        """Population variance, ddof=0."""

        if self.count <= 0:
            raise ValueError(
                "Statistics have not been fitted"
            )

        variance = self.m2 / self.count

        if not math.isfinite(variance):
            raise ValueError(
                "Calculated variance is not finite"
            )

        if variance < 0.0:
            raise ValueError(
                f"Calculated negative variance: {variance}"
            )

        return float(variance)

    @property
    def std(self) -> float:
        """Population standard deviation."""

        standard_deviation = math.sqrt(
            self.variance
        )

        if (
            not math.isfinite(standard_deviation)
            or standard_deviation <= 0.0
        ):
            raise ValueError(
                "Standard deviation must be finite and > 0"
            )

        return float(standard_deviation)


@dataclass(frozen=True)
class GlobalZScore:
    """Frozen parameters for global ECG standardization."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError(
                "Normalization mean must be finite"
            )

        if (
            not math.isfinite(self.std)
            or self.std <= 0.0
        ):
            raise ValueError(
                "Normalization std must be finite and > 0"
            )


def training_rows(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Return ONLY frozen training rows.

    There is intentionally no user-selectable fit split.
    """

    if "split" not in manifest.columns:
        raise ValueError(
            "Manifest does not contain a split column"
        )

    allowed = {
        "train",
        "validation",
        "test",
    }

    observed = set(
        manifest["split"].astype(str).unique()
    )

    if observed != allowed:
        raise ValueError(
            "Expected frozen manifest to contain exactly "
            f"{sorted(allowed)}, found {sorted(observed)}"
        )

    result = manifest.loc[
        manifest["split"].eq("train")
    ].copy()

    if result.empty:
        raise ValueError(
            "Frozen manifest contains no training records"
        )

    if not result["split"].eq("train").all():
        raise AssertionError(
            "Internal error: non-training row entered scaler fit"
        )

    return result


def standardize(
    values: np.ndarray,
    parameters: GlobalZScore,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Apply frozen training-only normalization parameters."""

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if not np.isfinite(x).all():
        raise ValueError(
            "Cannot standardize data containing NaN or Inf"
        )

    transformed = (
        x - parameters.mean
    ) / parameters.std

    if not np.isfinite(transformed).all():
        raise ValueError(
            "Standardization produced NaN or Inf"
        )

    return transformed.astype(
        dtype,
        copy=False,
    )
