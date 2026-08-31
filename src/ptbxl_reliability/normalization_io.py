"""Loading and validation of frozen normalization parameters."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ptbxl_reliability.normalization import GlobalZScore


def load_frozen_global_zscore(
    path: Path,
) -> tuple[GlobalZScore, dict]:
    """Load and validate a frozen global z-score parameter file."""

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    required = {
        "normalization",
        "fit_split",
        "variance_ddof",
        "mean_mV",
        "variance_mV2",
        "std_mV",
        "training_records",
        "training_scalar_values",
        "manifest_sha256",
        "preprocessing_config_sha256",
    }

    missing = required.difference(metadata)

    if missing:
        raise ValueError(
            "Normalization parameter file is missing fields: "
            f"{sorted(missing)}"
        )

    if metadata["normalization"] != "global_zscore":
        raise ValueError(
            "Expected global_zscore normalization"
        )

    if metadata["fit_split"] != "train":
        raise ValueError(
            "Frozen normalization parameters were not fit "
            "exclusively on the training split"
        )

    if int(metadata["variance_ddof"]) != 0:
        raise ValueError(
            "Expected population variance with ddof=0"
        )

    mean = float(metadata["mean_mV"])
    variance = float(metadata["variance_mV2"])
    std = float(metadata["std_mV"])

    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            "Frozen variance must be finite and > 0"
        )

    if not np.isclose(
        std * std,
        variance,
        rtol=1e-14,
        atol=1e-15,
    ):
        raise ValueError(
            "Frozen std^2 does not agree with frozen variance"
        )

    parameters = GlobalZScore(
        mean=mean,
        std=std,
    )

    return parameters, metadata
