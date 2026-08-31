#!/usr/bin/env python3
"""Verify application of frozen training-only normalization to all splits."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.normalization import (
    RunningPopulationStats,
    standardize,
)
from ptbxl_reliability.normalization_io import (
    load_frozen_global_zscore,
)
from ptbxl_reliability.paths import (
    PROJECT_ROOT,
    PTBXL_ROOT,
)
from ptbxl_reliability.waveforms import (
    EXPECTED_CHANNELS,
    EXPECTED_SAMPLES,
    load_ecg_100hz,
)


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort"
    / "v1"
    / "ptbxl_superdiagnostic_cohort.csv"
)

STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "v1"
    / "global_zscore_stats.json"
)

FIT_AUDIT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "normalization_fit_audit.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "normalization_application_audit.json"
)


EXPECTED_RECORD_COUNTS = {
    "train": 17084,
    "validation": 2146,
    "test": 2158,
}


def main() -> None:
    parameters, metadata = (
        load_frozen_global_zscore(
            STATS_PATH
        )
    )

    with FIT_AUDIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        fit_audit = json.load(handle)

    if (
        sha256_file(STATS_PATH)
        != fit_audit["parameters_sha256"]
    ):
        raise ValueError(
            "Frozen normalization parameter hash mismatch"
        )

    manifest_hash = sha256_file(
        MANIFEST_PATH
    )

    if manifest_hash != metadata[
        "manifest_sha256"
    ]:
        raise ValueError(
            "Frozen cohort manifest hash mismatch"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    accumulators = {
        split: RunningPopulationStats()
        for split in EXPECTED_RECORD_COUNTS
    }

    record_counts = {
        split: 0
        for split in EXPECTED_RECORD_COUNTS
    }

    transformed_min = {
        split: np.inf
        for split in EXPECTED_RECORD_COUNTS
    }

    transformed_max = {
        split: -np.inf
        for split in EXPECTED_RECORD_COUNTS
    }

    for index, row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        split = str(row.split)

        if split not in accumulators:
            raise ValueError(
                f"Unexpected split {split!r}"
            )

        record = load_ecg_100hz(
            PTBXL_ROOT
            / str(row.filename_lr)
        )

        normalized = standardize(
            record.signal,
            parameters,
            dtype=np.float64,
        )

        if normalized.shape != (
            EXPECTED_SAMPLES,
            EXPECTED_CHANNELS,
        ):
            raise ValueError(
                f"ECG {row.ecg_id}: normalization "
                "changed waveform shape"
            )

        if not np.isfinite(
            normalized
        ).all():
            raise ValueError(
                f"ECG {row.ecg_id}: normalized "
                "signal contains NaN or Inf"
            )

        # The eventual neural-network input will be float32.
        # Verify that this conversion is safe for every ECG.
        normalized_float32 = normalized.astype(
            np.float32
        )

        if not np.isfinite(
            normalized_float32
        ).all():
            raise ValueError(
                f"ECG {row.ecg_id}: float32 conversion "
                "produced NaN or Inf"
            )

        accumulators[split].update(
            normalized
        )

        transformed_min[split] = min(
            transformed_min[split],
            float(np.min(normalized)),
        )

        transformed_max[split] = max(
            transformed_max[split],
            float(np.max(normalized)),
        )

        record_counts[split] += 1

        if index % 1000 == 0:
            print(
                f"Checked {index}/{len(manifest)} ECGs"
            )

    if record_counts != EXPECTED_RECORD_COUNTS:
        raise ValueError(
            "Normalized split counts disagree with "
            f"frozen cohort: {record_counts}"
        )

    expected_scalar_counts = {
        split: count
        * EXPECTED_SAMPLES
        * EXPECTED_CHANNELS
        for split, count
        in EXPECTED_RECORD_COUNTS.items()
    }

    for split, expected in (
        expected_scalar_counts.items()
    ):
        observed = accumulators[
            split
        ].count

        if observed != expected:
            raise ValueError(
                f"{split}: expected {expected} scalar "
                f"values, observed {observed}"
            )

    train_mean = accumulators[
        "train"
    ].mean

    train_std = accumulators[
        "train"
    ].std

    # Because the same frozen training mean/std are applied back to the
    # training population, these should equal 0 and 1 up to numerical
    # floating-point error.
    if not np.isclose(
        train_mean,
        0.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            f"Normalized training mean is not zero: "
            f"{train_mean}"
        )

    if not np.isclose(
        train_std,
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            f"Normalized training std is not one: "
            f"{train_std}"
        )

    split_statistics = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        stats = accumulators[split]

        split_statistics[split] = {
            "records":
                record_counts[split],
            "scalar_values":
                stats.count,
            "mean":
                stats.mean,
            "variance":
                stats.variance,
            "std":
                stats.std,
            "min":
                float(
                    transformed_min[split]
                ),
            "max":
                float(
                    transformed_max[split]
                ),
        }

    report = {
        "audit_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "manifest_sha256":
            manifest_hash,
        "normalization_parameters_sha256":
            sha256_file(STATS_PATH),
        "frozen_mean_mV":
            parameters.mean,
        "frozen_std_mV":
            parameters.std,
        "split_statistics":
            split_statistics,
        "training_mean_zero_check":
            "PASS",
        "training_std_one_check":
            "PASS",
        "validation_used_for_fit": False,
        "test_used_for_fit": False,
        "float32_conversion":
            "PASS",
        "status":
            "PASS",
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print()
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
