#!/usr/bin/env python3
"""Fit and independently verify PTB-XL training-only normalization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from ptbxl_reliability.integrity import (
    sha256_file,
)
from ptbxl_reliability.normalization import (
    GlobalZScore,
    RunningPopulationStats,
    training_rows,
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

COHORT_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort"
    / "v1"
    / "cohort_metadata.json"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "preprocessing_v1.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "v1"
)

PARAMETERS_PATH = (
    OUTPUT_DIR
    / "global_zscore_stats.json"
)

AUDIT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "normalization_fit_audit.json"
)


EXPECTED_TRAIN_RECORDS = 17_084

EXPECTED_SCALAR_VALUES = (
    EXPECTED_TRAIN_RECORDS
    * EXPECTED_SAMPLES
    * EXPECTED_CHANNELS
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        result = json.load(handle)

    if not isinstance(result, dict):
        raise ValueError(
            f"{path} must contain a JSON object"
        )

    return result


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        result = yaml.safe_load(handle)

    if not isinstance(result, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping"
        )

    return result


def validate_preprocessing_config(
    config: dict,
) -> None:
    normalization = config[
        "normalization"
    ]

    expected = {
        "method": "global_zscore",
        "fit_split": "train",
        "aggregation":
            "all_training_records_all_leads_all_timesteps",
        "variance_ddof": 0,
        "statistics_dtype": "float64",
        "model_dtype": "float32",
        "epsilon": None,
    }

    for key, expected_value in expected.items():
        observed = normalization.get(key)

        if observed != expected_value:
            raise ValueError(
                f"Unexpected preprocessing setting "
                f"{key!r}: expected "
                f"{expected_value!r}, "
                f"found {observed!r}"
            )


def main() -> None:
    config = load_yaml(
        CONFIG_PATH
    )

    validate_preprocessing_config(
        config
    )

    cohort_metadata = load_json(
        COHORT_METADATA_PATH
    )

    manifest_hash = sha256_file(
        MANIFEST_PATH
    )

    if (
        manifest_hash
        != cohort_metadata["manifest_sha256"]
    ):
        raise ValueError(
            "Frozen cohort manifest hash mismatch"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    train = training_rows(
        manifest
    )

    if len(train) != EXPECTED_TRAIN_RECORDS:
        raise ValueError(
            "Unexpected number of training ECGs: "
            f"expected {EXPECTED_TRAIN_RECORDS}, "
            f"found {len(train)}"
        )

    if not train["split"].eq(
        "train"
    ).all():
        raise AssertionError(
            "Validation/test data entered normalization fit"
        )

    # Method 1:
    # our independent Chan/Welford accumulator.
    custom = RunningPopulationStats()

    # Method 2:
    # independent scikit-learn implementation.
    sklearn_scaler = StandardScaler(
        with_mean=True,
        with_std=True,
        copy=True,
    )

    records_seen = 0

    for row in train.itertuples(
        index=False
    ):
        record = load_ecg_100hz(
            PTBXL_ROOT
            / str(row.filename_lr)
        )

        signal = np.asarray(
            record.signal,
            dtype=np.float64,
        )

        if signal.shape != (
            EXPECTED_SAMPLES,
            EXPECTED_CHANNELS,
        ):
            raise ValueError(
                f"ECG {row.ecg_id}: "
                f"unexpected shape {signal.shape}"
            )

        if not np.isfinite(
            signal
        ).all():
            raise ValueError(
                f"ECG {row.ecg_id}: "
                "non-finite signal value"
            )

        flat = signal.reshape(
            -1
        )

        custom.update(
            flat
        )

        sklearn_scaler.partial_fit(
            flat.reshape(-1, 1)
        )

        records_seen += 1

        if records_seen % 1000 == 0:
            print(
                f"Fitted {records_seen}/"
                f"{EXPECTED_TRAIN_RECORDS} "
                "training ECGs"
            )

    if records_seen != EXPECTED_TRAIN_RECORDS:
        raise AssertionError(
            "Training iteration count mismatch"
        )

    if custom.count != EXPECTED_SCALAR_VALUES:
        raise ValueError(
            "Custom accumulator scalar count mismatch: "
            f"expected {EXPECTED_SCALAR_VALUES}, "
            f"found {custom.count}"
        )

    sklearn_count = int(
        np.asarray(
            sklearn_scaler.n_samples_seen_
        ).item()
    )

    if sklearn_count != EXPECTED_SCALAR_VALUES:
        raise ValueError(
            "scikit-learn scalar count mismatch: "
            f"expected {EXPECTED_SCALAR_VALUES}, "
            f"found {sklearn_count}"
        )

    custom_mean = custom.mean
    custom_variance = custom.variance
    custom_std = custom.std

    sklearn_mean = float(
        sklearn_scaler.mean_[0]
    )

    sklearn_variance = float(
        sklearn_scaler.var_[0]
    )

    sklearn_std = float(
        sklearn_scaler.scale_[0]
    )

    # Independent online algorithms need not be bit-identical because
    # floating-point addition order differs. These tolerances are far
    # below any practically meaningful ECG amplitude difference.
    mean_agrees = np.isclose(
        custom_mean,
        sklearn_mean,
        rtol=1e-10,
        atol=1e-12,
    )

    variance_agrees = np.isclose(
        custom_variance,
        sklearn_variance,
        rtol=1e-10,
        atol=1e-12,
    )

    std_agrees = np.isclose(
        custom_std,
        sklearn_std,
        rtol=1e-10,
        atol=1e-12,
    )

    if not (
        mean_agrees
        and variance_agrees
        and std_agrees
    ):
        raise ValueError(
            "Independent normalization calculations "
            "do not agree.\n"
            f"custom mean={custom_mean}, "
            f"sklearn mean={sklearn_mean}\n"
            f"custom variance={custom_variance}, "
            f"sklearn variance={sklearn_variance}\n"
            f"custom std={custom_std}, "
            f"sklearn std={sklearn_std}"
        )

    # Final defensive validation.
    _ = GlobalZScore(
        mean=custom_mean,
        std=custom_std,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameters = {
        "preprocessing_id":
            config["preprocessing_id"],
        "normalization":
            "global_zscore",
        "fit_split":
            "train",
        "variance_ddof":
            0,
        "mean_mV":
            custom_mean,
        "variance_mV2":
            custom_variance,
        "std_mV":
            custom_std,
        "training_records":
            records_seen,
        "training_scalar_values":
            custom.count,
        "manifest_sha256":
            manifest_hash,
        "preprocessing_config_sha256":
            sha256_file(CONFIG_PATH),
    }

    with PARAMETERS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            parameters,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    parameters_hash = sha256_file(
        PARAMETERS_PATH
    )

    audit = {
        "audit_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "fit_records":
            records_seen,
        "fit_split":
            "train",
        "validation_records_used":
            0,
        "test_records_used":
            0,
        "scalar_values":
            custom.count,
        "custom": {
            "mean":
                custom_mean,
            "variance":
                custom_variance,
            "std":
                custom_std,
        },
        "sklearn_independent_check": {
            "mean":
                sklearn_mean,
            "variance":
                sklearn_variance,
            "std":
                sklearn_std,
            "n_samples_seen":
                sklearn_count,
        },
        "absolute_differences": {
            "mean":
                abs(
                    custom_mean
                    - sklearn_mean
                ),
            "variance":
                abs(
                    custom_variance
                    - sklearn_variance
                ),
            "std":
                abs(
                    custom_std
                    - sklearn_std
                ),
        },
        "agreement_tolerance": {
            "rtol": 1e-10,
            "atol": 1e-12,
        },
        "parameters_sha256":
            parameters_hash,
        "manifest_sha256":
            manifest_hash,
        "status":
            "PASS",
    }

    AUDIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print()
    print(
        json.dumps(
            parameters,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
