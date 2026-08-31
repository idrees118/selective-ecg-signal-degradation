#!/usr/bin/env python3
"""Fit training-only reference distributions for waveform Q."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import wfdb
import yaml

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.waveform_signal_quality import (
    waveform_badness_features,
)


RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptb-xl"
    / "1.0.3"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort"
    / "v1"
    / "ptbxl_superdiagnostic_cohort.csv"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "waveform_signal_quality_v1.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "waveform_signal_quality"
    / "v1"
)

FEATURES_PATH = (
    OUTPUT_DIR
    / "training_features.csv"
)

REFERENCE_PATH = (
    OUTPUT_DIR
    / "training_reference.npz"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "training_reference_metadata.json"
)

EXPECTED_RECORDS = 17084

EXPECTED_LEADS = [
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def main():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        )

    if config["status"] != "FROZEN":
        raise ValueError(
            "Q-signal config is not FROZEN"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    train = (
        manifest.loc[
            manifest[
                "split"
            ].eq(
                "train"
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(train) != EXPECTED_RECORDS:
        raise ValueError(
            "Unexpected training size"
        )

    rows = []

    for index, row in enumerate(
        train.itertuples(
            index=False
        ),
        start=1,
    ):
        record_path = (
            RAW_ROOT
            / str(
                row.filename_lr
            )
        )

        signal, fields = (
            wfdb.rdsamp(
                str(
                    record_path
                )
            )
        )

        if signal.shape != (
            1000,
            12,
        ):
            raise ValueError(
                f"ECG {row.ecg_id}: "
                f"shape {signal.shape}"
            )

        if float(
            fields["fs"]
        ) != 100.0:
            raise ValueError(
                f"ECG {row.ecg_id}: "
                "sampling rate mismatch"
            )

        if list(
            fields["sig_name"]
        ) != EXPECTED_LEADS:
            raise ValueError(
                f"ECG {row.ecg_id}: "
                "lead order mismatch"
            )

        x = np.asarray(
            signal.T,
            dtype=np.float64,
        )

        features = (
            waveform_badness_features(
                x,
                fs=100.0,
            )
        )

        rows.append(
            {
                "ecg_id":
                    int(
                        row.ecg_id
                    ),

                **features,
            }
        )

        if (
            index % 500 == 0
            or index == EXPECTED_RECORDS
        ):
            print(
                f"training waveform SQI "
                f"{index}/{EXPECTED_RECORDS}"
            )

    features = pd.DataFrame(
        rows
    )

    if len(features) != EXPECTED_RECORDS:
        raise ValueError(
            "Feature count mismatch"
        )

    if features[
        "ecg_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate ECG IDs"
        )

    feature_names = [
        "baseline_burden",
        "high_frequency_burden",
        "lead_dropout_burden",
        "clipping_burden",
    ]

    for name in feature_names:
        values = features[
            name
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"{name} contains NaN/Inf"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        FEATURES_PATH,
        index=False,
        float_format="%.17g",
    )

    np.savez_compressed(
        REFERENCE_PATH,
        **{
            f"{name}_reference":
                np.sort(
                    features[
                        name
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            for name
            in feature_names
        },
    )

    feature_summary = {}

    for name in feature_names:
        values = features[
            name
        ].to_numpy(
            dtype=np.float64
        )

        feature_summary[
            name
        ] = {
            "min":
                float(
                    np.min(
                        values
                    )
                ),

            "q25":
                float(
                    np.quantile(
                        values,
                        0.25
                    )
                ),

            "median":
                float(
                    np.median(
                        values
                    )
                ),

            "mean":
                float(
                    np.mean(
                        values,
                        dtype=np.float64,
                    )
                ),

            "q75":
                float(
                    np.quantile(
                        values,
                        0.75
                    )
                ),

            "max":
                float(
                    np.max(
                        values
                    )
                ),
        }

    metadata = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "signal_quality_id":
            config[
                "signal_quality_id"
            ],

        "fit_split":
            "train",

        "records":
            EXPECTED_RECORDS,

        "validation_records_used":
            0,

        "test_records_used":
            0,

        "labels_used":
            False,

        "model_predictions_used":
            False,

        "feature_summary":
            feature_summary,

        "source_sha256": {
            "config":
                sha256_file(
                    CONFIG_PATH
                ),

            "manifest":
                sha256_file(
                    MANIFEST_PATH
                ),
        },

        "training_features_sha256":
            sha256_file(
                FEATURES_PATH
            ),

        "training_reference_sha256":
            sha256_file(
                REFERENCE_PATH
            ),

        "status":
            "FROZEN",
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write(
            "\n"
        )

    print()
    print(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
