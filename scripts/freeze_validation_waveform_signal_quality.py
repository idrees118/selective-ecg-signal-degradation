#!/usr/bin/env python3
"""Apply frozen training waveform-quality reference to validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import wfdb

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.selective_prediction import midrank_ecdf
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

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "waveform_signal_quality"
    / "v1"
    / "training_reference.npz"
)

REFERENCE_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "waveform_signal_quality"
    / "v1"
    / "training_reference_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "waveform_signal_quality"
    / "v1"
    / "validation"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "waveform_signal_quality.csv"
)

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "waveform_signal_quality_metadata.json"
)

EXPECTED_RECORDS = 2146

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

FEATURE_NAMES = [
    "baseline_burden",
    "high_frequency_burden",
    "lead_dropout_burden",
    "clipping_burden",
]


def main():
    with REFERENCE_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        reference_metadata = json.load(handle)

    if reference_metadata["status"] != "FROZEN":
        raise ValueError(
            "Training Q reference not frozen"
        )

    if reference_metadata["fit_split"] != "train":
        raise ValueError(
            "Reference was not fit on train"
        )

    if reference_metadata["validation_records_used"] != 0:
        raise ValueError(
            "Reference reports validation use"
        )

    if reference_metadata["test_records_used"] != 0:
        raise ValueError(
            "Reference reports test use"
        )

    reference = np.load(
        REFERENCE_PATH,
        allow_pickle=False,
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    validation = (
        manifest.loc[
            manifest["split"].eq(
                "validation"
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(validation) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} validation records, "
            f"found {len(validation)}"
        )

    rows = []

    for index, row in enumerate(
        validation.itertuples(
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

        signal, fields = wfdb.rdsamp(
            str(record_path)
        )

        if signal.shape != (
            1000,
            12,
        ):
            raise ValueError(
                f"ECG {row.ecg_id}: "
                f"unexpected shape {signal.shape}"
            )

        if float(
            fields["fs"]
        ) != 100.0:
            raise ValueError(
                f"ECG {row.ecg_id}: "
                "sampling-rate mismatch"
            )

        if list(
            fields["sig_name"]
        ) != EXPECTED_LEADS:
            raise ValueError(
                f"ECG {row.ecg_id}: "
                "lead-order mismatch"
            )

        x = np.asarray(
            signal.T,
            dtype=np.float64,
        )

        features = waveform_badness_features(
            x,
            fs=100.0,
        )

        percentiles = {}

        for name in FEATURE_NAMES:
            reference_values = np.asarray(
                reference[
                    f"{name}_reference"
                ],
                dtype=np.float64,
            )

            percentile = midrank_ecdf(
                reference_values,
                np.asarray(
                    [features[name]],
                    dtype=np.float64,
                ),
            )[0]

            percentiles[name] = float(
                percentile
            )

        composite_badness = float(
            np.mean(
                np.asarray(
                    [
                        percentiles[
                            name
                        ]
                        for name
                        in FEATURE_NAMES
                    ],
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )
        )

        q_signal = float(
            1.0
            - composite_badness
        )

        rows.append(
            {
                "ecg_id":
                    int(row.ecg_id),

                "patient_id":
                    int(row.patient_id),

                "split":
                    "validation",

                **features,

                **{
                    f"{name}_percentile":
                        percentiles[name]
                    for name
                    in FEATURE_NAMES
                },

                "waveform_badness":
                    composite_badness,

                "signal_quality":
                    q_signal,
            }
        )

        if (
            index % 500 == 0
            or index == EXPECTED_RECORDS
        ):
            print(
                f"validation waveform SQI "
                f"{index}/{EXPECTED_RECORDS}"
            )

    output = pd.DataFrame(
        rows
    )

    if len(output) != EXPECTED_RECORDS:
        raise ValueError(
            "Output count mismatch"
        )

    if output[
        "ecg_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate ECG ID"
        )

    expected_ids = validation[
        "ecg_id"
    ].to_numpy(
        dtype=np.int64
    )

    observed_ids = output[
        "ecg_id"
    ].to_numpy(
        dtype=np.int64
    )

    if not np.array_equal(
        expected_ids,
        observed_ids,
    ):
        raise ValueError(
            "ECG order mismatch"
        )

    q = output[
        "signal_quality"
    ].to_numpy(
        dtype=np.float64
    )

    badness = output[
        "waveform_badness"
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        q
    ).all():
        raise ValueError(
            "Q contains NaN/Inf"
        )

    if (
        (q < 0.0).any()
        or (q > 1.0).any()
    ):
        raise ValueError(
            "Q outside [0,1]"
        )

    np.testing.assert_allclose(
        q,
        1.0 - badness,
        rtol=0.0,
        atol=1e-15,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.17g",
    )

    metadata = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "signal_quality_id":
            "ptbxl_waveform_signal_quality_v1",

        "evaluation_split":
            "validation",

        "records":
            EXPECTED_RECORDS,

        "training_reference_records":
            int(
                reference_metadata[
                    "records"
                ]
            ),

        "labels_used_to_define_score":
            False,

        "prediction_errors_used_to_define_score":
            False,

        "test_records_used":
            0,

        "signal_quality_summary": {
            "min":
                float(
                    np.min(q)
                ),

            "q25":
                float(
                    np.quantile(
                        q,
                        0.25
                    )
                ),

            "median":
                float(
                    np.median(q)
                ),

            "mean":
                float(
                    np.mean(
                        q,
                        dtype=np.float64,
                    )
                ),

            "q75":
                float(
                    np.quantile(
                        q,
                        0.75
                    )
                ),

            "max":
                float(
                    np.max(q)
                ),

            "std":
                float(
                    np.std(
                        q,
                        ddof=0,
                    )
                ),
        },

        "training_reference_sha256":
            sha256_file(
                REFERENCE_PATH
            ),

        "training_reference_metadata_sha256":
            sha256_file(
                REFERENCE_METADATA_PATH
            ),

        "output_csv_sha256":
            sha256_file(
                OUTPUT_CSV
            ),

        "status":
            "FROZEN",
    }

    with OUTPUT_METADATA.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

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
