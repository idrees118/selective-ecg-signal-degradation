#!/usr/bin/env python3
"""Freeze metadata-derived PTB-XL signal quality Q on validation only."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.signal_quality import (
    QUALITY_FIELDS,
    metadata_annotation_present,
    signal_quality_from_presence,
)


DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptb-xl"
    / "1.0.3"
    / "ptbxl_database.csv"
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
    / "signal_quality_v1.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "signal_quality"
    / "v1"
    / "validation"
)

OUTPUT_CSV = OUTPUT_DIR / "signal_quality.csv"
OUTPUT_METADATA = OUTPUT_DIR / "signal_quality_metadata.json"

EXPECTED_RECORDS = 2146

EXPECTED_FIELD_COUNTS = {
    "baseline_drift": 226,
    "static_noise": 321,
    "burst_noise": 47,
    "electrodes_problems": 4,
}

EXPECTED_DISTRIBUTION = {
    0: 1618,
    1: 460,
    2: 66,
    3: 2,
    4: 0,
}


def load_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "Config must be a mapping"
        )

    if config.get("status") != "FROZEN":
        raise ValueError(
            "Signal-quality config is not FROZEN"
        )

    if tuple(
        config["source"]["quality_fields"]
    ) != QUALITY_FIELDS:
        raise ValueError(
            "Quality-field order changed"
        )

    if (
        config["development_split"][
            "validation_only"
        ]
        is not True
    ):
        raise ValueError(
            "Development split must be validation only"
        )

    if (
        config[
            "test_allowed_during_development"
        ]
        is not False
    ):
        raise ValueError(
            "Test use is forbidden"
        )

    return config


def main():
    config = load_config()

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
        .reset_index(drop=True)
    )

    if len(validation) != EXPECTED_RECORDS:
        raise ValueError(
            "Unexpected validation size"
        )

    database = pd.read_csv(
        DATABASE_PATH
    )

    required = {
        "ecg_id",
        *QUALITY_FIELDS,
    }

    missing = required.difference(
        database.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    merged = validation.merge(
        database[
            [
                "ecg_id",
                *QUALITY_FIELDS,
            ]
        ],
        on="ecg_id",
        how="left",
        validate="one_to_one",
    )

    rows = []

    for row in merged.itertuples(
        index=False
    ):
        presence = {
            field:
                metadata_annotation_present(
                    getattr(
                        row,
                        field
                    )
                )
            for field in QUALITY_FIELDS
        }

        score = signal_quality_from_presence(
            presence
        )

        rows.append(
            {
                "ecg_id":
                    int(row.ecg_id),

                "patient_id":
                    int(row.patient_id),

                "split":
                    "validation",

                **{
                    f"{field}_present":
                        bool(presence[field])
                    for field
                    in QUALITY_FIELDS
                },

                **score,
            }
        )

    output = pd.DataFrame(
        rows
    )

    if len(output) != EXPECTED_RECORDS:
        raise ValueError(
            "Output record count mismatch"
        )

    if output["ecg_id"].duplicated().any():
        raise ValueError(
            "Duplicate ecg_id"
        )

    if not np.array_equal(
        output[
            "ecg_id"
        ].to_numpy(
            dtype=np.int64
        ),
        validation[
            "ecg_id"
        ].to_numpy(
            dtype=np.int64
        ),
    ):
        raise ValueError(
            "ECG ordering changed"
        )

    observed_field_counts = {
        field:
            int(
                output[
                    f"{field}_present"
                ].sum()
            )
        for field in QUALITY_FIELDS
    }

    if (
        observed_field_counts
        != EXPECTED_FIELD_COUNTS
    ):
        raise ValueError(
            "Field counts differ from audit"
        )

    counts = (
        output[
            "n_quality_artifact_categories"
        ]
        .value_counts()
        .to_dict()
    )

    distribution = {
        value:
            int(
                counts.get(
                    value,
                    0
                )
            )
        for value in range(5)
    }

    if distribution != EXPECTED_DISTRIBUTION:
        raise ValueError(
            "Artifact-category distribution "
            "differs from audit"
        )

    any_artifact = int(
        output[
            "any_quality_artifact"
        ].sum()
    )

    if any_artifact != 528:
        raise ValueError(
            "Expected 528 artifact-annotated records"
        )

    q = output[
        "signal_quality"
    ].to_numpy(
        dtype=np.float64
    )

    burden = output[
        "artifact_burden"
    ].to_numpy(
        dtype=np.float64
    )

    np.testing.assert_allclose(
        q,
        1.0 - burden,
        rtol=0.0,
        atol=0.0,
    )

    if not np.isfinite(q).all():
        raise ValueError(
            "Q contains NaN/Inf"
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

    saved = pd.read_csv(
        OUTPUT_CSV
    )

    if len(saved) != EXPECTED_RECORDS:
        raise ValueError(
            "Saved artifact size mismatch"
        )

    metadata = {
        "signal_quality_id":
            config[
                "signal_quality_id"
            ],

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "evaluation_split":
            "validation",

        "records":
            EXPECTED_RECORDS,

        "test_records_used":
            0,

        "definition": {
            "artifact_burden":
                "number_of_present_quality_fields / 4",

            "signal_quality":
                "1 - artifact_burden",

            "severity_parsing":
                False,

            "lead_count_parsing":
                False,

            "interpretation":
                "metadata-derived relative quality proxy, not calibrated physical SQI",
        },

        "field_annotation_counts":
            observed_field_counts,

        "artifact_category_count_distribution":
            {
                str(key): value
                for key, value
                in distribution.items()
            },

        "records_with_any_quality_artifact":
            any_artifact,

        "fraction_with_any_quality_artifact":
            float(
                any_artifact
                / EXPECTED_RECORDS
            ),

        "records_with_no_recorded_quality_artifact":
            int(
                EXPECTED_RECORDS
                - any_artifact
            ),

        "signal_quality_summary": {
            "min":
                float(np.min(q)),

            "median":
                float(np.median(q)),

            "mean":
                float(
                    np.mean(
                        q,
                        dtype=np.float64
                    )
                ),

            "max":
                float(np.max(q)),

            "std":
                float(
                    np.std(
                        q,
                        ddof=0
                    )
                ),

            "unique_values":
                [
                    float(value)
                    for value in sorted(
                        np.unique(q)
                    )
                ],
        },

        "source_sha256": {
            "config":
                sha256_file(
                    CONFIG_PATH
                ),

            "ptbxl_database":
                sha256_file(
                    DATABASE_PATH
                ),

            "frozen_manifest":
                sha256_file(
                    MANIFEST_PATH
                ),
        },

        "output_csv_sha256":
            sha256_file(
                OUTPUT_CSV
            ),

        "audit_reproduction":
            "PASS",

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

    print(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
