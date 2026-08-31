#!/usr/bin/env python3
"""Freeze PTB-XL annotation-confidence L on validation data only."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.label_confidence import (
    positive_superclass_confidence,
    record_annotation_confidence,
)
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.metadata import parse_scp_codes
from ptbxl_reliability.paths import PROJECT_ROOT


RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptb-xl"
    / "1.0.3"
)

DATABASE_PATH = RAW_ROOT / "ptbxl_database.csv"
SCP_PATH = RAW_ROOT / "scp_statements.csv"

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
    / "label_confidence_v1.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "label_confidence"
    / "v1"
    / "validation"
)

OUTPUT_CSV = OUTPUT_DIR / "label_confidence.csv"
OUTPUT_METADATA = OUTPUT_DIR / "label_confidence_metadata.json"

EXPECTED_RECORDS = 2146


def load_yaml(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"{path} must contain a mapping"
        )

    return value


def main() -> None:
    config = load_yaml(
        CONFIG_PATH
    )

    if config.get("status") != "FROZEN":
        raise ValueError(
            "Label-confidence config is not FROZEN"
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
            f"Expected {EXPECTED_RECORDS} validation records, "
            f"found {len(validation)}"
        )

    database = pd.read_csv(
        DATABASE_PATH
    )

    source = database[
        [
            "ecg_id",
            "scp_codes",
        ]
    ].copy()

    merged = validation.merge(
        source,
        on="ecg_id",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_RECORDS:
        raise ValueError(
            "Validation/source merge changed record count"
        )

    if merged["scp_codes"].isna().any():
        raise ValueError(
            "Some validation ECGs lack scp_codes"
        )

    scp = pd.read_csv(
        SCP_PATH,
        index_col=0,
    )

    diagnostic = (
        pd.to_numeric(
            scp["diagnostic"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .eq(1)
    )

    code_to_class = {}

    for code, row in scp.loc[
        diagnostic
    ].iterrows():
        superclass = row[
            "diagnostic_class"
        ]

        if pd.isna(superclass):
            continue

        superclass = str(
            superclass
        ).strip()

        if superclass not in SUPERCLASS_ORDER:
            raise ValueError(
                f"Unexpected superclass {superclass!r}"
            )

        code_to_class[
            str(code)
        ] = superclass

    rows = []

    for row in merged.itertuples(
        index=False
    ):
        codes = parse_scp_codes(
            row.scp_codes
        )

        values_by_class = {
            label: []
            for label in SUPERCLASS_ORDER
        }

        for code, likelihood in codes.items():
            superclass = code_to_class.get(
                str(code)
            )

            if superclass is None:
                continue

            values_by_class[
                superclass
            ].append(
                float(likelihood)
            )

        positive_class_likelihoods = []
        class_confidences = {}

        for label in SUPERCLASS_ORDER:
            frozen_positive = int(
                getattr(
                    row,
                    label
                )
            )

            source_positive = int(
                len(
                    values_by_class[label]
                ) > 0
            )

            if (
                frozen_positive
                != source_positive
            ):
                raise ValueError(
                    f"ECG {row.ecg_id}: "
                    f"{label} target reconstruction mismatch"
                )

            if frozen_positive:
                values = values_by_class[
                    label
                ]

                positive_class_likelihoods.append(
                    values
                )

                class_confidences[
                    label
                ] = (
                    positive_superclass_confidence(
                        values
                    )
                )
            else:
                class_confidences[
                    label
                ] = float("nan")

        aggregate = (
            record_annotation_confidence(
                positive_class_likelihoods
            )
        )

        rows.append(
            {
                "ecg_id":
                    int(row.ecg_id),

                "patient_id":
                    int(row.patient_id),

                "split":
                    "validation",

                "raw_label_confidence":
                    float(
                        aggregate[
                            "label_confidence"
                        ]
                    ),

                "raw_label_ambiguity":
                    float(
                        aggregate[
                            "label_ambiguity"
                        ]
                    ),

                "positive_superclasses":
                    int(
                        aggregate[
                            "positive_superclasses"
                        ]
                    ),

                "known_superclass_likelihoods":
                    int(
                        aggregate[
                            "known_superclass_likelihoods"
                        ]
                    ),

                "coverage":
                    float(
                        aggregate[
                            "coverage"
                        ]
                    ),

                "complete_coverage":
                    bool(
                        aggregate[
                            "complete_coverage"
                        ]
                    ),

                **{
                    f"{label}_confidence":
                        float(
                            class_confidences[
                                label
                            ]
                        )
                    for label
                    in SUPERCLASS_ORDER
                },
            }
        )

    output = pd.DataFrame(
        rows
    )

    if len(output) != EXPECTED_RECORDS:
        raise ValueError(
            "Output record count mismatch"
        )

    if output[
        "ecg_id"
    ].duplicated().any():
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

    raw_confidence = output[
        "raw_label_confidence"
    ].to_numpy(
        dtype=np.float64
    )

    missing_mask = ~np.isfinite(
        raw_confidence
    )

    missing_count = int(
        np.sum(
            missing_mask
        )
    )

    if missing_count != 1:
        raise ValueError(
            "Expected exactly one validation record "
            f"with no usable likelihood; found {missing_count}"
        )

    observed = raw_confidence[
        ~missing_mask
    ]

    imputation_value = float(
        np.median(
            observed
        )
    )

    final_confidence = (
        raw_confidence.copy()
    )

    final_confidence[
        missing_mask
    ] = imputation_value

    final_ambiguity = (
        1.0
        - final_confidence
    )

    if not np.isfinite(
        final_confidence
    ).all():
        raise ValueError(
            "Final confidence contains NaN/Inf"
        )

    if (
        (final_confidence < 0.0).any()
        or (final_confidence > 1.0).any()
    ):
        raise ValueError(
            "Final confidence outside [0,1]"
        )

    output[
        "label_confidence"
    ] = final_confidence

    output[
        "label_ambiguity"
    ] = final_ambiguity

    output[
        "imputed"
    ] = missing_mask

    imputed_rows = output.loc[
        output["imputed"]
    ]

    if len(imputed_rows) != 1:
        raise AssertionError(
            "Expected exactly one imputed record"
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
            "Saved CSV size mismatch"
        )

    np.testing.assert_allclose(
        saved[
            "label_confidence"
        ].to_numpy(
            dtype=np.float64
        ),
        output[
            "label_confidence"
        ].to_numpy(
            dtype=np.float64
        ),
        rtol=1e-14,
        atol=1e-15,
    )

    confidence = output[
        "label_confidence"
    ].to_numpy(
        dtype=np.float64
    )

    ambiguity = output[
        "label_ambiguity"
    ].to_numpy(
        dtype=np.float64
    )

    coverage = output[
        "coverage"
    ].to_numpy(
        dtype=np.float64
    )

    metadata = {
        "label_confidence_id":
            config[
                "label_confidence_id"
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
            "per_positive_superclass":
                "maximum known constituent diagnostic likelihood divided by 100",

            "record_aggregation":
                "mean across positive superclasses with known likelihood",

            "zero_likelihood":
                "unavailable; not zero confidence",

            "ambiguity":
                "1 - label_confidence",
        },

        "missingness": {
            "records_with_no_known_superclass_likelihood":
                missing_count,

            "imputation_strategy":
                "validation median",

            "imputation_value":
                imputation_value,

            "imputed_ecg_ids":
                [
                    int(value)
                    for value
                    in imputed_rows[
                        "ecg_id"
                    ].tolist()
                ],
        },

        "coverage_summary": {
            "complete_coverage_records":
                int(
                    output[
                        "complete_coverage"
                    ].sum()
                ),

            "incomplete_coverage_records":
                int(
                    (
                        ~output[
                            "complete_coverage"
                        ]
                    ).sum()
                ),

            "mean":
                float(
                    np.mean(
                        coverage
                    )
                ),

            "min":
                float(
                    np.min(
                        coverage
                    )
                ),
        },

        "label_confidence_summary": {
            "min":
                float(
                    np.min(
                        confidence
                    )
                ),

            "q25":
                float(
                    np.quantile(
                        confidence,
                        0.25
                    )
                ),

            "median":
                float(
                    np.median(
                        confidence
                    )
                ),

            "mean":
                float(
                    np.mean(
                        confidence
                    )
                ),

            "q75":
                float(
                    np.quantile(
                        confidence,
                        0.75
                    )
                ),

            "max":
                float(
                    np.max(
                        confidence
                    )
                ),

            "std":
                float(
                    np.std(
                        confidence,
                        ddof=0
                    )
                ),
        },

        "label_ambiguity_summary": {
            "min":
                float(
                    np.min(
                        ambiguity
                    )
                ),

            "median":
                float(
                    np.median(
                        ambiguity
                    )
                ),

            "mean":
                float(
                    np.mean(
                        ambiguity
                    )
                ),

            "max":
                float(
                    np.max(
                        ambiguity
                    )
                ),
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

            "scp_statements":
                sha256_file(
                    SCP_PATH
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

        "target_reconstruction":
            "PASS",

        "saved_artifact_recheck":
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
