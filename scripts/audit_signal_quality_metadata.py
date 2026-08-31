#!/usr/bin/env python3
"""Audit PTB-XL signal-quality metadata before defining Q.

Descriptive only: no quality score is defined here.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT


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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "signal_quality_metadata_audit.json"
)

QUALITY_FIELDS = (
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
)

CONTEXT_ONLY_FIELDS = (
    "extra_beats",
    "pacemaker",
)


def missing_value(value) -> bool:
    if value is None:
        return True

    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))

    if isinstance(value, str):
        return value.strip().lower() in {
            "",
            "nan",
            "none",
        }

    return False


def parse_artifact_cell(value):
    if missing_value(value):
        return [], "missing"

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text], "raw_string"

    if parsed is None:
        return [], "missing"

    if isinstance(parsed, (list, tuple, set)):
        tokens = [
            str(item).strip()
            for item in parsed
            if str(item).strip()
        ]

        return tokens, "literal_list"

    return [
        str(parsed).strip()
    ], "literal_scalar"


def summarize_field(series):
    status_counts = Counter()
    token_counts = Counter()
    raw_counts = Counter()

    annotated = 0

    for value in series.tolist():
        tokens, status = parse_artifact_cell(
            value
        )

        status_counts[status] += 1

        if tokens:
            annotated += 1

            raw_counts[
                str(value).strip()
            ] += 1

            for token in tokens:
                token_counts[token] += 1

    records = len(series)

    return {
        "records":
            int(records),

        "records_with_annotation":
            int(annotated),

        "fraction_with_annotation":
            float(
                annotated / records
            ),

        "records_without_annotation":
            int(
                records - annotated
            ),

        "parse_status_counts":
            {
                key: int(value)
                for key, value
                in sorted(
                    status_counts.items()
                )
            },

        "tokens":
            {
                key: int(value)
                for key, value
                in token_counts.most_common()
            },

        "top_raw_values":
            [
                {
                    "value": key,
                    "count": int(value),
                }
                for key, value
                in raw_counts.most_common(
                    20
                )
            ],
    }


def main():
    database = pd.read_csv(
        DATABASE_PATH
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required = {
        "ecg_id",
        *QUALITY_FIELDS,
        *CONTEXT_ONLY_FIELDS,
    }

    missing = required.difference(
        database.columns
    )

    if missing:
        raise ValueError(
            f"Missing PTB-XL columns: "
            f"{sorted(missing)}"
        )

    merged = manifest.merge(
        database[
            [
                "ecg_id",
                *QUALITY_FIELDS,
                *CONTEXT_ONLY_FIELDS,
            ]
        ],
        on="ecg_id",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != 21388:
        raise ValueError(
            f"Expected 21388 records, "
            f"found {len(merged)}"
        )

    split_report = {}

    expected_counts = {
        "train": 17084,
        "validation": 2146,
        "test": 2158,
    }

    for split in (
        "train",
        "validation",
        "test",
    ):
        subset = merged.loc[
            merged["split"].eq(
                split
            )
        ].copy()

        if (
            len(subset)
            != expected_counts[split]
        ):
            raise ValueError(
                f"{split}: unexpected count"
            )

        quality_summary = {
            field:
                summarize_field(
                    subset[field]
                )
            for field in QUALITY_FIELDS
        }

        context_summary = {
            field:
                summarize_field(
                    subset[field]
                )
            for field
            in CONTEXT_ONLY_FIELDS
        }

        combinations = Counter()

        records_with_any = 0

        field_counts = Counter()

        for row in subset.itertuples(
            index=False
        ):
            present = []

            for field in QUALITY_FIELDS:
                tokens, _ = (
                    parse_artifact_cell(
                        getattr(
                            row,
                            field
                        )
                    )
                )

                if tokens:
                    present.append(
                        field
                    )

            if present:
                records_with_any += 1

            field_counts[
                len(present)
            ] += 1

            key = (
                "+".join(present)
                if present
                else "none"
            )

            combinations[key] += 1

        split_report[split] = {
            "records":
                int(len(subset)),

            "quality_fields":
                quality_summary,

            "context_only_not_used_for_Q":
                context_summary,

            "records_with_any_quality_artifact":
                int(
                    records_with_any
                ),

            "fraction_with_any_quality_artifact":
                float(
                    records_with_any
                    / len(subset)
                ),

            "number_of_quality_fields_present":
                {
                    str(key): int(value)
                    for key, value
                    in sorted(
                        field_counts.items()
                    )
                },

            "artifact_field_combinations":
                {
                    key: int(value)
                    for key, value
                    in combinations.most_common()
                },
        }

    report = {
        "audit_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "purpose":
            "inspect PTB-XL signal-quality metadata before defining Q",

        "quality_score_defined":
            False,

        "quality_fields_considered":
            list(
                QUALITY_FIELDS
            ),

        "context_fields_not_used_for_Q":
            list(
                CONTEXT_ONLY_FIELDS
            ),

        "interpretation_rules": {
            "metadata_presence":
                "artifact annotation present; severity not assumed",

            "metadata_absence":
                "no artifact annotation recorded; not proof of perfect signal",

            "extra_beats":
                "not treated as signal-quality degradation",

            "pacemaker":
                "not treated as signal-quality degradation",
        },

        "splits":
            split_report,

        "source_sha256": {
            "ptbxl_database":
                sha256_file(
                    DATABASE_PATH
                ),

            "frozen_manifest":
                sha256_file(
                    MANIFEST_PATH
                ),
        },

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

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
