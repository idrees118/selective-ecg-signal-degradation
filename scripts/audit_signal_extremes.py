#!/usr/bin/env python3
"""Audit unusually large raw and normalized ECG amplitudes.

This script is descriptive only. It does not clip, filter, remove, or modify
any ECG record.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ptbxl_reliability.normalization import standardize
from ptbxl_reliability.normalization_io import load_frozen_global_zscore
from ptbxl_reliability.paths import PROJECT_ROOT, PTBXL_ROOT
from ptbxl_reliability.waveforms import (
    EXPECTED_LEADS,
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "signal_extremes_audit.json"
)

Z_THRESHOLDS = (
    5.0,
    10.0,
    20.0,
    50.0,
)


def main() -> None:
    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    parameters, _ = (
        load_frozen_global_zscore(
            STATS_PATH
        )
    )

    record_threshold_counts = {
        split: Counter()
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    scalar_threshold_counts = {
        split: Counter()
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    scalar_totals = Counter()

    lead_extremes = {
        lead: {
            "min_mV": np.inf,
            "max_mV": -np.inf,
            "max_abs_z": 0.0,
        }
        for lead in EXPECTED_LEADS
    }

    extreme_records = []

    for index, row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        split = str(row.split)

        record = load_ecg_100hz(
            PTBXL_ROOT
            / str(row.filename_lr)
        )

        raw = np.asarray(
            record.signal,
            dtype=np.float64,
        )

        normalized = standardize(
            raw,
            parameters,
            dtype=np.float64,
        )

        abs_z = np.abs(
            normalized
        )

        scalar_totals[split] += int(
            abs_z.size
        )

        max_abs_z = float(
            np.max(abs_z)
        )

        raw_min = float(
            np.min(raw)
        )

        raw_max = float(
            np.max(raw)
        )

        for threshold in Z_THRESHOLDS:
            scalar_count = int(
                np.sum(
                    abs_z > threshold
                )
            )

            scalar_threshold_counts[
                split
            ][str(threshold)] += (
                scalar_count
            )

            if scalar_count > 0:
                record_threshold_counts[
                    split
                ][str(threshold)] += 1

        for lead_index, lead in enumerate(
            EXPECTED_LEADS
        ):
            lead_raw = raw[
                :,
                lead_index,
            ]

            lead_z = abs_z[
                :,
                lead_index,
            ]

            lead_extremes[
                lead
            ]["min_mV"] = min(
                lead_extremes[
                    lead
                ]["min_mV"],
                float(
                    np.min(
                        lead_raw
                    )
                ),
            )

            lead_extremes[
                lead
            ]["max_mV"] = max(
                lead_extremes[
                    lead
                ]["max_mV"],
                float(
                    np.max(
                        lead_raw
                    )
                ),
            )

            lead_extremes[
                lead
            ]["max_abs_z"] = max(
                lead_extremes[
                    lead
                ]["max_abs_z"],
                float(
                    np.max(
                        lead_z
                    )
                ),
            )

        if max_abs_z > 10.0:
            extreme_records.append(
                {
                    "ecg_id":
                        int(row.ecg_id),
                    "patient_id":
                        int(row.patient_id),
                    "split":
                        split,
                    "filename_lr":
                        str(
                            row.filename_lr
                        ),
                    "raw_min_mV":
                        raw_min,
                    "raw_max_mV":
                        raw_max,
                    "max_abs_z":
                        max_abs_z,
                }
            )

        if index % 1000 == 0:
            print(
                f"Checked {index}/"
                f"{len(manifest)} ECGs"
            )

    extreme_records.sort(
        key=lambda item:
            item["max_abs_z"],
        reverse=True,
    )

    threshold_report = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        threshold_report[
            split
        ] = {}

        total_records = int(
            manifest["split"]
            .eq(split)
            .sum()
        )

        total_scalars = int(
            scalar_totals[
                split
            ]
        )

        for threshold in Z_THRESHOLDS:
            key = str(
                threshold
            )

            records = int(
                record_threshold_counts[
                    split
                ][key]
            )

            scalars = int(
                scalar_threshold_counts[
                    split
                ][key]
            )

            threshold_report[
                split
            ][key] = {
                "records_with_abs_z_above_threshold":
                    records,
                "fraction_of_records":
                    records
                    / total_records,
                "scalar_values_above_threshold":
                    scalars,
                "fraction_of_scalar_values":
                    scalars
                    / total_scalars,
            }

    report = {
        "audit_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "records_checked":
            int(
                len(
                    manifest
                )
            ),
        "z_thresholds":
            list(
                Z_THRESHOLDS
            ),
        "threshold_statistics":
            threshold_report,
        "per_lead_extremes":
            {
                lead: {
                    key:
                        float(
                            value
                        )
                    for key, value
                    in values.items()
                }
                for lead, values
                in lead_extremes.items()
            },
        "top_50_extreme_records":
            extreme_records[
                :50
            ],
        "records_with_abs_z_above_10":
            len(
                extreme_records
            ),
        "action_taken":
            "none",
        "clipping_applied":
            False,
        "records_removed":
            0,
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
        handle.write(
            "\n"
        )

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
