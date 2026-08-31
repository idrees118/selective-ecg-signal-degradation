#!/usr/bin/env python3
"""Full waveform audit of the frozen PTB-XL modelling cohort."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT, PTBXL_ROOT
from ptbxl_reliability.waveforms import (
    EXPECTED_CHANNELS,
    EXPECTED_FS,
    EXPECTED_LEADS,
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "waveform_audit.json"
)


def main() -> None:
    with COHORT_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        cohort_metadata = json.load(handle)

    observed_manifest_hash = sha256_file(
        MANIFEST_PATH
    )

    expected_manifest_hash = cohort_metadata[
        "manifest_sha256"
    ]

    if observed_manifest_hash != expected_manifest_hash:
        raise ValueError(
            "Frozen cohort manifest hash mismatch"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    expected_records = 21_388

    if len(manifest) != expected_records:
        raise ValueError(
            f"Expected {expected_records} frozen records, "
            f"found {len(manifest)}"
        )

    split_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    global_min = np.inf
    global_max = -np.inf

    split_min = {
        key: np.inf
        for key in split_counts
    }

    split_max = {
        key: -np.inf
        for key in split_counts
    }

    total_scalar_values = 0

    for index, row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        split = str(row.split)

        if split not in split_counts:
            raise ValueError(
                f"Unexpected split {split!r}"
            )

        base = PTBXL_ROOT / str(
            row.filename_lr
        )

        record = load_ecg_100hz(base)

        signal = record.signal

        if signal.shape != (
            EXPECTED_SAMPLES,
            EXPECTED_CHANNELS,
        ):
            raise AssertionError(
                f"ECG {row.ecg_id}: invalid shape "
                f"{signal.shape}"
            )

        if record.sampling_rate_hz != EXPECTED_FS:
            raise AssertionError(
                f"ECG {row.ecg_id}: unexpected "
                f"sampling rate"
            )

        if record.lead_names != EXPECTED_LEADS:
            raise AssertionError(
                f"ECG {row.ecg_id}: unexpected "
                f"lead order"
            )

        if not np.isfinite(signal).all():
            raise ValueError(
                f"ECG {row.ecg_id}: contains "
                "non-finite values"
            )

        current_min = float(
            np.min(signal)
        )

        current_max = float(
            np.max(signal)
        )

        global_min = min(
            global_min,
            current_min,
        )

        global_max = max(
            global_max,
            current_max,
        )

        split_min[split] = min(
            split_min[split],
            current_min,
        )

        split_max[split] = max(
            split_max[split],
            current_max,
        )

        split_counts[split] += 1

        total_scalar_values += int(
            signal.size
        )

        if index % 1000 == 0:
            print(
                f"Checked {index}/{len(manifest)} ECGs"
            )

    expected_split_counts = {
        "train": 17_084,
        "validation": 2_146,
        "test": 2_158,
    }

    if split_counts != expected_split_counts:
        raise ValueError(
            "Waveform split counts disagree with "
            f"frozen cohort: {split_counts}"
        )

    expected_scalar_values = (
        len(manifest)
        * EXPECTED_SAMPLES
        * EXPECTED_CHANNELS
    )

    if total_scalar_values != expected_scalar_values:
        raise ValueError(
            "Unexpected total number of ECG values"
        )

    report = {
        "audit_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "manifest_sha256":
            observed_manifest_hash,
        "records_checked":
            int(len(manifest)),
        "split_counts":
            split_counts,
        "expected_native_shape": [
            EXPECTED_SAMPLES,
            EXPECTED_CHANNELS,
        ],
        "sampling_rate_hz":
            EXPECTED_FS,
        "lead_order":
            list(EXPECTED_LEADS),
        "total_scalar_values":
            int(total_scalar_values),
        "global_min_mV":
            float(global_min),
        "global_max_mV":
            float(global_max),
        "split_min_mV": {
            key: float(value)
            for key, value
            in split_min.items()
        },
        "split_max_mV": {
            key: float(value)
            for key, value
            in split_max.items()
        },
        "finite_values":
            "PASS",
        "shape_integrity":
            "PASS",
        "sampling_rate_integrity":
            "PASS",
        "lead_order_integrity":
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
