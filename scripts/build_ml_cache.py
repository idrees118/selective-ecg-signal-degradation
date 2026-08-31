#!/usr/bin/env python3
"""Build a deterministic cache of verified PTB-XL model tensors.

This is a storage optimization only.

Each cached signal is exactly the output of the already verified
PTBXLSuperdiagnosticDataset:

    raw PTB-XL
        -> frozen training-only normalization
        -> (12, 1000)
        -> float32

No new preprocessing, filtering, clipping, augmentation, or label
transformation occurs here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from ptbxl_reliability.dataset import (
    MANIFEST_PATH,
    NORMALIZATION_PATH,
    PTBXLSuperdiagnosticDataset,
)
from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.paths import PROJECT_ROOT


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_cache"
    / "v1"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "cache_metadata.json"
)

EXPECTED_COUNTS = {
    "train": 17084,
    "validation": 2146,
    "test": 2158,
}


def build_split(split: str) -> dict:
    dataset = PTBXLSuperdiagnosticDataset(
        split
    )

    expected = EXPECTED_COUNTS[split]

    if len(dataset) != expected:
        raise ValueError(
            f"{split}: expected {expected} records, "
            f"found {len(dataset)}"
        )

    split_dir = OUTPUT_DIR / split

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    signals_path = (
        split_dir / "signals.npy"
    )

    targets_path = (
        split_dir / "targets.npy"
    )

    ecg_ids_path = (
        split_dir / "ecg_ids.npy"
    )

    patient_ids_path = (
        split_dir / "patient_ids.npy"
    )

    signals = np.lib.format.open_memmap(
        signals_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            expected,
            12,
            1000,
        ),
    )

    targets = np.lib.format.open_memmap(
        targets_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            expected,
            5,
        ),
    )

    ecg_ids = np.lib.format.open_memmap(
        ecg_ids_path,
        mode="w+",
        dtype=np.int64,
        shape=(expected,),
    )

    patient_ids = np.lib.format.open_memmap(
        patient_ids_path,
        mode="w+",
        dtype=np.int64,
        shape=(expected,),
    )

    for index in range(expected):
        sample = dataset[index]

        signal = (
            sample["signal"]
            .detach()
            .cpu()
            .numpy()
        )

        target = (
            sample["target"]
            .detach()
            .cpu()
            .numpy()
        )

        if signal.shape != (
            12,
            1000,
        ):
            raise ValueError(
                f"{split} index {index}: "
                f"unexpected signal shape {signal.shape}"
            )

        if target.shape != (5,):
            raise ValueError(
                f"{split} index {index}: "
                f"unexpected target shape {target.shape}"
            )

        if signal.dtype != np.float32:
            raise ValueError(
                "Cached signal source is not float32"
            )

        if target.dtype != np.float32:
            raise ValueError(
                "Cached target source is not float32"
            )

        if not np.isfinite(
            signal
        ).all():
            raise ValueError(
                f"{split} index {index}: "
                "non-finite signal"
            )

        if not np.isin(
            target,
            [0.0, 1.0],
        ).all():
            raise ValueError(
                f"{split} index {index}: "
                "non-binary target"
            )

        signals[index] = signal
        targets[index] = target
        ecg_ids[index] = int(
            sample["ecg_id"]
        )
        patient_ids[index] = int(
            sample["patient_id"]
        )

        if (
            index == 0
            or (index + 1) % 1000 == 0
        ):
            print(
                f"{split}: cached "
                f"{index + 1}/{expected}"
            )

    signals.flush()
    targets.flush()
    ecg_ids.flush()
    patient_ids.flush()

    del signals
    del targets
    del ecg_ids
    del patient_ids

    # Reopen the completed cache read-only.
    saved_signals = np.load(
        signals_path,
        mmap_mode="r",
    )

    saved_targets = np.load(
        targets_path,
        mmap_mode="r",
    )

    saved_ecg_ids = np.load(
        ecg_ids_path,
        mmap_mode="r",
    )

    saved_patient_ids = np.load(
        patient_ids_path,
        mmap_mode="r",
    )

    if saved_signals.shape != (
        expected,
        12,
        1000,
    ):
        raise ValueError(
            f"{split}: saved signal shape mismatch"
        )

    if saved_targets.shape != (
        expected,
        5,
    ):
        raise ValueError(
            f"{split}: saved target shape mismatch"
        )

    # Independent exact comparison against the existing verified
    # Dataset at deterministic beginning/middle/end positions.
    verification_indices = sorted(
        {
            0,
            expected // 2,
            expected - 1,
        }
    )

    for index in verification_indices:
        reference = dataset[index]

        reference_signal = (
            reference["signal"]
            .numpy()
        )

        reference_target = (
            reference["target"]
            .numpy()
        )

        if not np.array_equal(
            saved_signals[index],
            reference_signal,
        ):
            raise ValueError(
                f"{split} index {index}: "
                "cached signal differs from "
                "verified Dataset output"
            )

        if not np.array_equal(
            saved_targets[index],
            reference_target,
        ):
            raise ValueError(
                f"{split} index {index}: "
                "cached target differs from "
                "verified Dataset output"
            )

        if (
            int(saved_ecg_ids[index])
            != int(reference["ecg_id"])
        ):
            raise ValueError(
                f"{split} index {index}: "
                "cached ecg_id mismatch"
            )

        if (
            int(saved_patient_ids[index])
            != int(reference["patient_id"])
        ):
            raise ValueError(
                f"{split} index {index}: "
                "cached patient_id mismatch"
            )

    positive_counts = {
        label: int(
            saved_targets[:, label_index]
            .sum(
                dtype=np.float64
            )
        )
        for label_index, label
        in enumerate(
            SUPERCLASS_ORDER
        )
    }

    return {
        "records": expected,
        "signal_shape": [
            expected,
            12,
            1000,
        ],
        "target_shape": [
            expected,
            5,
        ],
        "signal_dtype": "float32",
        "target_dtype": "float32",
        "class_positive_counts":
            positive_counts,
        "verification_indices":
            verification_indices,
        "sha256": {
            "signals":
                sha256_file(
                    signals_path
                ),
            "targets":
                sha256_file(
                    targets_path
                ),
            "ecg_ids":
                sha256_file(
                    ecg_ids_path
                ),
            "patient_ids":
                sha256_file(
                    patient_ids_path
                ),
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_metadata = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        split_metadata[split] = (
            build_split(split)
        )

    metadata = {
        "cache_id":
            "ptbxl_v103_ml_cache_v1",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "purpose":
            "lossless computational cache",

        "scientific_transformation":
            False,

        "source_manifest_sha256":
            sha256_file(
                MANIFEST_PATH
            ),

        "normalization_parameters_sha256":
            sha256_file(
                NORMALIZATION_PATH
            ),

        "model_signal_definition": {
            "shape": [
                12,
                1000,
            ],
            "dtype":
                "float32",
            "normalization":
                "frozen training-only global z-score",
        },

        "target_definition": {
            "shape": [5],
            "dtype": "float32",
            "class_order":
                list(
                    SUPERCLASS_ORDER
                ),
        },

        "splits":
            split_metadata,

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
