"""Auditable PyTorch Dataset for the frozen PTB-XL cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.normalization import standardize
from ptbxl_reliability.normalization_io import load_frozen_global_zscore
from ptbxl_reliability.paths import (
    PROJECT_ROOT,
    PTBXL_ROOT,
    RECORDS100_DIR,
)
from ptbxl_reliability.waveforms import (
    EXPECTED_CHANNELS,
    EXPECTED_SAMPLES,
    load_ecg_100hz,
    to_model_layout,
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

NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "v1"
    / "global_zscore_stats.json"
)

PREPROCESSING_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "v1"
    / "preprocessing_metadata.json"
)


EXPECTED_SPLIT_COUNTS = {
    "train": 17084,
    "validation": 2146,
    "test": 2158,
}


def validate_split_name(split: str) -> str:
    """Validate and return one frozen split name."""

    if split not in EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"split must be one of "
            f"{sorted(EXPECTED_SPLIT_COUNTS)}, "
            f"received {split!r}"
        )

    return split


def _load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"{path} must contain a JSON object"
        )

    return value


def validate_frozen_artifacts() -> None:
    """Verify cohort and preprocessing provenance before data loading."""

    cohort_metadata = _load_json(
        COHORT_METADATA_PATH
    )

    preprocessing_metadata = _load_json(
        PREPROCESSING_METADATA_PATH
    )

    if cohort_metadata.get("status") != "FROZEN":
        raise ValueError(
            "Cohort metadata is not marked FROZEN"
        )

    if preprocessing_metadata.get("status") != "FROZEN":
        raise ValueError(
            "Preprocessing metadata is not marked FROZEN"
        )

    manifest_hash = sha256_file(
        MANIFEST_PATH
    )

    if (
        manifest_hash
        != cohort_metadata["manifest_sha256"]
    ):
        raise ValueError(
            "Frozen cohort manifest SHA-256 mismatch"
        )

    preprocessing_hashes = (
        preprocessing_metadata["sha256"]
    )

    if (
        manifest_hash
        != preprocessing_hashes["cohort_manifest"]
    ):
        raise ValueError(
            "Preprocessing metadata references "
            "a different cohort manifest"
        )

    normalization_hash = sha256_file(
        NORMALIZATION_PATH
    )

    if (
        normalization_hash
        != preprocessing_hashes[
            "normalization_parameters"
        ]
    ):
        raise ValueError(
            "Frozen normalization SHA-256 mismatch"
        )


def resolve_record_base(
    filename_lr: str,
) -> Path:
    """Resolve one manifest waveform path safely.

    The path must remain inside PTB-XL records100.
    """

    relative = Path(
        str(filename_lr)
    )

    if relative.is_absolute():
        raise ValueError(
            "filename_lr must be relative"
        )

    if ".." in relative.parts:
        raise ValueError(
            "filename_lr may not contain '..'"
        )

    record_base = (
        PTBXL_ROOT / relative
    ).resolve()

    records_root = (
        RECORDS100_DIR.resolve()
    )

    if not record_base.is_relative_to(
        records_root
    ):
        raise ValueError(
            f"Waveform path escapes records100: "
            f"{filename_lr!r}"
        )

    return record_base


class PTBXLSuperdiagnosticDataset(Dataset):
    """Frozen five-superclass PTB-XL PyTorch dataset.

    No fitting, augmentation, filtering, clipping, or target
    transformation occurs inside this class.
    """

    def __init__(
        self,
        split: str,
    ) -> None:
        super().__init__()

        self.split = validate_split_name(
            split
        )

        validate_frozen_artifacts()

        self.normalization, normalization_metadata = (
            load_frozen_global_zscore(
                NORMALIZATION_PATH
            )
        )

        if normalization_metadata["fit_split"] != "train":
            raise ValueError(
                "Normalization was not fitted on train only"
            )

        manifest = pd.read_csv(
            MANIFEST_PATH
        )

        required_columns = {
            "ecg_id",
            "patient_id",
            "strat_fold",
            "split",
            "filename_lr",
            "n_superclass_labels",
            *SUPERCLASS_ORDER,
        }

        missing = required_columns.difference(
            manifest.columns
        )

        if missing:
            raise ValueError(
                "Frozen manifest is missing columns: "
                f"{sorted(missing)}"
            )

        rows = (
            manifest.loc[
                manifest["split"].eq(
                    self.split
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        expected_count = (
            EXPECTED_SPLIT_COUNTS[
                self.split
            ]
        )

        if len(rows) != expected_count:
            raise ValueError(
                f"{self.split}: expected "
                f"{expected_count} records, "
                f"found {len(rows)}"
            )

        if not rows["split"].eq(
            self.split
        ).all():
            raise AssertionError(
                "Dataset contains a record from "
                "the wrong split"
            )

        if rows["ecg_id"].duplicated().any():
            raise ValueError(
                f"{self.split}: duplicate ecg_id"
            )

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:

        if not isinstance(
            index,
            (int, np.integer),
        ):
            raise TypeError(
                "Dataset index must be an integer"
            )

        index = int(index)

        if (
            index < 0
            or index >= len(self.rows)
        ):
            raise IndexError(index)

        row = self.rows.iloc[index]

        record_base = resolve_record_base(
            str(row["filename_lr"])
        )

        record = load_ecg_100hz(
            record_base
        )

        # Standardization performs arithmetic in float64
        # and only then converts to float32.
        normalized_native = standardize(
            record.signal,
            self.normalization,
            dtype=np.float32,
        )

        model_signal = to_model_layout(
            normalized_native
        )

        expected_shape = (
            EXPECTED_CHANNELS,
            EXPECTED_SAMPLES,
        )

        if model_signal.shape != expected_shape:
            raise AssertionError(
                f"ECG {row['ecg_id']}: "
                f"expected model shape "
                f"{expected_shape}, "
                f"found {model_signal.shape}"
            )

        if model_signal.dtype != np.float32:
            raise AssertionError(
                f"ECG {row['ecg_id']}: "
                "model signal is not float32"
            )

        if not np.isfinite(
            model_signal
        ).all():
            raise ValueError(
                f"ECG {row['ecg_id']}: "
                "model signal contains NaN/Inf"
            )

        target_array = np.asarray(
            [
                row[label]
                for label in SUPERCLASS_ORDER
            ],
            dtype=np.float32,
        )

        if target_array.shape != (
            len(SUPERCLASS_ORDER),
        ):
            raise AssertionError(
                "Invalid target shape"
            )

        if not np.isin(
            target_array,
            [0.0, 1.0],
        ).all():
            raise ValueError(
                f"ECG {row['ecg_id']}: "
                "target is not binary"
            )

        if int(
            target_array.sum()
        ) != int(
            row["n_superclass_labels"]
        ):
            raise ValueError(
                f"ECG {row['ecg_id']}: "
                "target cardinality mismatch"
            )

        signal_tensor = torch.from_numpy(
            np.ascontiguousarray(
                model_signal
            )
        )

        target_tensor = torch.from_numpy(
            target_array
        )

        if signal_tensor.dtype != torch.float32:
            raise AssertionError(
                "Signal tensor is not torch.float32"
            )

        if target_tensor.dtype != torch.float32:
            raise AssertionError(
                "Target tensor is not torch.float32"
            )

        return {
            "signal": signal_tensor,
            "target": target_tensor,
            "ecg_id": int(
                row["ecg_id"]
            ),
            "patient_id": int(
                row["patient_id"]
            ),
            "split": self.split,
            "filename_lr": str(
                row["filename_lr"]
            ),
        }
