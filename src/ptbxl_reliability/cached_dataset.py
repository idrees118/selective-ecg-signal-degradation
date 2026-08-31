"""PyTorch access to the frozen, lossless PTB-XL ML cache."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT


CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_cache"
    / "v1"
)

CACHE_METADATA_PATH = (
    CACHE_ROOT
    / "cache_metadata.json"
)

EXPECTED_COUNTS = {
    "train": 17084,
    "validation": 2146,
    "test": 2158,
}


class CachedPTBXLDataset(Dataset):
    """Read-only access to the frozen PTB-XL tensor cache."""

    def __init__(
        self,
        split: str,
    ) -> None:

        if split not in EXPECTED_COUNTS:
            raise ValueError(
                f"Invalid split: {split!r}"
            )

        self.split = split

        with CACHE_METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            metadata = json.load(handle)

        if metadata["status"] != "FROZEN":
            raise ValueError(
                "ML cache is not frozen"
            )

        split_metadata = metadata[
            "splits"
        ][split]

        split_dir = (
            CACHE_ROOT / split
        )

        paths = {
            "signals":
                split_dir / "signals.npy",
            "targets":
                split_dir / "targets.npy",
            "ecg_ids":
                split_dir / "ecg_ids.npy",
            "patient_ids":
                split_dir / "patient_ids.npy",
        }

        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    path
                )

            observed_hash = (
                sha256_file(path)
            )

            expected_hash = (
                split_metadata[
                    "sha256"
                ][name]
            )

            if observed_hash != expected_hash:
                raise ValueError(
                    f"{split} cache SHA-256 "
                    f"mismatch for {name}"
                )

        self.signals = np.load(
            paths["signals"],
            mmap_mode="r",
        )

        self.targets = np.load(
            paths["targets"],
            mmap_mode="r",
        )

        self.ecg_ids = np.load(
            paths["ecg_ids"],
            mmap_mode="r",
        )

        self.patient_ids = np.load(
            paths["patient_ids"],
            mmap_mode="r",
        )

        expected = (
            EXPECTED_COUNTS[split]
        )

        if self.signals.shape != (
            expected,
            12,
            1000,
        ):
            raise ValueError(
                f"{split}: incorrect signal shape"
            )

        if self.targets.shape != (
            expected,
            5,
        ):
            raise ValueError(
                f"{split}: incorrect target shape"
            )

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:

        if (
            index < 0
            or index >= len(self)
        ):
            raise IndexError(index)

        signal = np.array(
            self.signals[index],
            dtype=np.float32,
            copy=True,
        )

        target = np.array(
            self.targets[index],
            dtype=np.float32,
            copy=True,
        )

        if not np.isfinite(
            signal
        ).all():
            raise ValueError(
                f"{self.split} index {index}: "
                "non-finite signal"
            )

        if not np.isin(
            target,
            [0.0, 1.0],
        ).all():
            raise ValueError(
                f"{self.split} index {index}: "
                "invalid target"
            )

        return {
            "signal":
                torch.from_numpy(signal),

            "target":
                torch.from_numpy(target),

            "ecg_id":
                int(
                    self.ecg_ids[index]
                ),

            "patient_id":
                int(
                    self.patient_ids[index]
                ),

            "split":
                self.split,
        }
