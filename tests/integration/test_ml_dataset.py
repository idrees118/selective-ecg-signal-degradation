import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader

from ptbxl_reliability.dataset import (
    MANIFEST_PATH,
    NORMALIZATION_PATH,
    PTBXLSuperdiagnosticDataset,
)
from ptbxl_reliability.labels import (
    SUPERCLASS_ORDER,
)
from ptbxl_reliability.normalization_io import (
    load_frozen_global_zscore,
)
from ptbxl_reliability.paths import (
    PTBXL_ROOT,
)


def test_dataset_lengths_are_frozen():
    assert len(
        PTBXLSuperdiagnosticDataset(
            "train"
        )
    ) == 17084

    assert len(
        PTBXLSuperdiagnosticDataset(
            "validation"
        )
    ) == 2146

    assert len(
        PTBXLSuperdiagnosticDataset(
            "test"
        )
    ) == 2158


def test_first_training_record_exact_contract():
    dataset = PTBXLSuperdiagnosticDataset(
        "train"
    )

    sample = dataset[0]

    # Frozen manifest begins with ECG 1.
    assert sample["ecg_id"] == 1
    assert sample["patient_id"] == 15709
    assert sample["split"] == "train"

    assert (
        sample["filename_lr"]
        == "records100/00000/00001_lr"
    )

    assert sample["signal"].shape == (
        12,
        1000,
    )

    assert sample["signal"].dtype == (
        torch.float32
    )

    assert sample["target"].shape == (
        5,
    )

    assert sample["target"].dtype == (
        torch.float32
    )

    torch.testing.assert_close(
        sample["target"],
        torch.tensor(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_dataset_signal_matches_independent_manual_formula():
    dataset = PTBXLSuperdiagnosticDataset(
        "train"
    )

    sample = dataset[0]

    parameters, _ = (
        load_frozen_global_zscore(
            NORMALIZATION_PATH
        )
    )

    raw_signal, _ = wfdb.rdsamp(
        str(
            PTBXL_ROOT
            / sample["filename_lr"]
        )
    )

    # Independent direct formula:
    expected = (
        (
            np.asarray(
                raw_signal,
                dtype=np.float64,
            )
            - parameters.mean
        )
        / parameters.std
    )

    expected = (
        expected
        .T
        .astype(
            np.float32
        )
    )

    np.testing.assert_allclose(
        sample["signal"].numpy(),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_dataset_targets_exactly_match_manifest():
    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    for split in (
        "train",
        "validation",
        "test",
    ):
        dataset = (
            PTBXLSuperdiagnosticDataset(
                split
            )
        )

        # Deterministic checks at beginning,
        # middle and end of each split.
        indices = [
            0,
            len(dataset) // 2,
            len(dataset) - 1,
        ]

        split_manifest = (
            manifest.loc[
                manifest["split"].eq(
                    split
                )
            ]
            .reset_index(drop=True)
        )

        for index in indices:
            sample = dataset[index]

            row = split_manifest.iloc[
                index
            ]

            assert (
                sample["ecg_id"]
                == int(row["ecg_id"])
            )

            assert (
                sample["patient_id"]
                == int(
                    row["patient_id"]
                )
            )

            expected_target = torch.tensor(
                [
                    float(
                        row[label]
                    )
                    for label
                    in SUPERCLASS_ORDER
                ],
                dtype=torch.float32,
            )

            torch.testing.assert_close(
                sample["target"],
                expected_target,
                rtol=0.0,
                atol=0.0,
            )


def test_dataloader_batch_contract():
    dataset = PTBXLSuperdiagnosticDataset(
        "train"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    batch = next(
        iter(loader)
    )

    assert batch["signal"].shape == (
        8,
        12,
        1000,
    )

    assert batch["target"].shape == (
        8,
        5,
    )

    assert batch["signal"].dtype == (
        torch.float32
    )

    assert batch["target"].dtype == (
        torch.float32
    )

    assert all(
        split == "train"
        for split in batch["split"]
    )

    assert torch.isfinite(
        batch["signal"]
    ).all()

    assert torch.isfinite(
        batch["target"]
    ).all()
