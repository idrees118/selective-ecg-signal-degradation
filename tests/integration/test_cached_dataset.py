import torch

from ptbxl_reliability.cached_dataset import (
    CachedPTBXLDataset,
)
from ptbxl_reliability.dataset import (
    PTBXLSuperdiagnosticDataset,
)


def test_cache_lengths():
    assert len(
        CachedPTBXLDataset("train")
    ) == 17084

    assert len(
        CachedPTBXLDataset("validation")
    ) == 2146

    assert len(
        CachedPTBXLDataset("test")
    ) == 2158


def test_cached_training_examples_equal_verified_dataset():
    cached = CachedPTBXLDataset(
        "train"
    )

    reference = (
        PTBXLSuperdiagnosticDataset(
            "train"
        )
    )

    for index in (
        0,
        len(cached) // 2,
        len(cached) - 1,
    ):
        a = cached[index]
        b = reference[index]

        assert (
            a["ecg_id"]
            == b["ecg_id"]
        )

        assert (
            a["patient_id"]
            == b["patient_id"]
        )

        torch.testing.assert_close(
            a["signal"],
            b["signal"],
            rtol=0.0,
            atol=0.0,
        )

        torch.testing.assert_close(
            a["target"],
            b["target"],
            rtol=0.0,
            atol=0.0,
        )
