import json

import numpy as np

from ptbxl_reliability.cached_dataset import CACHE_ROOT
from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.uncertainty import mc_uncertainty


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "uncertainty"
    / "v1"
    / "validation"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "mc_dropout_uncertainty.npz"
)

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "uncertainty_metadata.json"
)


def test_frozen_validation_uncertainty_artifact():
    with OUTPUT_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    assert metadata["status"] == "FROZEN"
    assert metadata["evaluation_split"] == "validation"
    assert metadata["records"] == 2146
    assert metadata["mc_passes"] == 30
    assert metadata["test_records_used"] == 0

    assert (
        metadata[
            "batchnorm_state_unchanged"
        ]
        is True
    )

    assert (
        metadata[
            "entire_model_state_unchanged"
        ]
        is True
    )

    assert (
        metadata[
            "saved_artifact_recomputation"
        ]
        == "PASS"
    )

    assert (
        sha256_file(
            OUTPUT_NPZ
        )
        == metadata[
            "output_npz_sha256"
        ]
    )

    saved = np.load(
        OUTPUT_NPZ,
        allow_pickle=False,
    )

    assert saved[
        "mc_probabilities"
    ].shape == (
        30,
        2146,
        5,
    )

    assert saved[
        "mean_probability"
    ].shape == (
        2146,
        5,
    )

    assert saved[
        "mutual_information"
    ].shape == (
        2146,
        5,
    )

    assert saved[
        "mean_mutual_information"
    ].shape == (
        2146,
    )

    assert tuple(
        str(value)
        for value
        in saved[
            "class_names"
        ].tolist()
    ) == SUPERCLASS_ORDER

    cached_targets = np.load(
        CACHE_ROOT
        / "validation"
        / "targets.npy",
        mmap_mode="r",
    )

    cached_ecg_ids = np.load(
        CACHE_ROOT
        / "validation"
        / "ecg_ids.npy",
        mmap_mode="r",
    )

    assert np.array_equal(
        saved["targets"],
        cached_targets,
    )

    assert np.array_equal(
        saved["ecg_ids"],
        cached_ecg_ids,
    )

    recomputed = mc_uncertainty(
        saved[
            "mc_probabilities"
        ]
    )

    np.testing.assert_allclose(
        saved[
            "mean_mutual_information"
        ],
        recomputed[
            "mean_mutual_information"
        ],
        rtol=1e-6,
        atol=1e-8,
    )

    assert np.all(
        saved[
            "mean_mutual_information"
        ] >= 0.0
    )

    assert float(
        np.max(
            saved[
                "mean_mutual_information"
            ]
        )
    ) > 0.0
