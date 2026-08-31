import json

import numpy as np
import pandas as pd

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "label_confidence"
    / "v1"
    / "validation"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "label_confidence.csv"
)

OUTPUT_METADATA = (
    OUTPUT_DIR
    / "label_confidence_metadata.json"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort"
    / "v1"
    / "ptbxl_superdiagnostic_cohort.csv"
)


def test_frozen_validation_label_confidence_artifact():
    with OUTPUT_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    assert metadata[
        "status"
    ] == "FROZEN"

    assert metadata[
        "evaluation_split"
    ] == "validation"

    assert metadata[
        "records"
    ] == 2146

    assert metadata[
        "test_records_used"
    ] == 0

    assert metadata[
        "target_reconstruction"
    ] == "PASS"

    assert (
        sha256_file(
            OUTPUT_CSV
        )
        == metadata[
            "output_csv_sha256"
        ]
    )

    data = pd.read_csv(
        OUTPUT_CSV
    )

    assert len(data) == 2146
    assert data[
        "ecg_id"
    ].is_unique

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    expected_ids = (
        manifest.loc[
            manifest["split"].eq(
                "validation"
            ),
            "ecg_id",
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    assert np.array_equal(
        data[
            "ecg_id"
        ].to_numpy(
            dtype=np.int64
        ),
        expected_ids,
    )

    confidence = data[
        "label_confidence"
    ].to_numpy(
        dtype=np.float64
    )

    ambiguity = data[
        "label_ambiguity"
    ].to_numpy(
        dtype=np.float64
    )

    coverage = data[
        "coverage"
    ].to_numpy(
        dtype=np.float64
    )

    assert np.isfinite(
        confidence
    ).all()

    assert np.isfinite(
        ambiguity
    ).all()

    assert np.all(
        (confidence >= 0.0)
        & (confidence <= 1.0)
    )

    assert np.all(
        (ambiguity >= 0.0)
        & (ambiguity <= 1.0)
    )

    assert np.all(
        (coverage >= 0.0)
        & (coverage <= 1.0)
    )

    np.testing.assert_allclose(
        ambiguity,
        1.0 - confidence,
        rtol=0.0,
        atol=1e-15,
    )

    assert int(
        data[
            "imputed"
        ].sum()
    ) == 1

    assert int(
        data[
            "raw_label_confidence"
        ].isna().sum()
    ) == 1

    assert (
        metadata[
            "missingness"
        ][
            "records_with_no_known_superclass_likelihood"
        ]
        == 1
    )
