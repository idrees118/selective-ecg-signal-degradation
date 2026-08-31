import json

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.normalization_io import (
    load_frozen_global_zscore,
)
from ptbxl_reliability.paths import PROJECT_ROOT


STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "v1"
    / "global_zscore_stats.json"
)

AUDIT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "normalization_fit_audit.json"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort"
    / "v1"
    / "ptbxl_superdiagnostic_cohort.csv"
)


def test_frozen_normalization_parameters_are_valid():
    parameters, metadata = (
        load_frozen_global_zscore(
            STATS_PATH
        )
    )

    assert parameters.std > 0.0

    assert metadata["fit_split"] == "train"

    assert (
        metadata["training_records"]
        == 17084
    )

    assert (
        metadata["training_scalar_values"]
        == 205008000
    )


def test_frozen_parameter_hash_matches_fit_audit():
    with AUDIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        audit = json.load(handle)

    assert (
        sha256_file(STATS_PATH)
        == audit["parameters_sha256"]
    )


def test_normalization_references_current_frozen_manifest():
    _, metadata = load_frozen_global_zscore(
        STATS_PATH
    )

    assert (
        sha256_file(MANIFEST_PATH)
        == metadata["manifest_sha256"]
    )
