import json

import pandas as pd
import yaml

from ptbxl_reliability.cohort import (
    build_cohort_manifest,
    validate_cohort_manifest,
)
from ptbxl_reliability.integrity import (
    sha256_file,
)
from ptbxl_reliability.metadata import (
    load_ptbxl_database,
    load_scp_statements,
)
from ptbxl_reliability.paths import (
    PROJECT_ROOT,
    PTBXL_DATABASE_CSV,
    SCP_STATEMENTS_CSV,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "cohort_v1.yaml"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort"
    / "v1"
    / "ptbxl_superdiagnostic_cohort.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort"
    / "v1"
    / "cohort_metadata.json"
)


def test_frozen_cohort_reproduces_exactly():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    frozen = pd.read_csv(
        MANIFEST_PATH
    )

    database = load_ptbxl_database(
        PTBXL_DATABASE_CSV
    )

    statements = load_scp_statements(
        SCP_STATEMENTS_CSV
    )

    regenerated = build_cohort_manifest(
        database,
        statements,
        config,
    )

    pd.testing.assert_frame_equal(
        frozen,
        regenerated,
        check_dtype=False,
        check_like=False,
    )

    validate_cohort_manifest(
        regenerated,
        config,
    )


def test_manifest_hash_matches_freeze_metadata():
    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    assert (
        sha256_file(MANIFEST_PATH)
        == metadata["manifest_sha256"]
    )


def test_frozen_test_split_is_patient_disjoint():
    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    train_patients = set(
        manifest.loc[
            manifest["split"].eq("train"),
            "patient_id",
        ]
    )

    validation_patients = set(
        manifest.loc[
            manifest["split"].eq("validation"),
            "patient_id",
        ]
    )

    test_patients = set(
        manifest.loc[
            manifest["split"].eq("test"),
            "patient_id",
        ]
    )

    assert train_patients.isdisjoint(
        validation_patients
    )

    assert train_patients.isdisjoint(
        test_patients
    )

    assert validation_patients.isdisjoint(
        test_patients
    )
