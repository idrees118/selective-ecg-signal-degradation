import pytest

from ptbxl_reliability.labels import (
    build_diagnostic_superclass_targets,
    select_diagnostic_superclass_records,
)
from ptbxl_reliability.metadata import (
    load_ptbxl_database,
    load_scp_statements,
    validate_patient_fold_integrity,
)
from ptbxl_reliability.paths import (
    PTBXL_DATABASE_CSV,
    SCP_STATEMENTS_CSV,
)


pytestmark = pytest.mark.integration


def _load_targets():
    database = load_ptbxl_database(
        PTBXL_DATABASE_CSV
    )

    statements = load_scp_statements(
        SCP_STATEMENTS_CSV
    )

    targets = build_diagnostic_superclass_targets(
        database,
        statements,
    )

    return database, targets


def test_selected_and_excluded_records_partition_dataset():
    database, targets = _load_targets()

    selected = select_diagnostic_superclass_records(
        targets
    )

    excluded = targets.loc[
        targets["n_superclass_labels"].eq(0)
    ].copy()

    selected_ids = set(selected["ecg_id"])
    excluded_ids = set(excluded["ecg_id"])
    database_ids = set(database["ecg_id"])

    # No ECG may be simultaneously included and excluded.
    assert selected_ids.isdisjoint(excluded_ids)

    # Every source ECG must belong to exactly one side.
    assert selected_ids | excluded_ids == database_ids

    assert len(selected) + len(excluded) == len(database)

    assert selected["n_superclass_labels"].ge(1).all()
    assert excluded["n_superclass_labels"].eq(0).all()


def test_selected_cohort_preserves_patient_fold_integrity():
    _, targets = _load_targets()

    selected = select_diagnostic_superclass_records(
        targets
    )

    # Selection by label availability must not introduce
    # patient overlap between PTB-XL folds.
    validate_patient_fold_integrity(selected)
