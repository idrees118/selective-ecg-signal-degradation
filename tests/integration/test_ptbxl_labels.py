import pytest

from ptbxl_reliability.labels import (
    EXPECTED_V103_SUPERCLASS_COUNTS,
    SUPERCLASS_ORDER,
    build_diagnostic_superclass_targets,
    select_diagnostic_superclass_records,
    validate_official_v103_class_counts,
)
from ptbxl_reliability.metadata import (
    load_ptbxl_database,
    load_scp_statements,
)
from ptbxl_reliability.paths import (
    PTBXL_DATABASE_CSV,
    SCP_STATEMENTS_CSV,
)


pytestmark = pytest.mark.integration


def test_official_v103_superclass_counts():
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

    observed = validate_official_v103_class_counts(
        targets
    )

    assert observed == EXPECTED_V103_SUPERCLASS_COUNTS


def test_superclass_task_is_multilabel_and_has_five_outputs():
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

    assert list(
        targets.loc[:, SUPERCLASS_ORDER].columns
    ) == list(SUPERCLASS_ORDER)

    # PTB-XL explicitly contains co-occurring diagnoses.
    assert (
        targets["n_superclass_labels"] > 1
    ).any()


def test_selected_superclass_records_have_at_least_one_label():
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

    selected = select_diagnostic_superclass_records(
        targets
    )

    assert (
        selected["n_superclass_labels"] >= 1
    ).all()

    assert len(selected) <= len(database)
