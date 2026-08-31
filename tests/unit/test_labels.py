import pandas as pd

from ptbxl_reliability.labels import (
    SUPERCLASS_ORDER,
    build_diagnostic_code_to_superclass,
    build_diagnostic_superclass_targets,
    diagnostic_superclasses_for_record,
    select_diagnostic_superclass_records,
)


def toy_statements():
    return pd.DataFrame(
        {
            "diagnostic": [
                1.0,
                1.0,
                1.0,
                0.0,
            ],
            "diagnostic_class": [
                "NORM",
                "MI",
                "MI",
                None,
            ],
        },
        index=[
            "NORM_CODE",
            "MI_CODE_A",
            "MI_CODE_B",
            "RHYTHM_CODE",
        ],
    )


def test_code_to_superclass_uses_only_diagnostic_rows():
    mapping = build_diagnostic_code_to_superclass(
        toy_statements()
    )

    assert mapping == {
        "NORM_CODE": "NORM",
        "MI_CODE_A": "MI",
        "MI_CODE_B": "MI",
    }


def test_multilabel_aggregation():
    mapping = build_diagnostic_code_to_superclass(
        toy_statements()
    )

    result = diagnostic_superclasses_for_record(
        "{'NORM_CODE': 100, 'MI_CODE_A': 80}",
        mapping,
    )

    assert result == ("NORM", "MI")


def test_two_codes_same_superclass_do_not_duplicate_label():
    mapping = build_diagnostic_code_to_superclass(
        toy_statements()
    )

    result = diagnostic_superclasses_for_record(
        "{'MI_CODE_A': 100, 'MI_CODE_B': 50}",
        mapping,
    )

    assert result == ("MI",)


def test_zero_likelihood_does_not_remove_task_label():
    mapping = build_diagnostic_code_to_superclass(
        toy_statements()
    )

    result = diagnostic_superclasses_for_record(
        "{'MI_CODE_A': 0}",
        mapping,
    )

    assert result == ("MI",)


def test_non_diagnostic_statement_does_not_create_target():
    mapping = build_diagnostic_code_to_superclass(
        toy_statements()
    )

    result = diagnostic_superclasses_for_record(
        "{'RHYTHM_CODE': 100}",
        mapping,
    )

    assert result == ()


def test_target_matrix_is_fixed_order_and_multilabel():
    database = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [10, 20, 30],
            "strat_fold": [1, 9, 10],
            "scp_codes": [
                "{'NORM_CODE': 100}",
                "{'MI_CODE_A': 80, 'NORM_CODE': 100}",
                "{'RHYTHM_CODE': 100}",
            ],
        }
    )

    targets = build_diagnostic_superclass_targets(
        database,
        toy_statements(),
    )

    assert SUPERCLASS_ORDER == (
        "NORM",
        "MI",
        "STTC",
        "CD",
        "HYP",
    )

    assert targets.loc[0, list(SUPERCLASS_ORDER)].tolist() == [
        1, 0, 0, 0, 0
    ]

    assert targets.loc[1, list(SUPERCLASS_ORDER)].tolist() == [
        1, 1, 0, 0, 0
    ]

    assert targets.loc[2, list(SUPERCLASS_ORDER)].tolist() == [
        0, 0, 0, 0, 0
    ]

    assert targets["n_superclass_labels"].tolist() == [
        1, 2, 0
    ]


def test_selection_explicitly_removes_zero_label_records():
    database = pd.DataFrame(
        {
            "ecg_id": [1, 2],
            "patient_id": [10, 20],
            "strat_fold": [1, 10],
            "scp_codes": [
                "{'MI_CODE_A': 100}",
                "{'RHYTHM_CODE': 100}",
            ],
        }
    )

    targets = build_diagnostic_superclass_targets(
        database,
        toy_statements(),
    )

    selected = select_diagnostic_superclass_records(
        targets
    )

    assert selected["ecg_id"].tolist() == [1]
