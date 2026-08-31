"""Canonical PTB-XL five-superclass diagnostic target construction.

The target construction implemented here follows the PTB-XL diagnostic
aggregation convention:

1. Identify SCP statements marked as diagnostic in scp_statements.csv.
2. Map those statements to diagnostic_class.
3. A superclass is present if at least one corresponding diagnostic SCP code
   occurs in the ECG's scp_codes dictionary.
4. SCP likelihood values do NOT determine target presence.

Likelihood is deliberately kept separate from target construction because it
will later serve as candidate annotation-reliability information.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ptbxl_reliability.metadata import parse_scp_codes


SUPERCLASS_ORDER: tuple[str, ...] = (
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
)

EXPECTED_V103_SUPERCLASS_COUNTS: dict[str, int] = {
    "NORM": 9514,
    "MI": 5469,
    "STTC": 5235,
    "CD": 4898,
    "HYP": 2649,
}


def build_diagnostic_code_to_superclass(
    statements: pd.DataFrame,
) -> dict[str, str]:
    """Construct SCP diagnostic-code -> superclass mapping.

    Only rows satisfying ``diagnostic == 1`` are eligible.

    Returns
    -------
    dict[str, str]
        Mapping from SCP code to one of:
        NORM, MI, STTC, CD, HYP.
    """

    required = {"diagnostic", "diagnostic_class"}

    missing = required.difference(statements.columns)

    if missing:
        raise ValueError(
            "SCP statement table is missing columns: "
            f"{sorted(missing)}"
        )

    diagnostic_rows = statements.loc[
        statements["diagnostic"].eq(1.0)
    ]

    mapping: dict[str, str] = {}

    for code, row in diagnostic_rows.iterrows():
        raw_superclass = row["diagnostic_class"]

        if pd.isna(raw_superclass):
            continue

        superclass = str(raw_superclass)

        if superclass not in SUPERCLASS_ORDER:
            raise ValueError(
                f"Diagnostic SCP code {code!r} maps to unexpected "
                f"superclass {superclass!r}"
            )

        mapping[str(code)] = superclass

    if not mapping:
        raise ValueError(
            "No diagnostic SCP-code-to-superclass mappings were found"
        )

    return mapping


def diagnostic_superclasses_for_record(
    raw_scp_codes: str,
    code_to_superclass: Mapping[str, str],
) -> tuple[str, ...]:
    """Aggregate one record into its diagnostic superclasses.

    Important
    ---------
    The likelihood values stored in ``scp_codes`` are intentionally ignored
    for target PRESENCE.

    For example, if a diagnostic code is present with likelihood 0, it remains
    present for benchmark-compatible target construction. The likelihood
    value itself is preserved elsewhere for later reliability analysis.

    The returned tuple always follows ``SUPERCLASS_ORDER``.
    """

    codes = parse_scp_codes(raw_scp_codes)

    observed = {
        code_to_superclass[code]
        for code in codes
        if code in code_to_superclass
    }

    return tuple(
        superclass
        for superclass in SUPERCLASS_ORDER
        if superclass in observed
    )


def build_diagnostic_superclass_targets(
    database: pd.DataFrame,
    statements: pd.DataFrame,
) -> pd.DataFrame:
    """Construct an auditable five-column multi-label target table.

    The returned table contains every PTB-XL record. Records without any
    diagnostic superclass remain explicitly identifiable through
    ``n_superclass_labels == 0``.

    No record is silently dropped by this function.
    """

    required_database_columns = {
        "ecg_id",
        "patient_id",
        "strat_fold",
        "scp_codes",
    }

    missing = required_database_columns.difference(database.columns)

    if missing:
        raise ValueError(
            "PTB-XL database is missing columns required for "
            f"target construction: {sorted(missing)}"
        )

    mapping = build_diagnostic_code_to_superclass(statements)

    rows: list[dict[str, object]] = []

    for row in database[
        ["ecg_id", "patient_id", "strat_fold", "scp_codes"]
    ].itertuples(index=False):

        classes = diagnostic_superclasses_for_record(
            row.scp_codes,
            mapping,
        )

        class_set = set(classes)

        output: dict[str, object] = {
            "ecg_id": int(row.ecg_id),
            "patient_id": row.patient_id,
            "strat_fold": int(row.strat_fold),
        }

        for superclass in SUPERCLASS_ORDER:
            output[superclass] = int(
                superclass in class_set
            )

        output["n_superclass_labels"] = len(classes)

        rows.append(output)

    targets = pd.DataFrame(rows)

    for superclass in SUPERCLASS_ORDER:
        targets[superclass] = targets[superclass].astype(
            np.int8
        )

    targets["n_superclass_labels"] = (
        targets["n_superclass_labels"].astype(np.int8)
    )

    validate_target_matrix(targets)

    return targets


def validate_target_matrix(targets: pd.DataFrame) -> None:
    """Validate mathematical and structural properties of targets."""

    missing = set(SUPERCLASS_ORDER).difference(targets.columns)

    if missing:
        raise ValueError(
            f"Missing superclass target columns: {sorted(missing)}"
        )

    matrix = targets.loc[:, SUPERCLASS_ORDER].to_numpy()

    if matrix.ndim != 2:
        raise ValueError("Target matrix must be two-dimensional")

    if matrix.shape[1] != len(SUPERCLASS_ORDER):
        raise ValueError(
            f"Expected {len(SUPERCLASS_ORDER)} target columns, "
            f"found {matrix.shape[1]}"
        )

    if not np.isin(matrix, [0, 1]).all():
        raise ValueError(
            "Diagnostic superclass targets must be binary"
        )

    calculated_cardinality = matrix.sum(axis=1)

    stored_cardinality = targets[
        "n_superclass_labels"
    ].to_numpy()

    if not np.array_equal(
        calculated_cardinality,
        stored_cardinality,
    ):
        raise ValueError(
            "n_superclass_labels disagrees with the target matrix"
        )

    if targets["ecg_id"].duplicated().any():
        raise ValueError(
            "Target table contains duplicate ecg_id values"
        )


def validate_official_v103_class_counts(
    targets: pd.DataFrame,
) -> dict[str, int]:
    """Verify target aggregation against official PTB-XL v1.0.3 counts."""

    observed = {
        superclass: int(targets[superclass].sum())
        for superclass in SUPERCLASS_ORDER
    }

    if observed != EXPECTED_V103_SUPERCLASS_COUNTS:
        raise ValueError(
            "Diagnostic superclass counts do not match official "
            "PTB-XL v1.0.3 counts.\n"
            f"Expected: {EXPECTED_V103_SUPERCLASS_COUNTS}\n"
            f"Observed: {observed}"
        )

    return observed


def select_diagnostic_superclass_records(
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Select records eligible for the five-superclass task.

    This explicitly implements the PTB-XL benchmark convention of retaining
    records with at least one diagnostic superclass.
    """

    selected = targets.loc[
        targets["n_superclass_labels"] > 0
    ].copy()

    if selected.empty:
        raise ValueError(
            "No records contain diagnostic superclass labels"
        )

    if (selected["n_superclass_labels"] <= 0).any():
        raise AssertionError(
            "Internal error: selected records contain zero labels"
        )

    return selected.reset_index(drop=True)
