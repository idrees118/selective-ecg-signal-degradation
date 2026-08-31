#!/usr/bin/env python3
"""Audit the PTB-XL five-diagnostic-superclass modelling cohort.

This script performs descriptive and structural auditing only.

It does NOT:
- train a model,
- normalize ECG signals,
- select thresholds,
- inspect predictions,
- define annotation reliability,
- modify the source dataset.

Output:
    logs/label_audit.json
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

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
    parse_scp_codes,
    validate_patient_fold_integrity,
)
from ptbxl_reliability.paths import (
    LOG_DIR,
    PTBXL_DATABASE_CSV,
    SCP_STATEMENTS_CSV,
)


def split_name(fold: int) -> str:
    """Map an official PTB-XL fold to our fixed experiment split."""

    fold = int(fold)

    if 1 <= fold <= 8:
        return "train"

    if fold == 9:
        return "validation"

    if fold == 10:
        return "test"

    raise ValueError(
        f"Unexpected PTB-XL strat_fold: {fold}"
    )


def label_combination(row: pd.Series) -> str:
    """Return deterministic textual representation of active labels."""

    active = [
        superclass
        for superclass in SUPERCLASS_ORDER
        if int(row[superclass]) == 1
    ]

    if not active:
        return "NONE"

    return "+".join(active)


def class_statistics(
    frame: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """Calculate class counts and prevalence for one cohort."""

    n_records = len(frame)

    if n_records == 0:
        raise ValueError(
            "Cannot calculate class statistics for an empty cohort"
        )

    output: dict[str, dict[str, float | int]] = {}

    for superclass in SUPERCLASS_ORDER:
        positive_count = int(frame[superclass].sum())

        output[superclass] = {
            "positive_records": positive_count,
            "prevalence": positive_count / n_records,
        }

    return output


def cardinality_distribution(
    frame: pd.DataFrame,
) -> dict[str, int]:
    """Count records by number of active superclass labels."""

    counts = (
        frame["n_superclass_labels"]
        .value_counts()
        .sort_index()
    )

    return {
        str(int(cardinality)): int(count)
        for cardinality, count in counts.items()
    }


def combination_distribution(
    frame: pd.DataFrame,
) -> dict[str, int]:
    """Count every observed superclass-label combination."""

    combinations = frame.apply(
        label_combination,
        axis=1,
    )

    counts = combinations.value_counts()

    return {
        str(combination): int(count)
        for combination, count in counts.items()
    }


def excluded_scp_code_summary(
    database: pd.DataFrame,
    excluded_ids: set[int],
    statements: pd.DataFrame,
) -> list[dict[str, object]]:
    """Describe SCP codes occurring in zero-superclass records.

    This does not assign a new label to excluded ECGs. It exists only to
    understand which annotation types are present in records that do not
    belong to the five diagnostic-superclass modelling task.
    """

    excluded_database = database.loc[
        database["ecg_id"].isin(excluded_ids)
    ]

    counter: Counter[str] = Counter()

    for raw in excluded_database["scp_codes"]:
        counter.update(parse_scp_codes(raw).keys())

    summary: list[dict[str, object]] = []

    for code, count in counter.most_common():
        if code in statements.index:
            row = statements.loc[code]

            diagnostic = (
                bool(row["diagnostic"] == 1.0)
                if "diagnostic" in row
                else False
            )

            form = (
                bool(row["form"] == 1.0)
                if "form" in row
                else False
            )

            rhythm = (
                bool(row["rhythm"] == 1.0)
                if "rhythm" in row
                else False
            )

            description = (
                None
                if pd.isna(row.get("description"))
                else str(row.get("description"))
            )

        else:
            # This should already have been prevented by the earlier
            # dataset audit, but remain defensive here.
            diagnostic = False
            form = False
            rhythm = False
            description = None

        summary.append(
            {
                "scp_code": code,
                "record_count": int(count),
                "description": description,
                "diagnostic": diagnostic,
                "form": form,
                "rhythm": rhythm,
            }
        )

    return summary


def main() -> None:
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

    # Independent agreement with the official PTB-XL v1.0.3
    # published superclass totals.
    official_counts = (
        validate_official_v103_class_counts(
            targets
        )
    )

    selected = select_diagnostic_superclass_records(
        targets
    )

    excluded = targets.loc[
        targets["n_superclass_labels"].eq(0)
    ].copy()

    if len(selected) + len(excluded) != len(database):
        raise AssertionError(
            "Selected and excluded cohort sizes do not sum "
            "to the source dataset size"
        )

    selected_ids = set(
        selected["ecg_id"].astype(int)
    )

    excluded_ids = set(
        excluded["ecg_id"].astype(int)
    )

    if not selected_ids.isdisjoint(excluded_ids):
        raise AssertionError(
            "An ECG occurs in both selected and excluded cohorts"
        )

    if selected_ids | excluded_ids != set(
        database["ecg_id"].astype(int)
    ):
        raise AssertionError(
            "Selected/excluded ECG IDs do not exhaust the source dataset"
        )

    # Re-check patient integrity after cohort selection.
    validate_patient_fold_integrity(selected)

    selected = selected.copy()
    excluded = excluded.copy()

    selected["split"] = selected[
        "strat_fold"
    ].map(split_name)

    excluded["split"] = excluded[
        "strat_fold"
    ].map(split_name)

    split_report: dict[str, object] = {}

    for name in ("train", "validation", "test"):
        subset = selected.loc[
            selected["split"].eq(name)
        ]

        if subset.empty:
            raise ValueError(
                f"Selected {name} split is unexpectedly empty"
            )

        split_report[name] = {
            "records": int(len(subset)),
            "patients": int(
                subset["patient_id"].nunique()
            ),
            "class_statistics": class_statistics(
                subset
            ),
            "label_cardinality_distribution":
                cardinality_distribution(subset),
        }

    excluded_by_split = {
        name: int(
            excluded["split"].eq(name).sum()
        )
        for name in (
            "train",
            "validation",
            "test",
        )
    }

    report = {
        "audit_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": "PTB-XL",
        "dataset_version": "1.0.3",
        "task": "diagnostic_superclass",
        "task_type": "multilabel",
        "superclass_order": list(
            SUPERCLASS_ORDER
        ),
        "target_definition": (
            "Superclass presence is determined by presence "
            "of diagnostic SCP codes mapped through "
            "scp_statements.csv; SCP likelihood values are "
            "not thresholded for target construction."
        ),
        "source_records": int(
            len(database)
        ),
        "source_patients": int(
            database["patient_id"].nunique()
        ),
        "selected_records": int(
            len(selected)
        ),
        "selected_patients": int(
            selected["patient_id"].nunique()
        ),
        "excluded_zero_superclass_records": int(
            len(excluded)
        ),
        "excluded_zero_superclass_patients": int(
            excluded["patient_id"].nunique()
        ),
        "excluded_records_by_split":
            excluded_by_split,
        "official_v103_superclass_counts_expected":
            EXPECTED_V103_SUPERCLASS_COUNTS,
        "official_v103_superclass_counts_observed":
            official_counts,
        "all_selected_class_statistics":
            class_statistics(selected),
        "all_selected_label_cardinality_distribution":
            cardinality_distribution(selected),
        "all_selected_label_combinations":
            combination_distribution(selected),
        "selected_split_statistics":
            split_report,
        "excluded_record_scp_codes":
            excluded_scp_code_summary(
                database,
                excluded_ids,
                statements,
            ),
        "patient_fold_integrity_after_selection":
            "PASS",
        "partition_integrity":
            "PASS",
        "status": "PASS",
    }

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        LOG_DIR / "label_audit.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        f"Label audit saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
