"""Validated PTB-XL metadata loading.

This module handles structural interpretation only. It does not define label
reliability or any experimental reliability score.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pandas as pd


EXPECTED_RECORD_COUNT = 21_799
EXPECTED_PATIENT_COUNT = 18_869

EXPECTED_FOLDS = frozenset(range(1, 11))

EXPECTED_DIAGNOSTIC_SUPERCLASSES = frozenset(
    {"NORM", "MI", "STTC", "CD", "HYP"}
)

REQUIRED_DATABASE_COLUMNS = frozenset(
    {
        "ecg_id",
        "patient_id",
        "scp_codes",
        "strat_fold",
        "filename_lr",
        "validated_by",
        "second_opinion",
        "validated_by_human",
        "static_noise",
        "burst_noise",
        "baseline_drift",
        "electrodes_problems",
    }
)

REQUIRED_SCP_COLUMNS = frozenset(
    {
        "description",
        "diagnostic",
        "form",
        "rhythm",
        "diagnostic_class",
        "diagnostic_subclass",
    }
)


def parse_scp_codes(raw: object) -> dict[str, float]:
    """Parse one PTB-XL ``scp_codes`` cell safely.

    PTB-XL stores SCP statements as a string representation of a Python
    dictionary:

        {'NORM': 100.0, 'SR': 0.0}

    Values are statement likelihoods in the range [0, 100].

    Important
    ---------
    A value of 0 is preserved exactly. This function does NOT interpret it
    as zero confidence because PTB-XL also uses 0 when likelihood information
    is unknown.

    ``ast.literal_eval`` is used instead of ``eval`` so arbitrary code cannot
    be executed from CSV contents.
    """

    if not isinstance(raw, str):
        raise TypeError(
            "scp_codes must be stored as a string; "
            f"received {type(raw).__name__}"
        )

    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid scp_codes value: {raw!r}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "scp_codes must decode to a dictionary; "
            f"decoded {type(parsed).__name__}"
        )

    validated: dict[str, float] = {}

    for code, likelihood in parsed.items():
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"Invalid SCP code: {code!r}")

        if isinstance(likelihood, bool) or not isinstance(
            likelihood, (int, float)
        ):
            raise ValueError(
                f"Likelihood for {code!r} must be numeric; "
                f"received {likelihood!r}"
            )

        likelihood_float = float(likelihood)

        if not math.isfinite(likelihood_float):
            raise ValueError(
                f"Likelihood for {code!r} is not finite: "
                f"{likelihood_float}"
            )

        if not 0.0 <= likelihood_float <= 100.0:
            raise ValueError(
                f"Likelihood for {code!r} lies outside [0, 100]: "
                f"{likelihood_float}"
            )

        validated[code] = likelihood_float

    return validated


def load_ptbxl_database(path: Path) -> pd.DataFrame:
    """Load and structurally validate ``ptbxl_database.csv``."""

    frame = pd.read_csv(path)

    missing_columns = REQUIRED_DATABASE_COLUMNS.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            "ptbxl_database.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(frame) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RECORD_COUNT} records, "
            f"found {len(frame)}"
        )

    if frame["ecg_id"].isna().any():
        raise ValueError("ecg_id contains missing values")

    if frame["ecg_id"].duplicated().any():
        duplicated = frame.loc[
            frame["ecg_id"].duplicated(keep=False), "ecg_id"
        ].tolist()
        raise ValueError(
            f"ecg_id is not unique. Examples: {duplicated[:10]}"
        )

    if frame["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values")

    patient_count = frame["patient_id"].nunique(dropna=False)

    if patient_count != EXPECTED_PATIENT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PATIENT_COUNT} unique patients, "
            f"found {patient_count}"
        )

    numeric_folds = pd.to_numeric(frame["strat_fold"], errors="raise")

    if ((numeric_folds % 1) != 0).any():
        raise ValueError("strat_fold contains non-integer values")

    frame = frame.copy()
    frame["strat_fold"] = numeric_folds.astype("int64")

    observed_folds = frozenset(frame["strat_fold"].unique().tolist())

    if observed_folds != EXPECTED_FOLDS:
        raise ValueError(
            f"Expected folds {sorted(EXPECTED_FOLDS)}, "
            f"found {sorted(observed_folds)}"
        )

    # Parse every SCP dictionary now so malformed metadata is discovered
    # before any label construction or model training.
    for row_number, raw in enumerate(frame["scp_codes"], start=2):
        try:
            parse_scp_codes(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid scp_codes at CSV row {row_number}"
            ) from exc

    return frame


def load_scp_statements(path: Path) -> pd.DataFrame:
    """Load and validate the PTB-XL SCP statement mapping."""

    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)

    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].tolist()
        raise ValueError(
            f"scp_statements.csv contains duplicate SCP codes: "
            f"{duplicates[:10]}"
        )

    missing_columns = REQUIRED_SCP_COLUMNS.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            "scp_statements.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    diagnostic_rows = frame.loc[frame["diagnostic"] == 1.0]

    observed_classes = frozenset(
        diagnostic_rows["diagnostic_class"].dropna().astype(str).unique()
    )

    if observed_classes != EXPECTED_DIAGNOSTIC_SUPERCLASSES:
        raise ValueError(
            "Unexpected diagnostic superclass mapping. "
            f"Expected {sorted(EXPECTED_DIAGNOSTIC_SUPERCLASSES)}, "
            f"found {sorted(observed_classes)}"
        )

    return frame


def validate_patient_fold_integrity(frame: pd.DataFrame) -> None:
    """Verify that every patient belongs to exactly one PTB-XL fold."""

    folds_per_patient = (
        frame.groupby("patient_id", dropna=False)["strat_fold"].nunique()
    )

    leaking_patients = folds_per_patient[folds_per_patient != 1]

    if not leaking_patients.empty:
        examples = leaking_patients.index.tolist()[:10]
        raise ValueError(
            "Patient leakage detected across strat_fold values. "
            f"Example patient IDs: {examples}"
        )


def split_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return counts under the recommended PTB-XL split."""

    train = frame["strat_fold"].between(1, 8)
    validation = frame["strat_fold"].eq(9)
    test = frame["strat_fold"].eq(10)

    membership_count = (
        train.astype(int)
        + validation.astype(int)
        + test.astype(int)
    )

    if not membership_count.eq(1).all():
        raise ValueError(
            "Recommended train/validation/test masks are not "
            "mutually exclusive and exhaustive"
        )

    return {
        "train_folds_1_8": int(train.sum()),
        "validation_fold_9": int(validation.sum()),
        "test_fold_10": int(test.sum()),
    }


def validate_scp_code_coverage(
    database: pd.DataFrame,
    statements: pd.DataFrame,
) -> set[str]:
    """Verify every observed SCP code exists in scp_statements.csv."""

    observed_codes: set[str] = set()

    for raw in database["scp_codes"]:
        observed_codes.update(parse_scp_codes(raw))

    known_codes = set(statements.index)

    unknown_codes = observed_codes.difference(known_codes)

    if unknown_codes:
        raise ValueError(
            "Observed SCP codes missing from scp_statements.csv: "
            f"{sorted(unknown_codes)}"
        )

    return observed_codes
