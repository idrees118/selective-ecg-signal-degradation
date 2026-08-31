"""Construction and validation of the frozen PTB-XL modelling cohort."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ptbxl_reliability.labels import (
    SUPERCLASS_ORDER,
    build_diagnostic_superclass_targets,
    select_diagnostic_superclass_records,
)


MANIFEST_COLUMNS = (
    "ecg_id",
    "patient_id",
    "strat_fold",
    "split",
    "filename_lr",
    *SUPERCLASS_ORDER,
    "n_superclass_labels",
)


def split_from_fold(
    fold: int,
    config: Mapping,
) -> str:
    """Map one PTB-XL fold to the frozen experimental split."""

    fold = int(fold)

    train = set(config["split"]["train_folds"])
    validation = set(config["split"]["validation_folds"])
    test = set(config["split"]["test_folds"])

    memberships = (
        fold in train,
        fold in validation,
        fold in test,
    )

    if sum(memberships) != 1:
        raise ValueError(
            f"Fold {fold} does not map to exactly one split"
        )

    if fold in train:
        return "train"

    if fold in validation:
        return "validation"

    return "test"


def build_cohort_manifest(
    database: pd.DataFrame,
    statements: pd.DataFrame,
    config: Mapping,
) -> pd.DataFrame:
    """Build the deterministic cohort-v1 manifest."""

    targets = build_diagnostic_superclass_targets(
        database,
        statements,
    )

    selected = select_diagnostic_superclass_records(
        targets
    )

    metadata = database.loc[
        :,
        [
            "ecg_id",
            "filename_lr",
        ],
    ].copy()

    if metadata["ecg_id"].duplicated().any():
        raise ValueError(
            "Source database contains duplicate ecg_id values"
        )

    manifest = selected.merge(
        metadata,
        on="ecg_id",
        how="left",
        validate="one_to_one",
    )

    if manifest["filename_lr"].isna().any():
        raise ValueError(
            "Selected cohort contains missing filename_lr values"
        )

    manifest["split"] = manifest[
        "strat_fold"
    ].map(
        lambda fold: split_from_fold(
            int(fold),
            config,
        )
    )

    manifest["ecg_id"] = manifest[
        "ecg_id"
    ].astype("int64")

    manifest["patient_id"] = manifest[
        "patient_id"
    ].astype("int64")

    manifest["strat_fold"] = manifest[
        "strat_fold"
    ].astype("int64")

    manifest["n_superclass_labels"] = manifest[
        "n_superclass_labels"
    ].astype("int64")

    for label in SUPERCLASS_ORDER:
        manifest[label] = manifest[
            label
        ].astype("int64")

    manifest = manifest.loc[
        :,
        MANIFEST_COLUMNS,
    ]

    # Deterministic ordering is critical because the manifest itself
    # receives a SHA-256 digest.
    manifest = (
        manifest
        .sort_values(
            "ecg_id",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    validate_cohort_manifest(
        manifest,
        config,
    )

    return manifest


def validate_cohort_manifest(
    manifest: pd.DataFrame,
    config: Mapping,
) -> None:
    """Validate all frozen cohort invariants."""

    if tuple(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(
            "Manifest column ordering does not match "
            "the frozen cohort schema"
        )

    if manifest["ecg_id"].duplicated().any():
        raise ValueError(
            "Manifest contains duplicate ecg_id values"
        )

    if manifest["filename_lr"].duplicated().any():
        raise ValueError(
            "Manifest contains duplicate waveform paths"
        )

    if manifest["ecg_id"].isna().any():
        raise ValueError(
            "Manifest contains missing ecg_id values"
        )

    if manifest["patient_id"].isna().any():
        raise ValueError(
            "Manifest contains missing patient_id values"
        )

    if manifest["filename_lr"].isna().any():
        raise ValueError(
            "Manifest contains missing filename_lr values"
        )

    label_matrix = manifest.loc[
        :,
        SUPERCLASS_ORDER,
    ]

    if not label_matrix.isin([0, 1]).all().all():
        raise ValueError(
            "Manifest diagnostic labels are not binary"
        )

    calculated_cardinality = label_matrix.sum(
        axis=1
    )

    if not calculated_cardinality.equals(
        manifest["n_superclass_labels"]
    ):
        raise ValueError(
            "Stored label cardinality disagrees "
            "with superclass labels"
        )

    if (
        manifest["n_superclass_labels"] < 1
    ).any():
        raise ValueError(
            "Frozen modelling cohort contains a "
            "zero-superclass ECG"
        )

    allowed_splits = {
        "train",
        "validation",
        "test",
    }

    observed_splits = set(
        manifest["split"].unique()
    )

    if observed_splits != allowed_splits:
        raise ValueError(
            f"Unexpected split values: {observed_splits}"
        )

    # Patient-level leakage test on the actual modelling cohort.
    split_count_per_patient = (
        manifest
        .groupby("patient_id")["split"]
        .nunique()
    )

    leaking = split_count_per_patient[
        split_count_per_patient != 1
    ]

    if not leaking.empty:
        raise ValueError(
            "Patient leakage detected in frozen cohort. "
            f"Example patient IDs: "
            f"{leaking.index.tolist()[:10]}"
        )

    expected = config[
        "verified_expected_counts"
    ]

    actual = {
        "selected_records": len(manifest),
        "selected_patients":
            manifest["patient_id"].nunique(),
        "train_records":
            int(manifest["split"].eq("train").sum()),
        "validation_records":
            int(
                manifest["split"]
                .eq("validation")
                .sum()
            ),
        "test_records":
            int(manifest["split"].eq("test").sum()),
    }

    for key, observed in actual.items():
        required = int(expected[key])

        if int(observed) != required:
            raise ValueError(
                f"Cohort invariant {key!r} failed: "
                f"expected {required}, observed {observed}"
            )
