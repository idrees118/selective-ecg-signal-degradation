#!/usr/bin/env python3
"""Structural audit of the local PTB-XL v1.0.3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version

from ptbxl_reliability.metadata import (
    load_ptbxl_database,
    load_scp_statements,
    split_counts,
    validate_patient_fold_integrity,
    validate_scp_code_coverage,
)
from ptbxl_reliability.paths import (
    LOG_DIR,
    PTBXL_DATABASE_CSV,
    PTBXL_ROOT,
    SCP_STATEMENTS_CSV,
    validate_required_paths,
)
from ptbxl_reliability.waveforms import (
    load_ecg_100hz,
    to_model_layout,
)


EXPECTED_METADATA_SHA256 = {
    "ptbxl_database.csv":
        "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
    "scp_statements.csv":
        "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
}


def sha256_file(path) -> str:
    """Compute SHA-256 without loading the entire file into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def verify_metadata_hashes() -> dict[str, str]:
    """Verify the two core metadata files against PhysioNet v1.0.3."""

    files = {
        "ptbxl_database.csv": PTBXL_DATABASE_CSV,
        "scp_statements.csv": SCP_STATEMENTS_CSV,
    }

    calculated: dict[str, str] = {}

    for name, path in files.items():
        digest = sha256_file(path)
        expected = EXPECTED_METADATA_SHA256[name]

        if digest != expected:
            raise ValueError(
                f"SHA-256 mismatch for {name}. "
                f"Expected {expected}, found {digest}"
            )

        calculated[name] = digest

    return calculated


def validate_all_waveform_paths(database) -> None:
    """Verify every metadata filename_lr has both WFDB files."""

    missing: list[str] = []

    for relative_path in database["filename_lr"]:
        base = PTBXL_ROOT / str(relative_path)

        dat_path = base.with_suffix(".dat")
        hea_path = base.with_suffix(".hea")

        if not dat_path.is_file():
            missing.append(str(dat_path))

        if not hea_path.is_file():
            missing.append(str(hea_path))

        if len(missing) >= 20:
            break

    if missing:
        raise FileNotFoundError(
            "Metadata references missing waveform files. "
            f"First examples: {missing}"
        )


def waveform_rows_to_check(database, full: bool):
    """Select either every ECG or one deterministic ECG from every fold."""

    if full:
        return database

    return (
        database.sort_values(["strat_fold", "ecg_id"])
        .groupby("strat_fold", sort=True, as_index=False)
        .head(1)
    )


def package_versions() -> dict[str, str]:
    distributions = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "wfdb",
        "PyYAML",
        "pytest",
    )

    return {
        distribution: version(distribution)
        for distribution in distributions
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full-waveforms",
        action="store_true",
        help=(
            "Read and validate all 21,799 waveforms. "
            "Without this flag, one deterministic record per fold is read."
        ),
    )

    args = parser.parse_args()

    validate_required_paths()

    database = load_ptbxl_database(PTBXL_DATABASE_CSV)
    statements = load_scp_statements(SCP_STATEMENTS_CSV)

    validate_patient_fold_integrity(database)
    observed_codes = validate_scp_code_coverage(database, statements)
    validate_all_waveform_paths(database)

    hashes = verify_metadata_hashes()

    checked_records: list[int] = []

    rows = waveform_rows_to_check(
        database,
        full=args.full_waveforms,
    )

    for _, row in rows.iterrows():
        record_base = PTBXL_ROOT / str(row["filename_lr"])

        record = load_ecg_100hz(record_base)
        model_signal = to_model_layout(record.signal)

        if model_signal.shape != (12, 1000):
            raise AssertionError(
                f"Invalid model tensor shape for ECG {row['ecg_id']}"
            )

        checked_records.append(int(row["ecg_id"]))

    report = {
        "audit_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": "PTB-XL",
        "dataset_version": "1.0.3",
        "dataset_root": str(PTBXL_ROOT),
        "record_count": int(len(database)),
        "patient_count": int(database["patient_id"].nunique()),
        "fold_counts": {
            str(int(fold)): int(count)
            for fold, count in (
                database["strat_fold"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "recommended_split_counts": split_counts(database),
        "observed_scp_code_count": len(observed_codes),
        "diagnostic_superclasses": sorted(
            statements.loc[
                statements["diagnostic"] == 1.0,
                "diagnostic_class",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "metadata_sha256": hashes,
        "waveform_mode": (
            "full"
            if args.full_waveforms
            else "one_record_per_fold"
        ),
        "waveform_records_checked": checked_records,
        "software_versions": package_versions(),
        "status": "PASS",
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    output_path = LOG_DIR / "dataset_audit.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    print()
    print(f"Audit report saved to: {output_path}")


if __name__ == "__main__":
    main()
