#!/usr/bin/env python3
"""Create the immutable PTB-XL modelling cohort manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ptbxl_reliability.cohort import (
    build_cohort_manifest,
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort"
    / "v1"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "ptbxl_superdiagnostic_cohort.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "cohort_metadata.json"
)


def load_config(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "Cohort configuration must be a mapping"
        )

    return config


def verify_source_hashes(config: dict) -> dict[str, str]:
    files = {
        "ptbxl_database.csv":
            PTBXL_DATABASE_CSV,
        "scp_statements.csv":
            SCP_STATEMENTS_CSV,
    }

    expected_hashes = config[
        "source_sha256"
    ]

    observed: dict[str, str] = {}

    for name, path in files.items():
        digest = sha256_file(path)
        expected = str(
            expected_hashes[name]
        )

        if digest != expected:
            raise ValueError(
                f"Source hash mismatch for {name}: "
                f"expected {expected}, "
                f"observed {digest}"
            )

        observed[name] = digest

    return observed


def main() -> None:
    config = load_config(
        CONFIG_PATH
    )

    source_hashes = verify_source_hashes(
        config
    )

    database = load_ptbxl_database(
        PTBXL_DATABASE_CSV
    )

    statements = load_scp_statements(
        SCP_STATEMENTS_CSV
    )

    manifest = build_cohort_manifest(
        database,
        statements,
        config,
    )

    expected = config[
        "verified_expected_counts"
    ]

    excluded = (
        int(expected["source_records"])
        - len(manifest)
    )

    if excluded != int(
        expected["excluded_records"]
    ):
        raise ValueError(
            "Excluded-record count does not match "
            "the independently verified cohort definition"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Fixed column ordering + fixed row ordering + fixed newline
    # make this CSV byte-deterministic for a fixed software stack.
    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        lineterminator="\n",
    )

    manifest_hash = sha256_file(
        MANIFEST_PATH
    )

    config_hash = sha256_file(
        CONFIG_PATH
    )

    metadata = {
        "cohort_id":
            config["cohort_id"],
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "dataset":
            config["dataset"],
        "task":
            config["task"],
        "inclusion_rule":
            config["inclusion_rule"],
        "split_definition":
            config["split"],
        "source_sha256":
            source_hashes,
        "config_sha256":
            config_hash,
        "manifest_sha256":
            manifest_hash,
        "manifest_file":
            str(
                MANIFEST_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        "records":
            int(len(manifest)),
        "patients":
            int(
                manifest[
                    "patient_id"
                ].nunique()
            ),
        "records_by_split": {
            split: int(
                manifest["split"]
                .eq(split)
                .sum()
            )
            for split in (
                "train",
                "validation",
                "test",
            )
        },
        "patients_by_split": {
            split: int(
                manifest.loc[
                    manifest["split"]
                    .eq(split),
                    "patient_id",
                ].nunique()
            )
            for split in (
                "train",
                "validation",
                "test",
            )
        },
        "status": "FROZEN",
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        f"Frozen manifest: {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
