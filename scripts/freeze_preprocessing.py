#!/usr/bin/env python3
"""Freeze verified preprocessing v1 and its provenance."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT


CONFIG = PROJECT_ROOT / "configs" / "preprocessing_v1.yaml"

COHORT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort"
    / "v1"
    / "ptbxl_superdiagnostic_cohort.csv"
)

PARAMETERS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "v1"
    / "global_zscore_stats.json"
)

FIT_AUDIT = (
    PROJECT_ROOT
    / "logs"
    / "normalization_fit_audit.json"
)

APPLICATION_AUDIT = (
    PROJECT_ROOT
    / "logs"
    / "normalization_application_audit.json"
)

EXTREMES_AUDIT = (
    PROJECT_ROOT
    / "logs"
    / "signal_extremes_audit.json"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "v1"
    / "preprocessing_metadata.json"
)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    required = (
        CONFIG,
        COHORT_MANIFEST,
        PARAMETERS,
        FIT_AUDIT,
        APPLICATION_AUDIT,
        EXTREMES_AUDIT,
    )

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    fit = load_json(FIT_AUDIT)
    application = load_json(APPLICATION_AUDIT)
    extremes = load_json(EXTREMES_AUDIT)

    for name, audit in (
        ("normalization fit", fit),
        ("normalization application", application),
        ("signal extremes", extremes),
    ):
        if audit.get("status") != "PASS":
            raise ValueError(
                f"{name} audit did not PASS"
            )

    parameters = load_json(PARAMETERS)

    if parameters["fit_split"] != "train":
        raise ValueError(
            "Normalization was not fitted on training data only"
        )

    if fit["validation_records_used"] != 0:
        raise ValueError(
            "Validation leakage detected"
        )

    if fit["test_records_used"] != 0:
        raise ValueError(
            "Test leakage detected"
        )

    if application["validation_used_for_fit"] is not False:
        raise ValueError(
            "Validation fit flag is invalid"
        )

    if application["test_used_for_fit"] is not False:
        raise ValueError(
            "Test fit flag is invalid"
        )

    if extremes["clipping_applied"] is not False:
        raise ValueError(
            "Unexpected clipping was applied"
        )

    if extremes["records_removed"] != 0:
        raise ValueError(
            "Unexpected ECG removal occurred"
        )

    metadata = {
        "preprocessing_id":
            "ptbxl_v103_preprocessing_v1",

        "frozen_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "input": {
            "sampling_rate_hz": 100,
            "samples": 1000,
            "leads": 12,
            "physical_unit": "mV",
        },

        "signal_processing": {
            "resampling": False,
            "bandpass_filter": False,
            "notch_filter": False,
            "detrending": False,
            "baseline_correction": False,
            "clipping": False,
            "record_removal_for_amplitude": False,
        },

        "normalization": {
            "method": "global_zscore",
            "fit_split": "train",
            "mean_mV":
                parameters["mean_mV"],
            "std_mV":
                parameters["std_mV"],
            "variance_ddof": 0,
        },

        "training_records_used_for_fit":
            parameters["training_records"],

        "training_scalar_values":
            parameters["training_scalar_values"],

        "validation_records_used_for_fit": 0,
        "test_records_used_for_fit": 0,

        "sha256": {
            "cohort_manifest":
                sha256_file(COHORT_MANIFEST),
            "preprocessing_config":
                sha256_file(CONFIG),
            "normalization_parameters":
                sha256_file(PARAMETERS),
            "normalization_fit_audit":
                sha256_file(FIT_AUDIT),
            "normalization_application_audit":
                sha256_file(APPLICATION_AUDIT),
            "signal_extremes_audit":
                sha256_file(EXTREMES_AUDIT),
        },

        "status": "FROZEN",
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
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


if __name__ == "__main__":
    main()
