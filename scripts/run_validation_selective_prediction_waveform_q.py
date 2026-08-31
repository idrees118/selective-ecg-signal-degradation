#!/usr/bin/env python3
"""Validation selective prediction using waveform-derived Q."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.selective_prediction import (
    grouped_risk_coverage,
    mean_bernoulli_confidence,
    midrank_ecdf,
    risk_at_or_above_coverage,
    sample_hamming_error,
)


BASELINE_DIR = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "v1"
    / "seed_20260826"
)

PREDICTIONS_PATH = (
    BASELINE_DIR
    / "best_validation_predictions.npz"
)

THRESHOLDS_PATH = (
    BASELINE_DIR
    / "validation_thresholds.json"
)

L_PATH = (
    PROJECT_ROOT
    / "results"
    / "label_confidence"
    / "v1"
    / "validation"
    / "label_confidence.csv"
)

Q_PATH = (
    PROJECT_ROOT
    / "results"
    / "waveform_signal_quality"
    / "v1"
    / "validation"
    / "waveform_signal_quality.csv"
)

U_PATH = (
    PROJECT_ROOT
    / "results"
    / "uncertainty"
    / "v1"
    / "validation"
    / "mc_dropout_uncertainty.npz"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "selective_prediction_waveform_q_v1.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "selective_prediction_waveform_q"
    / "v1"
    / "validation"
)

SUMMARY_PATH = OUTPUT_DIR / "summary.json"
SCORES_PATH = OUTPUT_DIR / "record_scores.csv"
CURVES_PATH = OUTPUT_DIR / "risk_coverage_curves.npz"
REFERENCE_PATH = OUTPUT_DIR / "combination_reference.npz"

EXPECTED_RECORDS = 2146

TARGET_COVERAGES = (
    0.50,
    0.70,
    0.80,
    0.90,
    1.00,
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"{path} must contain an object"
        )

    return value


def main() -> None:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if config["status"] != "FROZEN":
        raise ValueError(
            "Config is not FROZEN"
        )

    if config["development_split"] != "validation":
        raise ValueError(
            "Development split changed"
        )

    if config["test_allowed"] is not False:
        raise ValueError(
            "Test use is forbidden"
        )

    baseline = np.load(
        PREDICTIONS_PATH,
        allow_pickle=False,
    )

    probabilities = np.asarray(
        baseline["probabilities"],
        dtype=np.float64,
    )

    targets = np.asarray(
        baseline["targets"],
        dtype=np.int64,
    )

    ecg_ids = np.asarray(
        baseline["ecg_ids"],
        dtype=np.int64,
    )

    if probabilities.shape != (
        EXPECTED_RECORDS,
        5,
    ):
        raise ValueError(
            "Unexpected probability shape"
        )

    if targets.shape != (
        EXPECTED_RECORDS,
        5,
    ):
        raise ValueError(
            "Unexpected target shape"
        )

    class_names = tuple(
        str(x)
        for x in baseline[
            "class_names"
        ].tolist()
    )

    if class_names != SUPERCLASS_ORDER:
        raise ValueError(
            "Class order mismatch"
        )

    threshold_metadata = load_json(
        THRESHOLDS_PATH
    )

    if threshold_metadata[
        "test_records_used"
    ] != 0:
        raise ValueError(
            "Threshold artifact reports test use"
        )

    thresholds = np.asarray(
        [
            threshold_metadata[
                "thresholds"
            ][label]["threshold"]
            for label in SUPERCLASS_ORDER
        ],
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >= thresholds[None, :]
    ).astype(
        np.int64
    )

    losses = sample_hamming_error(
        targets,
        predictions,
    )

    confidence = mean_bernoulli_confidence(
        probabilities
    )

    l_data = pd.read_csv(
        L_PATH
    )

    q_data = pd.read_csv(
        Q_PATH
    )

    u_data = np.load(
        U_PATH,
        allow_pickle=False,
    )

    l_ids = l_data[
        "ecg_id"
    ].to_numpy(
        dtype=np.int64
    )

    q_ids = q_data[
        "ecg_id"
    ].to_numpy(
        dtype=np.int64
    )

    u_ids = np.asarray(
        u_data["ecg_ids"],
        dtype=np.int64,
    )

    for name, ids in (
        ("L", l_ids),
        ("Q_signal", q_ids),
        ("U", u_ids),
    ):
        if not np.array_equal(
            ids,
            ecg_ids,
        ):
            raise ValueError(
                f"{name} ECG-ID ordering mismatch"
            )

    if not np.array_equal(
        np.asarray(
            u_data["targets"],
            dtype=np.int64,
        ),
        targets,
    ):
        raise ValueError(
            "U targets differ from baseline targets"
        )

    L = l_data[
        "label_confidence"
    ].to_numpy(
        dtype=np.float64
    )

    Q = q_data[
        "signal_quality"
    ].to_numpy(
        dtype=np.float64
    )

    U = np.asarray(
        u_data[
            "mean_mutual_information"
        ],
        dtype=np.float64,
    )

    for name, values in (
        ("L", L),
        ("Q_signal", Q),
        ("U", U),
    ):
        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"{name} contains NaN/Inf"
            )

    L_bad = 1.0 - L
    Q_bad = 1.0 - Q
    U_bad = U

    L_pct = midrank_ecdf(
        L_bad,
        L_bad,
    )

    Q_pct = midrank_ecdf(
        Q_bad,
        Q_bad,
    )

    U_pct = midrank_ecdf(
        U_bad,
        U_bad,
    )

    LQ_bad = (
        L_pct
        + Q_pct
    ) / 2.0

    LU_bad = (
        L_pct
        + U_pct
    ) / 2.0

    QU_bad = (
        Q_pct
        + U_pct
    ) / 2.0

    LQU_bad = (
        L_pct
        + Q_pct
        + U_pct
    ) / 3.0

    methods = {
        "confidence":
            confidence,

        "L":
            L,

        "Q_signal":
            Q,

        "U":
            -U,

        "L+Q_signal":
            -LQ_bad,

        "L+U":
            -LU_bad,

        "Q_signal+U":
            -QU_bad,

        "L+Q_signal+U":
            -LQU_bad,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        {
            "ecg_id":
                ecg_ids,

            "hamming_error":
                losses,

            "ordinary_confidence":
                confidence,

            "L":
                L,

            "Q_signal":
                Q,

            "U":
                U,

            "L_badness_percentile":
                L_pct,

            "Q_signal_badness_percentile":
                Q_pct,

            "U_badness_percentile":
                U_pct,

            "LQ_badness":
                LQ_bad,

            "LU_badness":
                LU_bad,

            "QU_badness":
                QU_bad,

            "LQU_badness":
                LQU_bad,
        }
    ).to_csv(
        SCORES_PATH,
        index=False,
        float_format="%.17g",
    )

    np.savez_compressed(
        REFERENCE_PATH,

        L_badness_reference=
            np.sort(
                L_bad.astype(
                    np.float64
                )
            ),

        Q_signal_badness_reference=
            np.sort(
                Q_bad.astype(
                    np.float64
                )
            ),

        U_badness_reference=
            np.sort(
                U_bad.astype(
                    np.float64
                )
            ),
    )

    full_risk = float(
        np.mean(
            losses,
            dtype=np.float64,
        )
    )

    method_summary = {}
    curves = {}

    for name, reliability in methods.items():
        result = grouped_risk_coverage(
            reliability,
            losses,
        )

        risk_points = {}

        for requested in TARGET_COVERAGES:
            achieved, risk = (
                risk_at_or_above_coverage(
                    result["coverage"],
                    result["risk"],
                    requested,
                )
            )

            risk_points[
                f"{requested:.2f}"
            ] = {
                "requested_coverage":
                    float(requested),

                "achieved_coverage":
                    float(achieved),

                "risk":
                    float(risk),
            }

        aurc = float(
            result["aurc"]
        )

        method_summary[name] = {
            "aurc":
                aurc,

            "relative_aurc_reduction_vs_expected_random":
                float(
                    (
                        full_risk
                        - aurc
                    )
                    / full_risk
                )
                if full_risk > 0.0
                else 0.0,

            "risk_at_coverage":
                risk_points,

            "unique_reliability_levels":
                int(
                    len(
                        result[
                            "coverage"
                        ]
                    )
                ),
        }

        key = name.replace(
            "+",
            "_plus_",
        )

        curves[
            f"{key}_coverage"
        ] = result["coverage"]

        curves[
            f"{key}_risk"
        ] = result["risk"]

        curves[
            f"{key}_group_size"
        ] = result["group_size"]

    oracle = grouped_risk_coverage(
        -losses,
        losses,
    )

    np.savez_compressed(
        CURVES_PATH,
        **curves,
    )

    confidence_aurc = method_summary[
        "confidence"
    ]["aurc"]

    for name in method_summary:
        aurc = method_summary[
            name
        ]["aurc"]

        method_summary[
            name
        ][
            "relative_difference_vs_confidence"
        ] = float(
            (
                aurc
                - confidence_aurc
            )
            / confidence_aurc
        )

    summary = {
        "selective_prediction_id":
            config[
                "selective_prediction_id"
            ],

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "evaluation_split":
            "validation",

        "records":
            EXPECTED_RECORDS,

        "test_records_used":
            0,

        "full_coverage_hamming_risk":
            full_risk,

        "expected_random_ranking_aurc":
            full_risk,

        "oracle_hamming_ranking_aurc_diagnostic_only":
            float(
                oracle["aurc"]
            ),

        "Q_definition":
            "training-referenced waveform signal quality",

        "combination_definition":
            "equal mean of validation midrank badness percentiles",

        "methods":
            method_summary,

        "source_sha256": {
            "config":
                sha256_file(
                    CONFIG_PATH
                ),

            "baseline_predictions":
                sha256_file(
                    PREDICTIONS_PATH
                ),

            "thresholds":
                sha256_file(
                    THRESHOLDS_PATH
                ),

            "L":
                sha256_file(
                    L_PATH
                ),

            "Q_signal":
                sha256_file(
                    Q_PATH
                ),

            "U":
                sha256_file(
                    U_PATH
                ),

            "script":
                sha256_file(
                    Path(__file__).resolve()
                ),
        },

        "record_scores_sha256":
            sha256_file(
                SCORES_PATH
            ),

        "combination_reference_sha256":
            sha256_file(
                REFERENCE_PATH
            ),

        "risk_coverage_curves_sha256":
            sha256_file(
                CURVES_PATH
            ),

        "status":
            "FROZEN",
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print()
    print("METHOD RESULTS")

    for name, result in sorted(
        method_summary.items(),
        key=lambda item:
            item[1]["aurc"],
    ):
        print(
            f"{name:14s} "
            f"AURC={result['aurc']:.9f}  "
            f"vs_random="
            f"{result['relative_aurc_reduction_vs_expected_random']:.4f}  "
            f"vs_conf="
            f"{result['relative_difference_vs_confidence']:+.4f}"
        )

    print()
    print(
        f"FULL RISK     {full_risk:.9f}"
    )

    print(
        f"ORACLE AURC   "
        f"{oracle['aurc']:.9f}"
    )


if __name__ == "__main__":
    main()
