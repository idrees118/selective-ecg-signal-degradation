#!/usr/bin/env python3
"""Validation-only selective-prediction experiment.

Compares:
confidence, L, Q, U,
L+Q, L+U, Q+U, L+Q+U.

Fold 10 is never loaded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

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
    / "signal_quality"
    / "v1"
    / "validation"
    / "signal_quality.csv"
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
    / "selective_prediction_v1.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "selective_prediction"
    / "v1"
    / "validation"
)

SCORES_PATH = (
    OUTPUT_DIR
    / "record_scores.csv"
)

CURVES_PATH = (
    OUTPUT_DIR
    / "risk_coverage_curves.npz"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

REFERENCE_PATH = (
    OUTPUT_DIR
    / "combination_reference.npz"
)

EXPECTED_RECORDS = 2146

TARGET_COVERAGES = (
    0.50,
    0.70,
    0.80,
    0.90,
    1.00,
)


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "Config must be a mapping"
        )

    if config["status"] != "FROZEN":
        raise ValueError(
            "Selective-prediction config is not FROZEN"
        )

    if config["development_split"] != "validation":
        raise ValueError(
            "Development split changed"
        )

    if config["test_allowed"] is not False:
        raise ValueError(
            "Test use is forbidden"
        )

    return config


def main():
    config = load_config()

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
            "Unexpected baseline probability shape"
        )

    if targets.shape != (
        EXPECTED_RECORDS,
        5,
    ):
        raise ValueError(
            "Unexpected target shape"
        )

    class_names = tuple(
        str(value)
        for value
        in baseline[
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
            "Threshold metadata reports test usage"
        )

    thresholds = np.asarray(
        [
            threshold_metadata[
                "thresholds"
            ][label]["threshold"]
            for label
            in SUPERCLASS_ORDER
        ],
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >= thresholds[None, :]
    ).astype(
        np.int64
    )

    hamming_error = (
        sample_hamming_error(
            targets,
            predictions,
        )
    )

    ordinary_confidence = (
        mean_bernoulli_confidence(
            probabilities
        )
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

    for name, ids in (
        (
            "L",
            l_data[
                "ecg_id"
            ].to_numpy(
                dtype=np.int64
            ),
        ),
        (
            "Q",
            q_data[
                "ecg_id"
            ].to_numpy(
                dtype=np.int64
            ),
        ),
        (
            "U",
            np.asarray(
                u_data["ecg_ids"],
                dtype=np.int64,
            ),
        ),
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
            "U target matrix differs from baseline targets"
        )

    label_confidence = l_data[
        "label_confidence"
    ].to_numpy(
        dtype=np.float64
    )

    signal_quality = q_data[
        "signal_quality"
    ].to_numpy(
        dtype=np.float64
    )

    model_uncertainty = np.asarray(
        u_data[
            "mean_mutual_information"
        ],
        dtype=np.float64,
    )

    if not np.isfinite(
        label_confidence
    ).all():
        raise ValueError(
            "L contains NaN/Inf"
        )

    if not np.isfinite(
        signal_quality
    ).all():
        raise ValueError(
            "Q contains NaN/Inf"
        )

    if not np.isfinite(
        model_uncertainty
    ).all():
        raise ValueError(
            "U contains NaN/Inf"
        )

    l_badness = (
        1.0
        - label_confidence
    )

    q_badness = (
        1.0
        - signal_quality
    )

    u_badness = (
        model_uncertainty
    )

    l_pct = midrank_ecdf(
        l_badness,
        l_badness,
    )

    q_pct = midrank_ecdf(
        q_badness,
        q_badness,
    )

    u_pct = midrank_ecdf(
        u_badness,
        u_badness,
    )

    combined_lq_badness = (
        l_pct + q_pct
    ) / 2.0

    combined_lu_badness = (
        l_pct + u_pct
    ) / 2.0

    combined_qu_badness = (
        q_pct + u_pct
    ) / 2.0

    combined_lqu_badness = (
        l_pct
        + q_pct
        + u_pct
    ) / 3.0

    methods = {
        "confidence":
            ordinary_confidence,

        "L":
            label_confidence,

        "Q":
            signal_quality,

        "U":
            -model_uncertainty,

        "L+Q":
            -combined_lq_badness,

        "L+U":
            -combined_lu_badness,

        "Q+U":
            -combined_qu_badness,

        "L+Q+U":
            -combined_lqu_badness,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    record_table = pd.DataFrame(
        {
            "ecg_id":
                ecg_ids,

            "hamming_error":
                hamming_error,

            "ordinary_confidence":
                ordinary_confidence,

            "label_confidence_L":
                label_confidence,

            "signal_quality_Q":
                signal_quality,

            "model_uncertainty_U":
                model_uncertainty,

            "L_badness_percentile":
                l_pct,

            "Q_badness_percentile":
                q_pct,

            "U_badness_percentile":
                u_pct,

            "LQ_badness":
                combined_lq_badness,

            "LU_badness":
                combined_lu_badness,

            "QU_badness":
                combined_qu_badness,

            "LQU_badness":
                combined_lqu_badness,
        }
    )

    record_table.to_csv(
        SCORES_PATH,
        index=False,
        float_format="%.17g",
    )

    # Save the exact validation empirical distributions.
    # These are later used to map fold-10 L/Q/U values
    # without fitting anything on fold 10.
    np.savez_compressed(
        REFERENCE_PATH,
        L_badness_reference=
            np.sort(
                l_badness.astype(
                    np.float64
                )
            ),

        Q_badness_reference=
            np.sort(
                q_badness.astype(
                    np.float64
                )
            ),

        U_badness_reference=
            np.sort(
                u_badness.astype(
                    np.float64
                )
            ),
    )

    summary_methods = {}
    curve_payload = {}

    full_risk = float(
        np.mean(
            hamming_error,
            dtype=np.float64,
        )
    )

    for name, reliability in methods.items():
        result = grouped_risk_coverage(
            reliability,
            hamming_error,
        )

        coverage_results = {}

        for requested in TARGET_COVERAGES:
            achieved, risk = (
                risk_at_or_above_coverage(
                    result["coverage"],
                    result["risk"],
                    requested,
                )
            )

            coverage_results[
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

        summary_methods[name] = {
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

            "unique_reliability_levels":
                int(
                    len(
                        result[
                            "coverage"
                        ]
                    )
                ),

            "risk_at_coverage":
                coverage_results,
        }

        key = (
            name
            .replace("+", "_plus_")
        )

        curve_payload[
            f"{key}_coverage"
        ] = result[
            "coverage"
        ]

        curve_payload[
            f"{key}_risk"
        ] = result[
            "risk"
        ]

        curve_payload[
            f"{key}_group_size"
        ] = result[
            "group_size"
        ]

    # Expected AURC of a uniformly random ordering.
    # For every prefix size, expected retained risk equals
    # the full-dataset mean loss.
    random_expected_aurc = (
        full_risk
    )

    # Oracle is a diagnostic lower bound only.
    oracle = grouped_risk_coverage(
        -hamming_error,
        hamming_error,
    )

    np.savez_compressed(
        CURVES_PATH,
        **curve_payload,
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

        "risk_definition":
            "per-record Hamming error across five frozen binary decisions",

        "full_coverage_hamming_risk":
            full_risk,

        "expected_random_ranking_aurc":
            random_expected_aurc,

        "oracle_hamming_ranking_aurc_diagnostic_only":
            float(
                oracle["aurc"]
            ),

        "ordinary_confidence_definition":
            "mean_k max(p_k, 1-p_k)",

        "combination_definition":
            "equal mean of validation midrank badness percentiles",

        "tie_handling":
            "all exactly tied reliability scores added as one coverage group",

        "methods":
            summary_methods,

        "deployment_interpretation": {
            "confidence":
                "model-derived",

            "U":
                "model-derived",

            "Q":
                "metadata-assisted; not an automated physical SQI",

            "L":
                "retrospective annotation-aware",

            "L_containing_combinations":
                "retrospective analysis, not direct deployment-time selectors",
        },

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

            "Q":
                sha256_file(
                    Q_PATH
                ),

            "U":
                sha256_file(
                    U_PATH
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

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
