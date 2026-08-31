#!/usr/bin/env python3
"""Frozen 5,000-replicate paired patient-level AURC bootstrap."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ptbxl_reliability.bootstrap import (
    patient_record_weights,
    weighted_grouped_aurc,
)
from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.selective_prediction import (
    grouped_risk_coverage,
    midrank_ecdf,
)


FINAL_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "final_evaluation_v1.yaml"
)

FINAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_evaluation"
    / "v1"
    / "test"
)

FINAL_SUMMARY = (
    FINAL_DIR
    / "summary.json"
)

CONDITION_TABLE = (
    FINAL_DIR
    / "condition_metrics.csv"
)

L_TEST = (
    FINAL_DIR
    / "test_annotation_confidence.csv"
)

COMBINATION_REFERENCE = (
    PROJECT_ROOT
    / "results"
    / "selective_prediction_waveform_q"
    / "v1"
    / "validation"
    / "combination_reference.npz"
)

OUT = (
    PROJECT_ROOT
    / "results"
    / "bootstrap"
    / "v1"
    / "final_test"
)

CHECKPOINT = (
    OUT
    / "bootstrap_checkpoint.npz"
)

REPLICATES_PATH = (
    OUT
    / "primary_aurc_bootstrap_replicates.npz"
)

TABLE_PATH = (
    OUT
    / "primary_aurc_bootstrap_summary.csv"
)

SUMMARY_PATH = (
    OUT
    / "summary.json"
)

EXPECTED_RECORDS = 2158
EXPECTED_PATIENTS = 1877
EXPECTED_REPLICATES = 5000
EXPECTED_SEED = 20260826
SAVE_EVERY = 250

COMPARISONS = (
    (
        "L+U_vs_confidence",
        "L+U",
        "confidence",
    ),
    (
        "U_vs_confidence",
        "U",
        "confidence",
    ),
    (
        "L+Q_signal+U_vs_confidence",
        "L+Q_signal+U",
        "confidence",
    ),
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = json.load(handle)

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"{path} must contain an object"
        )

    return value


def selectors(
    *,
    confidence: np.ndarray,
    L: np.ndarray,
    Q: np.ndarray,
    U: np.ndarray,
    reference: np.lib.npyio.NpzFile,
) -> dict[str, np.ndarray]:
    L_pct = midrank_ecdf(
        np.asarray(
            reference[
                "L_badness_reference"
            ],
            dtype=np.float64,
        ),
        1.0 - L,
    )

    Q_pct = midrank_ecdf(
        np.asarray(
            reference[
                "Q_signal_badness_reference"
            ],
            dtype=np.float64,
        ),
        1.0 - Q,
    )

    U_pct = midrank_ecdf(
        np.asarray(
            reference[
                "U_badness_reference"
            ],
            dtype=np.float64,
        ),
        U,
    )

    return {
        "confidence":
            confidence,

        "U":
            -U,

        "L+U":
            -(L_pct + U_pct) / 2.0,

        "L+Q_signal+U":
            -(L_pct + Q_pct + U_pct) / 3.0,
    }


def main() -> None:
    with FINAL_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if config["status"] != "FROZEN":
        raise ValueError(
            "Final evaluation config is not FROZEN"
        )

    bootstrap_config = config[
        "bootstrap"
    ]

    if bootstrap_config[
        "unit"
    ] != "patient":
        raise ValueError(
            "Bootstrap unit must be patient"
        )

    if bootstrap_config[
        "paired"
    ] is not True:
        raise ValueError(
            "Bootstrap must be paired"
        )

    if int(
        bootstrap_config[
            "replicates"
        ]
    ) != EXPECTED_REPLICATES:
        raise ValueError(
            "Bootstrap replicate count changed"
        )

    if int(
        bootstrap_config[
            "random_seed"
        ]
    ) != EXPECTED_SEED:
        raise ValueError(
            "Bootstrap seed changed"
        )

    if bootstrap_config[
        "confidence_interval"
    ] != "percentile_95":
        raise ValueError(
            "Frozen CI method changed"
        )

    configured_comparisons = tuple(
        config[
            "primary_auc_comparisons"
        ]
    )

    expected_comparisons = tuple(
        value[0]
        for value in COMPARISONS
    )

    if configured_comparisons != expected_comparisons:
        raise ValueError(
            "Primary comparison list changed"
        )

    final_summary = load_json(
        FINAL_SUMMARY
    )

    if final_summary[
        "status"
    ] != "FROZEN":
        raise ValueError(
            "Final fold-10 results are not FROZEN"
        )

    if int(
        final_summary[
            "records"
        ]
    ) != EXPECTED_RECORDS:
        raise ValueError(
            "Unexpected final-test record count"
        )

    if int(
        final_summary[
            "patients"
        ]
    ) != EXPECTED_PATIENTS:
        raise ValueError(
            "Unexpected final-test patient count"
        )

    if final_summary[
        "post_test_tuning_allowed"
    ] is not False:
        raise ValueError(
            "Post-test tuning unexpectedly allowed"
        )

    condition_table = pd.read_csv(
        CONDITION_TABLE
    )

    condition_names = tuple(
        condition_table[
            "condition"
        ].astype(
            str
        ).tolist()
    )

    if len(
        condition_names
    ) != 13:
        raise ValueError(
            "Expected clean + 12 conditions"
        )

    if condition_names[
        0
    ] != "clean":
        raise ValueError(
            "First final condition must be clean"
        )

    # Preserve the exact IEEE-754 values written with %.17g.
    # Exact ECDF tie membership depends on round-trip parsing.
    l_data = pd.read_csv(
        L_TEST,
        float_precision="round_trip",
    )

    if len(
        l_data
    ) != EXPECTED_RECORDS:
        raise ValueError(
            "Unexpected test L count"
        )

    L = l_data[
        "label_confidence"
    ].to_numpy(
        dtype=np.float64
    )

    L_ids = l_data[
        "ecg_id"
    ].to_numpy(
        dtype=np.int64
    )

    reference = np.load(
        COMBINATION_REFERENCE,
        allow_pickle=False,
    )

    condition_data = []

    patient_ids_reference = None
    ecg_ids_reference = None

    print(
        "Loading and verifying 13 frozen "
        "final-test artifacts..."
    )

    for condition_name in condition_names:
        artifact = (
            FINAL_DIR
            / condition_name
            / "results.npz"
        )

        data = np.load(
            artifact,
            allow_pickle=False,
        )

        ecg_ids = np.asarray(
            data[
                "ecg_ids"
            ],
            dtype=np.int64,
        )

        patient_ids = np.asarray(
            data[
                "patient_ids"
            ],
            dtype=np.int64,
        )

        losses = np.asarray(
            data[
                "hamming_error"
            ],
            dtype=np.float64,
        )

        confidence = np.asarray(
            data[
                "confidence"
            ],
            dtype=np.float64,
        )

        Q = np.asarray(
            data[
                "Q_signal"
            ],
            dtype=np.float64,
        )

        U = np.asarray(
            data[
                "U"
            ],
            dtype=np.float64,
        )

        if ecg_ids.shape != (
            EXPECTED_RECORDS,
        ):
            raise ValueError(
                f"{condition_name}: bad ECG-ID shape"
            )

        if patient_ids.shape != (
            EXPECTED_RECORDS,
        ):
            raise ValueError(
                f"{condition_name}: bad patient-ID shape"
            )

        if losses.shape != (
            EXPECTED_RECORDS,
        ):
            raise ValueError(
                f"{condition_name}: bad loss shape"
            )

        if not np.array_equal(
            ecg_ids,
            L_ids,
        ):
            raise ValueError(
                f"{condition_name}: L ECG ordering mismatch"
            )

        if patient_ids_reference is None:
            patient_ids_reference = (
                patient_ids.copy()
            )
            ecg_ids_reference = (
                ecg_ids.copy()
            )
        else:
            if not np.array_equal(
                patient_ids,
                patient_ids_reference,
            ):
                raise ValueError(
                    f"{condition_name}: patient ordering mismatch"
                )

            if not np.array_equal(
                ecg_ids,
                ecg_ids_reference,
            ):
                raise ValueError(
                    f"{condition_name}: ECG ordering mismatch"
                )

        score = selectors(
            confidence=confidence,
            L=L,
            Q=Q,
            U=U,
            reference=reference,
        )

        # Verify point AURCs exactly reproduce the frozen final summary.
        for method in score:
            observed = float(
                grouped_risk_coverage(
                    score[
                        method
                    ],
                    losses,
                )[
                    "aurc"
                ]
            )

            expected = float(
                final_summary[
                    "condition_results"
                ][
                    condition_name
                ][
                    "selective_prediction"
                ][
                    method
                ][
                    "aurc"
                ]
            )

            if not np.isclose(
                observed,
                expected,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{condition_name}/{method}: "
                    "point AURC does not reproduce "
                    f"({observed} vs {expected})"
                )

        condition_data.append({
            "name":
                condition_name,

            "losses":
                losses,

            "scores":
                score,
        })

        print(
            f"  {condition_name}: PASS"
        )

    if np.unique(
        patient_ids_reference
    ).size != EXPECTED_PATIENTS:
        raise ValueError(
            "Unexpected number of unique patients"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    shape = (
        EXPECTED_REPLICATES,
        len(
            condition_names
        ),
        len(
            COMPARISONS
        ),
    )

    completed = 0

    deltas = np.full(
        shape,
        np.nan,
        dtype=np.float64,
    )

    if CHECKPOINT.exists():
        saved = np.load(
            CHECKPOINT,
            allow_pickle=False,
        )

        saved_conditions = tuple(
            str(x)
            for x in saved[
                "condition_names"
            ].tolist()
        )

        saved_comparisons = tuple(
            str(x)
            for x in saved[
                "comparison_names"
            ].tolist()
        )

        if saved_conditions != condition_names:
            raise ValueError(
                "Bootstrap checkpoint condition list mismatch"
            )

        if saved_comparisons != expected_comparisons:
            raise ValueError(
                "Bootstrap checkpoint comparison list mismatch"
            )

        if int(
            saved[
                "seed"
            ]
        ) != EXPECTED_SEED:
            raise ValueError(
                "Bootstrap checkpoint seed mismatch"
            )

        completed = int(
            saved[
                "completed"
            ]
        )

        saved_deltas = np.asarray(
            saved[
                "deltas"
            ],
            dtype=np.float64,
        )

        if saved_deltas.shape != shape:
            raise ValueError(
                "Bootstrap checkpoint shape mismatch"
            )

        deltas = saved_deltas

        print(
            f"Resuming from replicate "
            f"{completed}/{EXPECTED_REPLICATES}"
        )

    print(
        "\nRunning paired patient-level bootstrap..."
    )

    for replicate in range(
        completed,
        EXPECTED_REPLICATES,
    ):
        weights = patient_record_weights(
            patient_ids_reference,
            seed=EXPECTED_SEED,
            replicate=replicate,
        )

        for condition_index, data in enumerate(
            condition_data
        ):
            losses = data[
                "losses"
            ]

            scores = data[
                "scores"
            ]

            # Compute confidence once, since all three comparisons
            # use the exact same paired bootstrap sample.
            confidence_aurc = weighted_grouped_aurc(
                scores[
                    "confidence"
                ],
                losses,
                weights,
            )

            for comparison_index, (
                _comparison_name,
                method,
                baseline_method,
            ) in enumerate(
                COMPARISONS
            ):
                if baseline_method != "confidence":
                    raise AssertionError(
                        "Frozen primary baseline changed"
                    )

                method_aurc = weighted_grouped_aurc(
                    scores[
                        method
                    ],
                    losses,
                    weights,
                )

                deltas[
                    replicate,
                    condition_index,
                    comparison_index,
                ] = (
                    method_aurc
                    - confidence_aurc
                )

        done = (
            replicate + 1
        )

        if (
            done % SAVE_EVERY == 0
            or done == EXPECTED_REPLICATES
        ):
            np.savez_compressed(
                CHECKPOINT,
                completed=np.asarray(
                    done,
                    dtype=np.int64,
                ),
                seed=np.asarray(
                    EXPECTED_SEED,
                    dtype=np.int64,
                ),
                condition_names=np.asarray(
                    condition_names,
                    dtype="U64",
                ),
                comparison_names=np.asarray(
                    expected_comparisons,
                    dtype="U64",
                ),
                deltas=deltas,
            )

            print(
                f"  bootstrap "
                f"{done}/{EXPECTED_REPLICATES}"
            )

    if not np.isfinite(
        deltas
    ).all():
        raise ValueError(
            "Bootstrap output contains NaN/Inf"
        )

    np.savez_compressed(
        REPLICATES_PATH,
        seed=np.asarray(
            EXPECTED_SEED,
            dtype=np.int64,
        ),
        condition_names=np.asarray(
            condition_names,
            dtype="U64",
        ),
        comparison_names=np.asarray(
            expected_comparisons,
            dtype="U64",
        ),
        deltas=deltas,
    )

    rows = []
    nested = {}

    for condition_index, condition_name in enumerate(
        condition_names
    ):
        nested[
            condition_name
        ] = {}

        for comparison_index, (
            comparison_name,
            method,
            baseline_method,
        ) in enumerate(
            COMPARISONS
        ):
            values = deltas[
                :,
                condition_index,
                comparison_index,
            ]

            point_method = float(
                final_summary[
                    "condition_results"
                ][
                    condition_name
                ][
                    "selective_prediction"
                ][
                    method
                ][
                    "aurc"
                ]
            )

            point_baseline = float(
                final_summary[
                    "condition_results"
                ][
                    condition_name
                ][
                    "selective_prediction"
                ][
                    baseline_method
                ][
                    "aurc"
                ]
            )

            point_delta = (
                point_method
                - point_baseline
            )

            ci_lower = float(
                np.percentile(
                    values,
                    2.5,
                )
            )

            ci_upper = float(
                np.percentile(
                    values,
                    97.5,
                )
            )

            conclusion = (
                "method_lower_aurc"
                if ci_upper < 0.0
                else (
                    "method_higher_aurc"
                    if ci_lower > 0.0
                    else "inconclusive_ci_crosses_zero"
                )
            )

            result = {
                "method":
                    method,

                "baseline":
                    baseline_method,

                "delta_definition":
                    "method_aurc_minus_confidence_aurc",

                "point_delta":
                    float(
                        point_delta
                    ),

                "bootstrap_mean_delta":
                    float(
                        np.mean(
                            values,
                            dtype=np.float64,
                        )
                    ),

                "ci_95_percentile":
                    [
                        ci_lower,
                        ci_upper,
                    ],

                "conclusion":
                    conclusion,
            }

            nested[
                condition_name
            ][
                comparison_name
            ] = result

            rows.append({
                "condition":
                    condition_name,

                "comparison":
                    comparison_name,

                **result,
            })

    output_table = pd.DataFrame(
        rows
    )

    output_table.to_csv(
        TABLE_PATH,
        index=False,
        float_format="%.17g",
    )

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "bootstrap_id":
            "ptbxl_final_patient_bootstrap_v1",

        "evaluation_split":
            "test_fold_10",

        "unit":
            "patient",

        "paired":
            True,

        "unique_patients":
            EXPECTED_PATIENTS,

        "records":
            EXPECTED_RECORDS,

        "replicates":
            EXPECTED_REPLICATES,

        "random_seed":
            EXPECTED_SEED,

        "ci_method":
            "percentile_95",

        "delta_definition":
            (
                "method AURC minus ordinary-confidence AURC; "
                "negative favors method"
            ),

        "comparisons":
            nested,

        "source_sha256": {
            "final_config":
                sha256_file(
                    FINAL_CONFIG
                ),

            "final_summary":
                sha256_file(
                    FINAL_SUMMARY
                ),

            "condition_table":
                sha256_file(
                    CONDITION_TABLE
                ),

            "test_L":
                sha256_file(
                    L_TEST
                ),

            "validation_combination_reference":
                sha256_file(
                    COMBINATION_REFERENCE
                ),

            "script":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),
        },

        "replicates_sha256":
            sha256_file(
                REPLICATES_PATH
            ),

        "summary_table_sha256":
            sha256_file(
                TABLE_PATH
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
        handle.write(
            "\n"
        )

    # Successful completion: checkpoint is no longer needed.
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    print(
        "\nPATIENT-LEVEL BOOTSTRAP COMPLETE"
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )

    print(
        f"Table:   {TABLE_PATH}"
    )


if __name__ == "__main__":
    main()
