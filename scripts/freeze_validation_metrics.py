#!/usr/bin/env python3
"""Freeze seed-1 validation metrics and F1 thresholds.

Fold 9 only.
Fold 10 is never loaded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from ptbxl_reliability.cached_dataset import (
    CACHE_ROOT,
)
from ptbxl_reliability.integrity import (
    sha256_file,
)
from ptbxl_reliability.labels import (
    SUPERCLASS_ORDER,
)
from ptbxl_reliability.metrics import (
    classwise_auroc,
    macro_auroc,
)
from ptbxl_reliability.paths import (
    PROJECT_ROOT,
)
from ptbxl_reliability.validation_metrics import (
    classwise_average_precision,
    classwise_brier,
    macro_average_precision,
    macro_brier,
    select_classwise_f1_thresholds,
)


RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "v1"
    / "seed_20260826"
)

PREDICTIONS_PATH = (
    RUN_DIR
    / "best_validation_predictions.npz"
)

RUN_METADATA_PATH = (
    RUN_DIR
    / "run_metadata.json"
)

METRICS_OUTPUT = (
    RUN_DIR
    / "validation_metrics.json"
)

THRESHOLDS_OUTPUT = (
    RUN_DIR
    / "validation_thresholds.json"
)


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def main():
    run_metadata = load_json(
        RUN_METADATA_PATH
    )

    if run_metadata["status"] != "COMPLETE":
        raise ValueError(
            "Baseline run is not complete"
        )

    if run_metadata["seed"] != 20260826:
        raise ValueError(
            "Unexpected baseline seed"
        )

    if run_metadata[
        "test_records_used"
    ] != 0:
        raise ValueError(
            "Test-set leakage detected"
        )

    observed_prediction_hash = (
        sha256_file(
            PREDICTIONS_PATH
        )
    )

    if (
        observed_prediction_hash
        != run_metadata[
            "best_validation_predictions_sha256"
        ]
    ):
        raise ValueError(
            "Validation prediction SHA mismatch"
        )

    saved = np.load(
        PREDICTIONS_PATH,
        allow_pickle=False,
    )

    probabilities = np.asarray(
        saved["probabilities"],
        dtype=np.float64,
    )

    targets = np.asarray(
        saved["targets"],
        dtype=np.float64,
    )

    ecg_ids = np.asarray(
        saved["ecg_ids"],
        dtype=np.int64,
    )

    class_names = tuple(
        str(value)
        for value in saved[
            "class_names"
        ].tolist()
    )

    if class_names != SUPERCLASS_ORDER:
        raise ValueError(
            "Prediction class order mismatch"
        )

    if probabilities.shape != (
        2146,
        5,
    ):
        raise ValueError(
            "Unexpected probability shape"
        )

    if targets.shape != (
        2146,
        5,
    ):
        raise ValueError(
            "Unexpected target shape"
        )

    if ecg_ids.shape != (
        2146,
    ):
        raise ValueError(
            "Unexpected ECG-ID shape"
        )

    # Independent comparison with frozen validation cache.
    cached_targets = np.load(
        CACHE_ROOT
        / "validation"
        / "targets.npy",
        mmap_mode="r",
    )

    cached_ecg_ids = np.load(
        CACHE_ROOT
        / "validation"
        / "ecg_ids.npy",
        mmap_mode="r",
    )

    if not np.array_equal(
        targets.astype(
            np.float32
        ),
        cached_targets,
    ):
        raise ValueError(
            "Saved validation targets differ "
            "from frozen validation cache"
        )

    if not np.array_equal(
        ecg_ids,
        cached_ecg_ids,
    ):
        raise ValueError(
            "Saved validation ECG IDs differ "
            "from frozen validation cache"
        )

    auc = classwise_auroc(
        targets,
        probabilities,
    )

    macro_auc = macro_auroc(
        targets,
        probabilities,
    )

    if not np.isclose(
        macro_auc,
        run_metadata[
            "best_validation_macro_auroc"
        ],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Recomputed macro-AUROC does "
            "not reproduce run metadata"
        )

    ap = classwise_average_precision(
        targets,
        probabilities,
    )

    brier = classwise_brier(
        targets,
        probabilities,
    )

    threshold_results = (
        select_classwise_f1_thresholds(
            targets,
            probabilities,
        )
    )

    class_metrics = {}

    threshold_payload = {}

    for index, label in enumerate(
        SUPERCLASS_ORDER
    ):
        threshold = (
            threshold_results[index]
        )

        class_metrics[label] = {
            "auroc":
                float(auc[index]),
            "average_precision":
                float(ap[index]),
            "brier":
                float(brier[index]),
            "validation_f1":
                float(
                    threshold["f1"]
                ),
            "validation_precision":
                float(
                    threshold[
                        "precision"
                    ]
                ),
            "validation_recall":
                float(
                    threshold[
                        "recall"
                    ]
                ),
        }

        threshold_payload[label] = {
            "threshold":
                float(
                    threshold[
                        "threshold"
                    ]
                ),
            "selection_metric":
                "F1",
            "selection_split":
                "validation",
            "tie_policy":
                "highest_threshold",
            "tp":
                int(
                    threshold["tp"]
                ),
            "fp":
                int(
                    threshold["fp"]
                ),
            "fn":
                int(
                    threshold["fn"]
                ),
            "tn":
                int(
                    threshold["tn"]
                ),
        }

    macro_f1 = float(
        np.mean(
            [
                result["f1"]
                for result
                in threshold_results
            ],
            dtype=np.float64,
        )
    )

    metrics = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "experiment_id":
            run_metadata[
                "experiment_id"
            ],

        "seed":
            20260826,

        "checkpoint_epoch":
            int(
                run_metadata[
                    "best_epoch"
                ]
            ),

        "evaluation_split":
            "validation",

        "records":
            2146,

        "test_records_used":
            0,

        "macro_auroc":
            float(macro_auc),

        "macro_average_precision":
            macro_average_precision(
                targets,
                probabilities,
            ),

        "macro_brier":
            macro_brier(
                targets,
                probabilities,
            ),

        "macro_f1_at_validation_selected_thresholds":
            macro_f1,

        "class_metrics":
            class_metrics,

        "prediction_sha256":
            observed_prediction_hash,

        "status":
            "FROZEN",
    }

    thresholds = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "seed":
            20260826,

        "fit_split":
            "validation",

        "records_used":
            2146,

        "test_records_used":
            0,

        "decision_rule":
            "predict positive iff probability >= threshold",

        "optimization":
            "independent per-class maximum F1",

        "tie_policy":
            "highest threshold among equal maximum-F1 thresholds",

        "thresholds":
            threshold_payload,

        "source_prediction_sha256":
            observed_prediction_hash,

        "status":
            "FROZEN",
    }

    with METRICS_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    with THRESHOLDS_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            thresholds,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("VALIDATION THRESHOLDS")
    print(
        json.dumps(
            thresholds,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
