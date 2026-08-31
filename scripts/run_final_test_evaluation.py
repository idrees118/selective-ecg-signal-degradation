#!/usr/bin/env python3
"""Final frozen fold-10 evaluation for PTB-XL reliability study.

Modes
-----
--preflight
    Validate all schemas/artifacts and execute tiny inference/corruption checks.
    Does not create final scientific result artifacts.

(default)
    Evaluate clean fold 10 plus the exact 12 frozen corruption conditions.
    Completed condition artifacts are reused on rerun.
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
import yaml
from sklearn.metrics import average_precision_score, f1_score
from torch import nn

from ptbxl_reliability.corruptions import (
    add_baseline_wander,
    add_white_noise,
    amplitude_clip,
    mask_leads,
)
from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.metrics import classwise_auroc
from ptbxl_reliability.model import ResNet1DWang
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.selective_prediction import (
    grouped_risk_coverage,
    mean_bernoulli_confidence,
    midrank_ecdf,
    sample_hamming_error,
)
from ptbxl_reliability.uncertainty import (
    enable_mc_dropout,
    mc_uncertainty,
)
from ptbxl_reliability.waveform_signal_quality import (
    waveform_badness_features,
)


# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ptb-xl"
    / "1.0.3"
)

DATABASE_PATH = (
    RAW_ROOT
    / "ptbxl_database.csv"
)

SCP_PATH = (
    RAW_ROOT
    / "scp_statements.csv"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cohort"
    / "v1"
    / "ptbxl_superdiagnostic_cohort.csv"
)

NORM_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "v1"
    / "global_zscore_stats.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "v1"
    / "seed_20260826"
    / "best_checkpoint.pt"
)

THRESHOLDS_PATH = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "v1"
    / "seed_20260826"
    / "validation_thresholds.json"
)

L_VALIDATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "label_confidence"
    / "v1"
    / "validation"
    / "label_confidence.csv"
)

Q_TRAIN_REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "waveform_signal_quality"
    / "v1"
    / "training_reference.npz"
)

COMBINATION_REFERENCE_PATH = (
    PROJECT_ROOT
    / "results"
    / "selective_prediction_waveform_q"
    / "v1"
    / "validation"
    / "combination_reference.npz"
)

CORRUPTION_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "corruption_v1.yaml"
)

FINAL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "final_evaluation_v1.yaml"
)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_evaluation"
    / "v1"
    / "test"
)

SUMMARY_PATH = (
    OUT_DIR
    / "summary.json"
)

CONDITION_TABLE_PATH = (
    OUT_DIR
    / "condition_metrics.csv"
)

L_TEST_PATH = (
    OUT_DIR
    / "test_annotation_confidence.csv"
)

L_TEST_METADATA_PATH = (
    OUT_DIR
    / "test_annotation_confidence_metadata.json"
)


# ---------------------------------------------------------------------------
# Fixed protocol
# ---------------------------------------------------------------------------

EXPECTED_RECORDS = 2158
EXPECTED_PATIENTS = 1877
N_CLASSES = 5
BATCH_SIZE = 128
MC_PASSES = 30

EXPECTED_LEADS = [
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

Q_FEATURES = (
    "baseline_burden",
    "high_frequency_burden",
    "lead_dropout_burden",
    "clipping_burden",
)

METHODS = (
    "confidence",
    "U",
    "L",
    "Q_signal",
    "L+U",
    "L+Q_signal",
    "Q_signal+U",
    "L+Q_signal+U",
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
            f"{path} must contain a JSON object"
        )

    return value


def parse_scp_codes(value: str) -> dict[str, float]:
    parsed = ast.literal_eval(
        str(value)
    )

    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError(
            "scp_codes is not a dictionary"
        )

    result = {}

    for code, likelihood in parsed.items():
        number = float(
            likelihood
        )

        if (
            not np.isfinite(
                number
            )
            or number < 0.0
            or number > 100.0
        ):
            raise ValueError(
                f"Invalid SCP likelihood: "
                f"{code}={likelihood}"
            )

        result[
            str(code)
        ] = number

    return result


def diagnostic_code_map() -> dict[str, str]:
    statements = pd.read_csv(
        SCP_PATH,
        index_col=0,
    )

    required = {
        "diagnostic",
        "diagnostic_class",
    }

    missing = (
        required
        - set(
            statements.columns
        )
    )

    if missing:
        raise ValueError(
            "scp_statements.csv missing "
            f"columns {sorted(missing)}"
        )

    mapping = {}

    for code, row in statements.iterrows():
        diagnostic = row[
            "diagnostic"
        ]

        if (
            pd.isna(
                diagnostic
            )
            or int(
                diagnostic
            ) != 1
        ):
            continue

        diagnostic_class = row[
            "diagnostic_class"
        ]

        if pd.isna(
            diagnostic_class
        ):
            continue

        diagnostic_class = str(
            diagnostic_class
        )

        if diagnostic_class in SUPERCLASS_ORDER:
            mapping[
                str(code)
            ] = diagnostic_class

    if not mapping:
        raise ValueError(
            "No diagnostic superclass code map built"
        )

    return mapping


def build_test_annotation_confidence(
    test_manifest: pd.DataFrame,
    *,
    write_artifact: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Reproduce frozen retrospective annotation-confidence definition.

    For each positive superclass:
      - consider its positive diagnostic SCP codes,
      - likelihood==0 is unknown/unavailable and is NOT zero confidence,
      - use the maximum known positive likelihood within the superclass.
    Record L is the mean over positive superclasses with known confidence.
    If no positive superclass has known likelihood, use the frozen
    validation-derived median imputation value.
    """

    database = pd.read_csv(
        DATABASE_PATH,
        index_col="ecg_id",
    )

    if "scp_codes" not in database.columns:
        raise ValueError(
            "ptbxl_database.csv missing scp_codes"
        )

    code_map = diagnostic_code_map()

    validation_l = pd.read_csv(
        L_VALIDATION_PATH
    )

    if "label_confidence" not in validation_l.columns:
        raise ValueError(
            "Validation L artifact missing "
            "label_confidence"
        )

    validation_values = validation_l[
        "label_confidence"
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        validation_values
    ).all():
        raise ValueError(
            "Validation L contains NaN/Inf"
        )

    imputation_value = float(
        np.median(
            validation_values
        )
    )

    values = np.empty(
        len(
            test_manifest
        ),
        dtype=np.float64,
    )

    coverage = np.empty(
        len(
            test_manifest
        ),
        dtype=np.float64,
    )

    rows = []

    reconstructed_targets = np.zeros(
        (
            len(
                test_manifest
            ),
            N_CLASSES,
        ),
        dtype=np.int64,
    )

    for index, row in enumerate(
        test_manifest.itertuples(
            index=False
        )
    ):
        ecg_id = int(
            row.ecg_id
        )

        if ecg_id not in database.index:
            raise ValueError(
                f"ECG {ecg_id} missing from database"
            )

        codes = parse_scp_codes(
            database.at[
                ecg_id,
                "scp_codes",
            ]
        )

        positive_classes = []

        for class_index, label in enumerate(
            SUPERCLASS_ORDER
        ):
            present = any(
                code_map.get(
                    code
                ) == label
                for code in codes
            )

            reconstructed_targets[
                index,
                class_index,
            ] = int(
                present
            )

            if present:
                positive_classes.append(
                    label
                )

        known_class_confidences = []

        for label in positive_classes:
            candidates = [
                likelihood / 100.0
                for code, likelihood
                in codes.items()
                if (
                    code_map.get(
                        code
                    ) == label
                    and likelihood > 0.0
                )
            ]

            if candidates:
                known_class_confidences.append(
                    max(
                        candidates
                    )
                )

        if not positive_classes:
            raise ValueError(
                f"ECG {ecg_id} has no positive "
                "superclass in selected cohort"
            )

        if known_class_confidences:
            label_confidence = float(
                np.mean(
                    np.asarray(
                        known_class_confidences,
                        dtype=np.float64,
                    )
                )
            )

            was_imputed = False
        else:
            label_confidence = (
                imputation_value
            )

            was_imputed = True

        class_coverage = float(
            len(
                known_class_confidences
            )
            / len(
                positive_classes
            )
        )

        values[
            index
        ] = label_confidence

        coverage[
            index
        ] = class_coverage

        rows.append({
            "ecg_id":
                ecg_id,

            "patient_id":
                int(
                    row.patient_id
                ),

            "split":
                "test",

            "label_confidence":
                label_confidence,

            "ambiguity":
                float(
                    1.0
                    - label_confidence
                ),

            "known_positive_superclass_fraction":
                class_coverage,

            "imputed":
                bool(
                    was_imputed
                ),
        })

    expected_targets = test_manifest[
        list(
            SUPERCLASS_ORDER
        )
    ].to_numpy(
        dtype=np.int64
    )

    if not np.array_equal(
        reconstructed_targets,
        expected_targets,
    ):
        mismatch = np.argwhere(
            reconstructed_targets
            != expected_targets
        )[0]

        raise ValueError(
            "Test target reconstruction mismatch "
            f"at row={int(mismatch[0])}, "
            f"class={SUPERCLASS_ORDER[int(mismatch[1])]}"
        )

    if (
        not np.isfinite(
            values
        ).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
    ):
        raise ValueError(
            "Invalid test L values"
        )

    if write_artifact:
        OUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = pd.DataFrame(
            rows
        )

        output.to_csv(
            L_TEST_PATH,
            index=False,
            float_format="%.17g",
        )

        metadata = {
            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "evaluation_split":
                "test",

            "records":
                int(
                    len(
                        output
                    )
                ),

            "definition":
                (
                    "mean strongest known positive "
                    "diagnostic SCP likelihood per "
                    "positive superclass"
                ),

            "zero_likelihood_handling":
                "unknown_not_zero_confidence",

            "no_known_likelihood_imputation":
                "validation_median",

            "validation_median_imputation_value":
                imputation_value,

            "imputed_records":
                int(
                    output[
                        "imputed"
                    ].sum()
                ),

            "target_reconstruction_pass":
                True,

            "retrospective_annotation_aware":
                True,

            "output_sha256":
                sha256_file(
                    L_TEST_PATH
                ),

            "source_sha256": {
                "ptbxl_database":
                    sha256_file(
                        DATABASE_PATH
                    ),

                "scp_statements":
                    sha256_file(
                        SCP_PATH
                    ),

                "validation_L":
                    sha256_file(
                        L_VALIDATION_PATH
                    ),
            },

            "status":
                "FROZEN",
        }

        with L_TEST_METADATA_PATH.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                metadata,
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write(
                "\n"
            )

    return (
        values,
        coverage,
    )


def load_frozen_protocol() -> tuple[
    dict,
    dict,
]:
    with FINAL_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        final_config = yaml.safe_load(
            handle
        )

    with CORRUPTION_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        corruption_config = yaml.safe_load(
            handle
        )

    if final_config[
        "status"
    ] != "FROZEN":
        raise ValueError(
            "Final evaluation config is not FROZEN"
        )

    if final_config[
        "development_closed"
    ] is not True:
        raise ValueError(
            "Development must be closed"
        )

    if final_config[
        "post_test_tuning_allowed"
    ] is not False:
        raise ValueError(
            "Post-test tuning must be forbidden"
        )

    if int(
        final_config[
            "test_split"
        ][
            "strat_fold"
        ]
    ) != 10:
        raise ValueError(
            "Final split must be fold 10"
        )

    if int(
        final_config[
            "test_split"
        ][
            "records_expected"
        ]
    ) != EXPECTED_RECORDS:
        raise ValueError(
            "Final expected record count changed"
        )

    if int(
        final_config[
            "test_split"
        ][
            "patients_expected"
        ]
    ) != EXPECTED_PATIENTS:
        raise ValueError(
            "Final expected patient count changed"
        )

    if int(
        final_config[
            "uncertainty"
        ][
            "mc_passes"
        ]
    ) != MC_PASSES:
        raise ValueError(
            "MC pass count changed"
        )

    if list(
        final_config[
            "selectors"
        ]
    ) != list(
        METHODS
    ):
        raise ValueError(
            "Frozen selector list/order changed"
        )

    if final_config[
        "combination_scaling"
    ][
        "test_set_ranking_used"
    ] is not False:
        raise ValueError(
            "Test-set rank fitting is forbidden"
        )

    forbidden_rules = final_config[
        "test_set_rules"
    ]

    for key, value in forbidden_rules.items():
        if value is not False:
            raise ValueError(
                f"Forbidden test rule enabled: {key}"
            )

    if corruption_config[
        "status"
    ] != "FROZEN":
        raise ValueError(
            "Corruption config is not FROZEN"
        )

    # The corruption config itself was frozen during validation
    # and deliberately still says test_allowed:false. The final
    # evaluation config authorizes reusing that exact protocol.
    if corruption_config[
        "development"
    ][
        "test_allowed"
    ] is not False:
        raise ValueError(
            "Corruption development config was altered"
        )

    if final_config[
        "corruptions"
    ][
        "use_exact_frozen_12_conditions"
    ] is not True:
        raise ValueError(
            "Exact frozen corruption conditions required"
        )

    return (
        final_config,
        corruption_config,
    )


def load_normalization() -> tuple[
    float,
    float,
]:
    norm = load_json(
        NORM_PATH
    )

    if (
        "mean_mV"
        not in norm
        or "std_mV"
        not in norm
    ):
        raise ValueError(
            "Frozen normalization artifact must "
            "contain mean_mV/std_mV"
        )

    mean = float(
        norm[
            "mean_mV"
        ]
    )

    std = float(
        norm[
            "std_mV"
        ]
    )

    expected_mean = (
        -0.0008252533901116082
    )

    expected_std = (
        0.23222258117564865
    )

    if not np.isclose(
        mean,
        expected_mean,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "Frozen normalization mean mismatch"
        )

    if not np.isclose(
        std,
        expected_std,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "Frozen normalization std mismatch"
        )

    return (
        mean,
        std,
    )


def load_test_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required = {
        "ecg_id",
        "patient_id",
        "strat_fold",
        "split",
        "filename_lr",
        *SUPERCLASS_ORDER,
    }

    missing = (
        required
        - set(
            manifest.columns
        )
    )

    if missing:
        raise ValueError(
            f"Manifest missing columns: "
            f"{sorted(missing)}"
        )

    test = (
        manifest.loc[
            manifest[
                "split"
            ].eq(
                "test"
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(
        test
    ) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} test ECGs, "
            f"found {len(test)}"
        )

    if test[
        "patient_id"
    ].nunique() != EXPECTED_PATIENTS:
        raise ValueError(
            "Unexpected test patient count"
        )

    if not test[
        "strat_fold"
    ].eq(
        10
    ).all():
        raise ValueError(
            "Test contains non-fold-10 record"
        )

    if test[
        "ecg_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate test ECG ID"
        )

    targets = test[
        list(
            SUPERCLASS_ORDER
        )
    ].to_numpy(
        dtype=np.int64
    )

    if not np.isin(
        targets,
        [
            0,
            1,
        ],
    ).all():
        raise ValueError(
            "Non-binary targets"
        )

    if (
        targets.sum(
            axis=1
        )
        < 1
    ).any():
        raise ValueError(
            "Zero-superclass record in selected test cohort"
        )

    return test


def load_raw_test(
    test_manifest: pd.DataFrame,
) -> np.ndarray:
    raw = np.empty(
        (
            EXPECTED_RECORDS,
            12,
            1000,
        ),
        dtype=np.float32,
    )

    print(
        "Loading raw fold-10 ECGs once..."
    )

    for index, row in enumerate(
        test_manifest.itertuples(
            index=False
        )
    ):
        signal, fields = wfdb.rdsamp(
            str(
                RAW_ROOT
                / str(
                    row.filename_lr
                )
            )
        )

        if signal.shape != (
            1000,
            12,
        ):
            raise ValueError(
                f"ECG {row.ecg_id}: "
                f"unexpected shape {signal.shape}"
            )

        if float(
            fields[
                "fs"
            ]
        ) != 100.0:
            raise ValueError(
                f"ECG {row.ecg_id}: fs mismatch"
            )

        if list(
            fields[
                "sig_name"
            ]
        ) != EXPECTED_LEADS:
            raise ValueError(
                f"ECG {row.ecg_id}: "
                "lead-order mismatch"
            )

        if not np.isfinite(
            signal
        ).all():
            raise ValueError(
                f"ECG {row.ecg_id}: "
                "non-finite waveform"
            )

        raw[
            index
        ] = signal.T.astype(
            np.float32
        )

        if (
            (
                index + 1
            )
            % 500
            == 0
            or (
                index + 1
            )
            == EXPECTED_RECORDS
        ):
            print(
                f"  raw "
                f"{index + 1}/"
                f"{EXPECTED_RECORDS}"
            )

    return raw


def load_model(
    device: torch.device,
) -> ResNet1DWang:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "model_state_dict",
        "epoch",
    }

    missing = (
        required
        - set(
            checkpoint
        )
    )

    if missing:
        raise ValueError(
            f"Checkpoint missing: "
            f"{sorted(missing)}"
        )

    if int(
        checkpoint[
            "epoch"
        ]
    ) != 26:
        raise ValueError(
            "Unexpected frozen best epoch"
        )

    model = ResNet1DWang()

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    return model


def state_copy(
    model: nn.Module,
) -> dict[
    str,
    torch.Tensor,
]:
    return {
        name:
            value.detach().cpu().clone()
        for name, value
        in model.state_dict().items()
    }


def assert_same_state(
    before: dict[
        str,
        torch.Tensor,
    ],
    model: nn.Module,
) -> None:
    after = model.state_dict()

    if before.keys() != after.keys():
        raise ValueError(
            "Model state keys changed"
        )

    for name, value in before.items():
        if not torch.equal(
            value,
            after[
                name
            ].detach().cpu(),
        ):
            raise ValueError(
                f"Model state changed: {name}"
            )


def deterministic_probabilities(
    model: nn.Module,
    signals: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    probabilities = np.empty(
        (
            len(
                signals
            ),
            N_CLASSES,
        ),
        dtype=np.float32,
    )

    with torch.inference_mode():
        for start in range(
            0,
            len(
                signals
            ),
            BATCH_SIZE,
        ):
            end = min(
                start
                + BATCH_SIZE,
                len(
                    signals
                ),
            )

            inputs = torch.from_numpy(
                signals[
                    start:end
                ]
            ).to(
                device
            )

            logits = model(
                inputs
            )

            if not torch.isfinite(
                logits
            ).all():
                raise FloatingPointError(
                    "Non-finite deterministic logits"
                )

            probabilities[
                start:end
            ] = torch.sigmoid(
                logits
            ).cpu().numpy()

    return probabilities


def mc_probabilities(
    model: nn.Module,
    signals: np.ndarray,
    device: torch.device,
    *,
    passes: int = MC_PASSES,
    progress: bool = True,
) -> np.ndarray:
    probabilities = np.empty(
        (
            passes,
            len(
                signals
            ),
            N_CLASSES,
        ),
        dtype=np.float32,
    )

    # Common random-number schedule across clean/corruption
    # conditions to reduce Monte Carlo comparison noise.
    torch.manual_seed(
        20260826
    )

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(
            20260826
        )

    dropout_count = enable_mc_dropout(
        model
    )

    if dropout_count != 2:
        raise ValueError(
            "Expected exactly two Dropout modules"
        )

    if any(
        module.training
        for module
        in model.modules()
        if isinstance(
            module,
            nn.BatchNorm1d,
        )
    ):
        raise ValueError(
            "BatchNorm entered training mode"
        )

    with torch.inference_mode():
        for pass_index in range(
            passes
        ):
            for start in range(
                0,
                len(
                    signals
                ),
                BATCH_SIZE,
            ):
                end = min(
                    start
                    + BATCH_SIZE,
                    len(
                        signals
                    ),
                )

                inputs = torch.from_numpy(
                    signals[
                        start:end
                    ]
                ).to(
                    device
                )

                logits = model(
                    inputs
                )

                if not torch.isfinite(
                    logits
                ).all():
                    raise FloatingPointError(
                        "Non-finite MC logits"
                    )

                probabilities[
                    pass_index,
                    start:end,
                ] = torch.sigmoid(
                    logits
                ).cpu().numpy()

            if (
                progress
                and (
                    (
                        pass_index + 1
                    )
                    % 5
                    == 0
                    or (
                        pass_index + 1
                    )
                    == passes
                )
            ):
                print(
                    f"    MC "
                    f"{pass_index + 1:02d}/"
                    f"{passes}"
                )

    model.eval()

    return probabilities


def thresholds_array() -> np.ndarray:
    threshold_json = load_json(
        THRESHOLDS_PATH
    )

    if threshold_json.get(
        "test_records_used"
    ) != 0:
        raise ValueError(
            "Threshold artifact reports test use"
        )

    values = np.asarray(
        [
            threshold_json[
                "thresholds"
            ][
                label
            ][
                "threshold"
            ]
            for label
            in SUPERCLASS_ORDER
        ],
        dtype=np.float64,
    )

    if (
        values.shape
        != (
            N_CLASSES,
        )
        or not np.isfinite(
            values
        ).all()
        or (
            values < 0.0
        ).any()
        or (
            values > 1.0
        ).any()
    ):
        raise ValueError(
            "Invalid frozen thresholds"
        )

    return values


def corruption_conditions(
    config: dict,
) -> list[
    dict
]:
    corruptions = config[
        "corruptions"
    ]

    levels = (
        "mild",
        "moderate",
        "severe",
    )

    result = []

    for severity_index, level in enumerate(
        levels
    ):
        result.append({
            "name":
                f"baseline_wander_{level}",

            "type":
                "baseline_wander",

            "severity":
                level,

            "severity_index":
                severity_index,

            "snr_db":
                float(
                    corruptions[
                        "baseline_wander"
                    ][
                        "severity"
                    ][
                        level
                    ][
                        "snr_db"
                    ]
                ),

            "frequency_hz":
                float(
                    corruptions[
                        "baseline_wander"
                    ][
                        "frequency_hz"
                    ]
                ),
        })

    for severity_index, level in enumerate(
        levels
    ):
        result.append({
            "name":
                f"additive_white_noise_{level}",

            "type":
                "additive_white_noise",

            "severity":
                level,

            "severity_index":
                severity_index,

            "snr_db":
                float(
                    corruptions[
                        "additive_white_noise"
                    ][
                        "severity"
                    ][
                        level
                    ][
                        "snr_db"
                    ]
                ),
        })

    for severity_index, level in enumerate(
        levels
    ):
        result.append({
            "name":
                f"amplitude_clipping_{level}",

            "type":
                "amplitude_clipping",

            "severity":
                level,

            "severity_index":
                severity_index,

            "quantile":
                float(
                    corruptions[
                        "amplitude_clipping"
                    ][
                        "severity"
                    ][
                        level
                    ][
                        "absolute_centered_quantile"
                    ]
                ),
        })

    for severity_index, level in enumerate(
        levels
    ):
        result.append({
            "name":
                f"lead_masking_{level}",

            "type":
                "lead_masking",

            "severity":
                level,

            "severity_index":
                severity_index,

            "number_of_leads":
                int(
                    corruptions[
                        "lead_masking"
                    ][
                        "severity"
                    ][
                        level
                    ][
                        "number_of_leads"
                    ]
                ),
        })

    if len(
        result
    ) != 12:
        raise AssertionError(
            "Expected exactly 12 frozen "
            "corruption conditions"
        )

    return result


def apply_corruption(
    signal: np.ndarray,
    ecg_id: int,
    condition: dict,
) -> np.ndarray:
    kind = condition[
        "type"
    ]

    if kind == "baseline_wander":
        return add_baseline_wander(
            signal,
            ecg_id=ecg_id,
            snr_db=condition[
                "snr_db"
            ],
            severity_index=condition[
                "severity_index"
            ],
            frequency_hz=condition[
                "frequency_hz"
            ],
        )

    if kind == "additive_white_noise":
        return add_white_noise(
            signal,
            ecg_id=ecg_id,
            snr_db=condition[
                "snr_db"
            ],
            severity_index=condition[
                "severity_index"
            ],
        )

    if kind == "amplitude_clipping":
        return amplitude_clip(
            signal,
            quantile=condition[
                "quantile"
            ],
        )

    if kind == "lead_masking":
        corrupted, _ = mask_leads(
            signal,
            ecg_id=ecg_id,
            number_of_leads=condition[
                "number_of_leads"
            ],
            severity_index=condition[
                "severity_index"
            ],
        )

        return corrupted

    raise ValueError(
        f"Unknown corruption: {kind}"
    )


def q_from_feature_arrays(
    feature_arrays: dict[
        str,
        np.ndarray,
    ],
    q_reference: np.lib.npyio.NpzFile,
) -> tuple[
    np.ndarray,
    dict[
        str,
        np.ndarray,
    ],
]:
    percentiles = {}

    for feature in Q_FEATURES:
        reference_key = (
            f"{feature}_reference"
        )

        if reference_key not in q_reference:
            raise ValueError(
                f"Training Q reference missing "
                f"{reference_key}"
            )

        reference = np.asarray(
            q_reference[
                reference_key
            ],
            dtype=np.float64,
        )

        values = np.asarray(
            feature_arrays[
                feature
            ],
            dtype=np.float64,
        )

        percentiles[
            feature
        ] = midrank_ecdf(
            reference,
            values,
        )

    badness = np.mean(
        np.column_stack(
            [
                percentiles[
                    feature
                ]
                for feature
                in Q_FEATURES
            ]
        ),
        axis=1,
        dtype=np.float64,
    )

    quality = (
        1.0
        - badness
    )

    if (
        not np.isfinite(
            quality
        ).all()
        or (
            quality < 0.0
        ).any()
        or (
            quality > 1.0
        ).any()
    ):
        raise ValueError(
            "Invalid Q_signal"
        )

    return (
        quality,
        percentiles,
    )


def selectors_from_scores(
    *,
    confidence: np.ndarray,
    L: np.ndarray,
    Q: np.ndarray,
    U: np.ndarray,
    combination_reference: np.lib.npyio.NpzFile,
) -> dict[
    str,
    np.ndarray,
]:
    required = {
        "L_badness_reference",
        "Q_signal_badness_reference",
        "U_badness_reference",
    }

    missing = (
        required
        - set(
            combination_reference.files
        )
    )

    if missing:
        raise ValueError(
            f"Combination reference missing: "
            f"{sorted(missing)}"
        )

    L_pct = midrank_ecdf(
        np.asarray(
            combination_reference[
                "L_badness_reference"
            ],
            dtype=np.float64,
        ),
        1.0 - L,
    )

    Q_pct = midrank_ecdf(
        np.asarray(
            combination_reference[
                "Q_signal_badness_reference"
            ],
            dtype=np.float64,
        ),
        1.0 - Q,
    )

    U_pct = midrank_ecdf(
        np.asarray(
            combination_reference[
                "U_badness_reference"
            ],
            dtype=np.float64,
        ),
        U,
    )

    selectors = {
        "confidence":
            confidence,

        "U":
            -U,

        "L":
            L,

        "Q_signal":
            Q,

        "L+U":
            -(
                L_pct
                + U_pct
            )
            / 2.0,

        "L+Q_signal":
            -(
                L_pct
                + Q_pct
            )
            / 2.0,

        "Q_signal+U":
            -(
                Q_pct
                + U_pct
            )
            / 2.0,

        "L+Q_signal+U":
            -(
                L_pct
                + Q_pct
                + U_pct
            )
            / 3.0,
    }

    if tuple(
        selectors.keys()
    ) != METHODS:
        raise AssertionError(
            "Selector order changed"
        )

    return selectors


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    auc = np.asarray(
        classwise_auroc(
            targets,
            probabilities,
        ),
        dtype=np.float64,
    )

    ap = np.asarray(
        [
            average_precision_score(
                targets[
                    :,
                    index,
                ],
                probabilities[
                    :,
                    index,
                ],
            )
            for index in range(
                N_CLASSES
            )
        ],
        dtype=np.float64,
    )

    f1 = np.asarray(
        f1_score(
            targets,
            predictions,
            average=None,
            zero_division=0,
        ),
        dtype=np.float64,
    )

    return {
        "macro_auroc":
            float(
                auc.mean()
            ),

        "macro_average_precision":
            float(
                ap.mean()
            ),

        "macro_f1":
            float(
                f1.mean()
            ),

        "classwise": {
            label: {
                "auroc":
                    float(
                        auc[
                            index
                        ]
                    ),

                "average_precision":
                    float(
                        ap[
                            index
                        ]
                    ),

                "f1":
                    float(
                        f1[
                            index
                        ]
                    ),
            }
            for index, label
            in enumerate(
                SUPERCLASS_ORDER
            )
        },
    }


def result_metrics(
    *,
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    losses: np.ndarray,
    confidence: np.ndarray,
    L: np.ndarray,
    Q: np.ndarray,
    U: np.ndarray,
    combination_reference: np.lib.npyio.NpzFile,
) -> dict:
    selectors = selectors_from_scores(
        confidence=confidence,
        L=L,
        Q=Q,
        U=U,
        combination_reference=combination_reference,
    )

    selective = {}

    for method, reliability in selectors.items():
        curve = grouped_risk_coverage(
            reliability,
            losses,
        )

        selective[
            method
        ] = {
            "aurc":
                float(
                    curve[
                        "aurc"
                    ]
                ),

            "unique_reliability_levels":
                int(
                    len(
                        curve[
                            "coverage"
                        ]
                    )
                ),
        }

    return {
        "classification":
            classification_metrics(
                targets,
                probabilities,
                predictions,
            ),

        "full_hamming_risk":
            float(
                np.mean(
                    losses,
                    dtype=np.float64,
                )
            ),

        "mean_confidence":
            float(
                np.mean(
                    confidence,
                    dtype=np.float64,
                )
            ),

        "mean_L":
            float(
                np.mean(
                    L,
                    dtype=np.float64,
                )
            ),

        "mean_Q_signal":
            float(
                np.mean(
                    Q,
                    dtype=np.float64,
                )
            ),

        "mean_U":
            float(
                np.mean(
                    U,
                    dtype=np.float64,
                )
            ),

        "selective_prediction":
            selective,
    }


def compute_condition_arrays(
    *,
    raw: np.ndarray,
    test_manifest: pd.DataFrame,
    condition: dict | None,
    mean: float,
    std: float,
    model: nn.Module,
    device: torch.device,
    thresholds: np.ndarray,
    q_reference: np.lib.npyio.NpzFile,
) -> dict[
    str,
    np.ndarray,
]:
    n = len(
        raw
    )

    normalized = np.empty(
        (
            n,
            12,
            1000,
        ),
        dtype=np.float32,
    )

    feature_arrays = {
        feature:
            np.empty(
                n,
                dtype=np.float64,
            )
        for feature
        in Q_FEATURES
    }

    label = (
        "clean"
        if condition is None
        else condition[
            "name"
        ]
    )

    for index in range(
        n
    ):
        x = raw[
            index
        ].astype(
            np.float64,
            copy=False,
        )

        if condition is not None:
            x = apply_corruption(
                x,
                int(
                    test_manifest.iloc[
                        index
                    ][
                        "ecg_id"
                    ]
                ),
                condition,
            )

        features = waveform_badness_features(
            x,
            fs=100.0,
        )

        for feature in Q_FEATURES:
            feature_arrays[
                feature
            ][
                index
            ] = features[
                feature
            ]

        normalized[
            index
        ] = (
            (
                x
                - mean
            )
            / std
        ).astype(
            np.float32
        )

        if (
            (
                index + 1
            )
            % 500
            == 0
            or (
                index + 1
            )
            == n
        ):
            print(
                f"    {label} preprocess/Q "
                f"{index + 1}/{n}"
            )

    Q, _ = q_from_feature_arrays(
        feature_arrays,
        q_reference,
    )

    probabilities = deterministic_probabilities(
        model,
        normalized,
        device,
    )

    predictions = (
        probabilities
        >= thresholds[
            None,
            :,
        ]
    ).astype(
        np.int64
    )

    confidence = mean_bernoulli_confidence(
        probabilities
    )

    mc = mc_probabilities(
        model,
        normalized,
        device,
        passes=MC_PASSES,
        progress=True,
    )

    uncertainty = mc_uncertainty(
        mc
    )

    U = np.asarray(
        uncertainty[
            "mean_mutual_information"
        ],
        dtype=np.float64,
    )

    targets = test_manifest[
        list(
            SUPERCLASS_ORDER
        )
    ].to_numpy(
        dtype=np.int64
    )

    losses = sample_hamming_error(
        targets,
        predictions,
    )

    arrays = {
        "ecg_ids":
            test_manifest[
                "ecg_id"
            ].to_numpy(
                dtype=np.int64
            ),

        "patient_ids":
            test_manifest[
                "patient_id"
            ].to_numpy(
                dtype=np.int64
            ),

        "targets":
            targets.astype(
                np.int8
            ),

        "probabilities":
            probabilities.astype(
                np.float32
            ),

        "predictions":
            predictions.astype(
                np.int8
            ),

        "hamming_error":
            losses.astype(
                np.float64
            ),

        "confidence":
            confidence.astype(
                np.float64
            ),

        "Q_signal":
            Q.astype(
                np.float64
            ),

        "U":
            U.astype(
                np.float64
            ),

        "mc_probabilities":
            mc.astype(
                np.float32
            ),

        **{
            feature:
                feature_arrays[
                    feature
                ].astype(
                    np.float64
                )
            for feature
            in Q_FEATURES
        },
    }

    for name, values in arrays.items():
        if (
            name
            not in {
                "ecg_ids",
                "patient_ids",
                "targets",
                "predictions",
            }
            and not np.isfinite(
                values
            ).all()
        ):
            raise ValueError(
                f"{label}: {name} contains NaN/Inf"
            )

    return arrays


def validate_saved_artifact(
    artifact: Path,
    *,
    expected_ids: np.ndarray,
    expected_patients: np.ndarray,
    expected_targets: np.ndarray,
) -> dict[
    str,
    np.ndarray,
]:
    saved = np.load(
        artifact,
        allow_pickle=False,
    )

    required = {
        "ecg_ids",
        "patient_ids",
        "targets",
        "probabilities",
        "predictions",
        "hamming_error",
        "confidence",
        "Q_signal",
        "U",
        "mc_probabilities",
        *Q_FEATURES,
    }

    missing = (
        required
        - set(
            saved.files
        )
    )

    if missing:
        raise ValueError(
            f"{artifact} missing arrays: "
            f"{sorted(missing)}"
        )

    arrays = {
        key:
            np.asarray(
                saved[
                    key
                ]
            )
        for key in required
    }

    if not np.array_equal(
        arrays[
            "ecg_ids"
        ].astype(
            np.int64
        ),
        expected_ids,
    ):
        raise ValueError(
            f"{artifact}: ECG IDs mismatch"
        )

    if not np.array_equal(
        arrays[
            "patient_ids"
        ].astype(
            np.int64
        ),
        expected_patients,
    ):
        raise ValueError(
            f"{artifact}: patient IDs mismatch"
        )

    if not np.array_equal(
        arrays[
            "targets"
        ].astype(
            np.int64
        ),
        expected_targets,
    ):
        raise ValueError(
            f"{artifact}: targets mismatch"
        )

    if arrays[
        "probabilities"
    ].shape != (
        EXPECTED_RECORDS,
        N_CLASSES,
    ):
        raise ValueError(
            f"{artifact}: probability shape mismatch"
        )

    if arrays[
        "mc_probabilities"
    ].shape != (
        MC_PASSES,
        EXPECTED_RECORDS,
        N_CLASSES,
    ):
        raise ValueError(
            f"{artifact}: MC shape mismatch"
        )

    for name in (
        "probabilities",
        "hamming_error",
        "confidence",
        "Q_signal",
        "U",
        "mc_probabilities",
        *Q_FEATURES,
    ):
        if not np.isfinite(
            arrays[
                name
            ]
        ).all():
            raise ValueError(
                f"{artifact}: "
                f"{name} contains NaN/Inf"
            )

    return arrays


def save_arrays(
    artifact: Path,
    arrays: dict[
        str,
        np.ndarray,
    ],
) -> None:
    artifact.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        artifact,
        **arrays,
    )


def preflight() -> None:
    print(
        "FINAL EVALUATION PREFLIGHT"
    )

    _, corruption_config = (
        load_frozen_protocol()
    )

    mean, std = load_normalization()

    test_manifest = load_test_manifest()

    print(
        f"  fold 10 records: "
        f"{len(test_manifest)}"
    )

    print(
        f"  fold 10 patients: "
        f"{test_manifest['patient_id'].nunique()}"
    )

    L, coverage = (
        build_test_annotation_confidence(
            test_manifest,
            write_artifact=False,
        )
    )

    print(
        "  L target reconstruction: PASS"
    )

    print(
        f"  L mean: "
        f"{L.mean():.6f}"
    )

    print(
        f"  L mean coverage: "
        f"{coverage.mean():.6f}"
    )

    thresholds = thresholds_array()

    q_reference = np.load(
        Q_TRAIN_REFERENCE_PATH,
        allow_pickle=False,
    )

    combination_reference = np.load(
        COMBINATION_REFERENCE_PATH,
        allow_pickle=False,
    )

    # Force schema validation now.
    _ = selectors_from_scores(
        confidence=np.full(
            1,
            0.8,
            dtype=np.float64,
        ),
        L=np.full(
            1,
            L[0],
            dtype=np.float64,
        ),
        Q=np.full(
            1,
            0.5,
            dtype=np.float64,
        ),
        U=np.full(
            1,
            0.01,
            dtype=np.float64,
        ),
        combination_reference=
            combination_reference,
    )

    first = test_manifest.iloc[
        0
    ]

    signal, fields = wfdb.rdsamp(
        str(
            RAW_ROOT
            / str(
                first[
                    "filename_lr"
                ]
            )
        )
    )

    if signal.shape != (
        1000,
        12,
    ):
        raise ValueError(
            "Preflight waveform shape mismatch"
        )

    x = signal.T.astype(
        np.float64
    )

    features = waveform_badness_features(
        x,
        fs=100.0,
    )

    q_feature_arrays = {
        feature:
            np.asarray(
                [
                    features[
                        feature
                    ]
                ],
                dtype=np.float64,
            )
        for feature
        in Q_FEATURES
    }

    q, _ = q_from_feature_arrays(
        q_feature_arrays,
        q_reference,
    )

    normalized = (
        (
            x
            - mean
        )
        / std
    ).astype(
        np.float32
    )[
        None,
        :,
        :,
    ]

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = load_model(
        device
    )

    state = state_copy(
        model
    )

    probabilities = (
        deterministic_probabilities(
            model,
            normalized,
            device,
        )
    )

    if probabilities.shape != (
        1,
        N_CLASSES,
    ):
        raise ValueError(
            "Preflight deterministic output mismatch"
        )

    mc = mc_probabilities(
        model,
        normalized,
        device,
        passes=1,
        progress=False,
    )

    if mc.shape != (
        1,
        1,
        N_CLASSES,
    ):
        raise ValueError(
            "Preflight MC output mismatch"
        )

    # Check each corruption implementation on one fold-10 ECG.
    for condition in corruption_conditions(
        corruption_config
    ):
        corrupted = apply_corruption(
            x,
            int(
                first[
                    "ecg_id"
                ]
            ),
            condition,
        )

        if corrupted.shape != (
            12,
            1000,
        ):
            raise ValueError(
                f"{condition['name']}: "
                "preflight corruption shape mismatch"
            )

        if not np.isfinite(
            corrupted
        ).all():
            raise ValueError(
                f"{condition['name']}: "
                "preflight corruption NaN/Inf"
            )

    assert_same_state(
        state,
        model,
    )

    print(
        f"  frozen normalization: "
        f"mean={mean:.17g}, "
        f"std={std:.17g}"
    )

    print(
        f"  frozen thresholds: "
        f"{thresholds.tolist()}"
    )

    print(
        f"  first-record Q_signal: "
        f"{float(q[0]):.6f}"
    )

    print(
        "  checkpoint load/inference: PASS"
    )

    print(
        "  MC-dropout preflight: PASS"
    )

    print(
        "  12 corruption implementations: PASS"
    )

    print(
        "  model state unchanged: PASS"
    )

    print(
        "  test-set ECDF fitting: NOT USED"
    )

    print(
        "\nPREFLIGHT PASS"
    )

    print(
        "No final result artifacts were created."
    )


def full_run() -> None:
    final_config, corruption_config = (
        load_frozen_protocol()
    )

    mean, std = load_normalization()

    test_manifest = load_test_manifest()

    targets = test_manifest[
        list(
            SUPERCLASS_ORDER
        )
    ].to_numpy(
        dtype=np.int64
    )

    ecg_ids = test_manifest[
        "ecg_id"
    ].to_numpy(
        dtype=np.int64
    )

    patient_ids = test_manifest[
        "patient_id"
    ].to_numpy(
        dtype=np.int64
    )

    thresholds = thresholds_array()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    L, L_coverage = (
        build_test_annotation_confidence(
            test_manifest,
            write_artifact=True,
        )
    )

    raw = load_raw_test(
        test_manifest
    )

    q_reference = np.load(
        Q_TRAIN_REFERENCE_PATH,
        allow_pickle=False,
    )

    combination_reference = np.load(
        COMBINATION_REFERENCE_PATH,
        allow_pickle=False,
    )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = load_model(
        device
    )

    original_state = state_copy(
        model
    )

    all_results = {}
    rows = []

    run_items = [
        (
            "clean",
            None,
        ),
        *[
            (
                condition[
                    "name"
                ],
                condition,
            )
            for condition
            in corruption_conditions(
                corruption_config
            )
        ],
    ]

    for item_index, (
        name,
        condition,
    ) in enumerate(
        run_items,
        start=1,
    ):
        print(
            f"\n[{item_index:02d}/13] "
            f"{name}"
        )

        condition_dir = (
            OUT_DIR
            / name
        )

        artifact = (
            condition_dir
            / "results.npz"
        )

        if artifact.exists():
            print(
                "    reusing saved final-test artifact"
            )

            arrays = validate_saved_artifact(
                artifact,
                expected_ids=ecg_ids,
                expected_patients=
                    patient_ids,
                expected_targets=
                    targets,
            )
        else:
            arrays = (
                compute_condition_arrays(
                    raw=raw,
                    test_manifest=
                        test_manifest,
                    condition=condition,
                    mean=mean,
                    std=std,
                    model=model,
                    device=device,
                    thresholds=
                        thresholds,
                    q_reference=
                        q_reference,
                )
            )

            save_arrays(
                artifact,
                arrays,
            )

            print(
                "    artifact saved"
            )

        probabilities = np.asarray(
            arrays[
                "probabilities"
            ],
            dtype=np.float64,
        )

        predictions = np.asarray(
            arrays[
                "predictions"
            ],
            dtype=np.int64,
        )

        losses = np.asarray(
            arrays[
                "hamming_error"
            ],
            dtype=np.float64,
        )

        confidence = np.asarray(
            arrays[
                "confidence"
            ],
            dtype=np.float64,
        )

        Q = np.asarray(
            arrays[
                "Q_signal"
            ],
            dtype=np.float64,
        )

        U = np.asarray(
            arrays[
                "U"
            ],
            dtype=np.float64,
        )

        # Independently recompute losses from saved predictions/targets.
        recomputed_losses = (
            sample_hamming_error(
                targets,
                predictions,
            )
        )

        np.testing.assert_allclose(
            losses,
            recomputed_losses,
            rtol=0.0,
            atol=1e-15,
        )

        metrics = result_metrics(
            targets=targets,
            probabilities=
                probabilities,
            predictions=
                predictions,
            losses=losses,
            confidence=confidence,
            L=L,
            Q=Q,
            U=U,
            combination_reference=
                combination_reference,
        )

        all_results[
            name
        ] = {
            "condition":
                (
                    {
                        "type":
                            "clean",

                        "severity":
                            "clean",
                    }
                    if condition is None
                    else condition
                ),

            **metrics,

            "results_npz_sha256":
                sha256_file(
                    artifact
                ),
        }

        row = {
            "condition":
                name,

            "corruption_type":
                (
                    "clean"
                    if condition is None
                    else condition[
                        "type"
                    ]
                ),

            "severity":
                (
                    "clean"
                    if condition is None
                    else condition[
                        "severity"
                    ]
                ),

            "macro_auroc":
                metrics[
                    "classification"
                ][
                    "macro_auroc"
                ],

            "macro_average_precision":
                metrics[
                    "classification"
                ][
                    "macro_average_precision"
                ],

            "macro_f1":
                metrics[
                    "classification"
                ][
                    "macro_f1"
                ],

            "full_hamming_risk":
                metrics[
                    "full_hamming_risk"
                ],

            "mean_confidence":
                metrics[
                    "mean_confidence"
                ],

            "mean_L":
                metrics[
                    "mean_L"
                ],

            "mean_Q_signal":
                metrics[
                    "mean_Q_signal"
                ],

            "mean_U":
                metrics[
                    "mean_U"
                ],
        }

        for method in METHODS:
            row[
                f"aurc_{method}"
            ] = metrics[
                "selective_prediction"
            ][
                method
            ][
                "aurc"
            ]

        rows.append(
            row
        )

        print(
            "    "
            f"AUROC="
            f"{row['macro_auroc']:.6f} "
            f"F1="
            f"{row['macro_f1']:.6f} "
            f"risk="
            f"{row['full_hamming_risk']:.6f} "
            f"conf="
            f"{row['mean_confidence']:.6f} "
            f"Q="
            f"{row['mean_Q_signal']:.6f} "
            f"U="
            f"{row['mean_U']:.6f}"
        )

        assert_same_state(
            original_state,
            model,
        )

    table = pd.DataFrame(
        rows
    )

    table.to_csv(
        CONDITION_TABLE_PATH,
        index=False,
        float_format="%.17g",
    )

    clean_row = table.loc[
        table[
            "condition"
        ].eq(
            "clean"
        )
    ]

    if len(
        clean_row
    ) != 1:
        raise AssertionError(
            "Expected one clean row"
        )

    clean_row = clean_row.iloc[
        0
    ]

    deltas = {}

    for row in table.itertuples(
        index=False
    ):
        if row.condition == "clean":
            continue

        deltas[
            row.condition
        ] = {
            "delta_macro_auroc_vs_clean":
                float(
                    row.macro_auroc
                    - clean_row[
                        "macro_auroc"
                    ]
                ),

            "delta_macro_average_precision_vs_clean":
                float(
                    row.macro_average_precision
                    - clean_row[
                        "macro_average_precision"
                    ]
                ),

            "delta_macro_f1_vs_clean":
                float(
                    row.macro_f1
                    - clean_row[
                        "macro_f1"
                    ]
                ),

            "delta_full_hamming_risk_vs_clean":
                float(
                    row.full_hamming_risk
                    - clean_row[
                        "full_hamming_risk"
                    ]
                ),

            "delta_mean_confidence_vs_clean":
                float(
                    row.mean_confidence
                    - clean_row[
                        "mean_confidence"
                    ]
                ),

            "delta_mean_Q_signal_vs_clean":
                float(
                    row.mean_Q_signal
                    - clean_row[
                        "mean_Q_signal"
                    ]
                ),

            "delta_mean_U_vs_clean":
                float(
                    row.mean_U
                    - clean_row[
                        "mean_U"
                    ]
                ),
        }

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "final_evaluation_id":
            final_config[
                "final_evaluation_id"
            ],

        "evaluation_split":
            "test_fold_10",

        "records":
            EXPECTED_RECORDS,

        "patients":
            EXPECTED_PATIENTS,

        "conditions":
            13,

        "clean_plus_corruptions":
            "1+12",

        "mc_passes_per_condition":
            MC_PASSES,

        "thresholds_source":
            "frozen_validation_f1_thresholds",

        "combination_scaling":
            "frozen_validation_empirical_midrank_ecdf",

        "test_set_ecdf_fitting_used":
            False,

        "post_test_tuning_allowed":
            False,

        "L_interpretation":
            (
                "retrospective annotation-aware "
                "confidence/ambiguity; not a "
                "deployment-available selector"
            ),

        "Q_interpretation":
            (
                "training-referenced study-specific "
                "waveform quality score; not a "
                "clinically calibrated universal SQI"
            ),

        "condition_results":
            all_results,

        "deltas_vs_clean":
            deltas,

        "annotation_confidence": {
            "mean":
                float(
                    L.mean()
                ),

            "mean_known_positive_superclass_fraction":
                float(
                    L_coverage.mean()
                ),

            "artifact_sha256":
                sha256_file(
                    L_TEST_PATH
                ),
        },

        "source_sha256": {
            "final_config":
                sha256_file(
                    FINAL_CONFIG_PATH
                ),

            "corruption_config":
                sha256_file(
                    CORRUPTION_CONFIG_PATH
                ),

            "manifest":
                sha256_file(
                    MANIFEST_PATH
                ),

            "normalization":
                sha256_file(
                    NORM_PATH
                ),

            "checkpoint":
                sha256_file(
                    CHECKPOINT_PATH
                ),

            "thresholds":
                sha256_file(
                    THRESHOLDS_PATH
                ),

            "validation_L":
                sha256_file(
                    L_VALIDATION_PATH
                ),

            "training_Q_reference":
                sha256_file(
                    Q_TRAIN_REFERENCE_PATH
                ),

            "validation_combination_reference":
                sha256_file(
                    COMBINATION_REFERENCE_PATH
                ),

            "script":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),
        },

        "condition_table_sha256":
            sha256_file(
                CONDITION_TABLE_PATH
            ),

        "model_state_unchanged":
            True,

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

    print(
        "\nFINAL FOLD-10 EVALUATION COMPLETE"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Table:   "
        f"{CONDITION_TABLE_PATH}"
    )

    print(
        "Development remains CLOSED."
    )

    print(
        "Next permitted analysis: "
        "frozen patient-level paired bootstrap."
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Run schema/artifact/tiny-inference "
            "checks only."
        ),
    )

    args = parser.parse_args()

    if args.preflight:
        preflight()
    else:
        full_run()


if __name__ == "__main__":
    main()
