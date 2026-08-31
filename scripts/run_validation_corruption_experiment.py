#!/usr/bin/env python3
"""Validation-only 4x3 corruption stress test. Fold 10 is never loaded."""

from __future__ import annotations

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
    add_baseline_wander, add_white_noise, amplitude_clip, mask_leads,
)
from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.metrics import classwise_auroc
from ptbxl_reliability.model import ResNet1DWang
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.selective_prediction import (
    grouped_risk_coverage, mean_bernoulli_confidence, midrank_ecdf,
    sample_hamming_error,
)
from ptbxl_reliability.uncertainty import enable_mc_dropout, mc_uncertainty
from ptbxl_reliability.waveform_signal_quality import waveform_badness_features


RAW = PROJECT_ROOT / "data/raw/ptb-xl/1.0.3"
MANIFEST = PROJECT_ROOT / "data/processed/cohort/v1/ptbxl_superdiagnostic_cohort.csv"
NORM = PROJECT_ROOT / "data/processed/preprocessing/v1/global_zscore_stats.json"
RUN = PROJECT_ROOT / "results/baseline/v1/seed_20260826"
CHECKPOINT = RUN / "best_checkpoint.pt"
BASE_PREDS = RUN / "best_validation_predictions.npz"
THRESHOLDS = RUN / "validation_thresholds.json"
BASE_METRICS = RUN / "validation_metrics.json"
L_PATH = PROJECT_ROOT / "results/label_confidence/v1/validation/label_confidence.csv"
Q_CLEAN = PROJECT_ROOT / "results/waveform_signal_quality/v1/validation/waveform_signal_quality.csv"
Q_TRAIN_REF = PROJECT_ROOT / "data/processed/waveform_signal_quality/v1/training_reference.npz"
U_CLEAN = PROJECT_ROOT / "results/uncertainty/v1/validation/mc_dropout_uncertainty.npz"
SEL_DIR = PROJECT_ROOT / "results/selective_prediction_waveform_q/v1/validation"
SEL_CLEAN = SEL_DIR / "summary.json"
COMBO_REF = SEL_DIR / "combination_reference.npz"
CONFIG = PROJECT_ROOT / "configs/corruption_v1.yaml"

OUT = PROJECT_ROOT / "results/corruption/v1/validation"
SUMMARY = OUT / "summary.json"
TABLE = OUT / "condition_metrics.csv"

N = 2146
K = 5
BATCH = 128
MC_PASSES = 30
FEATURES = (
    "baseline_burden", "high_frequency_burden",
    "lead_dropout_burden", "clipping_burden",
)
METHODS = (
    "confidence", "L", "Q_signal", "U",
    "L+Q_signal", "L+U", "Q_signal+U", "L+Q_signal+U",
)
LEADS = ["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def state_copy(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def assert_same_state(before, model):
    after = model.state_dict()
    if before.keys() != after.keys():
        raise ValueError("Model state keys changed")
    for name, value in before.items():
        if not torch.equal(value, after[name].detach().cpu()):
            raise ValueError(f"Model state changed: {name}")


def deterministic_probs(model, signals, device):
    model.eval()
    out = np.empty((len(signals), K), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(signals), BATCH):
            end = min(start + BATCH, len(signals))
            logits = model(torch.from_numpy(signals[start:end]).to(device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite deterministic logits")
            out[start:end] = torch.sigmoid(logits).cpu().numpy()
    return out


def dropout_probs(model, signals, device):
    """Use common random dropout masks across corruption conditions."""
    out = np.empty((MC_PASSES, len(signals), K), dtype=np.float32)

    # Same MC seed schedule for every condition. This reduces Monte Carlo
    # noise in condition-to-condition comparisons.
    torch.manual_seed(20260826)
    torch.mps.manual_seed(20260826)

    if enable_mc_dropout(model) != 2:
        raise ValueError("Expected exactly two Dropout modules")

    if any(
        module.training
        for module in model.modules()
        if isinstance(module, nn.BatchNorm1d)
    ):
        raise ValueError("BatchNorm entered training mode")

    with torch.inference_mode():
        for t in range(MC_PASSES):
            for start in range(0, len(signals), BATCH):
                end = min(start + BATCH, len(signals))
                logits = model(torch.from_numpy(signals[start:end]).to(device))
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite MC logits")
                out[t, start:end] = torch.sigmoid(logits).cpu().numpy()

            if (t + 1) % 5 == 0:
                print(f"    MC {t + 1:02d}/{MC_PASSES}")

    model.eval()
    return out


def performance(targets, probs, preds):
    auc = np.asarray(classwise_auroc(targets, probs), dtype=np.float64)
    ap = np.asarray([
        average_precision_score(targets[:, j], probs[:, j])
        for j in range(K)
    ], dtype=np.float64)
    f1 = np.asarray(
        f1_score(targets, preds, average=None, zero_division=0),
        dtype=np.float64,
    )
    return {
        "macro_auroc": float(auc.mean()),
        "macro_average_precision": float(ap.mean()),
        "macro_f1": float(f1.mean()),
        "classwise": {
            label: {
                "auroc": float(auc[j]),
                "average_precision": float(ap[j]),
                "f1": float(f1[j]),
            }
            for j, label in enumerate(SUPERCLASS_ORDER)
        },
    }


def conditions(config):
    c = config["corruptions"]
    result = []
    levels = ("mild", "moderate", "severe")

    for i, level in enumerate(levels):
        result.append({
            "name": f"baseline_wander_{level}",
            "type": "baseline_wander",
            "severity": level,
            "severity_index": i,
            "snr_db": float(c["baseline_wander"]["severity"][level]["snr_db"]),
            "frequency_hz": float(c["baseline_wander"]["frequency_hz"]),
        })

    for i, level in enumerate(levels):
        result.append({
            "name": f"additive_white_noise_{level}",
            "type": "additive_white_noise",
            "severity": level,
            "severity_index": i,
            "snr_db": float(c["additive_white_noise"]["severity"][level]["snr_db"]),
        })

    for i, level in enumerate(levels):
        result.append({
            "name": f"amplitude_clipping_{level}",
            "type": "amplitude_clipping",
            "severity": level,
            "severity_index": i,
            "quantile": float(
                c["amplitude_clipping"]["severity"][level][
                    "absolute_centered_quantile"
                ]
            ),
        })

    for i, level in enumerate(levels):
        result.append({
            "name": f"lead_masking_{level}",
            "type": "lead_masking",
            "severity": level,
            "severity_index": i,
            "number_of_leads": int(
                c["lead_masking"]["severity"][level]["number_of_leads"]
            ),
        })

    if len(result) != 12:
        raise AssertionError("Expected 12 corruption conditions")

    return result


def corrupt(x, ecg_id, condition):
    kind = condition["type"]

    if kind == "baseline_wander":
        return add_baseline_wander(
            x,
            ecg_id=ecg_id,
            snr_db=condition["snr_db"],
            severity_index=condition["severity_index"],
            frequency_hz=condition["frequency_hz"],
        )

    if kind == "additive_white_noise":
        return add_white_noise(
            x,
            ecg_id=ecg_id,
            snr_db=condition["snr_db"],
            severity_index=condition["severity_index"],
        )

    if kind == "amplitude_clipping":
        return amplitude_clip(
            x,
            quantile=condition["quantile"],
        )

    if kind == "lead_masking":
        y, _ = mask_leads(
            x,
            ecg_id=ecg_id,
            number_of_leads=condition["number_of_leads"],
            severity_index=condition["severity_index"],
        )
        return y

    raise ValueError(f"Unknown corruption type: {kind}")


def main():
    with CONFIG.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config["status"] != "FROZEN":
        raise ValueError("Corruption config is not FROZEN")
    if config["development"]["split"] != "validation":
        raise ValueError("Development split must be validation")
    if config["development"]["test_allowed"] is not False:
        raise ValueError("Test use is forbidden")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    baseline = np.load(BASE_PREDS, allow_pickle=False)
    ecg_ids = np.asarray(baseline["ecg_ids"], dtype=np.int64)
    targets = np.asarray(baseline["targets"], dtype=np.int64)
    clean_probs = np.asarray(baseline["probabilities"], dtype=np.float64)

    if ecg_ids.shape != (N,) or targets.shape != (N, K):
        raise ValueError("Unexpected baseline validation shapes")

    threshold_json = load_json(THRESHOLDS)
    if threshold_json["test_records_used"] != 0:
        raise ValueError("Threshold artifact reports test use")

    thresholds = np.asarray([
        threshold_json["thresholds"][label]["threshold"]
        for label in SUPERCLASS_ORDER
    ], dtype=np.float64)

    norm = load_json(NORM)

    if "mean_mV" not in norm or "std_mV" not in norm:
        raise ValueError(
            "Frozen normalization artifact must contain "
            "top-level mean_mV and std_mV"
        )

    mean = float(norm["mean_mV"])
    std = float(norm["std_mV"])

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ValueError(
            "Invalid frozen normalization statistics"
        )

    expected_mean = -0.0008252533901116082
    expected_std = 0.23222258117564865

    if not np.isclose(
        mean,
        expected_mean,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            f"Frozen normalization mean mismatch: {mean}"
        )

    if not np.isclose(
        std,
        expected_std,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            f"Frozen normalization std mismatch: {std}"
        )

    manifest = pd.read_csv(MANIFEST)
    val = (
        manifest.loc[manifest["split"].eq("validation")]
        .reset_index(drop=True)
    )

    if len(val) != N:
        raise ValueError("Unexpected validation manifest size")
    if not np.array_equal(
        val["ecg_id"].to_numpy(np.int64),
        ecg_ids,
    ):
        raise ValueError("Manifest/baseline ECG ordering mismatch")

    print("Loading raw validation ECGs once...")
    raw = np.empty((N, 12, 1000), dtype=np.float32)

    for i, row in enumerate(val.itertuples(index=False)):
        signal, fields = wfdb.rdsamp(
            str(RAW / str(row.filename_lr))
        )

        if signal.shape != (1000, 12):
            raise ValueError(f"ECG {row.ecg_id}: bad shape")
        if float(fields["fs"]) != 100.0:
            raise ValueError(f"ECG {row.ecg_id}: bad fs")
        if list(fields["sig_name"]) != LEADS:
            raise ValueError(f"ECG {row.ecg_id}: bad lead order")
        if not np.isfinite(signal).all():
            raise ValueError(f"ECG {row.ecg_id}: non-finite signal")

        raw[i] = signal.T.astype(np.float32)

        if (i + 1) % 500 == 0 or i + 1 == N:
            print(f"  raw {i + 1}/{N}")

    l_df = pd.read_csv(L_PATH)
    q_clean_df = pd.read_csv(Q_CLEAN)
    u_clean_npz = np.load(U_CLEAN, allow_pickle=False)

    for name, ids in (
        ("L", l_df["ecg_id"].to_numpy(np.int64)),
        ("Q", q_clean_df["ecg_id"].to_numpy(np.int64)),
        ("U", np.asarray(u_clean_npz["ecg_ids"], dtype=np.int64)),
    ):
        if not np.array_equal(ids, ecg_ids):
            raise ValueError(f"{name} ECG ordering mismatch")

    L = l_df["label_confidence"].to_numpy(np.float64)
    train_q_ref = np.load(Q_TRAIN_REF, allow_pickle=False)
    combo_ref = np.load(COMBO_REF, allow_pickle=False)
    base_metrics = load_json(BASE_METRICS)
    clean_selective = load_json(SEL_CLEAN)

    clean_conf = mean_bernoulli_confidence(clean_probs)
    clean_q = q_clean_df["signal_quality"].to_numpy(np.float64)
    clean_u = np.asarray(
        u_clean_npz["mean_mutual_information"],
        dtype=np.float64,
    )

    clean = {
        "macro_auroc": float(base_metrics["macro_auroc"]),
        "macro_average_precision": float(
            base_metrics["macro_average_precision"]
        ),
        "macro_f1": float(
            base_metrics[
                "macro_f1_at_validation_selected_thresholds"
            ]
        ),
        "full_hamming_risk": float(
            clean_selective["full_coverage_hamming_risk"]
        ),
        "mean_confidence": float(clean_conf.mean()),
        "mean_Q_signal": float(clean_q.mean()),
        "mean_U": float(clean_u.mean()),
        "aurc": {
            name: float(
                clean_selective["methods"][name]["aurc"]
            )
            for name in METHODS
        },
    }

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if int(checkpoint["epoch"]) != 26:
        raise ValueError("Unexpected checkpoint epoch")

    model = ResNet1DWang()
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    device = torch.device("mps")
    model = model.to(device)
    model.eval()

    original_state = state_copy(model)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    all_results = {}

    for condition_index, condition in enumerate(
        conditions(config),
        start=1,
    ):
        name = condition["name"]
        print(f"\n[{condition_index:02d}/12] {name}")

        condition_dir = OUT / name
        condition_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifact = (
            condition_dir
            / "results.npz"
        )

        if artifact.exists():
            print(
                "    reusing saved condition artifact"
            )

            saved = np.load(
                artifact,
                allow_pickle=False,
            )

            saved_ids = np.asarray(
                saved["ecg_ids"],
                dtype=np.int64,
            )

            saved_targets = np.asarray(
                saved["targets"],
                dtype=np.int64,
            )

            if not np.array_equal(
                saved_ids,
                ecg_ids,
            ):
                raise ValueError(
                    f"{name}: saved ECG IDs mismatch"
                )

            if not np.array_equal(
                saved_targets,
                targets,
            ):
                raise ValueError(
                    f"{name}: saved targets mismatch"
                )

            probs = np.asarray(
                saved["probabilities"],
                dtype=np.float64,
            )

            preds = np.asarray(
                saved["predictions"],
                dtype=np.int64,
            )

            losses = np.asarray(
                saved["hamming_error"],
                dtype=np.float64,
            )

            confidence = np.asarray(
                saved["confidence"],
                dtype=np.float64,
            )

            Q = np.asarray(
                saved["Q_signal"],
                dtype=np.float64,
            )

            U = np.asarray(
                saved["U"],
                dtype=np.float64,
            )

            mc = np.asarray(
                saved["mc_probabilities"],
                dtype=np.float32,
            )

            feature_arrays = {
                feature:
                    np.asarray(
                        saved[feature],
                        dtype=np.float64,
                    )
                for feature
                in FEATURES
            }

            if probs.shape != (N, K):
                raise ValueError(
                    f"{name}: saved probability shape mismatch"
                )

            if mc.shape != (
                MC_PASSES,
                N,
                K,
            ):
                raise ValueError(
                    f"{name}: saved MC shape mismatch"
                )

            for values in (
                probs,
                losses,
                confidence,
                Q,
                U,
                mc,
            ):
                if not np.isfinite(
                    values
                ).all():
                    raise ValueError(
                        f"{name}: saved artifact contains NaN/Inf"
                    )

        else:
            normalized = np.empty(
                (N, 12, 1000),
                dtype=np.float32,
            )

            feature_arrays = {
                feature: np.empty(N, dtype=np.float64)
                for feature in FEATURES
            }

            for i in range(N):
                y = corrupt(
                    raw[i].astype(
                        np.float64,
                        copy=False,
                    ),
                    int(ecg_ids[i]),
                    condition,
                )

                feats = waveform_badness_features(
                    y,
                    fs=100.0,
                )

                for feature in FEATURES:
                    feature_arrays[feature][i] = feats[feature]

                normalized[i] = (
                    (y - mean) / std
                ).astype(np.float32)

                if (i + 1) % 500 == 0 or i + 1 == N:
                    print(
                        f"    corrupt/Q {i + 1}/{N}"
                    )

            feature_pct = {
                feature: midrank_ecdf(
                    np.asarray(
                        train_q_ref[
                            f"{feature}_reference"
                        ],
                        dtype=np.float64,
                    ),
                    feature_arrays[feature],
                )
                for feature in FEATURES
            }

            Q_bad = np.mean(
                np.column_stack(
                    [
                        feature_pct[f]
                        for f in FEATURES
                    ]
                ),
                axis=1,
                dtype=np.float64,
            )

            Q = 1.0 - Q_bad

            probs = deterministic_probs(
                model,
                normalized,
                device,
            )

            preds = (
                probs
                >= thresholds[None, :]
            ).astype(np.int64)

            losses = sample_hamming_error(
                targets,
                preds,
            )

            confidence = mean_bernoulli_confidence(
                probs
            )

            mc = dropout_probs(
                model,
                normalized,
                device,
            )

            U = np.asarray(
                mc_uncertainty(mc)[
                    "mean_mutual_information"
                ],
                dtype=np.float64,
            )

            assert_same_state(
                original_state,
                model,
            )

        perf = performance(
            targets,
            probs,
            preds,
        )

        L_pct = midrank_ecdf(
            np.asarray(
                combo_ref[
                    "L_badness_reference"
                ],
                dtype=np.float64,
            ),
            1.0 - L,
        )

        Q_pct = midrank_ecdf(
            np.asarray(
                combo_ref[
                    "Q_signal_badness_reference"
                ],
                dtype=np.float64,
            ),
            1.0 - Q,
        )

        U_pct = midrank_ecdf(
            np.asarray(
                combo_ref[
                    "U_badness_reference"
                ],
                dtype=np.float64,
            ),
            U,
        )

        selectors = {
            "confidence":
                confidence,

            "L":
                L,

            "Q_signal":
                Q,

            "U":
                -U,

            "L+Q_signal":
                -(L_pct + Q_pct) / 2.0,

            "L+U":
                -(L_pct + U_pct) / 2.0,

            "Q_signal+U":
                -(Q_pct + U_pct) / 2.0,

            "L+Q_signal+U":
                -(L_pct + Q_pct + U_pct) / 3.0,
        }

        aurc = {}

        for method, reliability in selectors.items():
            value = float(
                grouped_risk_coverage(
                    reliability,
                    losses,
                )["aurc"]
            )

            aurc[method] = {
                "aurc":
                    value,

                "delta_vs_clean":
                    value
                    - clean["aurc"][method],
            }

        condition_dir = OUT / name
        condition_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifact = (
            condition_dir
            / "results.npz"
        )

        np.savez_compressed(
            artifact,
            ecg_ids=ecg_ids,
            targets=targets.astype(np.int8),
            probabilities=probs,
            predictions=preds.astype(np.int8),
            hamming_error=losses.astype(np.float32),
            confidence=confidence.astype(np.float32),
            Q_signal=Q.astype(np.float32),
            U=U.astype(np.float32),
            mc_probabilities=mc,
            **{
                feature:
                    feature_arrays[
                        feature
                    ].astype(np.float32)
                for feature in FEATURES
            },
        )

        full_risk = float(losses.mean())
        mean_conf = float(confidence.mean())
        mean_q = float(Q.mean())
        mean_u = float(U.mean())

        result = {
            "condition":
                condition,

            "classification":
                perf,

            "full_hamming_risk":
                full_risk,

            "delta_full_hamming_risk_vs_clean":
                full_risk
                - clean[
                    "full_hamming_risk"
                ],

            "mean_confidence":
                mean_conf,

            "delta_mean_confidence_vs_clean":
                mean_conf
                - clean[
                    "mean_confidence"
                ],

            "mean_Q_signal":
                mean_q,

            "delta_mean_Q_signal_vs_clean":
                mean_q
                - clean[
                    "mean_Q_signal"
                ],

            "mean_U":
                mean_u,

            "delta_mean_U_vs_clean":
                mean_u
                - clean[
                    "mean_U"
                ],

            "delta_macro_auroc_vs_clean":
                perf["macro_auroc"]
                - clean["macro_auroc"],

            "delta_macro_average_precision_vs_clean":
                perf[
                    "macro_average_precision"
                ]
                - clean[
                    "macro_average_precision"
                ],

            "delta_macro_f1_vs_clean":
                perf["macro_f1"]
                - clean["macro_f1"],

            "selective_prediction":
                aurc,

            "results_npz_sha256":
                sha256_file(
                    artifact
                ),
        }

        all_results[name] = result

        rows.append({
            "condition":
                name,

            "corruption_type":
                condition["type"],

            "severity":
                condition["severity"],

            "macro_auroc":
                perf["macro_auroc"],

            "macro_average_precision":
                perf[
                    "macro_average_precision"
                ],

            "macro_f1":
                perf["macro_f1"],

            "full_hamming_risk":
                full_risk,

            "mean_confidence":
                mean_conf,

            "mean_Q_signal":
                mean_q,

            "mean_U":
                mean_u,

            **{
                f"aurc_{m}":
                    aurc[m]["aurc"]
                for m in METHODS
            },
        })

        print(
            f"    AUROC={perf['macro_auroc']:.6f} "
            f"F1={perf['macro_f1']:.6f} "
            f"risk={full_risk:.6f} "
            f"Q={mean_q:.6f} "
            f"U={mean_u:.6f}"
        )

    table = pd.DataFrame(rows)

    table.to_csv(
        TABLE,
        index=False,
        float_format="%.17g",
    )

    severity_order = [
        "mild",
        "moderate",
        "severe",
    ]

    monotonicity = {}

    for kind in (
        "baseline_wander",
        "additive_white_noise",
        "amplitude_clipping",
        "lead_masking",
    ):
        subset = (
            table.loc[
                table[
                    "corruption_type"
                ].eq(kind)
            ]
            .set_index("severity")
            .loc[severity_order]
        )

        q = subset[
            "mean_Q_signal"
        ].to_numpy(np.float64)

        u = subset[
            "mean_U"
        ].to_numpy(np.float64)

        c = subset[
            "mean_confidence"
        ].to_numpy(np.float64)

        r = subset[
            "full_hamming_risk"
        ].to_numpy(np.float64)

        monotonicity[kind] = {
            "Q_nonincreasing_with_severity":
                bool(
                    np.all(
                        np.diff(q) <= 0
                    )
                ),

            "U_nondecreasing_with_severity":
                bool(
                    np.all(
                        np.diff(u) >= 0
                    )
                ),

            "confidence_nonincreasing_with_severity":
                bool(
                    np.all(
                        np.diff(c) <= 0
                    )
                ),

            "risk_nondecreasing_with_severity":
                bool(
                    np.all(
                        np.diff(r) >= 0
                    )
                ),
        }

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "corruption_id":
            config[
                "corruption_id"
            ],

        "evaluation_split":
            "validation",

        "records":
            N,

        "conditions":
            12,

        "mc_passes_per_condition":
            MC_PASSES,

        "test_records_used":
            0,

        "clean_reference":
            clean,

        "condition_results":
            all_results,

        "severity_monotonicity":
            monotonicity,

        "source_sha256": {
            "config":
                sha256_file(CONFIG),

            "checkpoint":
                sha256_file(
                    CHECKPOINT
                ),

            "normalization":
                sha256_file(NORM),

            "manifest":
                sha256_file(
                    MANIFEST
                ),

            "thresholds":
                sha256_file(
                    THRESHOLDS
                ),

            "L":
                sha256_file(
                    L_PATH
                ),

            "training_Q_reference":
                sha256_file(
                    Q_TRAIN_REF
                ),

            "combination_reference":
                sha256_file(
                    COMBO_REF
                ),

            "script":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),
        },

        "condition_table_sha256":
            sha256_file(TABLE),

        "model_state_unchanged":
            True,

        "status":
            "FROZEN",
    }

    with SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    print(
        "\nCORRUPTION VALIDATION COMPLETE"
    )
    print(
        f"Summary: {SUMMARY}"
    )
    print(
        f"Table:   {TABLE}"
    )


if __name__ == "__main__":
    main()
