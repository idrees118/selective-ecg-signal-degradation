#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ptbxl_reliability.model import ResNet1DWang
from ptbxl_reliability.selective_prediction import (
    grouped_risk_coverage,
    mean_bernoulli_confidence,
    midrank_ecdf,
    sample_hamming_error,
)
from ptbxl_reliability.uncertainty import mc_uncertainty
from ptbxl_reliability.waveform_signal_quality import waveform_badness_features


PROJECT = Path.cwd()
ROOT = PROJECT / "results" / "external_validation" / "chapman" / "v1"
MODEL_DIR = ROOT / "model_seed_20260826"

MANIFEST_PATH = ROOT / "chapman_frozen_cohort.csv"
CACHE_PATH = ROOT / "chapman_100hz_mV.npy"
NORM_PATH = ROOT / "chapman_normalization.json"
CHECKPOINT_PATH = MODEL_DIR / "best_checkpoint.pt"
THRESHOLDS_PATH = MODEL_DIR / "validation_thresholds.json"
Q_REFERENCE_PATH = ROOT / "chapman_q_training_reference.npz"
COMBINATION_REFERENCE_PATH = ROOT / "chapman_combination_reference.npz"
PROTOCOL_PATH = ROOT / "chapman_test_protocol.json"

PTB_FINAL_SCRIPT = PROJECT / "scripts" / "run_final_test_evaluation.py"
CORRUPTION_CONFIG_PATH = PROJECT / "configs" / "corruption_v1.yaml"

OUT_DIR = ROOT / "final_test"
TABLE_PATH = OUT_DIR / "condition_metrics.csv"
SUMMARY_PATH = OUT_DIR / "summary.json"

LABELS = ["SB", "AFIB", "ST", "TWC", "LVHV"]
Q_FEATURES = (
    "baseline_burden",
    "high_frequency_burden",
    "lead_dropout_burden",
    "clipping_burden",
)
BATCH_SIZE = 128
MC_PASSES = 30


def load_ptb_helpers():
    spec = importlib.util.spec_from_file_location(
        "ptb_final_helpers",
        PTB_FINAL_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load PTB final-evaluation helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_metrics(targets, probs, preds):
    aucs, aps, f1s = [], [], []
    per_class = {}
    for k, label in enumerate(LABELS):
        auc = float(roc_auc_score(targets[:, k], probs[:, k]))
        ap = float(average_precision_score(targets[:, k], probs[:, k]))
        f1 = float(f1_score(targets[:, k], preds[:, k], zero_division=0))
        aucs.append(auc)
        aps.append(ap)
        f1s.append(f1)
        per_class[label] = {
            "auroc": auc,
            "average_precision": ap,
            "f1": f1,
        }
    return {
        "macro_auroc": float(np.mean(aucs)),
        "macro_average_precision": float(np.mean(aps)),
        "macro_f1": float(np.mean(f1s)),
        "classwise": per_class,
    }


def summarize(arrays, combo_ref):
    targets = arrays["targets"].astype(np.int64)
    probs = arrays["probabilities"].astype(np.float64)
    preds = arrays["predictions"].astype(np.int64)
    losses = arrays["hamming_error"].astype(np.float64)
    confidence = arrays["confidence"].astype(np.float64)
    Q = arrays["Q_signal"].astype(np.float64)
    U = arrays["U"].astype(np.float64)

    q_pct = midrank_ecdf(
        np.asarray(combo_ref["Q_signal_badness_reference"], dtype=np.float64),
        1.0 - Q,
    )
    u_pct = midrank_ecdf(
        np.asarray(combo_ref["U_badness_reference"], dtype=np.float64),
        U,
    )

    selectors = {
        "confidence": confidence,
        "Q": Q,
        "U": -U,
        "Q+U": -(q_pct + u_pct) / 2.0,
    }

    aurc = {}
    for name, selector in selectors.items():
        curve = grouped_risk_coverage(selector, losses)
        aurc[name] = float(curve["aurc"])

    return {
        "classification": class_metrics(targets, probs, preds),
        "full_hamming_risk": float(np.mean(losses)),
        "mean_confidence": float(np.mean(confidence)),
        "mean_Q": float(np.mean(Q)),
        "mean_U": float(np.mean(U)),
        "aurc": aurc,
    }


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main():
    required = [
        MANIFEST_PATH, CACHE_PATH, NORM_PATH, CHECKPOINT_PATH,
        THRESHOLDS_PATH, Q_REFERENCE_PATH, COMBINATION_REFERENCE_PATH,
        PROTOCOL_PATH, PTB_FINAL_SCRIPT, CORRUPTION_CONFIG_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    with PROTOCOL_PATH.open() as f:
        protocol = json.load(f)
    if protocol.get("status") != "FROZEN":
        raise ValueError("Chapman test protocol is not FROZEN")
    if protocol.get("test_tuning_allowed") is not False:
        raise ValueError("Test tuning must be forbidden")
    if protocol.get("primary_comparisons") != [
        "U_vs_confidence",
        "Q+U_vs_confidence",
    ]:
        raise ValueError("Primary comparison list changed")

    manifest = pd.read_csv(MANIFEST_PATH)
    cache = np.load(CACHE_PATH, mmap_mode="r")

    with NORM_PATH.open() as f:
        norm = json.load(f)
    mean = float(norm["mean_mV"])
    std = float(norm["std_mV"])

    with THRESHOLDS_PATH.open() as f:
        threshold_json = json.load(f)
    thresholds = np.asarray(
        [threshold_json[label]["threshold"] for label in LABELS],
        dtype=np.float64,
    )

    test_idx = np.flatnonzero(manifest["split"].eq("test").to_numpy())
    if len(test_idx) != 1065:
        raise ValueError(f"Expected 1065 test ECGs, found {len(test_idx)}")

    test_manifest = manifest.iloc[test_idx].reset_index(drop=True)
    targets = test_manifest[LABELS].to_numpy(dtype=np.int64)

    # Stable deterministic integer identifiers for corruption RNG.
    record_ids = (test_idx + 1).astype(np.int64)

    raw_test = np.asarray(cache[test_idx], dtype=np.float32)

    q_ref = np.load(Q_REFERENCE_PATH, allow_pickle=False)
    combo_ref = np.load(COMBINATION_REFERENCE_PATH, allow_pickle=False)

    helpers = load_ptb_helpers()

    with CORRUPTION_CONFIG_PATH.open() as f:
        corruption_config = yaml.safe_load(f)
    conditions = [None] + helpers.corruption_conditions(corruption_config)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model = ResNet1DWang()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    all_summaries = {}

    for condition in conditions:
        name = "clean" if condition is None else condition["name"]
        artifact = OUT_DIR / name / "results.npz"

        print("\n" + "=" * 68)
        print("CONDITION:", name)
        print("=" * 68)

        if artifact.exists():
            print("Reusing saved artifact:", artifact)
            saved = np.load(artifact, allow_pickle=False)
            arrays = {k: np.asarray(saved[k]) for k in saved.files}
        else:
            n = len(test_idx)
            normalized = np.empty((n, 12, 1000), dtype=np.float32)
            feature_arrays = {
                feature: np.empty(n, dtype=np.float64)
                for feature in Q_FEATURES
            }

            for i in range(n):
                x = raw_test[i].astype(np.float64, copy=False)

                if condition is not None:
                    x = helpers.apply_corruption(
                        x,
                        int(record_ids[i]),
                        condition,
                    )

                features = waveform_badness_features(x, fs=100.0)
                for feature in Q_FEATURES:
                    feature_arrays[feature][i] = features[feature]

                normalized[i] = ((x - mean) / std).astype(np.float32)

                if (i + 1) % 250 == 0 or i + 1 == n:
                    print(f"  preprocess/Q {i+1}/{n}", flush=True)

            Q, _ = helpers.q_from_feature_arrays(feature_arrays, q_ref)

            probs = helpers.deterministic_probabilities(
                model,
                normalized,
                device,
            ).astype(np.float64)

            preds = (probs >= thresholds[None, :]).astype(np.int64)
            confidence = mean_bernoulli_confidence(probs)

            mc = helpers.mc_probabilities(
                model,
                normalized,
                device,
                passes=MC_PASSES,
                progress=True,
            )

            uncertainty = mc_uncertainty(mc)
            U = np.asarray(
                uncertainty["mean_mutual_information"],
                dtype=np.float64,
            )

            losses = sample_hamming_error(targets, preds)

            arrays = {
                "record_ids": record_ids,
                "manifest_indices": test_idx.astype(np.int64),
                "targets": targets.astype(np.int8),
                "probabilities": probs.astype(np.float32),
                "predictions": preds.astype(np.int8),
                "hamming_error": losses.astype(np.float64),
                "confidence": confidence.astype(np.float64),
                "Q_signal": Q.astype(np.float64),
                "U": U.astype(np.float64),
                **{
                    feature: feature_arrays[feature].astype(np.float64)
                    for feature in Q_FEATURES
                },
            }
            save_npz(artifact, arrays)
            print("Saved:", artifact)

        summary = summarize(arrays, combo_ref)
        all_summaries[name] = summary

        c = summary["classification"]
        a = summary["aurc"]

        row = {
            "condition": name,
            "macro_auroc": c["macro_auroc"],
            "macro_average_precision": c["macro_average_precision"],
            "macro_f1": c["macro_f1"],
            "hamming_risk": summary["full_hamming_risk"],
            "mean_confidence": summary["mean_confidence"],
            "mean_Q": summary["mean_Q"],
            "mean_U": summary["mean_U"],
            "aurc_confidence": a["confidence"],
            "aurc_Q": a["Q"],
            "aurc_U": a["U"],
            "aurc_Q_plus_U": a["Q+U"],
            "delta_U_minus_confidence": a["U"] - a["confidence"],
            "delta_Q_plus_U_minus_confidence": a["Q+U"] - a["confidence"],
        }
        rows.append(row)

        print(
            f"macro-AUROC={row['macro_auroc']:.6f} | "
            f"Hamming={row['hamming_risk']:.6f}"
        )
        print(
            "AURC "
            f"C={row['aurc_confidence']:.6f} | "
            f"Q={row['aurc_Q']:.6f} | "
            f"U={row['aurc_U']:.6f} | "
            f"Q+U={row['aurc_Q_plus_U']:.6f}"
        )

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_PATH, index=False)

    output = {
        "status": "FROZEN_TEST_RESULTS",
        "dataset": "Chapman-Shaoxing ECG",
        "records": int(len(test_idx)),
        "labels": LABELS,
        "primary_comparisons": [
            "U_vs_confidence",
            "Q+U_vs_confidence",
        ],
        "descriptive_selector": "Q",
        "conditions": all_summaries,
    }
    with SUMMARY_PATH.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print("\n" + "=" * 68)
    print("CHAPMAN FINAL TEST COMPLETE")
    print("=" * 68)
    print(
        table[
            [
                "condition",
                "macro_auroc",
                "hamming_risk",
                "aurc_confidence",
                "aurc_Q",
                "aurc_U",
                "aurc_Q_plus_U",
                "delta_U_minus_confidence",
                "delta_Q_plus_U_minus_confidence",
            ]
        ].to_string(index=False)
    )
    print("\nSaved:", TABLE_PATH)
    print("Saved:", SUMMARY_PATH)
    print("\nTEST DEVELOPMENT IS CLOSED. Do not retune after these results.")


if __name__ == "__main__":
    main()
