#!/usr/bin/env python3
"""Inference for extra seeds, nested lead masking and recorded noise.

Same per-condition results.npz arrays as the v1 final evaluation, plus a
clean-validation artifact for thresholds, temperatures and the stacker.

Condition sets
--------------
validation   clean validation split (always needed by the post-hoc script)
frozen       clean test + the 12 frozen v1 corruption conditions
nested       lead masking with nested 1 -> 3 -> 6 lead masks
recorded     MIT-BIH NSTDB baseline wander (bw), electrode motion (em)
             and muscle artifact (ma) at 18, 12 and 6 dB SNR

Examples
--------
python scripts/extended/run_extended_inference.py --dataset ptbxl \
    --seed 20260826 --conditions validation nested recorded
python scripts/extended/run_extended_inference.py --dataset ptbxl \
    --seed 20260827 --conditions validation frozen
python scripts/extended/run_extended_inference.py --dataset chapman \
    --conditions validation nested recorded

Completed artifacts are reused on rerun.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
import yaml
from scipy.signal import resample_poly

from ptbxl_reliability.corruptions import (
    add_recorded_noise,
    mask_leads_nested,
)
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.model import ResNet1DWang
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.selective_prediction import (
    mean_bernoulli_confidence,
    sample_hamming_error,
)
from ptbxl_reliability.uncertainty import mc_uncertainty
from ptbxl_reliability.validation_metrics import (
    select_classwise_f1_thresholds,
)
from ptbxl_reliability.waveform_signal_quality import (
    waveform_badness_features,
)


PTB_FINAL_SCRIPT = PROJECT_ROOT / "scripts" / "run_final_test_evaluation.py"
CORRUPTION_CONFIG_PATH = PROJECT_ROOT / "configs" / "corruption_v1.yaml"
EXTENDED_CONFIG_PATH = PROJECT_ROOT / "configs" / "extended_v1.yaml"

PTB_BASELINE_DIR = PROJECT_ROOT / "results" / "baseline" / "v1"
PTB_Q_REFERENCE = (
    PROJECT_ROOT / "data" / "processed" / "waveform_signal_quality"
    / "v1" / "training_reference.npz"
)

CHAPMAN_ROOT = PROJECT_ROOT / "results" / "external_validation" / "chapman" / "v1"
CHAPMAN_LABELS = ["SB", "AFIB", "ST", "TWC", "LVHV"]

NSTDB_ROOT = PROJECT_ROOT / "data" / "raw" / "nstdb" / "1.0.0"
NSTDB_FS = 360

EXTENDED_ROOT = PROJECT_ROOT / "results" / "extended" / "v1"

PRIMARY_SEED = 20260826
MC_PASSES = 30
FS = 100.0

Q_FEATURES = (
    "baseline_burden",
    "high_frequency_burden",
    "lead_dropout_burden",
    "clipping_burden",
)


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


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------

class Split:
    """Raw 100-Hz mV waveforms and identifiers for one data split."""

    def __init__(self, raw, targets, record_ids, cluster_ids):
        self.raw = raw
        self.targets = targets
        self.record_ids = record_ids
        self.cluster_ids = cluster_ids


def ptbxl_split(helpers, split: str) -> Split:
    manifest = pd.read_csv(helpers.MANIFEST_PATH)
    part = manifest.loc[manifest["split"].eq(split)].reset_index(drop=True)
    expected = {"validation": 2146, "test": 2158}[split]
    if len(part) != expected:
        raise ValueError(f"Expected {expected} {split} ECGs, found {len(part)}")

    raw = np.empty((len(part), 12, 1000), dtype=np.float32)
    for i, row in enumerate(part.itertuples(index=False)):
        signal, fields = wfdb.rdsamp(str(helpers.RAW_ROOT / str(row.filename_lr)))
        if signal.shape != (1000, 12) or float(fields["fs"]) != FS:
            raise ValueError(f"ECG {row.ecg_id}: unexpected shape or fs")
        if list(fields["sig_name"]) != helpers.EXPECTED_LEADS:
            raise ValueError(f"ECG {row.ecg_id}: lead-order mismatch")
        raw[i] = signal.T.astype(np.float32)
        if (i + 1) % 500 == 0 or i + 1 == len(part):
            print(f"  raw {split} {i + 1}/{len(part)}", flush=True)

    return Split(
        raw=raw,
        targets=part[list(SUPERCLASS_ORDER)].to_numpy(dtype=np.int64),
        record_ids=part["ecg_id"].to_numpy(dtype=np.int64),
        cluster_ids=part["patient_id"].to_numpy(dtype=np.int64),
    )


def chapman_split(split: str) -> Split:
    manifest = pd.read_csv(CHAPMAN_ROOT / "chapman_frozen_cohort.csv")
    cache = np.load(CHAPMAN_ROOT / "chapman_100hz_mV.npy", mmap_mode="r")
    index = np.flatnonzero(manifest["split"].eq(split).to_numpy())
    if len(index) != 1065:
        raise ValueError(f"Expected 1065 {split} ECGs, found {len(index)}")

    # Same deterministic integer identifiers as run_chapman_final_test.py.
    record_ids = (index + 1).astype(np.int64)

    return Split(
        raw=np.asarray(cache[index], dtype=np.float32),
        targets=manifest.iloc[index][CHAPMAN_LABELS].to_numpy(dtype=np.int64),
        record_ids=record_ids,
        cluster_ids=record_ids,
    )


def dataset_paths(dataset: str, seed: int) -> dict[str, Path]:
    if dataset == "ptbxl":
        seed_dir = PTB_BASELINE_DIR / f"seed_{seed}"
        return {
            "checkpoint": seed_dir / "best_checkpoint.pt",
            "thresholds": seed_dir / "validation_thresholds.json",
            "q_reference": PTB_Q_REFERENCE,
        }
    if seed != PRIMARY_SEED:
        raise ValueError(
            "Only the primary Chapman checkpoint exists; retrain with "
            "scripts/chapman/train_chapman.py before evaluating other seeds"
        )
    model_dir = CHAPMAN_ROOT / f"model_seed_{seed}"
    return {
        "checkpoint": model_dir / "best_checkpoint.pt",
        "thresholds": model_dir / "validation_thresholds.json",
        "q_reference": CHAPMAN_ROOT / "chapman_q_training_reference.npz",
    }


def load_normalization(dataset: str, helpers) -> tuple[float, float]:
    path = (
        helpers.NORM_PATH if dataset == "ptbxl"
        else CHAPMAN_ROOT / "chapman_normalization.json"
    )
    norm = load_json(path)
    return float(norm["mean_mV"]), float(norm["std_mV"])


def frozen_thresholds(dataset: str, path: Path) -> np.ndarray:
    data = load_json(path)
    if dataset == "ptbxl":
        return np.asarray(
            [data["thresholds"][label]["threshold"] for label in SUPERCLASS_ORDER],
            dtype=np.float64,
        )
    return np.asarray(
        [data[label]["threshold"] for label in CHAPMAN_LABELS],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def load_nstdb() -> dict[str, np.ndarray]:
    """bw/em/ma noise records resampled from 360 Hz to 100 Hz, in mV."""
    bank = {}
    for name in ("bw", "em", "ma"):
        path = NSTDB_ROOT / name
        if not path.with_suffix(".hea").exists():
            raise FileNotFoundError(
                f"{path}.hea not found. Download with:\n"
                "  python -c \"import wfdb; wfdb.dl_database('nstdb', "
                "'data/raw/nstdb/1.0.0', records=['bw','em','ma'])\""
            )
        signal, fields = wfdb.rdsamp(str(path))
        if int(fields["fs"]) != NSTDB_FS or signal.shape[1] != 2:
            raise ValueError(f"Unexpected NSTDB record format: {name}")
        bank[name] = resample_poly(signal.T, up=5, down=18, axis=1)
    return bank


def condition_list(requested: list[str], helpers) -> list[dict | None]:
    extended = yaml.safe_load(EXTENDED_CONFIG_PATH.read_text())
    conditions: list[dict | None] = []

    if "frozen" in requested:
        with CORRUPTION_CONFIG_PATH.open() as f:
            frozen = yaml.safe_load(f)
        conditions.append(None)
        conditions.extend(helpers.corruption_conditions(frozen))

    if "nested" in requested:
        for level, k in extended["lead_masking_nested"]["severity"].items():
            conditions.append({
                "name": f"lead_masking_nested_{level}",
                "type": "lead_masking_nested",
                "number_of_leads": int(k["number_of_leads"]),
            })

    if "recorded" in requested:
        for noise in extended["recorded_noise"]["records"]:
            for index, (level, spec) in enumerate(
                extended["recorded_noise"]["severity"].items()
            ):
                conditions.append({
                    "name": f"nstdb_{noise}_{level}",
                    "type": "recorded_noise",
                    "noise": noise,
                    "snr_db": float(spec["snr_db"]),
                    "severity_index": index,
                })

    return conditions


def corrupt(x, record_id, condition, helpers, nstdb):
    if condition is None:
        return x
    if condition["type"] == "lead_masking_nested":
        return mask_leads_nested(
            x,
            ecg_id=record_id,
            number_of_leads=condition["number_of_leads"],
        )[0]
    if condition["type"] == "recorded_noise":
        return add_recorded_noise(
            x,
            ecg_id=record_id,
            noise=nstdb[condition["noise"]],
            noise_name=condition["noise"],
            snr_db=condition["snr_db"],
            severity_index=condition["severity_index"],
        )
    return helpers.apply_corruption(x, record_id, condition)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(split, condition, *, helpers, model, device, mean, std,
             thresholds, q_reference, nstdb):
    n = len(split.raw)
    normalized = np.empty((n, 12, 1000), dtype=np.float32)
    features = {name: np.empty(n, dtype=np.float64) for name in Q_FEATURES}

    for i in range(n):
        x = corrupt(
            split.raw[i].astype(np.float64, copy=False),
            int(split.record_ids[i]),
            condition,
            helpers,
            nstdb,
        )
        values = waveform_badness_features(x, fs=FS)
        for name in Q_FEATURES:
            features[name][i] = values[name]
        normalized[i] = ((x - mean) / std).astype(np.float32)
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"    preprocess/Q {i + 1}/{n}", flush=True)

    Q, _ = helpers.q_from_feature_arrays(features, q_reference)
    probabilities = helpers.deterministic_probabilities(
        model, normalized, device,
    ).astype(np.float64)
    mc = helpers.mc_probabilities(
        model, normalized, device, passes=MC_PASSES, progress=True,
    )
    U = np.asarray(mc_uncertainty(mc)["mean_mutual_information"], dtype=np.float64)

    arrays = {
        "record_ids": split.record_ids,
        "cluster_ids": split.cluster_ids,
        "targets": split.targets.astype(np.int8),
        "probabilities": probabilities.astype(np.float32),
        "confidence": mean_bernoulli_confidence(probabilities),
        "Q_signal": Q.astype(np.float64),
        "U": U,
        "mc_probabilities": mc.astype(np.float32),
        **{name: features[name] for name in Q_FEATURES},
    }

    if thresholds is not None:
        predictions = (probabilities >= thresholds[None, :]).astype(np.int64)
        arrays["predictions"] = predictions.astype(np.int8)
        arrays["hamming_error"] = sample_hamming_error(split.targets, predictions)

    return arrays


def load_model(path: Path, device: torch.device) -> ResNet1DWang:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ResNet1DWang()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ptbxl", "chapman"), required=True)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("validation", "frozen", "nested", "recorded"),
        required=True,
    )
    args = parser.parse_args()

    helpers = load_ptb_helpers()
    paths = dataset_paths(args.dataset, args.seed)
    for path in paths.values():
        if not path.exists() and path.name != "validation_thresholds.json":
            raise FileNotFoundError(path)

    out_dir = EXTENDED_ROOT / args.dataset / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    model = load_model(paths["checkpoint"], device)
    mean, std = load_normalization(args.dataset, helpers)
    q_reference = np.load(paths["q_reference"], allow_pickle=False)

    def get_split(split):
        return (
            ptbxl_split(helpers, split) if args.dataset == "ptbxl"
            else chapman_split(split)
        )

    common = dict(
        helpers=helpers, model=model, device=device, mean=mean, std=std,
        q_reference=q_reference,
    )

    # Clean validation; non-primary seeds get their thresholds here.
    thresholds_out = out_dir / "thresholds.json"
    validation_artifact = out_dir / "validation_clean" / "results.npz"

    if "validation" in args.conditions and not validation_artifact.exists():
        print("\n[validation_clean]")
        arrays = evaluate(get_split("validation"), None, thresholds=None,
                          nstdb=None, **common)
        if paths["thresholds"].exists():
            thresholds = frozen_thresholds(args.dataset, paths["thresholds"])
            source = str(paths["thresholds"].relative_to(PROJECT_ROOT))
        else:
            fits = select_classwise_f1_thresholds(
                arrays["targets"].astype(np.int64),
                arrays["probabilities"].astype(np.float64),
            )
            thresholds = np.asarray([fit["threshold"] for fit in fits])
            source = "fitted on clean validation (max F1 per class)"
        predictions = (arrays["probabilities"] >= thresholds[None, :]).astype(np.int64)
        arrays["predictions"] = predictions.astype(np.int8)
        arrays["hamming_error"] = sample_hamming_error(
            arrays["targets"].astype(np.int64), predictions,
        )
        validation_artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(validation_artifact, **arrays)
        thresholds_out.write_text(json.dumps({
            "thresholds": thresholds.tolist(),
            "source": source,
            "seed": args.seed,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        print("Saved:", validation_artifact)

    if not thresholds_out.exists():
        raise FileNotFoundError(
            "Run with --conditions validation first so thresholds are frozen"
        )
    thresholds = np.asarray(load_json(thresholds_out)["thresholds"], dtype=np.float64)

    test_sets = [c for c in args.conditions if c != "validation"]
    if not test_sets:
        return

    conditions = condition_list(test_sets, helpers)
    nstdb = load_nstdb() if "recorded" in test_sets else None
    test = get_split("test")

    for index, condition in enumerate(conditions, start=1):
        name = "clean" if condition is None else condition["name"]
        artifact = out_dir / name / "results.npz"
        print(f"\n[{index:02d}/{len(conditions)}] {name}")
        if artifact.exists():
            print("    reusing saved artifact")
            continue
        arrays = evaluate(test, condition, thresholds=thresholds,
                          nstdb=nstdb, **common)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact, **arrays)
        print("    saved:", artifact)


if __name__ == "__main__":
    main()
