from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from ptbxl_reliability.model import ResNet1DWang
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


ROOT = Path(
    "results/external_validation/chapman/v1"
)

MODEL_DIR = ROOT / "model_seed_20260826"

MANIFEST = ROOT / "chapman_frozen_cohort.csv"
CACHE = ROOT / "chapman_100hz_mV.npy"
NORM = ROOT / "chapman_normalization.json"

CHECKPOINT = MODEL_DIR / "best_checkpoint.pt"
VAL_PREDICTIONS = MODEL_DIR / "best_validation_predictions.npz"
THRESHOLDS = MODEL_DIR / "validation_thresholds.json"

Q_REFERENCE_OUT = ROOT / "chapman_q_training_reference.npz"
COMBINATION_OUT = ROOT / "chapman_combination_reference.npz"
SUMMARY_OUT = ROOT / "chapman_validation_reliability_summary.json"

LABELS = ["SB", "AFIB", "ST", "TWC", "LVHV"]

Q_FEATURES = (
    "baseline_burden",
    "high_frequency_burden",
    "lead_dropout_burden",
    "clipping_burden",
)

MC_PASSES = 30
BATCH_SIZE = 128


# ----------------------------------------------------------
# Inputs
# ----------------------------------------------------------

df = pd.read_csv(MANIFEST)

raw = np.load(
    CACHE,
    mmap_mode="r",
)

with NORM.open() as f:
    norm = json.load(f)

mean = float(norm["mean_mV"])
std = float(norm["std_mV"])

train_idx = np.flatnonzero(
    df["split"].eq("train").to_numpy()
)

val_idx = np.flatnonzero(
    df["split"].eq("validation").to_numpy()
)

assert len(train_idx) == 8516
assert len(val_idx) == 1065


# ----------------------------------------------------------
# 1. Training-derived waveform-quality references
# ----------------------------------------------------------

print("Computing training Q references...")

training_features = {
    name: np.empty(
        len(train_idx),
        dtype=np.float64,
    )
    for name in Q_FEATURES
}

for j, idx in enumerate(train_idx):

    x = np.asarray(
        raw[idx],
        dtype=np.float64,
    )

    features = waveform_badness_features(
        x,
        fs=100.0,
    )

    for name in Q_FEATURES:
        training_features[name][j] = features[name]

    if (
        (j + 1) % 500 == 0
        or j + 1 == len(train_idx)
    ):
        print(
            f"  train Q {j+1}/{len(train_idx)}",
            flush=True,
        )

np.savez_compressed(
    Q_REFERENCE_OUT,
    **{
        f"{name}_reference":
            training_features[name]
        for name in Q_FEATURES
    },
)


# ----------------------------------------------------------
# Helper: Q from training reference
# ----------------------------------------------------------

def compute_q(indices):

    feature_values = {
        name: np.empty(
            len(indices),
            dtype=np.float64,
        )
        for name in Q_FEATURES
    }

    for j, idx in enumerate(indices):

        x = np.asarray(
            raw[idx],
            dtype=np.float64,
        )

        features = waveform_badness_features(
            x,
            fs=100.0,
        )

        for name in Q_FEATURES:
            feature_values[name][j] = features[name]

    percentiles = []

    for name in Q_FEATURES:

        pct = midrank_ecdf(
            training_features[name],
            feature_values[name],
        )

        percentiles.append(pct)

    badness = np.mean(
        np.column_stack(percentiles),
        axis=1,
    )

    return 1.0 - badness


print("Computing validation Q...")

Q_val = compute_q(val_idx)


# ----------------------------------------------------------
# 2. Load frozen model
# ----------------------------------------------------------

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

model = ResNet1DWang()

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model = model.to(device)
model.eval()


# ----------------------------------------------------------
# Validation normalized signals
# ----------------------------------------------------------

val_raw = np.asarray(
    raw[val_idx],
    dtype=np.float32,
)

val_normalized = (
    (
        val_raw
        - np.float32(mean)
    )
    / np.float32(std)
).astype(np.float32)


# ----------------------------------------------------------
# 3. Frozen validation deterministic predictions
# ----------------------------------------------------------

saved = np.load(
    VAL_PREDICTIONS,
    allow_pickle=False,
)

targets = np.asarray(
    saved["y_true"],
    dtype=np.int64,
)

probabilities = np.asarray(
    saved["y_prob"],
    dtype=np.float64,
)

expected_targets = df.iloc[
    val_idx
][LABELS].to_numpy(
    dtype=np.int64,
)

assert np.array_equal(
    targets,
    expected_targets,
)


with THRESHOLDS.open() as f:
    threshold_json = json.load(f)

thresholds = np.asarray(
    [
        threshold_json[label]["threshold"]
        for label in LABELS
    ],
    dtype=np.float64,
)

predictions = (
    probabilities >= thresholds[None, :]
).astype(np.int64)

confidence = mean_bernoulli_confidence(
    probabilities
)

losses = sample_hamming_error(
    targets,
    predictions,
)


# ----------------------------------------------------------
# 4. Validation MC-dropout uncertainty
# ----------------------------------------------------------

print("Computing validation MC dropout...")

torch.manual_seed(20260826)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(20260826)

dropout_count = enable_mc_dropout(
    model
)

if dropout_count != 2:
    raise RuntimeError(
        f"Expected 2 Dropout modules, found {dropout_count}"
    )

if any(
    module.training
    for module in model.modules()
    if isinstance(module, nn.BatchNorm1d)
):
    raise RuntimeError(
        "BatchNorm unexpectedly entered training mode"
    )

mc = np.empty(
    (
        MC_PASSES,
        len(val_idx),
        5,
    ),
    dtype=np.float32,
)

with torch.inference_mode():

    for p in range(MC_PASSES):

        for start in range(
            0,
            len(val_idx),
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                len(val_idx),
            )

            x = torch.from_numpy(
                val_normalized[start:end]
            ).to(device)

            logits = model(x)

            mc[p, start:end] = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
            )

        if (
            (p + 1) % 5 == 0
            or p + 1 == MC_PASSES
        ):
            print(
                f"  MC {p+1:02d}/{MC_PASSES}",
                flush=True,
            )

model.eval()

uncertainty = mc_uncertainty(mc)

U_val = np.asarray(
    uncertainty["mean_mutual_information"],
    dtype=np.float64,
)


# ----------------------------------------------------------
# 5. Freeze clean-validation combination references
# ----------------------------------------------------------

np.savez_compressed(
    COMBINATION_OUT,
    Q_signal_badness_reference=(
        1.0 - Q_val
    ),
    U_badness_reference=U_val,
)


# ----------------------------------------------------------
# 6. Validation selectors / AURC sanity check
# ----------------------------------------------------------

Q_pct = midrank_ecdf(
    1.0 - Q_val,
    1.0 - Q_val,
)

U_pct = midrank_ecdf(
    U_val,
    U_val,
)

selectors = {
    "confidence": confidence,
    "Q": Q_val,
    "U": -U_val,
    "Q+U": -(
        Q_pct + U_pct
    ) / 2.0,
}

aurc = {}

for name, selector in selectors.items():

    curve = grouped_risk_coverage(
        selector,
        losses,
    )

    aurc[name] = float(
        curve["aurc"]
    )


summary = {
    "status": "FROZEN",
    "training_q_reference_records":
        int(len(train_idx)),
    "validation_reference_records":
        int(len(val_idx)),
    "mc_passes":
        MC_PASSES,
    "mean_validation_Q":
        float(Q_val.mean()),
    "mean_validation_U":
        float(U_val.mean()),
    "validation_full_hamming_risk":
        float(losses.mean()),
    "validation_aurc":
        aurc,
}

with SUMMARY_OUT.open("w") as f:
    json.dump(
        summary,
        f,
        indent=2,
    )
    f.write("\n")


print("\n" + "=" * 65)
print("CHAPMAN RELIABILITY REFERENCES FROZEN")
print("=" * 65)

print(
    "Training Q reference:",
    len(train_idx),
)

print(
    "Validation reference:",
    len(val_idx),
)

print(
    "Mean validation Q:",
    summary["mean_validation_Q"],
)

print(
    "Mean validation U:",
    summary["mean_validation_U"],
)

print(
    "Validation Hamming risk:",
    summary[
        "validation_full_hamming_risk"
    ],
)

print("\nValidation AURC:")

for name, value in aurc.items():
    print(
        f"  {name:10s}: {value:.6f}"
    )

print("\nSaved:")
print(Q_REFERENCE_OUT)
print(COMBINATION_OUT)
print(SUMMARY_OUT)

print("=" * 65)