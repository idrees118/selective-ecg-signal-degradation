from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ptbxl_reliability.model import ResNet1DWang


SEED = 20260826
EPOCHS = 50
BATCH_SIZE = 128
MAX_LR = 0.01

LABELS = ["SB", "AFIB", "ST", "TWC", "LVHV"]

ROOT = Path("results/external_validation/chapman/v1")

MANIFEST_PATH = ROOT / "chapman_frozen_cohort.csv"
CACHE_PATH = ROOT / "chapman_100hz_mV.npy"
NORM_PATH = ROOT / "chapman_normalization.json"

OUT = ROOT / "model_seed_20260826"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)


# ---------------------------------------------------------
# Load frozen artifacts
# ---------------------------------------------------------

manifest = pd.read_csv(MANIFEST_PATH)

with NORM_PATH.open() as f:
    norm = json.load(f)

mean = float(norm["mean_mV"])
std = float(norm["std_mV"])

if std <= 0:
    raise ValueError("Invalid normalization std")

cache = np.load(
    CACHE_PATH,
    mmap_mode="r",
)

if cache.shape != (10646, 12, 1000):
    raise ValueError(
        f"Unexpected cache shape: {cache.shape}"
    )

if len(manifest) != len(cache):
    raise ValueError(
        "Manifest/cache length mismatch"
    )


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class ChapmanDataset(Dataset):

    def __init__(self, split):
        self.indices = np.flatnonzero(
            manifest["split"].eq(split).to_numpy()
        )

        self.targets = manifest.loc[
            self.indices,
            LABELS,
        ].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])

        x = np.asarray(
            cache[idx],
            dtype=np.float32,
        ).copy()

        x = (
            x - np.float32(mean)
        ) / np.float32(std)

        y = self.targets[i]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
        )


train_ds = ChapmanDataset("train")
val_ds = ChapmanDataset("validation")

print("Train:", len(train_ds))
print("Validation:", len(val_ds))

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = ResNet1DWang().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=MAX_LR,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.30,
    anneal_strategy="cos",
    div_factor=25,
    final_div_factor=10000,
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validation_predictions():

    model.eval()

    ys = []
    ps = []

    with torch.no_grad():
        for x, y in val_loader:

            x = x.to(device)

            logits = model(x)

            probs = torch.sigmoid(
                logits
            ).cpu().numpy()

            ys.append(y.numpy())
            ps.append(probs)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    aurocs = []

    for k in range(len(LABELS)):
        aurocs.append(
            roc_auc_score(
                y_true[:, k],
                y_prob[:, k],
            )
        )

    macro_auroc = float(
        np.mean(aurocs)
    )

    return (
        y_true,
        y_prob,
        macro_auroc,
        aurocs,
    )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

best_auroc = -np.inf
history = []

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    seen = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()
        scheduler.step()

        n = x.shape[0]

        running_loss += (
            float(loss.item()) * n
        )

        seen += n

    train_loss = (
        running_loss / seen
    )

    (
        y_val,
        p_val,
        macro_auroc,
        class_aurocs,
    ) = validation_predictions()

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_macro_auroc":
            macro_auroc,
        **{
            f"auroc_{label}":
                float(value)
            for label, value
            in zip(
                LABELS,
                class_aurocs,
            )
        },
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"loss={train_loss:.5f} | "
        f"val AUROC={macro_auroc:.5f}",
        flush=True,
    )

    if macro_auroc > best_auroc:

        best_auroc = macro_auroc

        torch.save({
            "model_state_dict":
                model.state_dict(),
            "epoch":
                epoch,
            "validation_macro_auroc":
                macro_auroc,
            "seed":
                SEED,
            "labels":
                LABELS,
            "normalization":
                {
                    "mean_mV": mean,
                    "std_mV": std,
                },
        }, OUT / "best_checkpoint.pt")

        np.savez_compressed(
            OUT /
            "best_validation_predictions.npz",
            y_true=y_val,
            y_prob=p_val,
        )


# ---------------------------------------------------------
# Reload best checkpoint
# ---------------------------------------------------------

checkpoint = torch.load(
    OUT / "best_checkpoint.pt",
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

data = np.load(
    OUT /
    "best_validation_predictions.npz"
)

y_true = data["y_true"]
y_prob = data["y_prob"]


# ---------------------------------------------------------
# Validation thresholds: maximize classwise F1
# ---------------------------------------------------------

thresholds = {}

for k, label in enumerate(LABELS):

    candidates = np.unique(
        y_prob[:, k]
    )

    best_f1 = -1.0
    best_threshold = 0.5

    for threshold in candidates:

        pred = (
            y_prob[:, k] >= threshold
        ).astype(int)

        score = f1_score(
            y_true[:, k],
            pred,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(
                threshold
            )

    thresholds[label] = {
        "threshold":
            best_threshold,
        "validation_f1":
            float(best_f1),
    }


# ---------------------------------------------------------
# Final validation metrics
# ---------------------------------------------------------

class_metrics = {}
aps = []
f1s = []
aurocs = []

for k, label in enumerate(LABELS):

    threshold = thresholds[
        label
    ]["threshold"]

    pred = (
        y_prob[:, k] >= threshold
    ).astype(int)

    auc = roc_auc_score(
        y_true[:, k],
        y_prob[:, k],
    )

    ap = average_precision_score(
        y_true[:, k],
        y_prob[:, k],
    )

    f1 = f1_score(
        y_true[:, k],
        pred,
        zero_division=0,
    )

    aurocs.append(auc)
    aps.append(ap)
    f1s.append(f1)

    class_metrics[label] = {
        "auroc": float(auc),
        "average_precision":
            float(ap),
        "f1": float(f1),
        "threshold":
            float(threshold),
    }


summary = {
    "status": "FROZEN",
    "dataset":
        "Chapman-Shaoxing ECG",
    "seed":
        SEED,
    "best_epoch":
        int(checkpoint["epoch"]),
    "best_validation_macro_auroc":
        float(np.mean(aurocs)),
    "validation_macro_average_precision":
        float(np.mean(aps)),
    "validation_macro_f1":
        float(np.mean(f1s)),
    "labels":
        LABELS,
    "class_metrics":
        class_metrics,
}

with (
    OUT / "validation_summary.json"
).open("w") as f:

    json.dump(
        summary,
        f,
        indent=2,
    )

with (
    OUT / "validation_thresholds.json"
).open("w") as f:

    json.dump(
        thresholds,
        f,
        indent=2,
    )

pd.DataFrame(
    history
).to_csv(
    OUT / "history.csv",
    index=False,
)


print("\n" + "=" * 65)
print("CHAPMAN TRAINING COMPLETE")
print("=" * 65)

print(
    "Best epoch:",
    summary["best_epoch"],
)

print(
    "Validation macro-AUROC:",
    summary[
        "best_validation_macro_auroc"
    ],
)

print(
    "Validation macro-AP:",
    summary[
        "validation_macro_average_precision"
    ],
)

print(
    "Validation macro-F1:",
    summary[
        "validation_macro_f1"
    ],
)

print("\nThresholds:")

for label in LABELS:
    print(
        f"  {label:6s}: "
        f"{thresholds[label]['threshold']:.6f}"
    )

print("\nSaved:")
print(OUT)
print("=" * 65)