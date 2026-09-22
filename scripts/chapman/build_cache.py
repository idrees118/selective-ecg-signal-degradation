from pathlib import Path
import json
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.signal import resample_poly

ROOT = Path("data/raw/chapman")
MANIFEST = Path(
    "results/external_validation/chapman/v1/"
    "chapman_frozen_cohort.csv"
)
OUT = Path(
    "results/external_validation/chapman/v1"
)

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

df = pd.read_csv(MANIFEST)

assert len(df) == 10646
assert df["FileName"].is_unique

cache_path = OUT / "chapman_100hz_mV.npy"

cache = open_memmap(
    cache_path,
    mode="w+",
    dtype=np.float32,
    shape=(len(df), 12, 1000),
)

train_mask = df["split"].eq("train").to_numpy()

total_sum = 0.0
total_sumsq = 0.0
total_n = 0

global_min = np.inf
global_max = -np.inf

bad = []

for i, row in enumerate(df.itertuples(index=False)):

    path = ROOT / f"{row.FileName}.csv"

    try:
        x = pd.read_csv(path)

        if x.shape != (5000, 12):
            raise ValueError(
                f"shape={x.shape}, expected (5000, 12)"
            )

        if list(x.columns) != LEADS:
            raise ValueError(
                f"lead order mismatch: {list(x.columns)}"
            )

        x = x.to_numpy(dtype=np.float64)

        if not np.isfinite(x).all():
            raise ValueError("NaN/Inf present")

        # Original Chapman CSV values are microvolts.
        # Convert µV -> mV.
        x = x / 1000.0

        global_min = min(global_min, float(x.min()))
        global_max = max(global_max, float(x.max()))

        # 500 Hz -> 100 Hz.
        # resample_poly performs anti-alias filtering.
        x100 = resample_poly(
            x,
            up=1,
            down=5,
            axis=0,
        )

        if x100.shape != (1000, 12):
            raise ValueError(
                f"downsampled shape={x100.shape}"
            )

        if not np.isfinite(x100).all():
            raise ValueError(
                "NaN/Inf after downsampling"
            )

        # Model layout: (12, 1000)
        model_x = np.ascontiguousarray(
            x100.T,
            dtype=np.float32,
        )

        cache[i] = model_x

        # Fit normalization from TRAIN ONLY.
        if train_mask[i]:
            z = model_x.astype(
                np.float64,
                copy=False,
            )

            total_sum += z.sum()
            total_sumsq += np.square(z).sum()
            total_n += z.size

    except Exception as exc:
        bad.append(
            {
                "index": i,
                "FileName": row.FileName,
                "error": str(exc),
            }
        )

    if (i + 1) % 250 == 0 or i + 1 == len(df):
        print(
            f"{i+1:5d}/{len(df)} "
            f"processed | bad={len(bad)}",
            flush=True,
        )

cache.flush()

if bad:
    pd.DataFrame(bad).to_csv(
        OUT / "chapman_cache_errors.csv",
        index=False,
    )
    raise RuntimeError(
        f"{len(bad)} waveform(s) failed validation. "
        "See chapman_cache_errors.csv"
    )

mean = total_sum / total_n

variance = (
    total_sumsq / total_n
    - mean * mean
)

std = float(np.sqrt(variance))
mean = float(mean)

normalization = {
    "status": "FROZEN",
    "fit_split": "train",
    "train_records": int(train_mask.sum()),
    "sampling_rate_hz": 100,
    "input_unit": "mV",
    "mean_mV": mean,
    "std_mV": std,
    "source_sampling_rate_hz": 500,
    "source_samples": 5000,
    "output_samples": 1000,
    "resampling": "scipy.signal.resample_poly(up=1, down=5)",
}

with (
    OUT / "chapman_normalization.json"
).open("w") as f:
    json.dump(
        normalization,
        f,
        indent=2,
    )
    f.write("\n")

print("\n" + "=" * 65)
print("CHAPMAN CACHE COMPLETE")
print("=" * 65)

print("Records:        ", len(df))
print("Cache shape:    ", cache.shape)
print("Cache dtype:    ", cache.dtype)
print("Train records:  ", int(train_mask.sum()))
print("Raw min mV:     ", global_min)
print("Raw max mV:     ", global_max)
print("Train mean mV:  ", mean)
print("Train std mV:   ", std)
print("Failures:       ", len(bad))
print("Cache:          ", cache_path)
print("=" * 65)