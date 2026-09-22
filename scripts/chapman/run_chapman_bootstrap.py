#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ptbxl_reliability.selective_prediction import midrank_ecdf


ROOT = Path("results/external_validation/chapman/v1")
FINAL_DIR = ROOT / "final_test"

CONDITION_TABLE = FINAL_DIR / "condition_metrics.csv"
COMBINATION_REFERENCE = ROOT / "chapman_combination_reference.npz"
PROTOCOL_PATH = ROOT / "chapman_test_protocol.json"

OUT_DIR = ROOT / "bootstrap"
OUT_CSV = OUT_DIR / "primary_aurc_bootstrap_summary.csv"
OUT_NPZ = OUT_DIR / "primary_aurc_bootstrap_replicates.npz"
OUT_JSON = OUT_DIR / "summary.json"

N_BOOT = 5000
SEED = 20260826

PRIMARY = (
    ("U", "U"),
    ("Q+U", "Q_plus_U"),
)


def weighted_grouped_aurc_many(
    scores: np.ndarray,
    losses: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> np.ndarray:
    """
    Exact right-endpoint grouped AURC for many bootstrap resamples.

    Larger score = retained earlier.
    Exact score ties are retained as whole groups.
    bootstrap_counts shape: (B, N), integer multiplicities.
    """
    scores = np.asarray(scores, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    counts = np.asarray(bootstrap_counts)

    if scores.ndim != 1 or losses.ndim != 1:
        raise ValueError("scores/losses must be 1-D")
    if len(scores) != len(losses):
        raise ValueError("score/loss length mismatch")
    if counts.ndim != 2 or counts.shape[1] != len(scores):
        raise ValueError("bootstrap count matrix shape mismatch")
    if not np.isfinite(scores).all() or not np.isfinite(losses).all():
        raise ValueError("non-finite score/loss")
    if (losses < 0).any() or (losses > 1).any():
        raise ValueError("losses must lie in [0,1]")

    # Stable descending sort; exact ties stay adjacent.
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    l = losses[order]

    starts = np.r_[
        0,
        np.flatnonzero(s[1:] != s[:-1]) + 1,
    ]

    # Reorder bootstrap multiplicities to selector order.
    c = counts[:, order].astype(np.float64, copy=False)

    # Aggregate multiplicity and error mass within exact-score tie groups.
    group_n = np.add.reduceat(c, starts, axis=1)
    group_err = np.add.reduceat(c * l[None, :], starts, axis=1)

    cum_n = np.cumsum(group_n, axis=1)
    cum_err = np.cumsum(group_err, axis=1)

    # Every bootstrap replicate contains exactly N sampled records.
    n_total = counts.sum(axis=1).astype(np.float64)

    risk = np.divide(
        cum_err,
        cum_n,
        out=np.zeros_like(cum_err, dtype=np.float64),
        where=cum_n > 0,
    )

    coverage_increment = group_n / n_total[:, None]

    return np.sum(
        coverage_increment * risk,
        axis=1,
        dtype=np.float64,
    )


def point_grouped_aurc(
    scores: np.ndarray,
    losses: np.ndarray,
) -> float:
    ones = np.ones((1, len(scores)), dtype=np.int16)
    return float(
        weighted_grouped_aurc_many(
            scores,
            losses,
            ones,
        )[0]
    )


def load_condition_arrays(condition: str) -> dict[str, np.ndarray]:
    path = FINAL_DIR / condition / "results.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=False)

    required = {
        "hamming_error",
        "confidence",
        "Q_signal",
        "U",
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(
            f"{path} missing arrays: {sorted(missing)}"
        )

    return {
        key: np.asarray(data[key])
        for key in required
    }


def main() -> None:
    required_paths = [
        CONDITION_TABLE,
        COMBINATION_REFERENCE,
        PROTOCOL_PATH,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    with PROTOCOL_PATH.open() as f:
        protocol = json.load(f)

    if protocol.get("status") != "FROZEN":
        raise ValueError("Chapman test protocol is not FROZEN")

    expected_primary = [
        "U_vs_confidence",
        "Q+U_vs_confidence",
    ]
    if protocol.get("primary_comparisons") != expected_primary:
        raise ValueError(
            "Primary comparisons differ from frozen protocol"
        )

    table = pd.read_csv(CONDITION_TABLE)

    if len(table) != 13:
        raise ValueError(
            f"Expected 13 test conditions, found {len(table)}"
        )

    combo_ref = np.load(
        COMBINATION_REFERENCE,
        allow_pickle=False,
    )

    q_reference = np.asarray(
        combo_ref["Q_signal_badness_reference"],
        dtype=np.float64,
    )
    u_reference = np.asarray(
        combo_ref["U_badness_reference"],
        dtype=np.float64,
    )

    # Establish N from clean held-out test artifacts.
    clean = load_condition_arrays("clean")
    n = len(clean["hamming_error"])

    if n != 1065:
        raise ValueError(
            f"Expected 1065 test records, found {n}"
        )

    # Same 5,000 resamples are reused for all conditions and both
    # comparators. This preserves pairing and makes the analysis
    # deterministic/reproducible.
    print(
        f"Generating {N_BOOT} paired bootstrap resamples "
        f"of {n} held-out records..."
    )

    rng = np.random.default_rng(SEED)

    bootstrap_counts = rng.multinomial(
        n=n,
        pvals=np.full(n, 1.0 / n, dtype=np.float64),
        size=N_BOOT,
    ).astype(np.int16)

    if not np.all(
        bootstrap_counts.sum(axis=1) == n
    ):
        raise AssertionError(
            "Bootstrap replicate size mismatch"
        )

    rows = []
    replicate_store = {}

    print(
        "Running 26 frozen primary contrasts "
        "(2 comparisons x 13 conditions)..."
    )

    for row_i, table_row in table.iterrows():
        condition = str(table_row["condition"])
        arrays = (
            clean
            if condition == "clean"
            else load_condition_arrays(condition)
        )

        losses = np.asarray(
            arrays["hamming_error"],
            dtype=np.float64,
        )
        confidence = np.asarray(
            arrays["confidence"],
            dtype=np.float64,
        )
        Q = np.asarray(
            arrays["Q_signal"],
            dtype=np.float64,
        )
        U = np.asarray(
            arrays["U"],
            dtype=np.float64,
        )

        if not (
            len(losses)
            == len(confidence)
            == len(Q)
            == len(U)
            == n
        ):
            raise ValueError(
                f"{condition}: inconsistent record counts"
            )

        q_pct = midrank_ecdf(
            q_reference,
            1.0 - Q,
        )
        u_pct = midrank_ecdf(
            u_reference,
            U,
        )

        selectors = {
            "confidence": confidence,
            "U": -U,
            "Q+U": -(q_pct + u_pct) / 2.0,
        }

        # Point-estimate integrity checks against the already-frozen
        # condition_metrics.csv.
        direct_conf = point_grouped_aurc(
            selectors["confidence"],
            losses,
        )
        direct_u = point_grouped_aurc(
            selectors["U"],
            losses,
        )
        direct_qu = point_grouped_aurc(
            selectors["Q+U"],
            losses,
        )

        expected = {
            "confidence": float(
                table_row["aurc_confidence"]
            ),
            "U": float(
                table_row["aurc_U"]
            ),
            "Q+U": float(
                table_row["aurc_Q_plus_U"]
            ),
        }
        observed = {
            "confidence": direct_conf,
            "U": direct_u,
            "Q+U": direct_qu,
        }

        for key in observed:
            if not np.isclose(
                observed[key],
                expected[key],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"{condition}: AURC reconstruction mismatch "
                    f"for {key}: observed={observed[key]:.17g}, "
                    f"expected={expected[key]:.17g}"
                )

        conf_boot = weighted_grouped_aurc_many(
            selectors["confidence"],
            losses,
            bootstrap_counts,
        )

        for display_name, store_name in PRIMARY:
            alt_boot = weighted_grouped_aurc_many(
                selectors[display_name],
                losses,
                bootstrap_counts,
            )

            delta_boot = alt_boot - conf_boot

            point_delta = (
                observed[display_name]
                - observed["confidence"]
            )

            ci_low, ci_high = np.percentile(
                delta_boot,
                [2.5, 97.5],
                method="linear",
            )

            rows.append({
                "condition": condition,
                "comparison":
                    f"{display_name} vs confidence",
                "confidence_aurc":
                    observed["confidence"],
                "alternative_aurc":
                    observed[display_name],
                "delta_aurc_alternative_minus_confidence":
                    point_delta,
                "ci95_lower":
                    float(ci_low),
                "ci95_upper":
                    float(ci_high),
                "ci_excludes_zero":
                    bool(
                        ci_low > 0.0
                        or ci_high < 0.0
                    ),
                "direction":
                    (
                        "confidence_lower_aurc"
                        if point_delta > 0
                        else
                        "alternative_lower_aurc"
                        if point_delta < 0
                        else
                        "equal_point_estimate"
                    ),
            })

            replicate_store[
                f"{condition}__{store_name}__delta"
            ] = delta_boot.astype(
                np.float32
            )

        print(
            f"  {row_i + 1:02d}/13 {condition}"
        )

    summary = pd.DataFrame(rows)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUT_CSV,
        index=False,
        float_format="%.17g",
    )

    np.savez_compressed(
        OUT_NPZ,
        **replicate_store,
    )

    positive_point = int(
        (
            summary[
                "delta_aurc_alternative_minus_confidence"
            ] > 0
        ).sum()
    )
    positive_ci = int(
        (summary["ci95_lower"] > 0).sum()
    )
    contains_zero = int(
        (
            (summary["ci95_lower"] <= 0)
            & (summary["ci95_upper"] >= 0)
        ).sum()
    )
    negative_ci = int(
        (summary["ci95_upper"] < 0).sum()
    )

    meta = {
        "status": "FROZEN",
        "dataset": "Chapman-Shaoxing ECG",
        "test_records": n,
        "bootstrap_unit":
            "held-out ECG record",
        "paired": True,
        "replicates": N_BOOT,
        "confidence_interval":
            "95% percentile interval",
        "random_seed": SEED,
        "primary_comparisons": [
            "U_vs_confidence",
            "Q+U_vs_confidence",
        ],
        "conditions": 13,
        "total_contrasts": 26,
        "positive_point_estimates_favoring_confidence":
            positive_point,
        "intervals_entirely_above_zero":
            positive_ci,
        "intervals_containing_zero":
            contains_zero,
        "intervals_entirely_below_zero":
            negative_ci,
        "multiple_comparison_adjustment":
            "none",
        "interpretation":
            (
                "Delta AURC = alternative - confidence; "
                "positive values favor confidence because "
                "lower AURC is better."
            ),
    }

    with OUT_JSON.open("w") as f:
        json.dump(
            meta,
            f,
            indent=2,
        )
        f.write("\n")

    print("\n" + "=" * 78)
    print("CHAPMAN PAIRED BOOTSTRAP COMPLETE")
    print("=" * 78)

    display = summary[
        [
            "condition",
            "comparison",
            "delta_aurc_alternative_minus_confidence",
            "ci95_lower",
            "ci95_upper",
        ]
    ].copy()

    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSUMMARY")
    print(
        f"Positive point estimates favoring confidence: "
        f"{positive_point}/26"
    )
    print(
        f"95% intervals entirely above zero: "
        f"{positive_ci}/26"
    )
    print(
        f"95% intervals containing zero: "
        f"{contains_zero}/26"
    )
    print(
        f"95% intervals entirely below zero: "
        f"{negative_ci}/26"
    )

    print("\nSaved:")
    print(OUT_CSV)
    print(OUT_NPZ)
    print(OUT_JSON)
    print("=" * 78)


if __name__ == "__main__":
    main()
