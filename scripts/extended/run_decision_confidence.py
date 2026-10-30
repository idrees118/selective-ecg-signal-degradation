#!/usr/bin/env python3
"""Decision confidence: mean probability of the class actually predicted.

C_dec = mean_k [yhat_k * p_k + (1 - yhat_k) * (1 - p_k)], where yhat uses the
frozen class-specific thresholds. Compared with confidence C by paired
cluster bootstrap (single comparison per condition and metric, no Holm).

Uses saved prediction artifacts only; run after run_posthoc_selectors.py.

python scripts/extended/run_decision_confidence.py --dataset ptbxl \
    --conditions frozen nested recorded
python scripts/extended/run_decision_confidence.py --dataset chapman \
    --conditions frozen nested recorded
"""

from __future__ import annotations

import argparse
import importlib.util

import numpy as np
import pandas as pd
import yaml

from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.selective_metrics import cluster_weight_matrix


SUPPLEMENTARY_SCRIPT = PROJECT_ROOT / "scripts" / "extended" / "run_supplementary_checks.py"


def load_supplementary_module():
    spec = importlib.util.spec_from_file_location("supplementary", SUPPLEMENTARY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision_confidence(probabilities, predictions):
    p = np.asarray(probabilities, dtype=np.float64)
    y_hat = np.asarray(predictions, dtype=np.float64)
    return np.mean(y_hat * p + (1.0 - y_hat) * (1.0 - p), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ptbxl", "chapman"), required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--conditions", nargs="+",
                        choices=("frozen", "nested", "recorded"), default=["frozen"])
    parser.add_argument("--n-bootstrap", type=int, default=None)
    args = parser.parse_args()

    supplementary = load_supplementary_module()
    config = yaml.safe_load(supplementary.EXTENDED_CONFIG_PATH.read_text())["posthoc"]
    n_bootstrap = args.n_bootstrap or int(config["bootstrap_replicates"])
    posthoc = supplementary.load_posthoc_module()
    inference = posthoc.load_inference_module()
    helpers = inference.load_ptb_helpers()

    run_dir = inference.EXTENDED_ROOT / args.dataset / f"seed_{args.seed}"
    out_dir = run_dir / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = inference.condition_list(args.conditions, helpers)
    names = ["clean" if c is None else c["name"] for c in conditions]
    if "clean" not in names:
        names.insert(0, "clean")

    metric_rows, contrast_rows = [], []
    weights = reference_ids = None
    for name in names:
        print(f"[{name}]", flush=True)
        arrays = posthoc.load_npz(posthoc.find_test_artifact(
            args.dataset, args.seed, run_dir, name, inference.PRIMARY_SEED,
        ))
        ids = posthoc.record_ids(arrays)
        if reference_ids is None:
            reference_ids = ids
            weights = cluster_weight_matrix(
                posthoc.cluster_ids(arrays), n_replicates=n_bootstrap,
                seed=int(config["bootstrap_seed"]),
            )
        elif not np.array_equal(ids, reference_ids):
            raise ValueError(f"{name}: test records differ from clean")

        selectors = {
            "confidence": arrays["confidence"].astype(np.float64),
            "decision_confidence": decision_confidence(
                arrays["probabilities"], arrays["predictions"],
            ),
        }
        hamming = arrays["hamming_error"].astype(float)
        for endpoint, losses in (("hamming", hamming),
                                 ("any_error", (hamming > 0).astype(float))):
            m, c = supplementary.paired_contrasts(
                name, selectors, losses, weights, "confidence", posthoc,
            )
            metric_rows += [dict(row, endpoint=endpoint) for row in m]
            contrast_rows += [dict(row, endpoint=endpoint) for row in c]

    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics_decision_confidence.csv", index=False)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["direction"] = np.where(
        contrasts["ci_high"] < 0.0, "better_than_confidence",
        np.where(contrasts["ci_low"] > 0.0, "worse_than_confidence", "inconclusive"),
    )
    contrasts.to_csv(out_dir / "contrasts_decision_confidence.csv", index=False)
    summary = (contrasts.groupby(["endpoint", "metric", "direction"]).size()
               .unstack(fill_value=0))
    print(summary)


if __name__ == "__main__":
    main()
