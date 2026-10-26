#!/usr/bin/env python3
"""Endpoint, stacker, label-level, Q-component and ensemble checks.

Uses saved prediction artifacts only (no inference), so it runs after
run_posthoc_selectors.py. Everything fitted is fitted on clean validation.

1. any-error endpoint: AURC/AUGRC with loss = 1[any label wrong]
2. stacker_hamming: least-squares selector fitted to per-record Hamming
   error (the logistic stacker targets any-error)
3. bootstrap intervals for selective risk at 70/80/90% coverage
4. label-level AURC (loss = that label wrong)
5. Q components: mean value per condition and AURC as single selectors
6. --ensemble (PTB-XL): three-seed ensemble; C_E vs ensemble disagreement

python scripts/extended/run_supplementary_checks.py --dataset ptbxl \
    --conditions frozen nested recorded --ensemble
python scripts/extended/run_supplementary_checks.py --dataset chapman \
    --conditions frozen nested recorded
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.selective_metrics import (
    aurc_augrc,
    bootstrap_two_sided_p,
    cluster_weight_matrix,
    holm_adjust,
    percentile_selector,
)
from ptbxl_reliability.selective_prediction import (
    mean_bernoulli_confidence,
    sample_hamming_error,
)
from ptbxl_reliability.validation_metrics import select_classwise_f1_thresholds


POSTHOC_SCRIPT = PROJECT_ROOT / "scripts" / "extended" / "run_posthoc_selectors.py"
EXTENDED_CONFIG_PATH = PROJECT_ROOT / "configs" / "extended_v1.yaml"
ENSEMBLE_SEEDS = (20260826, 20260827, 20260828)
Q_FEATURES = (
    "baseline_burden",
    "high_frequency_burden",
    "lead_dropout_burden",
    "clipping_burden",
)
# Selectors whose risk at coverage gets bootstrap intervals.
COVERAGE_SELECTORS = ("confidence", "confidence_T", "stacker", "U", "Q+U", "bSQI")
LABEL_SELECTORS = ("confidence", "U", "Q", "bSQI")


def load_posthoc_module():
    spec = importlib.util.spec_from_file_location("posthoc", POSTHOC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HammingStacker:
    """Least-squares fit of per-record Hamming error; selector = -prediction."""

    def __init__(self, features: dict[str, np.ndarray], losses: np.ndarray):
        self.names = list(features)
        x = self._matrix(features)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        design = np.column_stack([np.ones(len(x)), (x - self.mean_) / self.scale_])
        self.coef_, *_ = np.linalg.lstsq(design, losses, rcond=None)

    def _matrix(self, features):
        return np.column_stack([np.asarray(features[n], float) for n in self.names])

    def selector(self, features):
        x = (self._matrix(features) - self.mean_) / self.scale_
        return -(self.coef_[0] + x @ self.coef_[1:])

    def coefficients(self):
        return {"intercept": float(self.coef_[0]),
                **{n: float(c) for n, c in zip(self.names, self.coef_[1:])}}


def weighted_risk_at_coverage(selector, losses, weights, targets, chunk=500):
    """(B, len(targets)) risk among the highest-ranked records.

    Ties are broken by input order, so this is used for intervals only;
    point estimates come from the tie-grouped curves in the posthoc run.
    """
    order = np.argsort(-np.asarray(selector, float), kind="stable")
    loss = np.asarray(losses, float)[order]
    out = []
    for start in range(0, len(weights), chunk):
        w = weights[start:start + chunk][:, order]
        cum_w = np.cumsum(w, axis=1)
        cum_l = np.cumsum(w * loss, axis=1)
        total = cum_w[:, -1:]
        cols = []
        for target in targets:
            idx = np.argmax(cum_w >= target * total, axis=1)
            rows = np.arange(len(w))
            cols.append(cum_l[rows, idx] / cum_w[rows, idx])
        out.append(np.column_stack(cols))
    return np.vstack(out)


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def paired_contrasts(name, selectors, losses, weights, reference, posthoc):
    """AURC/AUGRC point estimates and paired bootstrap deltas vs reference."""
    metric_rows, contrast_rows = [], []
    replicates = posthoc.bootstrap(selectors, losses, weights)
    point = {}
    for selector_name, selector in selectors.items():
        aurc, augrc = aurc_augrc(selector, losses)
        point[selector_name] = {"aurc": aurc, "augrc": augrc}
        metric_rows.append({"condition": name, "selector": selector_name,
                            "aurc": aurc, "augrc": augrc})
    for selector_name in selectors:
        if selector_name == reference:
            continue
        for metric in ("aurc", "augrc"):
            deltas = replicates[selector_name][metric] - replicates[reference][metric]
            low, high = np.percentile(deltas, [2.5, 97.5])
            contrast_rows.append({
                "condition": name, "selector": selector_name, "metric": metric,
                "delta_vs_reference": point[selector_name][metric] - point[reference][metric],
                "ci_low": float(low), "ci_high": float(high),
                "p_bootstrap": bootstrap_two_sided_p(deltas),
            })
    return metric_rows, contrast_rows


def finish_contrasts(rows):
    contrasts = pd.DataFrame(rows)
    contrasts["p_holm"] = np.nan
    for _, index in contrasts.groupby(["condition", "metric"]).groups.items():
        contrasts.loc[index, "p_holm"] = holm_adjust(
            contrasts.loc[index, "p_bootstrap"].to_numpy()
        )
    contrasts["direction"] = np.where(
        contrasts["ci_high"] < 0.0, "better_than_reference",
        np.where(contrasts["ci_low"] > 0.0, "worse_than_reference", "inconclusive"),
    )
    return contrasts


def single_model_checks(args, posthoc, inference, helpers, config):
    run_dir = inference.EXTENDED_ROOT / args.dataset / f"seed_{args.seed}"
    sqi_dir = inference.EXTENDED_ROOT / args.dataset / "published_sqi"
    out_dir = run_dir / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = tuple(float(c) for c in config["coverage_targets"])

    validation = posthoc.load_npz(run_dir / "validation_clean" / "results.npz")
    thresholds = np.asarray(
        json.loads((run_dir / "thresholds.json").read_text())["thresholds"], float,
    )
    validation_sqi = posthoc.load_sqi(
        sqi_dir, "validation_clean", posthoc.record_ids(validation),
    )
    frozen_path = posthoc.FROZEN_COMBINATION_REFERENCES[args.dataset]
    frozen_reference = (
        posthoc.load_npz(frozen_path)
        if args.seed == inference.PRIMARY_SEED and frozen_path.exists() else None
    )
    fitted = posthoc.Selectors(validation, thresholds, validation_sqi, frozen_reference)
    hamming_stacker = HammingStacker(
        fitted._stack_features(validation, validation_sqi),
        validation["hamming_error"].astype(float),
    )

    conditions = inference.condition_list(args.conditions, helpers)
    names = ["clean" if c is None else c["name"] for c in conditions]
    if "clean" not in names:
        names.insert(0, "clean")

    metric_rows = {"hamming": [], "any_error": []}
    contrast_rows = {"hamming": [], "any_error": []}
    coverage_rows, label_rows, component_rows = [], [], []
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
                posthoc.cluster_ids(arrays), n_replicates=args.n_bootstrap,
                seed=int(config["bootstrap_seed"]),
            )
        elif not np.array_equal(ids, reference_ids):
            raise ValueError(f"{name}: test records differ from clean")

        sqi = posthoc.load_sqi(sqi_dir, name, ids)
        selectors = fitted.compute(arrays, sqi)
        selectors["stacker_hamming"] = hamming_stacker.selector(
            fitted._stack_features(arrays, sqi)
        )
        hamming = arrays["hamming_error"].astype(float)
        endpoints = {"hamming": hamming, "any_error": (hamming > 0).astype(float)}

        for endpoint, losses in endpoints.items():
            m, c = paired_contrasts(name, selectors, losses, weights,
                                    "confidence", posthoc)
            metric_rows[endpoint] += m
            contrast_rows[endpoint] += c

        for selector_name in COVERAGE_SELECTORS:
            if selector_name not in selectors:
                continue
            risks = weighted_risk_at_coverage(
                selectors[selector_name], hamming, weights, targets,
            )
            for j, target in enumerate(targets):
                low, high = np.percentile(risks[:, j], [2.5, 97.5])
                coverage_rows.append({
                    "condition": name, "selector": selector_name,
                    "coverage": target, "ci_low": float(low), "ci_high": float(high),
                })

        # Label level: loss is whether that label is wrong.
        y = arrays["targets"].astype(np.int64)
        pred = arrays["predictions"].astype(np.int64)
        p = arrays["probabilities"].astype(float)
        for k in range(y.shape[1]):
            loss_k = (pred[:, k] != y[:, k]).astype(float)
            label_selectors = {s: selectors[s] for s in LABEL_SELECTORS if s in selectors}
            label_selectors["label_confidence"] = np.abs(p[:, k] - 0.5)
            for selector_name, selector in label_selectors.items():
                aurc, augrc = aurc_augrc(selector, loss_k)
                label_rows.append({
                    "condition": name, "label_index": k, "selector": selector_name,
                    "error_rate": float(loss_k.mean()), "aurc": aurc, "augrc": augrc,
                })

        for feature in Q_FEATURES:
            if feature not in arrays:
                continue
            values = arrays[feature].astype(float)
            aurc, augrc = aurc_augrc(-values, hamming)
            component_rows.append({
                "condition": name, "component": feature,
                "mean": float(values.mean()), "aurc": aurc, "augrc": augrc,
            })

    for endpoint in endpoints:
        pd.DataFrame(metric_rows[endpoint]).to_csv(
            out_dir / f"metrics_{endpoint}.csv", index=False)
        finish_contrasts(contrast_rows[endpoint]).to_csv(
            out_dir / f"contrasts_{endpoint}.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "risk_at_coverage_ci.csv", index=False)
    pd.DataFrame(label_rows).to_csv(out_dir / "label_level.csv", index=False)
    pd.DataFrame(component_rows).to_csv(out_dir / "q_components.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "dataset": args.dataset, "seed": args.seed, "conditions": names,
        "n_bootstrap": args.n_bootstrap,
        "logistic_stacker": fitted.stacker.coefficients(),
        "hamming_stacker": hamming_stacker.coefficients(),
    }, indent=2))
    print("Saved:", out_dir)


def ensemble_checks(args, posthoc, inference, helpers, config):
    if args.dataset != "ptbxl":
        raise ValueError("The ensemble needs the three PTB-XL seeds")
    root = inference.EXTENDED_ROOT / "ptbxl"
    sqi_dir = root / "published_sqi"
    out_dir = root / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {s: root / f"seed_{s}" for s in ENSEMBLE_SEEDS}

    def members(load):
        arrays = [load(s) for s in ENSEMBLE_SEEDS]
        ids = posthoc.record_ids(arrays[0])
        for a in arrays[1:]:
            if not np.array_equal(posthoc.record_ids(a), ids):
                raise ValueError("Ensemble members cover different records")
        return arrays

    def summarize(arrays):
        stack = np.stack([a["probabilities"].astype(float) for a in arrays])
        p = stack.mean(axis=0)
        disagreement = (binary_entropy(p) - binary_entropy(stack).mean(axis=0)).mean(axis=1)
        return p, disagreement

    validation = members(lambda s: posthoc.load_npz(
        run_dirs[s] / "validation_clean" / "results.npz"))
    p_val, d_val = summarize(validation)
    y_val = validation[0]["targets"].astype(np.int64)
    thresholds = np.asarray(
        [fit["threshold"] for fit in select_classwise_f1_thresholds(y_val, p_val)])
    references = {
        "C": 1.0 - mean_bernoulli_confidence(p_val),
        "D": d_val,
        "Q": 1.0 - validation[0]["Q_signal"].astype(float),
    }
    validation_sqi = posthoc.load_sqi(
        sqi_dir, "validation_clean", posthoc.record_ids(validation[0]))
    if validation_sqi is not None:
        references["S"] = 1.0 - validation_sqi

    def combo(badness):
        return percentile_selector([badness[k] for k in badness],
                                   [references[k] for k in badness])

    conditions = inference.condition_list(args.conditions, helpers)
    names = ["clean" if c is None else c["name"] for c in conditions]
    if "clean" not in names:
        names.insert(0, "clean")

    metric_rows, contrast_rows, class_rows = [], [], []
    weights = None
    for name in names:
        try:
            arrays = members(lambda s: posthoc.load_npz(posthoc.find_test_artifact(
                "ptbxl", s, run_dirs[s], name, inference.PRIMARY_SEED)))
        except FileNotFoundError as error:
            print(f"[{name}] skipped: {error}")
            continue
        print(f"[{name}]", flush=True)
        if weights is None:
            weights = cluster_weight_matrix(
                posthoc.cluster_ids(arrays[0]), n_replicates=args.n_bootstrap,
                seed=int(config["bootstrap_seed"]),
            )
        y = arrays[0]["targets"].astype(np.int64)
        p, disagreement = summarize(arrays)
        pred = (p >= thresholds[None, :]).astype(np.int64)
        losses = sample_hamming_error(y, pred).astype(float)
        C = mean_bernoulli_confidence(p)
        Q = arrays[0]["Q_signal"].astype(float)
        mc_U = np.mean([a["U"].astype(float) for a in arrays], axis=0)
        bad = {"C": 1.0 - C, "D": disagreement, "Q": 1.0 - Q}
        selectors = {
            "ensemble_confidence": C,
            "ensemble_disagreement": -disagreement,
            "mean_mc_dropout_U": -mc_U,
            "Q": Q,
            "C_E+D": combo({"C": bad["C"], "D": bad["D"]}),
            "C_E+Q": combo({"C": bad["C"], "Q": bad["Q"]}),
        }
        sqi = posthoc.load_sqi(sqi_dir, name, posthoc.record_ids(arrays[0]))
        if sqi is not None and "S" in references:
            selectors["bSQI"] = sqi
            selectors["C_E+bSQI"] = combo({"C": bad["C"], "S": 1.0 - sqi})

        m, c = paired_contrasts(name, selectors, losses, weights,
                                "ensemble_confidence", posthoc)
        metric_rows += m
        contrast_rows += c
        class_rows.append({
            "condition": name,
            **posthoc.classification({"targets": y, "probabilities": p,
                                      "predictions": pred, "hamming_error": losses}),
            "single_model_hamming_risk": float(np.mean(arrays[0]["hamming_error"])),
        })

    pd.DataFrame(metric_rows).to_csv(out_dir / "ensemble_metrics.csv", index=False)
    finish_contrasts(contrast_rows).to_csv(out_dir / "ensemble_contrasts.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out_dir / "ensemble_classification.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "seeds": list(ENSEMBLE_SEEDS), "thresholds": thresholds.tolist(),
        "threshold_source": "max F1 per class on averaged clean validation probabilities",
        "n_bootstrap": args.n_bootstrap,
    }, indent=2))
    print("Saved:", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ptbxl", "chapman"), required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--conditions", nargs="+",
                        choices=("frozen", "nested", "recorded"), default=["frozen"])
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--ensemble", action="store_true",
                        help="also run the three-seed PTB-XL ensemble")
    parser.add_argument("--ensemble-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(EXTENDED_CONFIG_PATH.read_text())["posthoc"]
    args.n_bootstrap = args.n_bootstrap or int(config["bootstrap_replicates"])

    posthoc = load_posthoc_module()
    inference = posthoc.load_inference_module()
    helpers = inference.load_ptb_helpers()

    if not args.ensemble_only:
        single_model_checks(args, posthoc, inference, helpers, config)
    if args.ensemble or args.ensemble_only:
        ensemble_checks(args, posthoc, inference, helpers, config)


if __name__ == "__main__":
    main()
