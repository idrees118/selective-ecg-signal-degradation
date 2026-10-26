#!/usr/bin/env python3
"""Post-hoc selectors, AURC/AUGRC, risk at coverage and paired bootstrap.

Uses saved prediction artifacts only (no inference). References,
temperatures and the stacker are fitted on clean validation.

Selectors (higher = retained first)
-----------------------------------
confidence       frozen v1 mean Bernoulli confidence (comparator)
U, Q, Q+U        frozen v1 auxiliary selectors
confidence_T     classwise temperature-scaled confidence
margin           mean |logit(p) - logit(threshold)|
bSQI             published beat-agreement SQI (if computed)
C+U, C+Q, C+Q+U  equal-weight percentile combinations including confidence
C+bSQI           (if bSQI computed)
stacker          logistic P(any error) from C, U, 1-Q (and 1-bSQI)

python scripts/extended/run_posthoc_selectors.py --dataset ptbxl \
    --seed 20260826 --conditions frozen nested recorded
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, roc_auc_score

from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.selective_metrics import (
    ErrorStacker,
    aurc_augrc,
    bootstrap_two_sided_p,
    cluster_weight_matrix,
    fit_classwise_temperature,
    holm_adjust,
    percentile_selector,
    risk_at_coverages,
    temperature_scaled_confidence,
    threshold_margin,
    weighted_aurc_augrc,
)


INFERENCE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "extended" / "run_extended_inference.py"
)
EXTENDED_CONFIG_PATH = PROJECT_ROOT / "configs" / "extended_v1.yaml"

FROZEN_TEST_DIRS = {
    "ptbxl": PROJECT_ROOT / "results" / "final_evaluation" / "v1" / "test",
    "chapman": (
        PROJECT_ROOT / "results" / "external_validation" / "chapman" / "v1"
        / "final_test"
    ),
}

# v1 Q/U references, used for the primary seed so Q+U matches v1.
FROZEN_COMBINATION_REFERENCES = {
    "ptbxl": (
        PROJECT_ROOT / "results" / "selective_prediction_waveform_q" / "v1"
        / "validation" / "combination_reference.npz"
    ),
    "chapman": (
        PROJECT_ROOT / "results" / "external_validation" / "chapman" / "v1"
        / "chapman_combination_reference.npz"
    ),
}

BOOTSTRAP_CHUNK = 500


def load_inference_module():
    spec = importlib.util.spec_from_file_location(
        "extended_inference",
        INFERENCE_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def cluster_ids(arrays: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("cluster_ids", "patient_ids", "record_ids"):
        if key in arrays:
            return arrays[key].astype(np.int64)
    raise KeyError("Artifact has no cluster/patient/record identifiers")


def record_ids(arrays: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("record_ids", "ecg_ids"):
        if key in arrays:
            return arrays[key].astype(np.int64)
    raise KeyError("Artifact has no record identifiers")


def find_test_artifact(dataset, seed, run_dir, name, primary_seed) -> Path:
    candidate = run_dir / name / "results.npz"
    if candidate.exists():
        return candidate
    frozen = FROZEN_TEST_DIRS[dataset] / name / "results.npz"
    if seed == primary_seed and frozen.exists():
        return frozen
    raise FileNotFoundError(
        f"No artifact for {name}: run run_extended_inference.py first "
        f"(looked in {candidate} and {frozen})"
    )


def load_sqi(sqi_dir: Path, name: str, ids: np.ndarray) -> np.ndarray | None:
    path = sqi_dir / f"{name}.npz"
    if not path.exists():
        return None
    data = load_npz(path)
    if not np.array_equal(data["record_ids"].astype(np.int64), ids):
        raise ValueError(f"{path}: record IDs do not match predictions")
    return data["bsqi_mean"].astype(np.float64)


class Selectors:
    """Selectors fitted on clean validation."""

    def __init__(self, validation, thresholds, validation_sqi,
                 frozen_reference=None):
        self.thresholds = thresholds
        self.use_sqi = validation_sqi is not None

        probabilities = validation["probabilities"].astype(np.float64)
        self.temperatures = fit_classwise_temperature(
            probabilities,
            validation["targets"].astype(np.int64),
        )

        self.references = {
            "C": 1.0 - validation["confidence"],
            "Q": 1.0 - validation["Q_signal"],
            "U": validation["U"].astype(np.float64),
        }
        if frozen_reference is not None:
            self.references["Q"] = frozen_reference["Q_signal_badness_reference"]
            self.references["U"] = frozen_reference["U_badness_reference"]
        if self.use_sqi:
            self.references["S"] = 1.0 - validation_sqi

        names = ["confidence", "U", "Q_badness"]
        if self.use_sqi:
            names.append("bSQI_badness")
        self.stacker = ErrorStacker(names).fit(
            self._stack_features(validation, validation_sqi),
            validation["hamming_error"],
        )

    def _stack_features(self, arrays, sqi):
        features = {
            "confidence": arrays["confidence"],
            "U": arrays["U"],
            "Q_badness": 1.0 - arrays["Q_signal"],
        }
        if self.use_sqi:
            features["bSQI_badness"] = 1.0 - sqi
        return features

    def _combo(self, badness: dict[str, np.ndarray]) -> np.ndarray:
        keys = list(badness)
        return percentile_selector(
            [badness[k] for k in keys],
            [self.references[k] for k in keys],
        )

    def compute(self, arrays, sqi) -> dict[str, np.ndarray]:
        C = arrays["confidence"].astype(np.float64)
        Q = arrays["Q_signal"].astype(np.float64)
        U = arrays["U"].astype(np.float64)
        probabilities = arrays["probabilities"].astype(np.float64)

        bad = {"C": 1.0 - C, "Q": 1.0 - Q, "U": U}

        selectors = {
            "confidence": C,
            "U": -U,
            "Q": Q,
            "Q+U": self._combo({"Q": bad["Q"], "U": bad["U"]}),
            "confidence_T": temperature_scaled_confidence(
                probabilities, self.temperatures,
            ),
            "margin": threshold_margin(probabilities, self.thresholds),
            "C+U": self._combo({"C": bad["C"], "U": bad["U"]}),
            "C+Q": self._combo({"C": bad["C"], "Q": bad["Q"]}),
            "C+Q+U": self._combo(bad),
        }

        if self.use_sqi:
            if sqi is None:
                raise ValueError("bSQI missing for a test condition")
            selectors["bSQI"] = sqi
            selectors["C+bSQI"] = self._combo({"C": bad["C"], "S": 1.0 - sqi})

        selectors["stacker"] = self.stacker.selector(
            self._stack_features(arrays, sqi)
        )
        return selectors


def classification(arrays) -> dict[str, float]:
    y = arrays["targets"].astype(np.int64)
    p = arrays["probabilities"].astype(np.float64)
    pred = arrays["predictions"].astype(np.int64)
    return {
        "macro_auroc": float(np.mean(
            [roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1])]
        )),
        "macro_f1": float(np.mean(
            [f1_score(y[:, k], pred[:, k], zero_division=0)
             for k in range(y.shape[1])]
        )),
        "full_hamming_risk": float(np.mean(arrays["hamming_error"])),
    }


def bootstrap(selectors, losses, weights):
    """(B,) AURC and AUGRC per selector on shared paired replicates."""
    result = {}
    for name, selector in selectors.items():
        aurc_parts, augrc_parts = [], []
        for start in range(0, len(weights), BOOTSTRAP_CHUNK):
            a, g = weighted_aurc_augrc(
                selector, losses, weights[start:start + BOOTSTRAP_CHUNK],
            )
            aurc_parts.append(a)
            augrc_parts.append(g)
        result[name] = {
            "aurc": np.concatenate(aurc_parts),
            "augrc": np.concatenate(augrc_parts),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ptbxl", "chapman"), required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("frozen", "nested", "recorded"),
        default=["frozen"],
    )
    parser.add_argument("--n-bootstrap", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(EXTENDED_CONFIG_PATH.read_text())["posthoc"]
    n_bootstrap = args.n_bootstrap or int(config["bootstrap_replicates"])
    coverage_targets = tuple(float(c) for c in config["coverage_targets"])

    inference = load_inference_module()
    helpers = inference.load_ptb_helpers()

    run_dir = inference.EXTENDED_ROOT / args.dataset / f"seed_{args.seed}"
    sqi_dir = inference.EXTENDED_ROOT / args.dataset / "published_sqi"
    out_dir = run_dir / "posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = load_npz(run_dir / "validation_clean" / "results.npz")
    thresholds = np.asarray(
        json.loads((run_dir / "thresholds.json").read_text())["thresholds"],
        dtype=np.float64,
    )
    validation_sqi = load_sqi(sqi_dir, "validation_clean", record_ids(validation))
    if validation_sqi is None:
        print("bSQI not found; run compute_published_sqi.py to include it")

    frozen_path = FROZEN_COMBINATION_REFERENCES[args.dataset]
    frozen_reference = (
        load_npz(frozen_path)
        if args.seed == inference.PRIMARY_SEED and frozen_path.exists()
        else None
    )
    selectors_fit = Selectors(
        validation, thresholds, validation_sqi, frozen_reference,
    )

    conditions = inference.condition_list(args.conditions, helpers)
    names = ["clean" if c is None else c["name"] for c in conditions]
    if "clean" not in names:
        names.insert(0, "clean")

    metric_rows, class_rows, contrast_rows = [], [], []
    weights = None
    reference_ids = None

    for name in names:
        print(f"[{name}]", flush=True)
        arrays = load_npz(find_test_artifact(
            args.dataset, args.seed, run_dir, name, inference.PRIMARY_SEED,
        ))
        ids = record_ids(arrays)
        losses = arrays["hamming_error"].astype(np.float64)

        if reference_ids is None:
            reference_ids = ids
            weights = cluster_weight_matrix(
                cluster_ids(arrays),
                n_replicates=n_bootstrap,
                seed=int(config["bootstrap_seed"]),
            )
        elif not np.array_equal(ids, reference_ids):
            raise ValueError(f"{name}: test records differ from clean")

        selectors = selectors_fit.compute(arrays, load_sqi(sqi_dir, name, ids))

        class_rows.append({"condition": name, **classification(arrays)})

        for selector_name, selector in selectors.items():
            aurc, augrc = aurc_augrc(selector, losses)
            row = {
                "condition": name,
                "selector": selector_name,
                "aurc": aurc,
                "augrc": augrc,
            }
            for target, (coverage, risk) in risk_at_coverages(
                selector, losses, coverage_targets,
            ).items():
                row[f"risk_at_{int(round(target * 100))}"] = risk
                row[f"achieved_coverage_{int(round(target * 100))}"] = coverage
            metric_rows.append(row)

        replicates = bootstrap(selectors, losses, weights)
        point = {r["selector"]: r for r in metric_rows if r["condition"] == name}

        for selector_name in selectors:
            if selector_name == "confidence":
                continue
            for metric in ("aurc", "augrc"):
                deltas = (
                    replicates[selector_name][metric]
                    - replicates["confidence"][metric]
                )
                low, high = np.percentile(deltas, [2.5, 97.5])
                contrast_rows.append({
                    "condition": name,
                    "selector": selector_name,
                    "metric": metric,
                    "delta_vs_confidence": (
                        point[selector_name][metric] - point["confidence"][metric]
                    ),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "p_bootstrap": bootstrap_two_sided_p(deltas),
                })

    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_holm"] = np.nan
    # Holm within each condition and metric (one family per stress test).
    for _, index in contrasts.groupby(["condition", "metric"]).groups.items():
        contrasts.loc[index, "p_holm"] = holm_adjust(
            contrasts.loc[index, "p_bootstrap"].to_numpy()
        )
    contrasts["direction"] = np.where(
        contrasts["ci_high"] < 0.0, "better_than_confidence",
        np.where(contrasts["ci_low"] > 0.0, "worse_than_confidence", "inconclusive"),
    )

    pd.DataFrame(metric_rows).to_csv(out_dir / "selector_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out_dir / "classification.csv", index=False)
    contrasts.to_csv(out_dir / "contrasts.csv", index=False)

    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "conditions": names,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": int(config["bootstrap_seed"]),
        "coverage_targets": list(coverage_targets),
        "thresholds": thresholds.tolist(),
        "temperatures": selectors_fit.temperatures.tolist(),
        "stacker_coefficients": selectors_fit.stacker.coefficients(),
        "bsqi_included": selectors_fit.use_sqi,
        "frozen_v1_q_u_references": frozen_reference is not None,
        "counts_by_direction": {
            metric: contrasts.loc[contrasts["metric"].eq(metric), "direction"]
            .value_counts().to_dict()
            for metric in ("aurc", "augrc")
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    clean = pd.DataFrame(metric_rows).query("condition == 'clean'")
    print("\nClean test AURC / AUGRC:")
    print(clean[["selector", "aurc", "augrc", "risk_at_80"]].to_string(index=False))
    print("\nWrote", out_dir)


if __name__ == "__main__":
    main()
