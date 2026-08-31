#!/usr/bin/env python3
"""Run frozen Monte Carlo dropout uncertainty on validation data only.

This script never instantiates the PTB-XL test split.

It:
- loads the frozen seed-20260826 best checkpoint,
- keeps BatchNorm in evaluation mode,
- activates only nn.Dropout,
- performs 30 stochastic validation passes,
- saves every MC probability,
- computes multilabel Bernoulli uncertainty,
- verifies model state is unchanged,
- independently recomputes uncertainty from the saved MC draws.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ptbxl_reliability.cached_dataset import CachedPTBXLDataset
from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.model import ResNet1DWang
from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.uncertainty import (
    enable_mc_dropout,
    mc_uncertainty,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "uncertainty_v1.yaml"

RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "v1"
    / "seed_20260826"
)

CHECKPOINT_PATH = RUN_DIR / "best_checkpoint.pt"
BASELINE_METADATA_PATH = RUN_DIR / "run_metadata.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "uncertainty"
    / "v1"
    / "validation"
)

OUTPUT_NPZ = OUTPUT_DIR / "mc_dropout_uncertainty.npz"
OUTPUT_METADATA = OUTPUT_DIR / "uncertainty_metadata.json"

EXPECTED_RECORDS = 2146
EXPECTED_CLASSES = 5
BATCH_SIZE = 128


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")

    return value


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Uncertainty config must be a YAML mapping")

    if config.get("status") != "FROZEN":
        raise ValueError("Uncertainty config is not FROZEN")

    if config["source_model"]["seed"] != 20260826:
        raise ValueError("Unexpected source-model seed")

    if config["source_model"]["checkpoint_epoch"] != 26:
        raise ValueError("Unexpected source-model checkpoint epoch")

    if config["method"]["name"] != "monte_carlo_dropout":
        raise ValueError("Unexpected uncertainty method")

    if config["method"]["mc_passes"] != 30:
        raise ValueError("MC passes changed from frozen value 30")

    if config["inference"]["batchnorm_mode"] != "eval":
        raise ValueError("BatchNorm must remain in eval mode")

    if config["inference"]["dropout_mode"] != "train":
        raise ValueError("Dropout must be active")

    if config["inference"]["gradients"] != "disabled":
        raise ValueError("Gradients must be disabled")

    if config["development_split"]["validation_only"] is not True:
        raise ValueError("Development split must be validation only")

    if config["test_allowed_during_development"] is not False:
        raise ValueError("Test use is forbidden")

    return config


def snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def assert_state_unchanged(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> None:
    if before.keys() != after.keys():
        raise ValueError("Model state keys changed during inference")

    for name in before:
        if not torch.equal(before[name], after[name]):
            raise ValueError(
                f"Model state changed during MC-dropout inference: {name}"
            )


def main() -> None:
    config = load_config()
    baseline_metadata = load_json(BASELINE_METADATA_PATH)

    if baseline_metadata["status"] != "COMPLETE":
        raise ValueError("Baseline run is not COMPLETE")

    if baseline_metadata["seed"] != 20260826:
        raise ValueError("Unexpected baseline seed")

    if baseline_metadata["best_epoch"] != 26:
        raise ValueError("Unexpected best baseline epoch")

    if baseline_metadata["test_records_used"] != 0:
        raise ValueError("Baseline metadata reports test usage")

    if (
        sha256_file(CHECKPOINT_PATH)
        != baseline_metadata["best_checkpoint_sha256"]
    ):
        raise ValueError("Best-checkpoint SHA-256 mismatch")

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required but not available")

    seed = int(config["random_seed"])

    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    dataset = CachedPTBXLDataset("validation")

    if len(dataset) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} validation records, "
            f"found {len(dataset)}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint["epoch"] != 26:
        raise ValueError("Checkpoint epoch mismatch")

    model = ResNet1DWang()

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    device = torch.device("mps")
    model = model.to(device)

    state_before = snapshot_state(model)

    dropout_modules_activated = enable_mc_dropout(model)

    if dropout_modules_activated != 2:
        raise ValueError(
            f"Expected 2 Dropout modules, "
            f"found {dropout_modules_activated}"
        )

    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) and module.training:
            raise ValueError("BatchNorm is in training mode")

    mc_passes = int(config["method"]["mc_passes"])

    all_probabilities = np.empty(
        (
            mc_passes,
            EXPECTED_RECORDS,
            EXPECTED_CLASSES,
        ),
        dtype=np.float32,
    )

    reference_ecg_ids = None
    reference_targets = None

    with torch.inference_mode():
        for pass_index in range(mc_passes):
            offset = 0

            pass_ecg_ids = np.empty(
                EXPECTED_RECORDS,
                dtype=np.int64,
            )

            pass_targets = np.empty(
                (
                    EXPECTED_RECORDS,
                    EXPECTED_CLASSES,
                ),
                dtype=np.float32,
            )

            for batch in loader:
                x = batch["signal"].to(device)
                y = batch["target"]

                logits = model(x)

                if not torch.isfinite(logits).all():
                    raise FloatingPointError(
                        f"Non-finite logits in MC pass "
                        f"{pass_index + 1}"
                    )

                probabilities = torch.sigmoid(logits)

                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError(
                        f"Non-finite probabilities in MC pass "
                        f"{pass_index + 1}"
                    )

                probabilities_cpu = (
                    probabilities
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                batch_size = probabilities_cpu.shape[0]
                end = offset + batch_size

                all_probabilities[
                    pass_index,
                    offset:end,
                    :,
                ] = probabilities_cpu

                ids = batch["ecg_id"]

                if isinstance(ids, torch.Tensor):
                    ids = ids.cpu().numpy()

                pass_ecg_ids[offset:end] = np.asarray(
                    ids,
                    dtype=np.int64,
                )

                pass_targets[offset:end] = (
                    y.cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                offset = end

            if offset != EXPECTED_RECORDS:
                raise ValueError(
                    f"MC pass {pass_index + 1}: expected "
                    f"{EXPECTED_RECORDS} predictions, "
                    f"found {offset}"
                )

            if pass_index == 0:
                reference_ecg_ids = pass_ecg_ids.copy()
                reference_targets = pass_targets.copy()

            else:
                if not np.array_equal(
                    pass_ecg_ids,
                    reference_ecg_ids,
                ):
                    raise ValueError(
                        f"ECG ordering changed in MC pass "
                        f"{pass_index + 1}"
                    )

                if not np.array_equal(
                    pass_targets,
                    reference_targets,
                ):
                    raise ValueError(
                        f"Targets changed in MC pass "
                        f"{pass_index + 1}"
                    )

            print(
                f"MC pass {pass_index + 1:02d}/"
                f"{mc_passes} complete"
            )

    torch.mps.synchronize()

    state_after = snapshot_state(model)

    assert_state_unchanged(
        state_before,
        state_after,
    )

    if not np.isfinite(all_probabilities).all():
        raise ValueError(
            "Saved MC probabilities contain NaN/Inf"
        )

    if (
        (all_probabilities < 0.0).any()
        or (all_probabilities > 1.0).any()
    ):
        raise ValueError(
            "MC probabilities fall outside [0,1]"
        )

    consecutive_difference = float(
        np.mean(
            np.abs(
                all_probabilities[0].astype(np.float64)
                - all_probabilities[1].astype(np.float64)
            ),
            dtype=np.float64,
        )
    )

    if consecutive_difference <= 0.0:
        raise ValueError(
            "First two MC passes are identical; "
            "dropout stochasticity is absent"
        )

    result = mc_uncertainty(
        all_probabilities
    )

    mean_mi = result[
        "mean_mutual_information"
    ]

    if not np.isfinite(mean_mi).all():
        raise ValueError(
            "Mean MI contains NaN/Inf"
        )

    if float(np.max(mean_mi)) <= 0.0:
        raise ValueError(
            "All ECG uncertainty values are zero"
        )

    if float(
        np.std(
            mean_mi,
            ddof=0,
        )
    ) <= 0.0:
        raise ValueError(
            "ECG uncertainty has zero variance"
        )

    ln2 = float(
        np.log(2.0)
    )

    if (
        float(
            np.max(
                result[
                    "predictive_entropy"
                ]
            )
        )
        > ln2 + 1e-10
    ):
        raise ValueError(
            "Bernoulli predictive entropy exceeds ln(2)"
        )

    if (
        float(
            np.max(
                result[
                    "mutual_information"
                ]
            )
        )
        > ln2 + 1e-10
    ):
        raise ValueError(
            "Mutual information exceeds ln(2)"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,

        mc_probabilities=
            all_probabilities,

        mean_probability=
            result[
                "mean_probability"
            ].astype(
                np.float32
            ),

        probability_std=
            result[
                "probability_std"
            ].astype(
                np.float32
            ),

        predictive_entropy=
            result[
                "predictive_entropy"
            ].astype(
                np.float32
            ),

        expected_entropy=
            result[
                "expected_entropy"
            ].astype(
                np.float32
            ),

        mutual_information=
            result[
                "mutual_information"
            ].astype(
                np.float32
            ),

        mean_mutual_information=
            result[
                "mean_mutual_information"
            ].astype(
                np.float32
            ),

        max_mutual_information=
            result[
                "max_mutual_information"
            ].astype(
                np.float32
            ),

        mean_predictive_entropy=
            result[
                "mean_predictive_entropy"
            ].astype(
                np.float32
            ),

        ecg_ids=
            reference_ecg_ids,

        targets=
            reference_targets,

        class_names=
            np.asarray(
                SUPERCLASS_ORDER
            ),
    )

    # Reopen saved MC draws and independently recompute U.
    saved = np.load(
        OUTPUT_NPZ,
        allow_pickle=False,
    )

    recomputed = mc_uncertainty(
        saved["mc_probabilities"]
    )

    np.testing.assert_allclose(
        saved[
            "mean_mutual_information"
        ].astype(
            np.float64
        ),
        recomputed[
            "mean_mutual_information"
        ],
        rtol=1e-6,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        saved[
            "mutual_information"
        ].astype(
            np.float64
        ),
        recomputed[
            "mutual_information"
        ],
        rtol=1e-6,
        atol=1e-8,
    )

    metadata = {
        "uncertainty_id":
            config[
                "uncertainty_id"
            ],

        "created_utc":
            utc_now(),

        "evaluation_split":
            "validation",

        "records":
            EXPECTED_RECORDS,

        "classes":
            list(
                SUPERCLASS_ORDER
            ),

        "mc_passes":
            mc_passes,

        "random_seed":
            seed,

        "device":
            "mps",

        "dropout_modules_activated":
            dropout_modules_activated,

        "batchnorm_state_unchanged":
            True,

        "entire_model_state_unchanged":
            True,

        "first_two_pass_mean_absolute_probability_difference":
            consecutive_difference,

        "mean_mutual_information_summary": {
            "min":
                float(
                    np.min(
                        mean_mi
                    )
                ),

            "mean":
                float(
                    np.mean(
                        mean_mi,
                        dtype=np.float64,
                    )
                ),

            "median":
                float(
                    np.median(
                        mean_mi
                    )
                ),

            "std":
                float(
                    np.std(
                        mean_mi,
                        ddof=0,
                        dtype=np.float64,
                    )
                ),

            "max":
                float(
                    np.max(
                        mean_mi
                    )
                ),
        },

        "mean_predictive_entropy_summary": {
            "min":
                float(
                    np.min(
                        result[
                            "mean_predictive_entropy"
                        ]
                    )
                ),

            "mean":
                float(
                    np.mean(
                        result[
                            "mean_predictive_entropy"
                        ],
                        dtype=np.float64,
                    )
                ),

            "median":
                float(
                    np.median(
                        result[
                            "mean_predictive_entropy"
                        ]
                    )
                ),

            "std":
                float(
                    np.std(
                        result[
                            "mean_predictive_entropy"
                        ],
                        ddof=0,
                        dtype=np.float64,
                    )
                ),

            "max":
                float(
                    np.max(
                        result[
                            "mean_predictive_entropy"
                        ]
                    )
                ),
        },

        "test_records_used":
            0,

        "source_checkpoint_sha256":
            sha256_file(
                CHECKPOINT_PATH
            ),

        "config_sha256":
            sha256_file(
                CONFIG_PATH
            ),

        "output_npz_sha256":
            sha256_file(
                OUTPUT_NPZ
            ),

        "script_sha256":
            sha256_file(
                Path(__file__).resolve()
            ),

        "saved_artifact_recomputation":
            "PASS",

        "status":
            "FROZEN",
    }

    with OUTPUT_METADATA.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print()
    print(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
