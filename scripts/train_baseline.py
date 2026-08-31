#!/usr/bin/env python3
"""Train one pre-specified PTB-XL baseline seed.

SCIENTIFIC BOUNDARY
-------------------
This script uses ONLY:
- training split: folds 1-8
- validation split: fold 9

It NEVER instantiates or evaluates fold 10.

The best checkpoint is selected only by validation macro-AUROC.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ptbxl_reliability.cached_dataset import (
    CACHE_METADATA_PATH,
    CachedPTBXLDataset,
)
from ptbxl_reliability.integrity import sha256_file
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.metrics import (
    classwise_auroc,
    macro_auroc,
)
from ptbxl_reliability.model import (
    ResNet1DWang,
    count_trainable_parameters,
)
from ptbxl_reliability.paths import PROJECT_ROOT


CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "baseline_v1.yaml"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "v1"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "Baseline config must be a YAML mapping"
        )

    return config


def validate_config(config: dict) -> None:
    """Fail if the frozen scientific protocol has changed."""

    if config["task"]["type"] != "multilabel":
        raise ValueError("Task must be multilabel")

    if tuple(
        config["task"]["classes"]
    ) != SUPERCLASS_ORDER:
        raise ValueError(
            "Class order differs from frozen label order"
        )

    if config["model"]["architecture"] != "ResNet1DWang":
        raise ValueError(
            "Unexpected model architecture"
        )

    if config["model"]["input_shape"] != [12, 1000]:
        raise ValueError(
            "Unexpected model input shape"
        )

    if config["loss"]["name"] != "BCEWithLogitsLoss":
        raise ValueError(
            "Unexpected loss"
        )

    if (
        config["loss"]["positive_class_weights"]
        is not None
    ):
        raise ValueError(
            "Baseline v1 must not use class weights"
        )

    training = config["training"]

    expected_training = {
        "epochs": 50,
        "batch_size": 128,
        "optimizer": "AdamW",
        "max_learning_rate": 0.01,
        "weight_decay": 0.01,
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1.0e-8,
        "gradient_clipping": None,
    }

    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(
                f"Frozen training setting {key!r} "
                f"changed: expected {expected!r}, "
                f"found {training.get(key)!r}"
            )

    scheduler = config["scheduler"]

    expected_scheduler = {
        "name": "OneCycleLR",
        "step_frequency": "batch",
        "pct_start": 0.30,
        "anneal_strategy": "cos",
        "div_factor": 25.0,
        "final_div_factor": 10000.0,
        "cycle_momentum": True,
        "base_momentum": 0.85,
        "max_momentum": 0.95,
        "three_phase": False,
    }

    for key, expected in expected_scheduler.items():
        if scheduler.get(key) != expected:
            raise ValueError(
                f"Frozen scheduler setting {key!r} changed"
            )

    checkpoint = config[
        "checkpoint_selection"
    ]

    if checkpoint != {
        "split": "validation",
        "metric": "macro_auroc",
        "mode": "max",
        "tie_policy": "keep_earliest",
        "test_allowed": False,
    }:
        raise ValueError(
            "Checkpoint-selection protocol changed"
        )

    if config["splits"]["train"] != "folds_1_to_8":
        raise ValueError(
            "Unexpected training split"
        )

    if config["splits"]["validation"] != "fold_9":
        raise ValueError(
            "Unexpected validation split"
        )

    if config["splits"]["test"] != "fold_10":
        raise ValueError(
            "Unexpected test split declaration"
        )

    if config["thresholds"][
        "optimized_during_training"
    ] is not False:
        raise ValueError(
            "Threshold optimization is forbidden here"
        )


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is required by baseline_v1 "
            "but is not available"
        )

    torch.mps.manual_seed(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def make_loaders(
    config: dict,
    seed: int,
) -> tuple[
    DataLoader,
    DataLoader,
]:
    """Construct train and validation loaders only."""

    train_dataset = CachedPTBXLDataset(
        "train"
    )

    validation_dataset = CachedPTBXLDataset(
        "validation"
    )

    if len(train_dataset) != 17084:
        raise ValueError(
            "Unexpected training-set size"
        )

    if len(validation_dataset) != 2146:
        raise ValueError(
            "Unexpected validation-set size"
        )

    batch_size = int(
        config["training"]["batch_size"]
    )

    loader_config = config[
        "data_loader"
    ]

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(
            loader_config["num_workers"]
        ),
        drop_last=bool(
            loader_config["drop_last"]
        ),
        pin_memory=bool(
            loader_config["pin_memory"]
        ),
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    return (
        train_loader,
        validation_loader,
    )


def build_optimizer_and_scheduler(
    model: nn.Module,
    train_loader: DataLoader,
    config: dict,
):
    training = config["training"]
    schedule = config["scheduler"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            training["max_learning_rate"]
        ),
        betas=tuple(
            float(value)
            for value in training["adam_betas"]
        ),
        eps=float(
            training["adam_epsilon"]
        ),
        weight_decay=float(
            training["weight_decay"]
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(
                training["max_learning_rate"]
            ),
            epochs=int(
                training["epochs"]
            ),
            steps_per_epoch=len(
                train_loader
            ),
            pct_start=float(
                schedule["pct_start"]
            ),
            anneal_strategy=str(
                schedule["anneal_strategy"]
            ),
            cycle_momentum=bool(
                schedule["cycle_momentum"]
            ),
            base_momentum=float(
                schedule["base_momentum"]
            ),
            max_momentum=float(
                schedule["max_momentum"]
            ),
            div_factor=float(
                schedule["div_factor"]
            ),
            final_div_factor=float(
                schedule["final_div_factor"]
            ),
            three_phase=bool(
                schedule["three_phase"]
            ),
        )
    )

    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:

    model.train()

    total_loss_sum = 0.0
    total_elements = 0

    learning_rates: list[float] = []

    for batch in loader:
        x = batch["signal"].to(
            device
        )

        y = batch["target"].to(
            device
        )

        if x.shape[1:] != (
            12,
            1000,
        ):
            raise ValueError(
                f"Invalid training input shape: "
                f"{tuple(x.shape)}"
            )

        if y.shape[1:] != (5,):
            raise ValueError(
                f"Invalid training target shape: "
                f"{tuple(y.shape)}"
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = loss_fn(
            logits,
            y,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss"
            )

        loss.backward()

        # max_norm=inf performs no clipping.
        # It is used only to detect NaN/Inf gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
            error_if_nonfinite=True,
            foreach=False,
        )

        learning_rates.append(
            float(
                optimizer.param_groups[0][
                    "lr"
                ]
            )
        )

        optimizer.step()

        # PyTorch OneCycleLR is stepped after
        # every optimizer update.
        scheduler.step()

        elements = int(
            y.numel()
        )

        total_loss_sum += (
            float(
                loss.detach()
                .cpu()
                .item()
            )
            * elements
        )

        total_elements += elements

    if total_elements != (
        len(loader.dataset) * 5
    ):
        raise AssertionError(
            "Training loss denominator mismatch"
        )

    return {
        "loss":
            total_loss_sum
            / total_elements,

        "lr_first":
            learning_rates[0],

        "lr_last":
            learning_rates[-1],

        "lr_min":
            min(learning_rates),

        "lr_max":
            max(learning_rates),
    }


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, object]:

    model.eval()

    probability_parts = []
    target_parts = []
    ecg_id_parts = []

    total_loss_sum = 0.0
    total_elements = 0

    for batch in loader:
        x = batch["signal"].to(
            device
        )

        y = batch["target"].to(
            device
        )

        logits = model(x)

        loss = loss_fn(
            logits,
            y,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite validation loss"
            )

        probabilities = torch.sigmoid(
            logits
        )

        if not torch.isfinite(
            probabilities
        ).all():
            raise FloatingPointError(
                "Non-finite validation probabilities"
            )

        probability_parts.append(
            probabilities.cpu().numpy()
        )

        target_parts.append(
            y.cpu().numpy()
        )

        ids = batch["ecg_id"]

        if isinstance(
            ids,
            torch.Tensor,
        ):
            ids = ids.cpu().numpy()

        ecg_id_parts.append(
            np.asarray(
                ids,
                dtype=np.int64,
            )
        )

        elements = int(
            y.numel()
        )

        total_loss_sum += (
            float(
                loss.cpu().item()
            )
            * elements
        )

        total_elements += elements

    probabilities = np.concatenate(
        probability_parts,
        axis=0,
    )

    targets = np.concatenate(
        target_parts,
        axis=0,
    )

    ecg_ids = np.concatenate(
        ecg_id_parts,
        axis=0,
    )

    if probabilities.shape != (
        2146,
        5,
    ):
        raise ValueError(
            "Unexpected validation prediction shape"
        )

    if targets.shape != (
        2146,
        5,
    ):
        raise ValueError(
            "Unexpected validation target shape"
        )

    if ecg_ids.shape != (
        2146,
    ):
        raise ValueError(
            "Unexpected validation ECG-ID shape"
        )

    if total_elements != 2146 * 5:
        raise AssertionError(
            "Validation loss denominator mismatch"
        )

    class_auc = classwise_auroc(
        targets,
        probabilities,
    )

    macro_auc = macro_auroc(
        targets,
        probabilities,
    )

    # Independent consistency check.
    if not np.isclose(
        macro_auc,
        np.mean(
            class_auc,
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError(
            "Macro-AUROC aggregation mismatch"
        )

    return {
        "loss":
            total_loss_sum
            / total_elements,

        "macro_auroc":
            float(macro_auc),

        "classwise_auroc": {
            label: float(score)
            for label, score
            in zip(
                SUPERCLASS_ORDER,
                class_auc,
                strict=True,
            )
        },

        "probabilities":
            probabilities,

        "targets":
            targets,

        "ecg_ids":
            ecg_ids,
    }


def cpu_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name:
            tensor.detach()
            .cpu()
            .clone()
        for name, tensor
        in model.state_dict().items()
    }


def package_versions() -> dict[str, str]:
    packages = (
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "PyYAML",
    )

    return {
        name: version(name)
        for name in packages
    }


def preflight(
    config: dict,
    seed: int,
) -> None:
    """One full batch at the real batch size.

    This is NOT a scientific result.
    """

    set_reproducibility(seed)

    train_loader, _ = make_loaders(
        config,
        seed,
    )

    device = torch.device(
        "mps"
    )

    model = ResNet1DWang().to(
        device
    )

    optimizer, scheduler = (
        build_optimizer_and_scheduler(
            model,
            train_loader,
            config,
        )
    )

    loss_fn = nn.BCEWithLogitsLoss()

    batch = next(
        iter(train_loader)
    )

    x = batch["signal"].to(
        device
    )

    y = batch["target"].to(
        device
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(x)

    loss = loss_fn(
        logits,
        y,
    )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Preflight loss is non-finite"
        )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float("inf"),
        error_if_nonfinite=True,
        foreach=False,
    )

    optimizer.step()
    scheduler.step()

    torch.mps.synchronize()

    print(
        "PRE-FLIGHT PASS\n"
        f"batch_size={x.shape[0]}\n"
        f"signal_shape={tuple(x.shape)}\n"
        f"target_shape={tuple(y.shape)}\n"
        f"loss={float(loss.detach().cpu().item()):.6f}\n"
        "No scientific result was saved."
    )


def run_training(
    config: dict,
    seed: int,
) -> None:

    if seed not in set(
        int(value)
        for value in config["seeds"]
    ):
        raise ValueError(
            f"Seed {seed} is not pre-specified"
        )

    set_reproducibility(seed)

    run_dir = (
        RESULT_ROOT
        / f"seed_{seed}"
    )

    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists:\n"
            f"{run_dir}\n"
            "Refusing to overwrite a prior run."
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    device = torch.device(
        "mps"
    )

    train_loader, validation_loader = (
        make_loaders(
            config,
            seed,
        )
    )

    model = ResNet1DWang().to(
        device
    )

    if count_trainable_parameters(
        model
    ) != 473349:
        raise ValueError(
            "Model parameter count changed"
        )

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer, scheduler = (
        build_optimizer_and_scheduler(
            model,
            train_loader,
            config,
        )
    )

    history = []

    best_macro_auc = -np.inf
    best_epoch = None

    best_checkpoint_path = (
        run_dir
        / "best_checkpoint.pt"
    )

    status_path = (
        run_dir
        / "run_metadata.json"
    )

    metadata = {
        "experiment_id":
            config["experiment_id"],

        "seed":
            seed,

        "started_utc":
            utc_now(),

        "status":
            "RUNNING",

        "device":
            "mps",

        "train_records":
            17084,

        "validation_records":
            2146,

        "test_records_used":
            0,

        "config_sha256":
            sha256_file(
                CONFIG_PATH
            ),

        "cache_metadata_sha256":
            sha256_file(
                CACHE_METADATA_PATH
            ),

        "model_source_sha256":
            sha256_file(
                PROJECT_ROOT
                / "src"
                / "ptbxl_reliability"
                / "model.py"
            ),

        "metrics_source_sha256":
            sha256_file(
                PROJECT_ROOT
                / "src"
                / "ptbxl_reliability"
                / "metrics.py"
            ),

        "training_script_sha256":
            sha256_file(
                Path(__file__).resolve()
            ),

        "software_versions":
            package_versions(),

        "python":
            sys.version,

        "best_epoch":
            None,

        "best_validation_macro_auroc":
            None,
    }

    with status_path.open(
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

    epochs = int(
        config["training"]["epochs"]
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        training_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
        )

        validation_metrics = (
            evaluate_validation(
                model,
                validation_loader,
                loss_fn,
                device,
            )
        )

        current_auc = float(
            validation_metrics[
                "macro_auroc"
            ]
        )

        improved = (
            current_auc
            > best_macro_auc
        )

        if improved:
            best_macro_auc = current_auc
            best_epoch = epoch

            checkpoint = {
                "experiment_id":
                    config["experiment_id"],

                "seed":
                    seed,

                "epoch":
                    epoch,

                "validation_macro_auroc":
                    current_auc,

                "validation_classwise_auroc":
                    validation_metrics[
                        "classwise_auroc"
                    ],

                "model_state_dict":
                    cpu_state_dict(
                        model
                    ),

                "config_sha256":
                    sha256_file(
                        CONFIG_PATH
                    ),
            }

            torch.save(
                checkpoint,
                best_checkpoint_path,
            )

        epoch_record = {
            "epoch":
                epoch,

            "train_loss":
                float(
                    training_metrics["loss"]
                ),

            "validation_loss":
                float(
                    validation_metrics["loss"]
                ),

            "validation_macro_auroc":
                current_auc,

            "validation_classwise_auroc":
                validation_metrics[
                    "classwise_auroc"
                ],

            "lr_first":
                training_metrics["lr_first"],

            "lr_last":
                training_metrics["lr_last"],

            "lr_min":
                training_metrics["lr_min"],

            "lr_max":
                training_metrics["lr_max"],

            "best_so_far":
                bool(improved),
        }

        history.append(
            epoch_record
        )

        with (
            run_dir
            / "history.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                history,
                handle,
                indent=2,
            )
            handle.write("\n")

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss="
            f"{training_metrics['loss']:.6f} | "
            f"val_loss="
            f"{validation_metrics['loss']:.6f} | "
            f"val_macro_AUROC="
            f"{current_auc:.6f}"
            + (
                " | BEST"
                if improved
                else ""
            )
        )

    if best_epoch is None:
        raise RuntimeError(
            "No best checkpoint was created"
        )

    # Re-load the SAVED checkpoint and independently
    # re-evaluate it. This verifies that the persisted
    # checkpoint reproduces the selected score.
    saved = torch.load(
        best_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    verification_model = (
        ResNet1DWang()
    )

    verification_model.load_state_dict(
        saved["model_state_dict"],
        strict=True,
    )

    verification_model = (
        verification_model.to(
            device
        )
    )

    verification = evaluate_validation(
        verification_model,
        validation_loader,
        loss_fn,
        device,
    )

    reproduced_auc = float(
        verification["macro_auroc"]
    )

    if not np.isclose(
        reproduced_auc,
        best_macro_auc,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Saved best checkpoint did not reproduce "
            "the selected validation AUROC:\n"
            f"selected={best_macro_auc}\n"
            f"reproduced={reproduced_auc}"
        )

    np.savez_compressed(
        run_dir
        / "best_validation_predictions.npz",

        probabilities=verification[
            "probabilities"
        ].astype(
            np.float32
        ),

        targets=verification[
            "targets"
        ].astype(
            np.float32
        ),

        ecg_ids=verification[
            "ecg_ids"
        ].astype(
            np.int64
        ),

        class_names=np.asarray(
            SUPERCLASS_ORDER
        ),
    )

    checkpoint_hash = sha256_file(
        best_checkpoint_path
    )

    prediction_hash = sha256_file(
        run_dir
        / "best_validation_predictions.npz"
    )

    metadata.update(
        {
            "completed_utc":
                utc_now(),

            "status":
                "COMPLETE",

            "epochs_completed":
                epochs,

            "best_epoch":
                int(best_epoch),

            "best_validation_macro_auroc":
                float(best_macro_auc),

            "reproduced_validation_macro_auroc":
                reproduced_auc,

            "best_validation_classwise_auroc":
                verification[
                    "classwise_auroc"
                ],

            "best_checkpoint_sha256":
                checkpoint_hash,

            "best_validation_predictions_sha256":
                prediction_hash,

            "test_records_used":
                0,
        }
    )

    with status_path.open(
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
        "TRAINING COMPLETE\n"
        f"seed={seed}\n"
        f"best_epoch={best_epoch}\n"
        f"best_validation_macro_AUROC="
        f"{best_macro_auc:.6f}\n"
        f"checkpoint_reproduced="
        f"{reproduced_auc:.6f}\n"
        "test_records_used=0"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=20260826,
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Run one real-size MPS batch only; "
            "save no scientific result."
        ),
    )

    args = parser.parse_args()

    config = load_config()

    validate_config(
        config
    )

    if args.preflight:
        preflight(
            config,
            args.seed,
        )
        return

    run_training(
        config,
        args.seed,
    )


if __name__ == "__main__":
    main()
