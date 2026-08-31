#!/usr/bin/env python3
"""Training-pipeline sanity checks before real PTB-XL training.

This script is NOT a scientific experiment.

It verifies:
1. correct forward pass,
2. BCEWithLogitsLoss is finite,
3. gradients are finite,
4. parameters receive gradients,
5. an optimizer step changes parameters,
6. Apple MPS can execute forward/backward/update,
7. the model can deliberately overfit a tiny fixed training subset.

No validation or test ECG is used.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from ptbxl_reliability.dataset import (
    PTBXLSuperdiagnosticDataset,
)
from ptbxl_reliability.labels import SUPERCLASS_ORDER
from ptbxl_reliability.model import (
    ResNet1DWang,
    count_trainable_parameters,
)
from ptbxl_reliability.paths import PROJECT_ROOT


SEED = 20260826
TINY_SUBSET_SIZE = 16
OVERFIT_STEPS = 300
OVERFIT_LR = 1e-3

OUTPUT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "training_sanity_audit.json"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_finite_tensor(
    tensor: torch.Tensor,
    name: str,
) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{name} contains NaN or Inf"
        )


def make_tiny_training_batch(
    dataset: PTBXLSuperdiagnosticDataset,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Select a deterministic tiny subset containing all five labels.

    Selection uses training data only.

    We greedily add training ECGs until every superclass has at least
    two positive examples, then fill deterministically to 16 records.
    """

    selected_indices: list[int] = []
    positive_counts = np.zeros(
        len(SUPERCLASS_ORDER),
        dtype=np.int64,
    )

    for index in range(len(dataset)):
        row = dataset.rows.iloc[index]

        target = np.asarray(
            [
                int(row[label])
                for label in SUPERCLASS_ORDER
            ],
            dtype=np.int64,
        )

        contributes = np.any(
            (target == 1)
            & (positive_counts < 2)
        )

        if contributes:
            selected_indices.append(index)
            positive_counts += target

        if np.all(positive_counts >= 2):
            break

    if not np.all(
        positive_counts >= 2
    ):
        raise ValueError(
            "Could not construct tiny subset with "
            "at least two positives per superclass"
        )

    selected_set = set(selected_indices)

    for index in range(len(dataset)):
        if len(selected_indices) >= TINY_SUBSET_SIZE:
            break

        if index not in selected_set:
            selected_indices.append(index)
            selected_set.add(index)

    if len(selected_indices) != TINY_SUBSET_SIZE:
        raise ValueError(
            f"Expected {TINY_SUBSET_SIZE} tiny-subset ECGs, "
            f"found {len(selected_indices)}"
        )

    samples = [
        dataset[index]
        for index in selected_indices
    ]

    signals = torch.stack(
        [
            sample["signal"]
            for sample in samples
        ],
        dim=0,
    )

    targets = torch.stack(
        [
            sample["target"]
            for sample in samples
        ],
        dim=0,
    )

    ecg_ids = [
        int(sample["ecg_id"])
        for sample in samples
    ]

    if signals.shape != (
        TINY_SUBSET_SIZE,
        12,
        1000,
    ):
        raise AssertionError(
            f"Unexpected tiny signal shape: "
            f"{signals.shape}"
        )

    if targets.shape != (
        TINY_SUBSET_SIZE,
        5,
    ):
        raise AssertionError(
            f"Unexpected tiny target shape: "
            f"{targets.shape}"
        )

    # Every class must have positive AND negative examples.
    positives = targets.sum(dim=0)

    if not torch.all(positives >= 2):
        raise ValueError(
            "Tiny subset lacks sufficient positive labels"
        )

    negatives = (
        TINY_SUBSET_SIZE - positives
    )

    if not torch.all(negatives >= 2):
        raise ValueError(
            "Tiny subset lacks sufficient negative labels"
        )

    return signals, targets, ecg_ids


def gradient_and_update_check(
    device: torch.device,
    signal: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, object]:
    """Verify one complete optimization step."""

    set_seed(SEED)

    model = ResNet1DWang().to(device)

    model.train()

    x = signal.to(device)
    y = target.to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    }

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(x)

    assert_finite_tensor(
        logits,
        "logits",
    )

    loss = loss_fn(
        logits,
        y,
    )

    assert_finite_tensor(
        loss,
        "loss",
    )

    loss.backward()

    parameters_with_gradient = 0
    gradient_elements = 0
    maximum_absolute_gradient = 0.0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        gradient = parameter.grad

        if gradient is None:
            raise ValueError(
                f"Trainable parameter {name!r} "
                "received no gradient"
            )

        assert_finite_tensor(
            gradient,
            f"gradient:{name}",
        )

        parameters_with_gradient += 1
        gradient_elements += gradient.numel()

        maximum_absolute_gradient = max(
            maximum_absolute_gradient,
            float(
                gradient.detach()
                .abs()
                .max()
                .cpu()
                .item()
            ),
        )

    if parameters_with_gradient == 0:
        raise ValueError(
            "No trainable parameters received gradients"
        )

    if not math.isfinite(
        maximum_absolute_gradient
    ):
        raise ValueError(
            "Maximum gradient is not finite"
        )

    optimizer.step()

    changed_parameters = 0
    unchanged_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        previous = before[name]

        current = (
            parameter.detach()
            .cpu()
        )

        if torch.equal(
            previous,
            current,
        ):
            unchanged_parameters.append(
                name
            )
        else:
            changed_parameters += 1

    if changed_parameters == 0:
        raise ValueError(
            "Optimizer step changed no trainable parameters"
        )

    return {
        "device": str(device),
        "loss": float(
            loss.detach()
            .cpu()
            .item()
        ),
        "parameters_with_gradient":
            parameters_with_gradient,
        "gradient_elements":
            gradient_elements,
        "maximum_absolute_gradient":
            maximum_absolute_gradient,
        "changed_parameter_tensors":
            changed_parameters,
        "unchanged_parameter_tensors":
            unchanged_parameters,
        "status": "PASS",
    }


def tiny_overfit_check(
    signals: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, object]:
    """Deliberately overfit 16 training ECGs on CPU.

    This is a software sanity test, not a performance experiment.
    """

    set_seed(SEED)

    device = torch.device(
        "cpu"
    )

    model = ResNet1DWang().to(
        device
    )

    model.train()

    x = signals.to(
        device
    )

    y = targets.to(
        device
    )

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=OVERFIT_LR,
    )

    losses: list[float] = []

    for step in range(
        OVERFIT_STEPS
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = loss_fn(
            logits,
            y,
        )

        assert_finite_tensor(
            loss,
            "tiny-overfit loss",
        )

        loss.backward()

        for name, parameter in model.named_parameters():
            if (
                parameter.requires_grad
                and parameter.grad is not None
            ):
                assert_finite_tensor(
                    parameter.grad,
                    f"tiny gradient:{name}",
                )

        optimizer.step()

        loss_value = float(
            loss.detach().cpu().item()
        )

        losses.append(
            loss_value
        )

        if (
            step == 0
            or (step + 1) % 50 == 0
        ):
            print(
                f"Tiny overfit step "
                f"{step + 1:03d}/"
                f"{OVERFIT_STEPS}: "
                f"loss={loss_value:.6f}"
            )

    model.eval()

    with torch.no_grad():
        final_logits = model(x)

        final_loss = loss_fn(
            final_logits,
            y,
        )

        probabilities = torch.sigmoid(
            final_logits
        )

        predictions = (
            probabilities >= 0.5
        ).to(
            dtype=torch.float32
        )

    assert_finite_tensor(
        final_logits,
        "final logits",
    )

    assert_finite_tensor(
        probabilities,
        "final probabilities",
    )

    initial_loss = losses[0]

    final_eval_loss = float(
        final_loss.item()
    )

    reduction_ratio = (
        final_eval_loss
        / initial_loss
    )

    exact_binary_accuracy = float(
        (
            predictions == y
        )
        .float()
        .mean()
        .item()
    )

    exact_record_match = float(
        (
            predictions == y
        )
        .all(dim=1)
        .float()
        .mean()
        .item()
    )

    # The purpose is not to demand a scientifically meaningful threshold.
    # We only require strong evidence that the network can memorize
    # a tiny fixed dataset.
    if reduction_ratio >= 0.35:
        raise ValueError(
            "Tiny-subset loss did not decrease enough: "
            f"initial={initial_loss:.6f}, "
            f"final={final_eval_loss:.6f}, "
            f"ratio={reduction_ratio:.6f}"
        )

    if exact_binary_accuracy < 0.95:
        raise ValueError(
            "Tiny-subset binary accuracy is too low "
            f"for an overfit sanity test: "
            f"{exact_binary_accuracy:.6f}"
        )

    return {
        "device": "cpu",
        "seed": SEED,
        "records": TINY_SUBSET_SIZE,
        "steps": OVERFIT_STEPS,
        "learning_rate": OVERFIT_LR,
        "initial_training_loss":
            initial_loss,
        "final_eval_loss":
            final_eval_loss,
        "loss_ratio":
            reduction_ratio,
        "binary_label_accuracy":
            exact_binary_accuracy,
        "exact_record_match_fraction":
            exact_record_match,
        "status": "PASS",
    }


def main() -> None:
    set_seed(SEED)

    dataset = PTBXLSuperdiagnosticDataset(
        "train"
    )

    signals, targets, ecg_ids = (
        make_tiny_training_batch(
            dataset
        )
    )

    if not torch.isfinite(
        signals
    ).all():
        raise ValueError(
            "Tiny subset signals contain NaN/Inf"
        )

    if not torch.isfinite(
        targets
    ).all():
        raise ValueError(
            "Tiny subset targets contain NaN/Inf"
        )

    positive_counts = {
        label: int(
            targets[:, index]
            .sum()
            .item()
        )
        for index, label
        in enumerate(
            SUPERCLASS_ORDER
        )
    }

    cpu_step = gradient_and_update_check(
        torch.device("cpu"),
        signals[:8],
        targets[:8],
    )

    mps_available = (
        torch.backends.mps.is_available()
    )

    if not mps_available:
        raise RuntimeError(
            "MPS was previously available but "
            "is not available now"
        )

    mps_step = gradient_and_update_check(
        torch.device("mps"),
        signals[:8],
        targets[:8],
    )

    overfit = tiny_overfit_check(
        signals,
        targets,
    )

    report = {
        "audit_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "purpose":
            "pre-training software sanity test",

        "scientific_result":
            False,

        "seed":
            SEED,

        "tiny_subset_ecg_ids":
            ecg_ids,

        "tiny_subset_positive_counts":
            positive_counts,

        "model_trainable_parameters":
            count_trainable_parameters(
                ResNet1DWang()
            ),

        "cpu_single_step":
            cpu_step,

        "mps_single_step":
            mps_step,

        "tiny_cpu_overfit":
            overfit,

        "validation_records_used":
            0,

        "test_records_used":
            0,

        "status":
            "PASS",
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print()
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
