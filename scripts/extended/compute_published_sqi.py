#!/usr/bin/env python3
"""Record-level SQIs (bSQI etc.) on the evaluated 100-Hz waveforms.

Model-independent: run once per dataset.

python scripts/extended/compute_published_sqi.py --dataset ptbxl \
    --conditions validation frozen nested recorded --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ptbxl_reliability.paths import PROJECT_ROOT
from ptbxl_reliability.published_sqi import SQI_FIELDS, record_sqi


INFERENCE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "extended" / "run_extended_inference.py"
)


def load_inference_module():
    spec = importlib.util.spec_from_file_location(
        "extended_inference",
        INFERENCE_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STATE: dict = {}


def _init(raw, record_ids, condition, nstdb):
    inference = load_inference_module()
    _STATE.update(
        raw=raw,
        record_ids=record_ids,
        condition=condition,
        nstdb=nstdb,
        inference=inference,
        helpers=inference.load_ptb_helpers(),
    )


def _work(index: int) -> list[float]:
    s = _STATE
    x = s["inference"].corrupt(
        s["raw"][index].astype(np.float64, copy=False),
        int(s["record_ids"][index]),
        s["condition"],
        s["helpers"],
        s["nstdb"],
    )
    values = record_sqi(x, fs=100.0)
    return [values[field] for field in SQI_FIELDS]


def compute(split, condition, nstdb, workers: int) -> dict[str, np.ndarray]:
    n = len(split.raw)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init,
        initargs=(split.raw, split.record_ids, condition, nstdb),
    ) as pool:
        rows = list(pool.map(_work, range(n), chunksize=32))
    table = np.asarray(rows, dtype=np.float64)
    return {
        "record_ids": split.record_ids,
        **{field: table[:, j] for j, field in enumerate(SQI_FIELDS)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ptbxl", "chapman"), required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("validation", "frozen", "nested", "recorded"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    inference = load_inference_module()
    helpers = inference.load_ptb_helpers()

    out_dir = inference.EXTENDED_ROOT / args.dataset / "published_sqi"
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_split(split):
        return (
            inference.ptbxl_split(helpers, split) if args.dataset == "ptbxl"
            else inference.chapman_split(split)
        )

    if "validation" in args.conditions:
        artifact = out_dir / "validation_clean.npz"
        if not artifact.exists():
            print("[validation_clean]", flush=True)
            np.savez_compressed(
                artifact, **compute(get_split("validation"), None, None, args.workers)
            )

    test_sets = [c for c in args.conditions if c != "validation"]
    if not test_sets:
        return

    conditions = inference.condition_list(test_sets, helpers)
    nstdb = inference.load_nstdb() if "recorded" in test_sets else None
    test = get_split("test")

    for condition in conditions:
        name = "clean" if condition is None else condition["name"]
        artifact = out_dir / f"{name}.npz"
        if artifact.exists():
            print(f"[{name}] reusing", flush=True)
            continue
        print(f"[{name}]", flush=True)
        np.savez_compressed(
            artifact, **compute(test, condition, nstdb, args.workers)
        )


if __name__ == "__main__":
    main()
