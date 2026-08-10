#!/usr/bin/env python3
"""Merge train, validation and test regional predictions after evaluation.

The merged file is only a convenient post-processing view.  It does not alter
the split used for fitting or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CASE_KEYS = (
    "condition_id",
    "condition_normalized",
    "baseline_state_normalized",
    "target_state_normalized",
)
SHARED_KEYS = ("node_type", "node_volume_m3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        missing = sorted(set(CASE_KEYS + SHARED_KEYS) - set(values.files))
        if missing:
            raise ValueError(f"{path} lacks prediction arrays: {missing}")
        return {key: values[key] for key in CASE_KEYS + SHARED_KEYS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    inputs = [path.resolve() for path in args.input]
    if len(inputs) < 2:
        raise ValueError("at least two prediction split files are required")
    loaded = [load(path) for path in inputs]
    reference = loaded[0]
    for path, current in zip(inputs[1:], loaded[1:]):
        if not np.array_equal(current["node_type"], reference["node_type"]):
            raise ValueError(f"node_type differs in {path}")
        if not np.array_equal(current["node_volume_m3"], reference["node_volume_m3"]):
            raise ValueError(f"node_volume_m3 differs in {path}")

    combined = {
        key: np.concatenate([current[key] for current in loaded], axis=0)
        for key in CASE_KEYS
    }
    identifiers = [str(value) for value in combined["condition_id"]]
    if len(identifiers) != len(set(identifiers)):
        repeated = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        raise ValueError(f"prediction splits repeat conditions: {repeated}")
    order = np.argsort(np.asarray(identifiers))
    arrays = {key: value[order] for key, value in combined.items()}
    arrays.update({key: reference[key] for key in SHARED_KEYS})

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else output.with_name(output.stem + "_summary.json")
    )
    payload = {
        "status": "prediction_splits_merged_for_postprocessing",
        "condition_count": len(identifiers),
        "condition_ids": sorted(identifiers),
        "input_files": [
            {"path": str(path), "sha256": sha256(path)} for path in inputs
        ],
        "output_file": str(output),
        "output_sha256": sha256(output),
        "interpretation_cn": (
            "该文件只合并已经分别生成的训练、检查和独立预测结果，"
            "不改变原来的模型训练和工况划分。"
        ),
        "new_physical_parameters": [],
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
