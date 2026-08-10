#!/usr/bin/env python3
"""Assemble one corrected steady split with the unchanged formal split results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


METHODS = ("response_surface", "pinn_data_only", "pinn", "graph", "transolver")
SPLITS = (
    "interleaved_all_ranges",
    "temperature_extrapolation",
    "velocity_extrapolation",
    "heat_source_interpolation",
    "heat_source_extrapolation",
)
REQUIRED_FILES = (
    "summary.json",
    "train_regional_predictions.npz",
    "validation_regional_predictions.npz",
    "test_regional_predictions.npz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_dir(root: Path, namespace: str, method: str, split: str, epochs: int) -> Path:
    return root / f"{namespace}_{method}_{split}_{epochs}epoch"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--source-namespace", default="hccb_p418_60")
    parser.add_argument("--corrected-namespace", required=True)
    parser.add_argument("--output-namespace", required=True)
    parser.add_argument("--corrected-split", default="heat_source_extrapolation")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    root = args.results_root.resolve()
    if args.corrected_split not in SPLITS:
        raise ValueError(f"unknown corrected split: {args.corrected_split}")

    records: list[dict[str, object]] = []
    for method in METHODS:
        for split in SPLITS:
            namespace = (
                args.corrected_namespace
                if split == args.corrected_split
                else args.source_namespace
            )
            source = result_dir(root, namespace, method, split, args.epochs)
            target = result_dir(root, args.output_namespace, method, split, args.epochs)
            if not source.is_dir():
                raise FileNotFoundError(f"missing steady result directory: {source}")
            hashes: dict[str, str] = {}
            for relative in REQUIRED_FILES:
                path = source / relative
                if not path.is_file():
                    raise FileNotFoundError(f"missing completed steady result: {path}")
                hashes[relative] = sha256(path)
            summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
            status = str(summary.get("status", ""))
            if "complete" not in status:
                raise ValueError(f"incomplete steady result status for {source}: {status}")
            records.append(
                {
                    "method": method,
                    "split": split,
                    "source_namespace": namespace,
                    "source_dir": str(source),
                    "target_link": str(target),
                    "status": status,
                    "files": hashes,
                }
            )

    for record in records:
        source = Path(str(record["source_dir"]))
        target = Path(str(record["target_link"]))
        if target.is_symlink():
            if target.resolve() != source:
                raise FileExistsError(f"target link points elsewhere: {target}")
            continue
        if target.exists():
            raise FileExistsError(f"target exists and is not the expected link: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)

    payload = {
        "status": "corrected_steady_result_assembly_complete",
        "source_namespace": args.source_namespace,
        "corrected_namespace": args.corrected_namespace,
        "output_namespace": args.output_namespace,
        "corrected_split": args.corrected_split,
        "epochs": args.epochs,
        "method_count": len(METHODS),
        "split_count": len(SPLITS),
        "result_count": len(records),
        "records": records,
        "new_physical_parameters": [],
    }
    manifest = args.manifest.resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
