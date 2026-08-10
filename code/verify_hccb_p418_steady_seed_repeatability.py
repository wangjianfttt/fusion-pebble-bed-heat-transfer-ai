#!/usr/bin/env python3
"""Compare repeated same-seed and different-seed steady neural predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


ARCHITECTURES = ("pinn_data_only", "pinn", "graph", "transolver")


def directory(
    root: Path,
    prefix: str,
    architecture: str,
    split_name: str,
    epochs: int,
    seed: Optional[int] = None,
) -> Path:
    suffix = "" if seed is None else f"_seed{seed}"
    return root / f"{prefix}_{architecture}_{split_name}_{epochs}epoch{suffix}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(directory_path: Path, expected_seed: int) -> tuple[dict, np.ndarray]:
    summary_path = directory_path / "summary.json"
    prediction_path = directory_path / "test_regional_predictions.npz"
    if not summary_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"missing steady repeatability result in {directory_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("training_seed", -1)) != expected_seed:
        raise ValueError(f"{summary_path} does not record seed {expected_seed}")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        prediction = loaded["baseline_state_normalized"].astype(np.float32)
    if not np.all(np.isfinite(prediction)):
        raise ValueError(f"non-finite prediction in {prediction_path}")
    return summary, prediction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--primary-prefix", required=True)
    parser.add_argument("--repeat-prefix", required=True)
    parser.add_argument("--split-name", default="completed_smoke")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--primary-seed", type=int, default=20260717)
    parser.add_argument("--other-seeds", type=int, nargs="+", default=[20260718, 20260719])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.primary_prefix == args.repeat_prefix:
        raise ValueError("same-seed repeat must use a separate result prefix")
    if len(set(args.other_seeds)) != len(args.other_seeds) or args.primary_seed in args.other_seeds:
        raise ValueError("different seeds must be unique and exclude the primary seed")

    rows = []
    for architecture in ARCHITECTURES:
        primary_dir = directory(
            args.results_root,
            args.primary_prefix,
            architecture,
            args.split_name,
            args.epochs,
        )
        repeat_dir = directory(
            args.results_root,
            args.repeat_prefix,
            architecture,
            args.split_name,
            args.epochs,
        )
        primary_summary, primary = load(primary_dir, args.primary_seed)
        repeat_summary, repeat = load(repeat_dir, args.primary_seed)
        if primary.shape != repeat.shape:
            raise ValueError(f"same-seed {architecture} prediction shapes differ")
        if primary_summary.get("split_case_ids") != repeat_summary.get("split_case_ids"):
            raise ValueError(f"same-seed {architecture} runs use different cases")
        if (
            primary_summary.get("run_provenance", {}).get("common_comparison_fingerprint")
            != repeat_summary.get("run_provenance", {}).get("common_comparison_fingerprint")
        ):
            raise ValueError(f"same-seed {architecture} runs use different input fields")

        scale = max(1.0, float(np.max(np.abs(primary))), float(np.max(np.abs(repeat))))
        same_seed_difference = float(np.max(np.abs(primary - repeat)))
        float32_tolerance = 64.0 * float(np.finfo(np.float32).eps) * scale
        if same_seed_difference > float32_tolerance:
            raise ValueError(
                f"same-seed {architecture} difference {same_seed_difference} exceeds "
                f"the float32 accumulation tolerance {float32_tolerance}"
            )

        different_seed_differences = []
        for seed in args.other_seeds:
            other_dir = directory(
                args.results_root,
                args.primary_prefix,
                architecture,
                args.split_name,
                args.epochs,
                seed,
            )
            other_summary, other = load(other_dir, seed)
            if other.shape != primary.shape:
                raise ValueError(f"different-seed {architecture} prediction shapes differ")
            if other_summary.get("split_case_ids") != primary_summary.get("split_case_ids"):
                raise ValueError(f"different-seed {architecture} runs use different cases")
            different_seed_differences.append(float(np.max(np.abs(primary - other))))
        minimum_different_seed_difference = min(different_seed_differences)
        if minimum_different_seed_difference <= 100.0 * float32_tolerance:
            raise ValueError(f"different seeds do not change {architecture} beyond numerical noise")

        rows.append(
            {
                "architecture": architecture,
                "primary_seed": args.primary_seed,
                "other_seeds": args.other_seeds,
                "same_seed_max_abs_prediction_difference": same_seed_difference,
                "float32_accumulation_tolerance": float32_tolerance,
                "different_seed_max_abs_prediction_differences": different_seed_differences,
                "minimum_different_seed_difference": minimum_different_seed_difference,
                "different_to_same_or_tolerance_ratio": minimum_different_seed_difference
                / max(same_seed_difference, float32_tolerance),
                "same_seed_checkpoint_sha256_equal": sha256(primary_dir / "best.pt")
                == sha256(repeat_dir / "best.pt"),
                "same_fields_and_cases": True,
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "steady_neural_seed_repeatability_verified",
        "scope": "one-epoch real P418 3D software check; not a model-accuracy result",
        "primary_seed": args.primary_seed,
        "other_seeds": args.other_seeds,
        "architectures": list(ARCHITECTURES),
        "results": rows,
        "new_physical_parameters": [],
        "interpretation": (
            "Repeating the same seed reproduces normalized regional predictions within a "
            "float32 GPU accumulation tolerance, while changing the seed alters predictions "
            "far beyond that numerical difference."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 稳态神经网络随机初值实测",
        "",
        "本检查使用已有8个真实P418三维工况，每种网络只训练1轮。它只说明随机初值和重复运行程序工作正常，不代表正式模型精度。",
        "",
        "| 模型 | 同种子最大差 | 不同种子最小差 | 不同/同种子数值差比例 | 同种子模型文件完全相同 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['architecture']} | "
            f"{row['same_seed_max_abs_prediction_difference']:.3e} | "
            f"{row['minimum_different_seed_difference']:.3e} | "
            f"{row['different_to_same_or_tolerance_ratio']:.3e} | "
            f"{'是' if row['same_seed_checkpoint_sha256_equal'] else '否（GPU并行求和末位差）'} |"
        )
    lines.extend(
        [
            "",
            "结论：同一个随机种子重复运行时，预测差异只处于float32 GPU并行累加的末位范围；更换种子后，预测变化远大于该数值差。因此正式三次独立初值既不是重复复制同一模型，也不会把GPU末位变化误当成模型差异。",
        ]
    )
    (output / "README_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
