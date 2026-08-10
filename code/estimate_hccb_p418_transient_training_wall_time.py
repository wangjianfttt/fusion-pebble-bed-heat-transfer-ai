#!/usr/bin/env python3
"""Estimate formal P418 transient GPU training time from full-curve measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        type=Path,
        default=ROOT / "parameters/hccb_p418_step_response_splits.json",
    )
    parser.add_argument(
        "--data-only-graph-summary",
        type=Path,
        default=(
            ROOT
            / "results/hccb_p418_actual_spatiotemporal_operator_56time_gpu_data_only/summary.json"
        ),
    )
    parser.add_argument(
        "--repeated-graph-summary",
        type=Path,
        default=(
            ROOT
            / "results/hccb_p418_actual_spatiotemporal_operator_56time_gpu_repeated/summary.json"
        ),
    )
    parser.add_argument(
        "--factorized-graph-summary",
        type=Path,
        default=(
            ROOT
            / "results/hccb_p418_actual_spatiotemporal_operator_56time_gpu_factorized/summary.json"
        ),
    )
    parser.add_argument(
        "--diffusion-summary",
        type=Path,
        default=(
            ROOT
            / "results/hccb_p418_actual_temporal_diffusion_56time_gpu_batch1_bfloat16_chunk2048/summary.json"
        ),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.seed_count < 1:
        raise ValueError("epochs and seed count must be positive")

    splits = read_json(args.splits)["splits"]
    primary_splits = list(splits)
    train_counts = {name: len(splits[name]["train"]) for name in primary_splits}
    if len(set(train_counts.values())) != 1:
        raise ValueError("runtime projection requires a common training-curve count")
    train_curves = next(iter(train_counts.values()))
    if train_curves <= 0:
        raise ValueError("no training curves are registered")

    repeated = read_json(args.repeated_graph_summary)
    factorized = read_json(args.factorized_graph_summary)
    diffusion = read_json(args.diffusion_summary)
    data_only = (
        read_json(args.data_only_graph_summary)
        if args.data_only_graph_summary.is_file()
        else None
    )
    measured_summaries = {
        "repeated_graph": repeated,
        "factorized_graph": factorized,
        "diffusion": diffusion,
    }
    if data_only is not None:
        measured_summaries["data_only_graph"] = data_only
    reference_shape = {
        key: int(repeated[key]) for key in ("nodes", "edges", "time_points")
    }
    for name, summary in measured_summaries.items():
        shape = {key: int(summary[key]) for key in reference_shape}
        if shape != reference_shape:
            raise ValueError(
                f"{name} measurement shape {shape} does not match {reference_shape}"
            )
    repeated_seconds = float(repeated["elapsed_seconds"])
    factorized_seconds = float(factorized["elapsed_seconds"])
    diffusion_seconds = float(diffusion["elapsed_seconds"])
    if min(repeated_seconds, factorized_seconds, diffusion_seconds) <= 0.0:
        raise ValueError("measured full-curve times must be positive")

    split_count = len(primary_splits)
    extra_seeds = args.seed_count - 1
    runs = {
        "data_only_graph": split_count + extra_seeds,
        "energy_flux_graph": split_count + extra_seeds,
        "factorized_energy_flux_graph": split_count,
        "diffusion_residual": split_count + extra_seeds,
    }
    rows = []
    if data_only is not None:
        data_only_seconds_low = data_only_seconds_high = float(
            data_only["elapsed_seconds"]
        )
        data_only_basis = "measured repeated-query graph data-loss backward update"
    else:
        # Keep a conservative range until the same full-size data-only update is timed.
        data_only_seconds_low = factorized_seconds
        data_only_seconds_high = repeated_seconds
        data_only_basis = (
            "bounded by measured factorized and repeated full-physics curve updates"
        )
    specifications = (
        (
            "data_only_graph",
            runs["data_only_graph"],
            data_only_seconds_low,
            data_only_seconds_high,
            data_only_basis,
        ),
        (
            "energy_flux_graph",
            runs["energy_flux_graph"],
            repeated_seconds,
            repeated_seconds,
            "measured repeated-query graph, transient physics and backward update",
        ),
        (
            "factorized_energy_flux_graph",
            runs["factorized_energy_flux_graph"],
            factorized_seconds,
            factorized_seconds,
            "measured factorized graph, transient physics and backward update",
        ),
        (
            "diffusion_residual",
            runs["diffusion_residual"],
            diffusion_seconds,
            diffusion_seconds,
            "measured bfloat16 temporal diffusion backward update",
        ),
    )
    for name, run_count, low_seconds, high_seconds, basis in specifications:
        curve_updates = run_count * train_curves * args.epochs
        rows.append(
            {
                "model_family": name,
                "training_runs": run_count,
                "train_curves_per_epoch": train_curves,
                "epochs": args.epochs,
                "full_curve_updates": curve_updates,
                "measured_or_bounding_seconds_per_curve_update_low": low_seconds,
                "measured_or_bounding_seconds_per_curve_update_high": high_seconds,
                "projected_training_hours_low": curve_updates * low_seconds / 3600.0,
                "projected_training_hours_high": curve_updates * high_seconds / 3600.0,
                "timing_basis": basis,
            }
        )
    low_total = sum(float(row["projected_training_hours_low"]) for row in rows)
    high_total = sum(float(row["projected_training_hours_high"]) for row in rows)
    payload = {
        "status": "measured_full_curve_transient_training_time_projection",
        "primary_splits": primary_splits,
        "train_curves_per_split": train_curves,
        "seed_count_on_pair_disjoint_split": args.seed_count,
        "epochs": args.epochs,
        "measured_graph_nodes": int(repeated["nodes"]),
        "measured_graph_edges": int(repeated["edges"]),
        "measured_time_points": int(repeated["time_points"]),
        "measured_seconds_per_full_curve_update": {
            "data_only_graph": (
                float(data_only["elapsed_seconds"]) if data_only is not None else None
            ),
            "energy_flux_graph": repeated_seconds,
            "factorized_energy_flux_graph": factorized_seconds,
            "diffusion_residual": diffusion_seconds,
        },
        "measured_peak_gpu_GB": {
            name: float(summary["peak_gpu_GB"])
            for name, summary in measured_summaries.items()
            if "peak_gpu_GB" in summary
        },
        "data_only_timing_measured": data_only is not None,
        "projected_training_hours_low": low_total,
        "projected_training_hours_high": high_total,
        "projected_uninterrupted_training_days_low": low_total / 24.0,
        "projected_uninterrupted_training_days_high": high_total / 24.0,
        "excluded_time": [
            "validation after each epoch",
            "final train/validation/test prediction",
            "32-member diffusion sampling and convergence comparison",
            "observable Transformer, DMDc and low-rank residual calculation",
            "data loading, checkpoint writing and common energy-equation evaluation",
        ],
        "interpretation": (
            "The range estimates the GPU training-update component from measured full-size "
            "curves. It is not a guaranteed completion time; the complete transient model "
            "workflow will take longer because the listed evaluation work is excluded."
        ),
        "new_physical_parameters": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_components.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if abs(high_total - low_total) < 1.0e-12:
        training_time_text = f"约`{low_total:.1f} h`，即不间断运行约`{low_total/24:.1f}天`"
    else:
        training_time_text = (
            f"`{low_total:.1f}--{high_total:.1f} h`，即不间断运行约"
            f"`{low_total/24:.1f}--{high_total/24:.1f}天`"
        )
    complete_stage_days_low = math.ceil(low_total / 24.0) + 2
    complete_stage_days_high = math.ceil(high_total / 24.0) + 4
    (args.output_dir / "README_CN.md").write_text(
        "# P418正式瞬态模型训练时间估计\n\n"
        f"实测对象为`{payload['measured_graph_nodes']}`个区域节点、"
        f"`{payload['measured_graph_edges']}`条连接和`{payload['measured_time_points']}`个时刻的一条完整曲线。\n\n"
        f"按当前三种工况划分、500轮训练和三次随机初值重复，仅GPU参数更新部分需{training_time_text}。\n\n"
        "这个时间还不包含每轮检查、最终独立预测、32个扩散样本、DMDc/POD和通用能量方程计算，"
        f"所以完整模型阶段应按至少{complete_stage_days_low}--{complete_stage_days_high}天安排，"
        "不再使用早期3--5天的乐观估计。\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
