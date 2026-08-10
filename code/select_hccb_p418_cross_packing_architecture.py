#!/usr/bin/env python3
"""Freeze the architecture after seed202 and before any seed303 field is read."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PRIMARY_METRIC = "solid_temperature_volume_weighted_rmse_K"
PARETO_METRICS = (
    "fluid_temperature_volume_weighted_rmse_K",
    "solid_temperature_volume_weighted_rmse_K",
    "solid_hotspot_location_error_m",
    "engineering_absolute_errors.pressure_drop_Pa",
    "engineering_absolute_errors.cooling_wall_heat_into_fluid_W",
    "local_mass_l1_over_two_inlet",
    "local_energy_l1_over_two_generated_power",
)


def dominates(first: dict[str, float], second: dict[str, float]) -> bool:
    """Return true when first is no worse everywhere and better somewhere."""

    no_worse = all(first[name] <= second[name] for name in PARETO_METRICS)
    strictly_better = any(first[name] < second[name] for name in PARETO_METRICS)
    return no_worse and strictly_better


def select(
    summary: dict[str, object], model_sources: dict[str, object] | None = None
) -> dict[str, object]:
    if summary.get("status") != "cross_packing_model_summary_complete":
        raise ValueError("seed202 model comparison is not complete")
    runs = list(summary.get("runs", []))
    if not runs:
        raise ValueError("seed202 model comparison contains no runs")
    if {int(run["packing_seed"]) for run in runs} != {202}:
        raise ValueError("architecture selection may read seed202 results only")

    values: dict[str, dict[str, float]] = {}
    for run in runs:
        architecture = str(run["architecture"])
        if architecture in values:
            raise ValueError(f"duplicate seed202 architecture: {architecture}")
        metrics = run.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{architecture} has no metric summary")
        values[architecture] = {}
        for name in PARETO_METRICS:
            record = metrics.get(name)
            if not isinstance(record, dict) or "p95" not in record:
                raise ValueError(f"{architecture} lacks p95 {name}")
            value = float(record["p95"])
            if value < 0.0:
                raise ValueError(f"{architecture} has a negative p95 {name}")
            values[architecture][name] = value

    pareto = sorted(
        architecture
        for architecture, metrics in values.items()
        if not any(
            dominates(other_metrics, metrics)
            for other_architecture, other_metrics in values.items()
            if other_architecture != architecture
        )
    )
    if not pareto:
        raise RuntimeError("no non-dominated seed202 architecture was found")
    source_selection = None
    if model_sources is not None:
        if model_sources.get("status") != "cross_packing_seed101_model_sources_selected":
            raise ValueError("seed101 model-source map is not ready")
        if model_sources.get("independent_test_used_for_selection") is not False:
            raise ValueError("seed101 model sources used independent test conditions")
        source_models = model_sources.get("models", {})
        if set(source_models) != set(values):
            raise ValueError("seed101 model-source map and seed202 architectures differ")
        source_selection = {
            name: {
                "selected_epochs": int(source_models[name]["selected_epochs"]),
                "selected_epoch": int(source_models[name]["selected_epoch"]),
                "selected_validation_total_loss": float(
                    source_models[name]["selected_validation_total_loss"]
                ),
            }
            for name in sorted(source_models)
        }
    selected = min(pareto, key=lambda name: (values[name][PRIMARY_METRIC], name))
    result = {
        "status": "seed202_architecture_fixed_before_seed303",
        "selection_data": "seed202 nine-condition development packing only",
        "selected_architecture": selected,
        "pareto_architectures": pareto,
        "pareto_metrics": list(PARETO_METRICS),
        "primary_metric_after_pareto_filter": PRIMARY_METRIC,
        "statistic": "p95 across the nine declared seed202 conditions",
        "metric_values": values,
        "selection_rule": (
            "Retain architectures that are not worse than another architecture in all "
            "declared temperature, hotspot, pressure, wall-heat, mass-balance and "
            "energy-balance quantities. Among those, select the smallest seed202 p95 "
            "solid-temperature field RMSE. Architecture, weights and seed101 scaling "
            "are then frozen before seed303 is read."
        ),
        "composite_score_used": False,
        "seed303_fields_read": False,
        "new_physical_parameter_values_added": [],
    }
    if source_selection is not None:
        result["seed101_checkpoint_selection"] = source_selection
        result["seed101_checkpoint_selection_data"] = "validation conditions only"
        result["seed101_independent_test_used_for_checkpoint_selection"] = False
    return result


def write_chinese(path: Path, result: dict[str, object]) -> None:
    values = result["metric_values"]
    lines = [
        "# seed202模型选择结果",
        "",
        "这里只使用seed202的9个工况，尚未读取seed303温度场。",
        "温度、热点、压降、壁面热量以及质量和能量关系分别比较，不合成一个总分。",
        "先去掉在全部这些量上都被另一种模型压过的方案，再以固体温度场95%分位误差选择最终模型。",
        "",
        f"最终固定模型：`{result['selected_architecture']}`",
        "",
        "进入最后比较的模型：" + ", ".join(f"`{name}`" for name in result["pareto_architectures"]),
        "",
        "| 模型 | 固体温度RMSE (K) | 热点偏差 (m) | 压降偏差 (Pa) | 壁面热量偏差 (W) |",
        "|---|---:|---:|---:|---:|",
    ]
    for architecture in sorted(values):
        row = values[architecture]
        lines.append(
            f"| {architecture} | {row['solid_temperature_volume_weighted_rmse_K']:.6g} | "
            f"{row['solid_hotspot_location_error_m']:.6g} | "
            f"{row['engineering_absolute_errors.pressure_drop_Pa']:.6g} | "
            f"{row['engineering_absolute_errors.cooling_wall_heat_into_fluid_W']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-sources", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chinese-output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    model_sources = None
    if args.model_sources is not None:
        model_sources = json.loads(args.model_sources.read_text(encoding="utf-8"))
    result = select(summary, model_sources)
    if args.model_sources is not None:
        result["seed101_model_sources_file"] = str(args.model_sources.resolve())
        result["seed101_model_sources_sha256"] = hashlib.sha256(
            args.model_sources.read_bytes()
        ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.chinese_output is not None:
        args.chinese_output.parent.mkdir(parents=True, exist_ok=True)
        write_chinese(args.chinese_output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
