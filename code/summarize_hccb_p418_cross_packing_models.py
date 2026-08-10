#!/usr/bin/env python3
"""Summarize independent-packing model errors without a composite score."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = {
    "state_normalized_rmse": ("-", "regional state"),
    "fluid_temperature_volume_weighted_rmse_K": ("K", "fluid temperature field"),
    "solid_temperature_volume_weighted_rmse_K": ("K", "solid temperature field"),
    "solid_hotspot_location_error_m": ("m", "solid hotspot position"),
    "engineering_absolute_errors.pressure_drop_Pa": ("Pa", "pressure drop"),
    "engineering_absolute_errors.outlet_temperature_K": ("K", "outlet temperature"),
    "engineering_absolute_errors.solid_maximum_temperature_K": ("K", "maximum solid temperature"),
    "engineering_absolute_errors.cooling_wall_heat_into_fluid_W": ("W", "cooling-wall heat flow"),
    "engineering_absolute_errors.solid_to_fluid_interphase_net_W": ("W", "net solid-to-fluid heat flow"),
    "local_mass_l1_over_two_inlet": ("-", "regional mass balance"),
    "global_mass_imbalance_over_inlet": ("-", "global mass balance"),
    "local_energy_l1_over_two_generated_power": ("-", "regional energy balance"),
    "global_energy_imbalance_over_generated_power": ("-", "global energy balance"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(case: dict[str, object], name: str) -> float:
    value: object = case
    for key in name.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"case {case.get('condition_id')} lacks metric {name}")
        value = value[key]
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"metric {name} must be finite and non-negative")
    return number


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 9:
        raise ValueError("each packing evaluation must contain nine cases")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def load_run(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "cross_packing_conservative_evaluation_complete":
        raise ValueError(f"{path} is not a completed conservative evaluation")
    cases = list(payload.get("cases", []))
    condition_ids = [str(case["condition_id"]) for case in cases]
    if len(cases) != 9 or len(condition_ids) != len(set(condition_ids)):
        raise ValueError(f"{path} must contain nine unique conditions")
    metric_summary = {
        name: summarize([nested(case, name) for case in cases]) for name in METRICS
    }
    return {
        "source_file": str(path.resolve()),
        "source_sha256": sha256(path),
        "packing_seed": int(payload["packing_seed"]),
        "packing_role": str(payload["packing_role"]),
        "architecture": str(payload["architecture"]),
        "condition_ids": condition_ids,
        "metrics": metric_summary,
    }


def write_csv(path: Path, runs: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "packing_seed",
                "packing_role",
                "architecture",
                "metric",
                "physical_quantity",
                "unit",
                "mean",
                "median",
                "p95",
                "maximum",
            ),
        )
        writer.writeheader()
        for run in runs:
            for metric, statistics in run["metrics"].items():
                unit, quantity = METRICS[metric]
                writer.writerow(
                    {
                        "packing_seed": run["packing_seed"],
                        "packing_role": run["packing_role"],
                        "architecture": run["architecture"],
                        "metric": metric,
                        "physical_quantity": quantity,
                        "unit": unit,
                        **statistics,
                    }
                )


def write_chinese_summary(
    path: Path,
    runs: list[dict[str, object]],
    transfer: list[dict[str, object]],
) -> None:
    lines = [
        "# P418不同颗粒排列的模型结果",
        "",
        "下表分别列出各物理量，不将温度、压降、热量和守恒误差合成一个总分。",
        "",
        "| 装填 | 模型 | 物理量 | 单位 | 平均 | 95%分位 | 最差 |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for run in runs:
        for metric, statistics in run["metrics"].items():
            unit, quantity = METRICS[metric]
            lines.append(
                f"| seed{run['packing_seed']} | {run['architecture']} | {quantity} | "
                f"{unit} | {statistics['mean']:.6g} | {statistics['p95']:.6g} | "
                f"{statistics['maximum']:.6g} |"
            )
    if transfer:
        lines.extend(
            [
                "",
                "## seed303相对seed202的误差变化",
                "",
                "比值大于1表示在最终未见颗粒排列上误差增大。",
                "",
                "| 模型 | 物理量 | 平均误差比 | 95%分位误差比 |",
                "|---|---|---:|---:|",
            ]
        )
        for row in transfer:
            lines.append(
                f"| {row['architecture']} | {METRICS[row['metric']][1]} | "
                f"{row['mean_ratio_seed303_to_seed202']:.6g} | "
                f"{row['p95_ratio_seed303_to_seed202']:.6g} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = [load_run(path.resolve()) for path in args.input]
    keys = [(run["packing_seed"], run["architecture"]) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("packing and architecture pairs must be unique")
    for seed in sorted({int(run["packing_seed"]) for run in runs}):
        identifiers = {
            tuple(run["condition_ids"])
            for run in runs
            if int(run["packing_seed"]) == seed
        }
        if len(identifiers) != 1:
            raise ValueError(f"models on seed{seed} use different condition orders")
    lookup = {
        (int(run["packing_seed"]), str(run["architecture"])): run for run in runs
    }
    transfer = []
    architectures = sorted(
        {architecture for seed, architecture in lookup if (202, architecture) in lookup and (303, architecture) in lookup}
    )
    for architecture in architectures:
        development = lookup[(202, architecture)]
        final = lookup[(303, architecture)]
        for metric in METRICS:
            mean_denominator = float(development["metrics"][metric]["mean"])
            p95_denominator = float(development["metrics"][metric]["p95"])
            transfer.append(
                {
                    "architecture": architecture,
                    "metric": metric,
                    "mean_ratio_seed303_to_seed202": (
                        float(final["metrics"][metric]["mean"]) / mean_denominator
                        if mean_denominator > 0.0
                        else None
                    ),
                    "p95_ratio_seed303_to_seed202": (
                        float(final["metrics"][metric]["p95"]) / p95_denominator
                        if p95_denominator > 0.0
                        else None
                    ),
                }
            )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "cross_packing_model_metrics.csv", runs)
    comparable_transfer = [
        row
        for row in transfer
        if row["mean_ratio_seed303_to_seed202"] is not None
        and row["p95_ratio_seed303_to_seed202"] is not None
    ]
    write_chinese_summary(output / "P418_跨装填模型结果_CN.md", runs, comparable_transfer)
    summary = {
        "status": "cross_packing_model_summary_complete",
        "run_count": len(runs),
        "packing_seeds": sorted({int(run["packing_seed"]) for run in runs}),
        "architectures": sorted({str(run["architecture"]) for run in runs}),
        "metric_definition": METRICS,
        "runs": runs,
        "seed303_to_seed202": transfer,
        "composite_score_used": False,
        "new_physical_parameter_values_added": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
