#!/usr/bin/env python3
"""Summarize three independent initializations on the main steady split."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np


ARCHITECTURES = ("pinn_data_only", "pinn", "graph", "transolver")
LABELS = {
    "pinn_data_only": "Data-only PINN",
    "pinn": "Physics PINN",
    "graph": "Graph operator",
    "transolver": "Transolver",
}
METRICS = (
    ("solid_temperature_normalized_rmse", "Solid-$T$ nRMSE", ""),
    ("solid_maximum_temperature_p95_K", "Max-$T$ p95", "K"),
    ("pressure_drop_p95_Pa", "Pressure p95", "Pa"),
    ("wall_heat_p95_percent", "Wall-heat p95", "\\%"),
    ("regional_energy_difference_percent", "Energy difference", "\\%"),
)


def result_path(
    results_root: Path,
    result_prefix: str,
    architecture: str,
    split_name: str,
    epochs: int,
    seed: int,
    primary_seed: int,
) -> Path:
    suffix = "" if seed == primary_seed else f"_seed{seed}"
    return (
        results_root
        / f"{result_prefix}_{architecture}_{split_name}_{epochs}epoch{suffix}"
        / "summary.json"
    )


def exact_split(summary: dict, expected: dict[str, list[str]], architecture: str) -> None:
    recorded = summary.get("split_case_ids")
    if not isinstance(recorded, dict):
        raise ValueError(f"{architecture} does not record the complete steady split")
    for role, identifiers in expected.items():
        actual = [str(value) for value in recorded.get(role, [])]
        if actual != identifiers or len(actual) != len(set(actual)):
            raise ValueError(f"{architecture} {role} cases differ from the registered split")


def p95(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("metric values must be finite and non-empty")
    return float(np.quantile(array, 0.95))


def extract_metrics(summary: dict) -> dict[str, float]:
    evaluation = summary.get("evaluations", {}).get("test", {})
    metrics = evaluation.get("metrics", {})
    cases = evaluation.get("cases", [])
    channels = metrics.get("state_channel_rmse")
    if not isinstance(channels, list) or len(channels) != 6 or not cases:
        raise ValueError("steady summary lacks common test fields or case metrics")
    solid_nrmse = float(channels[5])
    maximum_temperature = [
        float(case["engineering_absolute_errors"]["solid_maximum_temperature_K"])
        for case in cases
    ]
    pressure = [
        float(case["engineering_absolute_errors"]["pressure_drop_Pa"])
        for case in cases
    ]
    wall_heat = [
        100.0
        * float(case["engineering_absolute_errors"]["cooling_wall_heat_into_fluid_W"])
        / float(case["generated_power_W"])
        for case in cases
    ]
    regional_energy = [
        100.0 * float(case["local_energy_l1_over_two_generated_power"])
        for case in cases
    ]
    values = {
        "solid_temperature_normalized_rmse": solid_nrmse,
        "solid_maximum_temperature_p95_K": p95(maximum_temperature),
        "pressure_drop_p95_Pa": p95(pressure),
        "wall_heat_p95_percent": p95(wall_heat),
        "regional_energy_difference_percent": float(np.mean(regional_energy)),
    }
    if not np.all(np.isfinite(np.asarray(list(values.values()), dtype=float))):
        raise ValueError("steady seed metrics contain non-finite values")
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if abs(value) < 0.01 or abs(value) >= 1000.0:
        return f"{value:.2e}"
    return f"{value:.3g}"


def write_tex(path: Path, aggregate: list[dict], split_name: str, seed_count: int) -> None:
    lookup = {(row["architecture"], row["metric"]): row for row in aggregate}
    lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4.0pt}",
        (
            "\\caption{Sensitivity of the main steady-condition comparison to neural-network "
            f"initialization. Values are mean $\\pm$ sample standard deviation over {seed_count} "
            "independent training seeds on the same registered split; OpenFOAM fields, case "
            "partition and normalization are unchanged.}"
        ),
        "\\label{tab:steady_seed_robustness}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Solid-$T$ nRMSE & Max-$T$ p95 (K) & Wall-heat p95 (\\%) & Energy diff. (\\%) \\\\",
        "\\midrule",
    ]
    shown_metrics = (
        "solid_temperature_normalized_rmse",
        "solid_maximum_temperature_p95_K",
        "wall_heat_p95_percent",
        "regional_energy_difference_percent",
    )
    for architecture in ARCHITECTURES:
        entries = []
        for metric in shown_metrics:
            row = lookup[(architecture, metric)]
            entries.append(f"{fmt(float(row['mean']))} $\\pm$ {fmt(float(row['sample_std']))}")
        lines.append(f"{LABELS[architecture]} & " + " & ".join(entries) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--result-prefix", default="hccb_p418_60")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split-name", default="interleaved_all_ranges")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--primary-seed", type=int, default=20260717)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260717, 20260718, 20260719])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    args = parser.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) != 3 or args.primary_seed not in seeds:
        raise ValueError("steady robustness requires exactly three seeds including the primary seed")
    split_payload = json.loads(args.split_file.read_text(encoding="utf-8"))["splits"]
    if args.split_name not in split_payload:
        raise ValueError(f"unknown steady split {args.split_name}")
    expected = {
        role: [str(value) for value in split_payload[args.split_name][role]]
        for role in ("train", "validation", "test")
    }

    rows: list[dict] = []
    fingerprints: set[str] = set()
    for architecture in ARCHITECTURES:
        for seed in seeds:
            source = result_path(
                args.results_root,
                args.result_prefix,
                architecture,
                args.split_name,
                args.epochs,
                seed,
                args.primary_seed,
            )
            if not source.is_file():
                raise FileNotFoundError(f"missing steady seed result: {source}")
            summary = json.loads(source.read_text(encoding="utf-8"))
            if summary.get("architecture") != architecture:
                raise ValueError(f"wrong architecture in {source}")
            if summary.get("split_name") != args.split_name:
                raise ValueError(f"wrong split in {source}")
            if int(summary.get("epochs", -1)) != args.epochs:
                raise ValueError(f"wrong epoch count in {source}")
            if int(summary.get("training_seed", -1)) != seed:
                raise ValueError(f"{source} does not record seed {seed}")
            exact_split(summary, expected, architecture)
            evaluation_ids = [
                str(case.get("condition_id"))
                for case in summary.get("evaluations", {}).get("test", {}).get("cases", [])
            ]
            if evaluation_ids != expected["test"]:
                raise ValueError(f"{source} evaluated different independent cases")
            fingerprint = str(
                summary.get("run_provenance", {}).get("common_comparison_fingerprint", "")
            )
            if not fingerprint:
                raise ValueError(f"{source} lacks the common data fingerprint")
            fingerprints.add(fingerprint)
            values = extract_metrics(summary)
            rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    **values,
                    "source_summary": str(source.resolve()),
                }
            )
    if len(fingerprints) != 1:
        raise ValueError("steady seed runs used different fields, splits or normalization files")

    aggregate: list[dict] = []
    for architecture in ARCHITECTURES:
        selected = [row for row in rows if row["architecture"] == architecture]
        for metric, label, unit in METRICS:
            values = [float(row[metric]) for row in selected]
            aggregate.append(
                {
                    "architecture": architecture,
                    "label": LABELS[architecture],
                    "metric": metric,
                    "metric_label": label,
                    "unit": unit,
                    "seed_count": len(values),
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
            )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "steady_seed_metrics.csv", rows)
    write_csv(output / "steady_seed_summary.csv", aggregate)
    if args.tex_output is not None:
        write_tex(args.tex_output.resolve(), aggregate, args.split_name, len(seeds))
    payload = {
        "status": "completed_p418_main_steady_split_seed_robustness",
        "split_name": args.split_name,
        "seeds": seeds,
        "architectures": list(ARCHITECTURES),
        "split_case_ids": expected,
        "common_comparison_fingerprint": next(iter(fingerprints)),
        "metrics": aggregate,
        "new_physical_parameters": [],
        "scientific_scope": (
            "Three independent neural-network initializations on the main steady split. "
            "The OpenFOAM fields, physical cases, normalization and test cases are unchanged."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    chinese = [
        "# 稳态模型三次独立初值结果",
        "",
        f"- 工况划分：`{args.split_name}`",
        f"- 随机种子：{', '.join(str(seed) for seed in seeds)}",
        "- 三次训练使用完全相同的OpenFOAM场、训练/验证/独立预测工况和归一化量。",
        "- 响应面是确定性方法，不做没有物理意义的随机重复。",
        "",
        "| 模型 | 指标 | 均值 | 样本标准差 | 最小值 | 最大值 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        chinese.append(
            f"| {row['label']} | {row['metric_label']} {row['unit']} | "
            f"{fmt(float(row['mean']))} | {fmt(float(row['sample_std']))} | "
            f"{fmt(float(row['minimum']))} | {fmt(float(row['maximum']))} |"
        )
    (output / "README_CN.md").write_text("\n".join(chinese) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
