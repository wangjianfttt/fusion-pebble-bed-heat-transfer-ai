#!/usr/bin/env python3
"""Collect native-cell temperature errors from the formal model comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "fluid_representation_rmse_K",
    "fluid_regional_model_rmse_K",
    "fluid_native_total_rmse_K",
    "fluid_limited_native_total_rmse_K",
    "solid_representation_rmse_K",
    "solid_regional_model_rmse_K",
    "solid_native_total_rmse_K",
    "solid_limited_native_total_rmse_K",
    "predicted_solid_max_temperature_error_K",
    "predicted_hotspot_nearest_cell_distance_dp",
    "limited_predicted_solid_max_temperature_error_K",
    "limited_predicted_hotspot_distance_dp",
)


def parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("result must use model=summary.json")
    model, path = value.split("=", 1)
    if not model.strip() or not path.strip():
        raise argparse.ArgumentTypeError("result must use model=summary.json")
    return model.strip(), Path(path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=parse_result, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for model, path in args.result:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "native_cell_prediction_metrics_ready":
            raise ValueError(f"native-cell result is incomplete: {path}")
        metrics = payload["metrics"]
        row: dict[str, object] = {
            "model": model,
            "split_name": payload["split_name"],
            "case_count": int(payload["case_count"]),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(metrics[metric]["mean"])
            row[f"{metric}_maximum_absolute"] = float(
                metrics[metric]["maximum_absolute"]
            )
        rows.append(row)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "native_cell_model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "native_cell_model_comparison_ready",
        "models": [row["model"] for row in rows],
        "split_name": rows[0]["split_name"],
        "rows": rows,
        "error_relation": (
            "piecewise_constant_native_total_rmse^2 = representation_rmse^2 + "
            "regional_model_rmse^2"
        ),
        "new_physical_parameters": [],
        "result_csv": csv_path.name,
    }
    (output / "native_cell_model_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 原始OpenFOAM网格上的模型温度误差比较",
        "",
        "区域节点预测已还原到原始流体和颗粒单元。下表中的总误差同时包含区域化造成的信息损失和模型本身的预测误差。",
        "",
        "| 模型 | 流体限制重构RMSE (K) | 颗粒限制重构RMSE (K) | 最高温度误差 (K) | 热点偏移 (d_p) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['fluid_limited_native_total_rmse_K_mean']:.4g} | "
            f"{row['solid_limited_native_total_rmse_K_mean']:.4g} | "
            f"{row['limited_predicted_solid_max_temperature_error_K_mean']:.4g} | "
            f"{row['limited_predicted_hotspot_distance_dp_mean']:.4g} |"
        )
    lines.extend(
        [
            "",
            "其中 d_p 为文献给定的颗粒直径。这里没有增加新的物理参数。",
        ]
    )
    (output / "原始网格模型比较_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
