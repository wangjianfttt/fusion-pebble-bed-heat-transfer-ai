#!/usr/bin/env python3
"""Compare three frozen fixed-flow models on the same six high-Re curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from hccb_p418_comparison_contract import file_record


EXPECTED_STATUS = "completed_p418_frozen_high_re_independent_evaluation"
ENERGY_STATUS = "completed_p418_common_transient_energy_balance"
ENERGY_UNAVAILABLE_STATUS = (
    "p418_energy_evaluation_unavailable_outside_registered_temperature_range"
)
MODEL_ORDER = ("data_only", "physics_constrained", "factorized")
MODEL_LABELS = {
    "data_only": "Data-only graph Transformer",
    "physics_constrained": "Physics-constrained graph Transformer",
    "factorized": "Factorized graph Transformer",
}
METRICS = (
    (
        "fluid_temperature_volume_weighted_RMSE_K",
        "Fluid temperature RMSE (K)",
    ),
    (
        "solid_temperature_volume_weighted_RMSE_K",
        "Solid temperature RMSE (K)",
    ),
    (
        "solid_maximum_temperature_history_RMSE_K",
        "Solid maximum-history RMSE (K)",
    ),
    (
        "solid_regional_hotspot_location_mean_error_m",
        "Mean hotspot-location error (m)",
    ),
    (
        "solid_regional_hotspot_location_p95_error_m",
        "Hotspot-location p95 error (m)",
    ),
)
PER_CURVE_METRICS = (
    "fluid_temperature_volume_weighted_RMSE_K",
    "solid_temperature_volume_weighted_RMSE_K",
)
ENERGY_METRICS = (
    (
        "projection_aware_volume_weighted_energy_equation_normalized_RMSE",
        "Projection-aware energy-equation RMSE",
    ),
    (
        "prediction_global_energy_closure_normalized_RMSE",
        "Predicted global energy-closure RMSE",
    ),
)


def finite_number(record: dict[str, object], name: str) -> float:
    value = float(record[name])
    if not math.isfinite(value):
        raise ValueError(f"metric {name} is not finite")
    return value


def load_evaluation(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != EXPECTED_STATUS or data.get("mode") != "fixed":
        raise ValueError(f"{path} is not a completed fixed-flow evaluation")
    if data.get("training_or_model_selection_performed") is not False:
        raise ValueError(f"{path} is not a frozen-model evaluation")
    sequence_ids = list(map(str, data.get("independent_sequence_ids", [])))
    if len(sequence_ids) != 6 or len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError(f"{path} must contain six unique independent curves")
    rows = data.get("per_curve_metrics")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError(f"{path} per-curve metrics are incomplete")
    row_ids = [str(row.get("sequence_id")) for row in rows]
    if row_ids != sequence_ids:
        raise ValueError(f"{path} per-curve order differs from its declared test set")
    return data


def load_energy(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") == ENERGY_UNAVAILABLE_STATUS:
        fractions = [
            float(
                data.get(
                    "predicted_fluid_temperature_outside_registered_range_fraction",
                    0.0,
                )
            ),
            float(
                data.get(
                    "predicted_solid_temperature_outside_registered_range_fraction",
                    0.0,
                )
            ),
        ]
        if not all(math.isfinite(value) and value >= 0.0 for value in fractions):
            raise ValueError(f"{path} has a non-finite unavailable-energy record")
        if max(fractions) <= 0.0:
            raise ValueError(f"{path} has an invalid unavailable-energy record")
        return data
    if data.get("status") != ENERGY_STATUS:
        raise ValueError(f"{path} is not a completed energy evaluation")
    roles = list(map(str, data.get("evaluated_roles", [])))
    if roles != ["test"]:
        raise ValueError(f"{path} must contain only the independent test role")
    metrics = data.get("role_metrics", {}).get("test")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} has no test energy metrics")
    for name, _ in ENERGY_METRICS:
        finite_number(metrics, name)
    return data


def energy_number(data: dict[str, object], name: str) -> float | None:
    if data.get("status") == ENERGY_UNAVAILABLE_STATUS:
        return None
    return finite_number(dict(data["role_metrics"]["test"]), name)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: float) -> str:
    absolute = abs(value)
    if absolute != 0.0 and (absolute < 1.0e-3 or absolute >= 1.0e4):
        return f"{value:.3e}"
    return f"{value:.4g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only-summary", type=Path, required=True)
    parser.add_argument("--physics-summary", type=Path, required=True)
    parser.add_argument("--factorized-summary", type=Path, required=True)
    parser.add_argument("--data-only-energy", type=Path, required=True)
    parser.add_argument("--physics-energy", type=Path, required=True)
    parser.add_argument("--factorized-energy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path)
    args = parser.parse_args()

    summary_paths = {
        "data_only": args.data_only_summary.resolve(),
        "physics_constrained": args.physics_summary.resolve(),
        "factorized": args.factorized_summary.resolve(),
    }
    energy_paths = {
        "data_only": args.data_only_energy.resolve(),
        "physics_constrained": args.physics_energy.resolve(),
        "factorized": args.factorized_energy.resolve(),
    }
    evaluations = {
        label: load_evaluation(path) for label, path in summary_paths.items()
    }
    energies = {label: load_energy(path) for label, path in energy_paths.items()}

    sequence_ids = list(
        map(str, evaluations["data_only"]["independent_sequence_ids"])
    )
    for label in MODEL_ORDER[1:]:
        if list(map(str, evaluations[label]["independent_sequence_ids"])) != sequence_ids:
            raise ValueError("all three models must use the same ordered curves")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, object]] = []
    for name, display_name in METRICS:
        row: dict[str, object] = {"metric": name, "quantity": display_name}
        for label in MODEL_ORDER:
            row[label] = finite_number(
                dict(evaluations[label]["aggregate_metrics"]), name
            )
        aggregate_rows.append(row)
    for name, display_name in ENERGY_METRICS:
        row = {"metric": name, "quantity": display_name}
        for label in MODEL_ORDER:
            value = energy_number(energies[label], name)
            row[label] = "" if value is None else value
        aggregate_rows.append(row)

    per_curve_rows: list[dict[str, object]] = []
    rows_by_model = {
        label: {
            str(row["sequence_id"]): row
            for row in evaluations[label]["per_curve_metrics"]
        }
        for label in MODEL_ORDER
    }
    for sequence_id in sequence_ids:
        row: dict[str, object] = {"sequence_id": sequence_id}
        for metric in PER_CURVE_METRICS:
            for label in MODEL_ORDER:
                row[f"{label}_{metric}"] = finite_number(
                    rows_by_model[label][sequence_id], metric
                )
        per_curve_rows.append(row)

    aggregate_csv = output_dir / "aggregate_fixed_model_comparison.csv"
    per_curve_csv = output_dir / "per_curve_fixed_model_comparison.csv"
    write_csv(aggregate_csv, aggregate_rows)
    write_csv(per_curve_csv, per_curve_rows)

    report_lines = [
        "# P418高速端三种固定流场模型独立测试",
        "",
        "## 数据范围",
        "",
        "- 三种模型使用同一组6条高流速OpenFOAM独立曲线。",
        "- 这6条曲线没有参与训练、归一化、模型选择或超参数调整。",
        "- 比较对象为数据模型、带能量关系约束的模型和时空分解模型。",
        "- 全耦合启动短算仅用于说明固定流场近似的适用范围，不参加精度排名。",
        "",
        "## 汇总结果",
        "",
        "| 物理量 | 数据模型 | 物理约束模型 | 时空分解模型 |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        values = {
            label: (
                "--"
                if row[label] == ""
                else format_value(float(row[label]))
            )
            for label in MODEL_ORDER
        }
        report_lines.append(
            f"| {row['quantity']} | {values['data_only']} | "
            f"{values['physics_constrained']} | {values['factorized']} |"
        )
    report_lines.extend(
        [
            "",
            "## 解释原则",
            "",
            "- 温度、热点位置和能量关系分别报告，不压缩成一个人为总分。",
            "- 模型优劣只按预先保留的6条高流速曲线判断。",
            "- 固定流场方法的结论限定为热响应建模，不外推为全耦合流动预测。",
            "",
        ]
    )
    report_path = output_dir / "P418_高速端三种固定流场模型比较_CN.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Frozen fixed-flow model comparison on six independent "
            r"high-Reynolds-number OpenFOAM histories.}"
        ),
        r"\label{tab:p418_high_re_frozen_comparison}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Quantity & Data only & Physics constrained & Factorized \\",
        r"\hline",
    ]
    for row in aggregate_rows:
        values = {
            label: (
                "--"
                if row[label] == ""
                else format_value(float(row[label]))
            )
            for label in MODEL_ORDER
        }
        tex_lines.append(
            f"{row['quantity']} & {values['data_only']} & "
            f"{values['physics_constrained']} & {values['factorized']} \\\\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    tex_text = "\n".join(tex_lines)
    tex_path = output_dir / "p418_high_re_fixed_model_comparison_table.tex"
    tex_path.write_text(tex_text, encoding="utf-8")
    manuscript_record = None
    if args.latex_output is not None:
        manuscript_path = args.latex_output.resolve()
        manuscript_path.parent.mkdir(parents=True, exist_ok=True)
        manuscript_path.write_text(tex_text, encoding="utf-8")
        manuscript_record = file_record(manuscript_path)

    output = {
        "status": "completed_p418_high_re_three_fixed_model_comparison",
        "curve_count": 6,
        "sequence_ids": sequence_ids,
        "model_order": list(MODEL_ORDER),
        "model_labels": MODEL_LABELS,
        "training_or_model_selection_performed": False,
        "same_ordered_independent_curves": True,
        "fully_coupled_model_used_for_accuracy_ranking": False,
        "evaluation_summary_records": {
            label: file_record(summary_paths[label]) for label in MODEL_ORDER
        },
        "energy_summary_records": {
            label: file_record(energy_paths[label]) for label in MODEL_ORDER
        },
        "aggregate_comparison_file": aggregate_csv.name,
        "per_curve_comparison_file": per_curve_csv.name,
        "chinese_report_file": report_path.name,
        "latex_table_file": tex_path.name,
        "manuscript_latex_table": manuscript_record,
        "new_physical_parameters": [],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
