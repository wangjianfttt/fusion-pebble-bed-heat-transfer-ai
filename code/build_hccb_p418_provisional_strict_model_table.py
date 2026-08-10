#!/usr/bin/env python3
"""Build a provisional strict-split model table from completed formal summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


MODEL_SOURCES = (
    (
        "initial_temperature_persistence",
        "Initial-temperature persistence",
        "regional_persistence_pair_disjoint_stress_test/summary.json",
    ),
    (
        "dmdc",
        "DMDc",
        "regional_dmdc_pair_disjoint_stress_test/summary.json",
    ),
    (
        "bounded_data_only_graph_transformer",
        "Bounded data-only graph Transformer",
        "regional_graph_transformer_bounded_data_only_pair_disjoint_stress_test/summary.json",
    ),
    (
        "bounded_physics_graph_transformer",
        "Bounded physics graph Transformer",
        "regional_graph_transformer_bounded_physics_pair_disjoint_stress_test/summary.json",
    ),
    (
        "bounded_factorized_graph_transformer",
        "Bounded factorized graph Transformer",
        "regional_graph_transformer_bounded_factorized_pair_disjoint_stress_test/summary.json",
    ),
    (
        "low_rank_temperature_residual",
        "Low-rank temperature residual",
        "low_rank_temperature_residual_pair_disjoint_stress_test/summary.json",
    ),
)

CSV_FIELDS = (
    "model_id",
    "model_label",
    "result_status",
    "selected_epoch_or_rank",
    "selection_split",
    "split_train_count",
    "split_validation_count",
    "split_test_count",
    "compute_device",
    "training_seconds",
    "fluid_temperature_RMSE_K",
    "solid_temperature_RMSE_K",
    "maximum_absolute_temperature_error_K",
    "solid_maximum_temperature_history_RMSE_K",
    "solid_regional_hotspot_exact_match_fraction",
    "inference_seconds_per_curve",
    "common_energy_test_volume_weighted_residual_ratio",
    "test_temperature_range_status",
    "common_energy_rejected_roles",
    "source_summary",
    "source_summary_sha256",
    "source_energy_summary",
    "source_energy_summary_sha256",
)

EXPECTED_SPLIT_COUNTS = {"train": 6, "validation": 2, "test": 4}
EXPECTED_TEMPERATURE_METRIC = (
    "regional-volume-weighted RMSE, reported separately for fluid and solid"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_value(summary: dict) -> int | None:
    for key in ("selected_epoch", "best_epoch", "selected_rank"):
        if key in summary and summary[key] is not None:
            return int(summary[key])
    return None


def finite_nonnegative(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def split_counts(summary: dict) -> dict[str, int]:
    counts = summary.get("split_case_counts")
    if counts is None:
        counts = {
            role: len(identifiers)
            for role, identifiers in summary["split_case_ids"].items()
        }
    normalized = {role: int(counts[role]) for role in EXPECTED_SPLIT_COUNTS}
    if normalized != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"unexpected split counts {normalized}; expected {EXPECTED_SPLIT_COUNTS}"
        )
    return normalized


def test_temperature_range_status(summary: dict, energy: dict) -> str:
    rejected = {str(role) for role in energy.get("rejected_roles", [])}
    if "test" in rejected:
        return "test_prediction_outside_registered_range"

    metrics = summary["metrics"]["test"]
    outside_keys = (
        "predicted_fluid_temperature_outside_registered_range_fraction",
        "predicted_solid_temperature_outside_registered_range_fraction",
    )
    outside = [float(metrics[key]) for key in outside_keys if key in metrics]
    if outside and max(outside) > 0.0:
        return "test_prediction_outside_registered_range"
    return "test_prediction_accepted_by_common_energy_evaluator"


def load_completed_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    common_split: dict[str, tuple[str, ...]] | None = None
    for model_id, label, relative_path in MODEL_SOURCES:
        path = results_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing formal summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("split_name") != "pair_disjoint_stress_test":
            raise ValueError(f"{model_id} does not use pair_disjoint_stress_test")
        if summary.get("temperature_metric_definition") != EXPECTED_TEMPERATURE_METRIC:
            raise ValueError(f"{model_id} uses a different temperature metric")
        counts = split_counts(summary)
        selection_split = str(summary.get("selection_split"))
        if selection_split not in {"validation", "not_applicable"}:
            raise ValueError(f"{model_id} uses invalid selection split {selection_split}")
        split_ids = {
            role: tuple(sorted(str(value) for value in identifiers))
            for role, identifiers in summary["split_case_ids"].items()
        }
        if common_split is None:
            common_split = split_ids
        elif split_ids != common_split:
            raise ValueError(f"{model_id} uses different train/validation/test curves")

        metrics = summary["metrics"]["test"]
        energy_path = path.with_name("energy_balance_summary.json")
        if not energy_path.is_file():
            raise FileNotFoundError(f"missing common energy evaluation: {energy_path}")
        energy = json.loads(energy_path.read_text(encoding="utf-8"))
        if energy.get("split_name") != summary["split_name"]:
            raise ValueError(f"{model_id} energy evaluation uses a different split")
        energy_test = energy.get("role_metrics", {}).get("test")
        if not isinstance(energy_test, dict) or int(energy_test.get("curve_count", -1)) != 4:
            raise ValueError(f"{model_id} lacks a four-curve common test energy result")
        energy_ratio = finite_nonnegative(
            energy_test["prediction_to_openfoam_volume_weighted_energy_residual_ratio"],
            name=f"{model_id} common energy residual ratio",
        )
        rejected_roles = sorted(str(role) for role in energy.get("rejected_roles", []))
        row = {
            "model_id": model_id,
            "model_label": label,
            "result_status": "completed_formal_pair_disjoint_result",
            "selected_epoch_or_rank": selected_value(summary),
            "selection_split": selection_split,
            "split_train_count": counts["train"],
            "split_validation_count": counts["validation"],
            "split_test_count": counts["test"],
            "compute_device": str(summary.get("compute_device", "not_recorded")),
            "training_seconds": finite_nonnegative(
                summary.get("training_seconds", 0.0), name=f"{model_id} training time"
            ),
            "fluid_temperature_RMSE_K": finite_nonnegative(
                metrics["fluid_temperature_RMSE_K"], name=f"{model_id} fluid RMSE"
            ),
            "solid_temperature_RMSE_K": finite_nonnegative(
                metrics["solid_temperature_RMSE_K"], name=f"{model_id} solid RMSE"
            ),
            "maximum_absolute_temperature_error_K": finite_nonnegative(
                metrics["maximum_absolute_temperature_error_K"],
                name=f"{model_id} maximum absolute error",
            ),
            "solid_maximum_temperature_history_RMSE_K": finite_nonnegative(
                metrics["solid_maximum_temperature_history_RMSE_K"],
                name=f"{model_id} hotspot-history RMSE",
            ),
            "solid_regional_hotspot_exact_match_fraction": finite_nonnegative(
                metrics["solid_regional_hotspot_exact_match_fraction"],
                name=f"{model_id} hotspot match fraction",
            ),
            "inference_seconds_per_curve": finite_nonnegative(
                metrics["inference_seconds_per_curve"],
                name=f"{model_id} inference time",
            ),
            "common_energy_test_volume_weighted_residual_ratio": energy_ratio,
            "test_temperature_range_status": test_temperature_range_status(
                summary, energy
            ),
            "common_energy_rejected_roles": ";".join(rejected_roles),
            "source_summary": str(path),
            "source_summary_sha256": sha256(path),
            "source_energy_summary": str(energy_path),
            "source_energy_summary_sha256": sha256(energy_path),
        }
        rows.append(row)
    return rows


def best_model(rows: list[dict], metric: str, *, maximize: bool = False) -> dict:
    available = [row for row in rows if row[metric] is not None]
    selected = (max if maximize else min)(available, key=lambda row: row[metric])
    return {
        "metric": metric,
        "model_id": selected["model_id"],
        "value": selected[metric],
        "direction": "maximum" if maximize else "minimum",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict) -> None:
    rows = payload["completed_models"]
    lines = [
        "# 严格成对留出模型阶段对比",
        "",
        "这张表只汇总已经完成的正式结果。扩散温度修正尚未完成，因此现在不做最终综合排名。",
        "",
        "六种方法使用完全相同的6/2/4条训练、验证和独立测试曲线，温度误差均按区域体积加权计算。",
        "",
        "| 方法 | 流体RMSE (K) | 颗粒RMSE (K) | 最大误差 (K) | 最高温度历程RMSE (K) | 热点命中率 | 统一能量残差比 | 测试温度范围 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model_label} | {fluid_temperature_RMSE_K:.3f} | "
            "{solid_temperature_RMSE_K:.3f} | "
            "{maximum_absolute_temperature_error_K:.3f} | "
            "{solid_maximum_temperature_history_RMSE_K:.3f} | "
            "{solid_regional_hotspot_exact_match_fraction:.3f} | "
            "{common_energy_test_volume_weighted_residual_ratio:.3f} | "
            "{test_temperature_range_status} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 当前可以确认的现象",
            "",
            "- 持续性基线在当前严格划分上仍很强，说明这些热响应具有明显的短时惯性，神经网络必须超过这个基线才有实际意义。",
            "- 纯数据图Transformer的流体和颗粒整体RMSE与持续性基线接近，但推理时间更长。",
            "- 按统一能量后处理程序重算后，当前物理约束和因子化模型都没有优于纯数据模型的能量残差比。因此不能声称“加入物理项必然改善能量守恒”。",
            "- 物理约束模型改善了颗粒最高温度历程，但流体整体RMSE和最大误差明显增大，说明当前损失权重仍有竞争。",
            "- DMDc和低秩残差模型在当前严格外推划分上弱于持续性基线。",
            "- 低秩模型的独立测试集可用，但训练集和验证集中有少量颗粒温度低于298 K的登记下限，最终论文必须明确报告。",
            "- CPU和GPU的训练时间只用于说明计算成本，因硬件不同，不用它给方法排名。",
            "- 上述判断均为阶段结论；扩散温度修正、三次独立初值和轨迹数量曲线完成后才形成论文最终排序。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/hccb_p418_physical_steps_12"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/hccb_p418_physical_steps_12/provisional_strict_pair_disjoint_comparison"
        ),
    )
    args = parser.parse_args()

    rows = load_completed_rows(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "provisional_strict_pair_disjoint_comparison",
        "split_name": "pair_disjoint_stress_test",
        "completed_model_count": len(rows),
        "temperature_metric_definition": EXPECTED_TEMPERATURE_METRIC,
        "split_case_counts": EXPECTED_SPLIT_COUNTS,
        "common_energy_metric_definition": (
            "test prediction-to-OpenFOAM volume-weighted energy residual ratio "
            "from evaluate_hccb_p418_temporal_energy_balance.py"
        ),
        "pending_model_families": ["diffusion_temperature_correction"],
        "final_ranking_allowed": False,
        "hardware_normalized_training_time_available": False,
        "completed_models": rows,
        "descriptive_metric_extrema_not_overall_ranking": [
            best_model(rows, "fluid_temperature_RMSE_K"),
            best_model(rows, "solid_temperature_RMSE_K"),
            best_model(rows, "maximum_absolute_temperature_error_K"),
            best_model(rows, "solid_maximum_temperature_history_RMSE_K"),
            best_model(rows, "solid_regional_hotspot_exact_match_fraction", maximize=True),
            best_model(rows, "inference_seconds_per_curve"),
        ],
        "interpretation_boundary": (
            "The six completed methods may be compared metric by metric, but no final "
            "overall ranking is reported before the diffusion model and robustness runs finish. "
            "Training time is not hardware-normalized and is excluded from method ranking."
        ),
    }
    json_path = args.output_dir / "provisional_model_comparison.json"
    csv_path = args.output_dir / "provisional_model_comparison.csv"
    markdown_path = args.output_dir / "README_CN.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(csv_path, rows)
    write_markdown(markdown_path, payload)
    print(json.dumps({"status": payload["status"], "models": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
