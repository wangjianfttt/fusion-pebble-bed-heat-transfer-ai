#!/usr/bin/env python3
"""Compare source-backed steady PINN loss-weight settings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


def load_settings(path: Path, project_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.resolve().open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            source = project_root.resolve() / raw["source_path"]
            if not source.is_file():
                raise ValueError(f"missing loss-weight source {source}")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            if source_hash != raw["source_sha256"]:
                raise ValueError(f"loss-weight source changed: {source}")
            source_text = source.read_text(encoding="utf-8")
            for assignment in raw["source_fields"].split(";"):
                name, expected_text = assignment.split("=", 1)
                matched = re.search(
                    rf"^\s*{re.escape(name)}:\s*([-+0-9.eE_]+)\s*$",
                    source_text,
                    flags=re.MULTILINE,
                )
                if not matched or float(matched.group(1).replace("_", "")) != float(expected_text):
                    raise ValueError(f"{source} does not contain {assignment}")
            expected_mapping = (
                "xy_loss->state_data;ic_loss->face_flux;"
                "f_loss->physics_balance;ratio_only"
            )
            if raw.get("transfer_mapping") != expected_mapping:
                raise ValueError("loss-weight transfer mapping is missing or changed")
            rows.append(
                {
                    **raw,
                    "state_data_weight": float(raw["state_data_weight"]),
                    "face_flux_weight": float(raw["face_flux_weight"]),
                    "physics_balance_weight": float(raw["physics_balance_weight"]),
                    "source_absolute_path": str(source),
                }
            )
    if len(rows) < 2 or len({str(row["setting_id"]) for row in rows}) != len(rows):
        raise ValueError("loss-weight comparison needs unique source-backed settings")
    return rows


def p95(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("metric list is empty or non-finite")
    return float(np.percentile(np.abs(values), 95.0))


def case_metric(cases: list[dict[str, object]], name: str) -> float:
    return p95(
        [float(case["engineering_absolute_errors"][name]) for case in cases]
    )


def summarize(
    *,
    project_root: Path,
    settings_path: Path,
    selection_path: Path,
    result_root: Path,
) -> dict[str, object]:
    settings = load_settings(settings_path, project_root)
    selection = json.loads(selection_path.resolve().read_text(encoding="utf-8"))
    if selection.get("status") != "steady_PINN_chain_source_selected":
        raise ValueError("steady PINN training length has not been selected")
    epochs = int(selection["selected_epochs"])
    split_name = str(selection["split_name"])
    standard_summary = Path(str(selection["selected_summary"])).resolve()
    summaries: list[tuple[dict[str, object], Path]] = []
    for index, setting in enumerate(settings):
        summary_path = (
            standard_summary
            if index == 0
            else result_root.resolve()
            / f"hccb_p418_loss_weight_{setting['setting_id']}_{epochs}epoch"
            / "summary.json"
        )
        summaries.append((setting, summary_path))

    rows: list[dict[str, object]] = []
    common: dict[str, object] | None = None
    initial_hash: str | None = None
    for setting, summary_path in summaries:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("architecture") != "pinn":
            raise ValueError(f"loss-weight result is not a physics PINN: {summary_path}")
        if payload.get("split_name") != split_name or int(payload.get("epochs", -1)) != epochs:
            raise ValueError(f"loss-weight result uses a different split or schedule: {summary_path}")
        expected_weights = {
            "state_data": setting["state_data_weight"],
            "face_flux": setting["face_flux_weight"],
            "physics_balance": setting["physics_balance_weight"],
        }
        if payload.get("loss_group_weights") != expected_weights:
            raise ValueError(f"loss weights differ from their source row: {summary_path}")
        current_common = {
            "split_case_ids": payload.get("split_case_ids"),
            "model_parameter_count": payload.get("model_parameter_count"),
            "optimizer_name": payload.get("optimizer_name"),
            "training_seed": payload.get("training_seed"),
            "effective_batch_size": payload.get("effective_batch_size"),
            "microbatch_size": payload.get("microbatch_size"),
            "updates_per_epoch": payload.get("updates_per_epoch"),
            "total_parameter_updates": payload.get("total_parameter_updates"),
            "field_architecture": payload.get("field_architecture"),
            "settings_from_archived_source": payload.get("settings_from_archived_source"),
            "normalization": payload.get("normalization"),
            "metric_contract": payload.get("metric_contract"),
            "common_comparison_fingerprint": payload.get("run_provenance", {}).get(
                "common_comparison_fingerprint"
            ),
        }
        if common is None:
            common = current_common
            initial_hash = str(payload.get("initial_model_state_sha256", ""))
        elif current_common != common or payload.get("initial_model_state_sha256") != initial_hash:
            raise ValueError("loss-weight runs do not use the same data, model and initialization")
        test = payload["evaluations"]["test"]
        cases = test["cases"]
        metrics = test["metrics"]
        best_epoch = int(payload["best_epoch"])
        rows.append(
            {
                "setting_id": setting["setting_id"],
                "state_data_weight": setting["state_data_weight"],
                "face_flux_weight": setting["face_flux_weight"],
                "physics_balance_weight": setting["physics_balance_weight"],
                "state_normalized_RMSE": float(metrics["state_normalized_rmse"]),
                "outlet_temperature_p95_K": case_metric(cases, "outlet_temperature_K"),
                "solid_maximum_temperature_p95_K": case_metric(
                    cases, "solid_maximum_temperature_K"
                ),
                "cooling_wall_heat_p95_percent_generated": p95(
                    [
                        100.0
                        * float(
                            case["engineering_absolute_errors"][
                                "cooling_wall_heat_into_fluid_W"
                            ]
                        )
                        / float(case["generated_power_W"])
                        for case in cases
                    ]
                ),
                "global_mass_imbalance_p95_percent_inlet": p95(
                    [100.0 * float(case["global_mass_imbalance_over_inlet"]) for case in cases]
                ),
                "global_energy_imbalance_p95_percent_generated": p95(
                    [
                        100.0 * float(case["global_energy_imbalance_over_generated_power"])
                        for case in cases
                    ]
                ),
                "best_epoch": best_epoch,
                "best_epoch_is_final_epoch": best_epoch == epochs,
                "summary_file": str(summary_path),
                "source_path": setting["source_path"],
            }
        )
    return {
        "status": "p418_source_backed_loss_weight_sensitivity_complete",
        "split_name": split_name,
        "epochs": epochs,
        "setting_count": len(rows),
        "same_data_network_optimizer_seed_and_initial_state": True,
        "common_training_contract": common,
        "all_settings_best_epoch_is_not_final": all(
            not bool(row["best_epoch_is_final_epoch"]) for row in rows
        ),
        "initial_model_state_sha256": initial_hash,
        "rows": rows,
        "interpretation": (
            "Each physical quantity is reported separately. No cross-unit score is used, and the "
            "three dimensionless numerical ratios are transferred from archived official PINO "
            "configurations by supervised-data/constrained-data/equation-loss role. This does not "
            "assert physical equivalence between the original PINO initial-condition loss and the "
            "pebble-bed face-flow loss."
        ),
        "new_physical_parameters": [],
    }


def write_outputs(payload: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = payload["rows"]
    with (output_dir / "loss_weight_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 稳态PINN损失比例比较",
        "",
        "三组无量纲比例均直接取自项目保存的公开PINO配置。它们按状态数据、受约束数据和方程关系的顺序转移到本项目，并使用相同的OpenFOAM工况、坐标网络、初始权重、Adam设置和训练轮数。该转移只用于数值敏感性比较，不表示原PINO初值损失与本项目面流量损失具有相同物理含义。",
        "",
        "| 比例编号 | 状态:面流量:收支 | 状态RMSE | 出口温度p95 (K) | 颗粒最高温度p95 (K) | 冷却壁热量p95 (%发热) | 全局质量差p95 (%入口) | 全局能量差p95 (%发热) | 最低验证误差在最后一轮 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['setting_id']} | {row['state_data_weight']:g}:{row['face_flux_weight']:g}:{row['physics_balance_weight']:g} | "
            f"{row['state_normalized_RMSE']:.4g} | {row['outlet_temperature_p95_K']:.4g} | "
            f"{row['solid_maximum_temperature_p95_K']:.4g} | {row['cooling_wall_heat_p95_percent_generated']:.4g} | "
            f"{row['global_mass_imbalance_p95_percent_inlet']:.4g} | {row['global_energy_imbalance_p95_percent_generated']:.4g} | "
            f"{'是' if row['best_epoch_is_final_epoch'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "这些量分别比较，不把K、质量差和热量差混成一个总分。如果某组比例的最低验证误差恰好出现在最后一轮，该组需要按同一来源轮数继续训练，不能据此判断比例优劣。比例变化后的模型精度只能在60个稳态工况完成后解释。",
        ]
    )
    (output_dir / "损失比例比较_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        project_root=args.project_root,
        settings_path=args.settings,
        selection_path=args.selection,
        result_root=args.result_root,
    )
    write_outputs(payload, args.output_dir.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
