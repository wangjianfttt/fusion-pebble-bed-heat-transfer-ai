#!/usr/bin/env python3
"""Summarize which converged P418 endpoints are ready for thermal steps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def endpoint_state(matrix_root: Path, condition_id: str) -> dict[str, object]:
    case = matrix_root / condition_id
    marker_path = case / "formal_sample_complete.json"
    state: dict[str, object] = {
        "condition_id": condition_id,
        "marker_path": str(marker_path),
        "ready": False,
        "reason": "completion marker is missing",
    }
    if not marker_path.is_file():
        return state
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        state["reason"] = f"completion marker cannot be read: {exc}"
        return state
    iteration = marker.get(
        "reported_iteration",
        marker.get("steady_iteration_end", marker.get("time")),
    )
    try:
        state["steady_iteration"] = int(float(iteration))
    except (TypeError, ValueError):
        state["reason"] = "steady iteration is missing or invalid"
        return state
    state["solver_time_semantics"] = marker.get(
        "solver_time_semantics", "steady_iteration_index"
    )
    state["physical_time_s"] = marker.get("physical_time_s")
    if state["solver_time_semantics"] != "steady_iteration_index":
        state["reason"] = "endpoint is not marked as a steady-iteration result"
        return state
    if marker.get("solver_finished") is not True:
        state["reason"] = "latest solver attempt is not marked as finished"
        return state
    if marker.get("training_sample_schema_version") != 3:
        state["reason"] = "training sample is not schema version 3"
        return state
    for key in ("relative_mass_difference", "relative_energy_difference"):
        value = marker.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            state["reason"] = f"{key} is missing or non-finite"
            return state
        state[key] = float(value)
    sample = Path(str(marker.get("training_sample", "")))
    if not sample.is_file():
        state["reason"] = "training sample file is missing"
        state["training_sample"] = str(sample)
        return state
    state.update(
        {
            "ready": True,
            "reason": "complete 3D endpoint and schema-3 sample are available",
            "training_sample": str(sample),
        }
    )
    return state


def build_summary(matrix_root: Path, plan: dict[str, object]) -> dict[str, object]:
    endpoint_ids = sorted(
        {
            str(sequence[key])
            for sequence in plan["sequences"]
            for key in ("source_condition_id", "target_condition_id")
        }
    )
    endpoint_rows = [endpoint_state(matrix_root, condition_id) for condition_id in endpoint_ids]
    endpoints = {str(row["condition_id"]): row for row in endpoint_rows}
    sequence_rows = []
    for sequence in plan["sequences"]:
        source_id = str(sequence["source_condition_id"])
        target_id = str(sequence["target_condition_id"])
        source = endpoints[source_id]
        target = endpoints[target_id]
        sequence_rows.append(
            {
                "sequence_id": str(sequence["sequence_id"]),
                "family": str(sequence["family"]),
                "source_condition_id": source_id,
                "source_ready": bool(source["ready"]),
                "source_reason": str(source["reason"]),
                "target_condition_id": target_id,
                "target_ready": bool(target["ready"]),
                "target_reason": str(target["reason"]),
                "sequence_ready": bool(source["ready"] and target["ready"]),
            }
        )
    return {
        "status": "p418_step_endpoint_readiness_summarized",
        "source_doi": plan["source_doi"],
        "sequence_count": len(sequence_rows),
        "ready_sequence_count": sum(bool(row["sequence_ready"]) for row in sequence_rows),
        "waiting_sequence_count": sum(not bool(row["sequence_ready"]) for row in sequence_rows),
        "unique_endpoint_count": len(endpoint_rows),
        "ready_endpoint_count": sum(bool(row["ready"]) for row in endpoint_rows),
        "waiting_endpoint_count": sum(not bool(row["ready"]) for row in endpoint_rows),
        "sequences": sequence_rows,
        "endpoints": endpoint_rows,
        "new_physical_parameters": [],
    }


def write_outputs(summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = summary["sequences"]
    with (output_dir / "sequence_readiness.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# P418热阶跃稳态端点准备情况",
        "",
        f"计划热阶跃：{summary['sequence_count']}条；当前可启动：{summary['ready_sequence_count']}条；仍需等待：{summary['waiting_sequence_count']}条。",
        f"共使用{summary['unique_endpoint_count']}个不同稳态端点，其中{summary['ready_endpoint_count']}个已经完成。",
        "",
        "| 热阶跃 | 类型 | 初始稳态 | 初始完成 | 目标稳态 | 目标完成 | 可启动 |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {sequence_id} | {family} | {source_condition_id} | {source_ready} | "
            "{target_condition_id} | {target_ready} | {sequence_ready} |".format(**row)
        )
    lines.extend(
        [
            "",
            "这里的“完成”要求稳态求解正常结束、质量和能量结果为有限数值，并且三维schema-3训练样本真实存在。",
            "端点中的200表示第200次稳态非线性迭代，不是200秒；物理时间只用于后续独立的瞬态热阶跃。",
            "本表只检查已有计算文件，不增加任何物理参数。",
        ]
    )
    (output_dir / "P418_热阶跃端点准备情况_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "parameters/hccb_p418_transient_step_plan.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    summary = build_summary(args.matrix_root.resolve(), plan)
    write_outputs(summary, args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
