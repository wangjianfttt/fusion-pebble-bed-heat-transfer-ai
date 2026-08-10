#!/usr/bin/env python3
"""Report finite-volume mass/energy scales used by the steady P418 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hccb_p418_comparison_contract import (
    file_record,
    split_indices,
    validate_split_and_statistics,
)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def oriented_balance(
    internal: np.ndarray,
    boundary: np.ndarray,
    internal_owner: np.ndarray,
    internal_neighbour: np.ndarray,
    boundary_owner: np.ndarray,
    node_count: int,
    source: np.ndarray | None = None,
) -> np.ndarray:
    """Return owner-positive, neighbour-negative finite-volume balances."""
    result = np.zeros((internal.shape[0], node_count), dtype=np.float64)
    for case_index in range(internal.shape[0]):
        np.add.at(result[case_index], internal_owner, internal[case_index])
        np.add.at(result[case_index], internal_neighbour, -internal[case_index])
        np.add.at(result[case_index], boundary_owner, boundary[case_index])
    if source is not None:
        result -= np.asarray(source, dtype=np.float64)
    return result


def incident_flux_rms(
    internal: np.ndarray,
    boundary: np.ndarray,
    internal_owner: np.ndarray,
    internal_neighbour: np.ndarray,
    boundary_owner: np.ndarray,
    node_count: int,
    source: np.ndarray | None = None,
) -> float:
    """Match the incident-flow normalization used by the training program."""
    incident = np.zeros((internal.shape[0], node_count), dtype=np.float64)
    for case_index in range(internal.shape[0]):
        np.add.at(incident[case_index], internal_owner, np.abs(internal[case_index]))
        np.add.at(
            incident[case_index], internal_neighbour, np.abs(internal[case_index])
        )
        np.add.at(incident[case_index], boundary_owner, np.abs(boundary[case_index]))
    if source is not None:
        incident += np.abs(np.asarray(source, dtype=np.float64))
    scale = rms(incident)
    if not scale > 0.0:
        raise ValueError("incident-flow normalization is not positive")
    return scale


def finite_range(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("reported finite-volume values are empty or non-finite")
    return {
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def summarize(
    *,
    state_targets: Path,
    mass_targets: Path,
    energy_targets: Path,
    split_file: Path,
    training_statistics: Path,
    split_name: str,
) -> dict[str, object]:
    paths = {
        "state_targets": state_targets.resolve(),
        "mass_targets": mass_targets.resolve(),
        "energy_targets": energy_targets.resolve(),
        "split_file": split_file.resolve(),
        "training_statistics": training_statistics.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(paths["state_targets"], allow_pickle=False) as loaded:
        condition_id = loaded["condition_id"].astype(str)
        node_type = loaded["node_type"].astype(np.int64)
    with np.load(paths["mass_targets"], allow_pickle=False) as loaded:
        mass_id = loaded["condition_id"].astype(str)
        internal_mass = loaded["internal_mass_flow_kg_s"].astype(np.float64)
        boundary_mass = loaded["boundary_mass_flow_kg_s"].astype(np.float64)
        mass_owner = loaded["internal_owner"].astype(np.int64)
        mass_neighbour = loaded["internal_neighbour"].astype(np.int64)
        mass_boundary_owner = loaded["boundary_owner"].astype(np.int64)
        boundary_patch = loaded["boundary_patch"].astype(np.int64)
    with np.load(paths["energy_targets"], allow_pickle=False) as loaded:
        energy_id = loaded["condition_id"].astype(str)
        internal_energy = loaded["internal_energy_flow_W"].astype(np.float64)
        boundary_energy = loaded["boundary_energy_flow_W"].astype(np.float64)
        source_power = loaded["node_source_power_W"].astype(np.float64)
        energy_owner = loaded["internal_owner"].astype(np.int64)
        energy_neighbour = loaded["internal_neighbour"].astype(np.int64)
        energy_boundary_owner = loaded["boundary_owner"].astype(np.int64)

    if not np.array_equal(condition_id, mass_id) or not np.array_equal(
        condition_id, energy_id
    ):
        raise ValueError("state, mass and energy condition orders differ")
    split_case_ids, _ = validate_split_and_statistics(
        split_file=paths["split_file"],
        training_statistics=paths["training_statistics"],
        split_name=split_name,
        condition_ids=condition_id,
    )
    indices = split_indices(split_case_ids, condition_id)
    train = indices["train"]
    fluid_count = int(np.count_nonzero(node_type == 0))
    node_count = len(node_type)

    scales = {
        "internal_mass_flow_rms_kg_s": rms(internal_mass[train]),
        "boundary_mass_flow_rms_kg_s": rms(boundary_mass[train]),
        "regional_incident_mass_rms_kg_s": incident_flux_rms(
            internal_mass[train],
            boundary_mass[train],
            mass_owner,
            mass_neighbour,
            mass_boundary_owner,
            fluid_count,
        ),
        "internal_energy_flow_rms_W": rms(internal_energy[train]),
        "boundary_energy_flow_rms_W": rms(boundary_energy[train]),
        "regional_incident_energy_rms_W": incident_flux_rms(
            internal_energy[train],
            boundary_energy[train],
            energy_owner,
            energy_neighbour,
            energy_boundary_owner,
            node_count,
            source_power[train],
        ),
    }
    if not all(value > 0.0 for value in scales.values()):
        raise ValueError("one or more finite-volume normalization scales are not positive")

    mass_balance = oriented_balance(
        internal_mass,
        boundary_mass,
        mass_owner,
        mass_neighbour,
        mass_boundary_owner,
        fluid_count,
    )
    energy_balance = oriented_balance(
        internal_energy,
        boundary_energy,
        energy_owner,
        energy_neighbour,
        energy_boundary_owner,
        node_count,
        source_power,
    )

    case_rows: list[dict[str, object]] = []
    role_by_case = {
        int(case_index): role
        for role, role_indices in indices.items()
        for case_index in role_indices
    }
    for case_index, identifier in enumerate(condition_id):
        inlet_mass = abs(
            float(np.sum(boundary_mass[case_index][boundary_patch == 0]))
        )
        generated_power = float(np.sum(source_power[case_index]))
        if inlet_mass <= 0.0 or generated_power <= 0.0:
            raise ValueError(
                f"condition {identifier} has non-positive inlet mass or generated power"
            )
        role = role_by_case.get(case_index, "unused")
        case_rows.append(
            {
                "condition_id": str(identifier),
                "role": role,
                "inlet_mass_flow_kg_s": inlet_mass,
                "generated_power_W": generated_power,
                "target_mass_balance_rms_kg_s": rms(mass_balance[case_index]),
                "target_mass_balance_normalized_rms": (
                    rms(mass_balance[case_index])
                    / scales["regional_incident_mass_rms_kg_s"]
                ),
                "target_local_mass_l1_over_two_inlet": float(
                    np.sum(np.abs(mass_balance[case_index])) / (2.0 * inlet_mass)
                ),
                "target_global_mass_imbalance_over_inlet": float(
                    abs(np.sum(mass_balance[case_index])) / inlet_mass
                ),
                "target_energy_balance_rms_W": rms(energy_balance[case_index]),
                "target_energy_balance_normalized_rms": (
                    rms(energy_balance[case_index])
                    / scales["regional_incident_energy_rms_W"]
                ),
                "target_local_energy_l1_over_two_generated_power": float(
                    np.sum(np.abs(energy_balance[case_index]))
                    / (2.0 * generated_power)
                ),
                "target_global_energy_imbalance_over_generated_power": float(
                    abs(np.sum(energy_balance[case_index])) / generated_power
                ),
            }
        )

    split_summary: dict[str, object] = {}
    metric_names = tuple(
        name
        for name in case_rows[0]
        if name not in {"condition_id", "role"}
    )
    for role in ("train", "validation", "test"):
        selected = [row for row in case_rows if row["role"] == role]
        split_summary[role] = {
            "condition_count": len(selected),
            "condition_ids": [row["condition_id"] for row in selected],
            "metrics": {
                name: finite_range([float(row[name]) for row in selected])
                for name in metric_names
            },
        }

    return {
        "status": "p418_finite_volume_loss_scales_reported",
        "split_name": split_name,
        "condition_count": len(condition_id),
        "fluid_node_count": fluid_count,
        "regional_node_count": node_count,
        "formula": {
            "mass_balance": "sum(owner +m_dot, neighbour -m_dot, boundary +m_dot)",
            "energy_balance": "sum(owner +Q_dot, neighbour -Q_dot, boundary +Q_dot) - source_power",
            "physics_loss": "mean square of predicted minus finite-volume target balance, divided by the training-only incident-flow RMS",
        },
        "training_only_normalization": {
            "condition_ids": split_case_ids["train"],
            "scales": scales,
            "validation_or_test_values_used_in_scale": False,
        },
        "unused_conditions": {
            "condition_count": sum(row["role"] == "unused" for row in case_rows),
            "condition_ids": [
                row["condition_id"] for row in case_rows if row["role"] == "unused"
            ],
        },
        "split_summary": split_summary,
        "cases": case_rows,
        "files": {name: file_record(path) for name, path in paths.items()},
        "new_physical_parameters": [],
    }


def write_chinese_summary(payload: dict[str, object], output: Path) -> None:
    scales = payload["training_only_normalization"]["scales"]
    lines = [
        "# P418 三维有限体积质量与能量收支量级",
        "",
        "这份结果直接读取 OpenFOAM 后处理得到的区域面质量流量、区域面能量流量和体热源。",
        "连续性按控制体各面的有向质量流量求和；能量收支按有向能流求和后减去体热源。",
        "PINN 使用的归一化尺度只由训练工况计算，验证和测试工况不参与尺度计算。",
        "",
        "## 训练工况给出的尺度",
        "",
        f"- 内部面质量流量均方根：{scales['internal_mass_flow_rms_kg_s']:.6e} kg/s",
        f"- 边界面质量流量均方根：{scales['boundary_mass_flow_rms_kg_s']:.6e} kg/s",
        f"- 区域入射质量流量尺度：{scales['regional_incident_mass_rms_kg_s']:.6e} kg/s",
        f"- 内部面能量流量均方根：{scales['internal_energy_flow_rms_W']:.6e} W",
        f"- 边界面能量流量均方根：{scales['boundary_energy_flow_rms_W']:.6e} W",
        f"- 区域入射能量尺度：{scales['regional_incident_energy_rms_W']:.6e} W",
        "",
        "## 各数据部分的有限体积收支",
        "",
    ]
    for role, label in (("train", "训练"), ("validation", "验证"), ("test", "测试")):
        metrics = payload["split_summary"][role]["metrics"]
        lines.extend(
            [
                f"### {label}工况",
                "",
                f"- 工况数：{payload['split_summary'][role]['condition_count']}",
                "- 质量收支归一化均方根范围："
                f"{metrics['target_mass_balance_normalized_rms']['minimum']:.3e}--"
                f"{metrics['target_mass_balance_normalized_rms']['maximum']:.3e}",
                "- 能量收支归一化均方根范围："
                f"{metrics['target_energy_balance_normalized_rms']['minimum']:.3e}--"
                f"{metrics['target_energy_balance_normalized_rms']['maximum']:.3e}",
                "",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-targets", type=Path, required=True)
    parser.add_argument("--mass-targets", type=Path, required=True)
    parser.add_argument("--energy-targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--training-statistics", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chinese-summary", type=Path)
    args = parser.parse_args()
    payload = summarize(
        state_targets=args.state_targets,
        mass_targets=args.mass_targets,
        energy_targets=args.energy_targets,
        split_file=args.split_file,
        training_statistics=args.training_statistics,
        split_name=args.split_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.chinese_summary is not None:
        write_chinese_summary(payload, args.chinese_summary)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
