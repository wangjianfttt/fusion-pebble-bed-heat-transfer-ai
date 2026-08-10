#!/usr/bin/env python3
"""Quantify information lost when native P418 cells are volume-averaged.

This is a representation check, not a neural-network accuracy result.  It
compares each solved OpenFOAM temperature field with the exact regional state
that is supplied to the PINN and graph operators.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def volume_mean(
    values: np.ndarray,
    volume: np.ndarray,
    regional_global: np.ndarray,
    selected_global: np.ndarray,
) -> np.ndarray:
    """Use the same volume-average definition as the regional-state builder."""
    global_to_local = np.full(int(selected_global.max()) + 1, -1, dtype=np.int64)
    global_to_local[selected_global] = np.arange(len(selected_global), dtype=np.int64)
    local = global_to_local[regional_global]
    if np.any(local < 0):
        raise ValueError("fine cells map outside the selected material regions")
    denominator = np.bincount(local, weights=volume, minlength=len(selected_global))
    numerator = np.bincount(
        local, weights=values * volume, minlength=len(selected_global)
    )
    return numerator / denominator


def error_metrics(
    native: np.ndarray,
    reconstructed: np.ndarray,
    volume: np.ndarray,
) -> dict[str, float]:
    error = reconstructed - native
    temperature_range = float(np.max(native) - np.min(native))
    rmse = float(np.sqrt(weighted_mean(error * error, volume)))
    return {
        "volume_weighted_mean_error_K": weighted_mean(error, volume),
        "volume_weighted_mae_K": weighted_mean(np.abs(error), volume),
        "volume_weighted_rmse_K": rmse,
        "maximum_absolute_error_K": float(np.max(np.abs(error))),
        "native_temperature_range_K": temperature_range,
        "rmse_over_native_range_percent": (
            100.0 * rmse / temperature_range if temperature_range > 0.0 else 0.0
        ),
    }


def particle_diameter_m(path: Path) -> float:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = {row["parameter_id"]: row for row in csv.DictReader(stream)}
    row = rows["P048"]
    if row["单位"].strip() != "mm":
        raise ValueError("P048 must be recorded in mm")
    return float(row["采用值或关系式"]) * 1.0e-3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--subface-geometry", type=Path, required=True)
    parser.add_argument("--regional-state-targets", type=Path, required=True)
    parser.add_argument(
        "--parameter-source",
        type=Path,
        default=ROOT / "parameters/hccb_p418_physical_parameter_sources.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_index.resolve()
    dataset_root = dataset_path.parent
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    diameter_m = particle_diameter_m(args.parameter_source.resolve())

    with np.load(dataset_root / dataset["shared_topology_file"], allow_pickle=False) as topology:
        fluid_volume = topology["fluid_cell_volume_m3"].astype(np.float64)
        solid_volume = topology["solid_cell_volume_m3"].astype(np.float64)
        solid_centroid = topology["solid_cell_centroid_m"].astype(np.float64)
    with np.load(args.subface_geometry.resolve(), allow_pickle=False) as geometry:
        parent = geometry["fine_to_regional_global"].astype(np.int64)
        fluid_global = geometry["fluid_global_region"].astype(np.int64)
        solid_global = geometry["solid_global_region"].astype(np.int64)
        regional_solid_centroid = geometry["solid_cell_centroid_m"].astype(np.float64)
    with np.load(args.regional_state_targets.resolve(), allow_pickle=False) as targets:
        target_ids = targets["condition_id"].astype(str)
        regional_states = targets["state_physical"].astype(np.float64)

    fine_fluid_count = len(fluid_volume)
    if len(parent) != fine_fluid_count + len(solid_volume):
        raise ValueError("fine-to-regional map does not cover both native meshes")
    fluid_parent = parent[:fine_fluid_count]
    solid_parent = parent[fine_fluid_count:]
    dataset_ids = np.asarray([row["condition_id"] for row in dataset["conditions"]])
    if not np.array_equal(dataset_ids, target_ids):
        raise ValueError("dataset and regional target condition order differ")

    rows: list[dict[str, object]] = []
    exact_regional_state_match = True
    for case_index, record in enumerate(dataset["conditions"]):
        with np.load(dataset_root / record["field_file"], allow_pickle=False) as fields:
            native_fluid = fields["fluid_temperature_K"].astype(np.float64)
            native_solid = fields["solid_temperature_K"].astype(np.float64)

        state = regional_states[case_index]
        regional_fluid = state[fluid_global, 4]
        regional_solid = state[solid_global, 4]
        recalculated_fluid = volume_mean(
            native_fluid, fluid_volume, fluid_parent, fluid_global
        )
        recalculated_solid = volume_mean(
            native_solid, solid_volume, solid_parent, solid_global
        )
        if not np.array_equal(recalculated_fluid, regional_fluid):
            exact_regional_state_match = False
        if not np.array_equal(recalculated_solid, regional_solid):
            exact_regional_state_match = False

        reconstructed_fluid = state[fluid_parent, 4]
        reconstructed_solid = state[solid_parent, 4]
        fluid_metrics = error_metrics(native_fluid, reconstructed_fluid, fluid_volume)
        solid_metrics = error_metrics(native_solid, reconstructed_solid, solid_volume)

        native_hot_index = int(np.argmax(native_solid))
        regional_hot_local = int(np.argmax(regional_solid))
        regional_hot_global = int(solid_global[regional_hot_local])
        centroid_distance_m = float(
            np.linalg.norm(
                solid_centroid[native_hot_index]
                - regional_solid_centroid[regional_hot_local]
            )
        )
        regional_hot_fine_cells = np.flatnonzero(solid_parent == regional_hot_global)
        nearest_cell_distance_m = float(
            np.min(
                np.linalg.norm(
                    solid_centroid[regional_hot_fine_cells]
                    - solid_centroid[native_hot_index],
                    axis=1,
                )
            )
        )
        native_max = float(native_solid[native_hot_index])
        regional_max = float(regional_solid[regional_hot_local])
        row: dict[str, object] = {
            "condition_id": record["condition_id"],
            "inlet_velocity_m_s": float(record["inlet_velocity_m_s"]),
            "inlet_temperature_K": float(record["inlet_temperature_K"]),
            "solid_heat_source_W_m3": float(record["solid_heat_source_W_m3"]),
        }
        row.update({f"fluid_{key}": value for key, value in fluid_metrics.items()})
        row.update({f"solid_{key}": value for key, value in solid_metrics.items()})
        row.update(
            {
                "native_solid_max_temperature_K": native_max,
                "regional_solid_max_temperature_K": regional_max,
                "solid_hotspot_temperature_loss_K": native_max - regional_max,
                "solid_hotspot_temperature_loss_over_native_range_percent": (
                    100.0 * (native_max - regional_max)
                    / solid_metrics["native_temperature_range_K"]
                    if solid_metrics["native_temperature_range_K"] > 0.0
                    else 0.0
                ),
                "solid_hotspot_region_centroid_distance_m": centroid_distance_m,
                "solid_hotspot_region_centroid_distance_dp": centroid_distance_m / diameter_m,
                "solid_hotspot_nearest_cell_distance_m": nearest_cell_distance_m,
                "solid_hotspot_nearest_cell_distance_dp": nearest_cell_distance_m / diameter_m,
                "native_hot_cell_is_in_hottest_regional_node": bool(
                    solid_parent[native_hot_index] == regional_hot_global
                ),
            }
        )
        rows.append(row)

    if not exact_regional_state_match:
        raise ValueError("saved regional temperatures differ from direct volume averaging")

    csv_path = output / "regional_representation_fidelity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def aggregate(key: str) -> dict[str, float | str]:
        values = np.asarray([float(row[key]) for row in rows])
        maximum_index = int(np.argmax(values))
        return {
            "mean": float(np.mean(values)),
            "maximum": float(values[maximum_index]),
            "maximum_condition_id": str(rows[maximum_index]["condition_id"]),
        }

    def maximum_absolute(key: str) -> dict[str, float | str]:
        values = np.abs(np.asarray([float(row[key]) for row in rows]))
        maximum_index = int(np.argmax(values))
        return {
            "maximum_absolute": float(values[maximum_index]),
            "maximum_condition_id": str(rows[maximum_index]["condition_id"]),
        }

    summary = {
        "status": "regional_representation_fidelity_ready",
        "scope": "OpenFOAM-to-regional representation only; not neural prediction accuracy",
        "counts": {
            "cases": len(rows),
            "native_fluid_cells": len(fluid_volume),
            "native_solid_cells": len(solid_volume),
            "regional_fluid_nodes": len(fluid_global),
            "regional_solid_nodes": len(solid_global),
        },
        "compression_ratio": {
            "fluid_native_cells_per_regional_node": len(fluid_volume) / len(fluid_global),
            "solid_native_cells_per_regional_node": len(solid_volume) / len(solid_global),
            "total_native_cells_per_regional_node": len(parent) / (len(fluid_global) + len(solid_global)),
        },
        "exact_saved_regional_state_matches_direct_volume_average": True,
        "global_volume_mean_temperature_error_K": {
            "fluid": maximum_absolute("fluid_volume_weighted_mean_error_K"),
            "solid": maximum_absolute("solid_volume_weighted_mean_error_K"),
        },
        "metrics": {
            key: aggregate(key)
            for key in (
                "fluid_volume_weighted_rmse_K",
                "solid_volume_weighted_rmse_K",
                "fluid_rmse_over_native_range_percent",
                "solid_rmse_over_native_range_percent",
                "solid_hotspot_temperature_loss_K",
                "solid_hotspot_temperature_loss_over_native_range_percent",
                "solid_hotspot_region_centroid_distance_dp",
                "solid_hotspot_nearest_cell_distance_dp",
            )
        },
        "hottest_native_cell_region_match_fraction": float(
            np.mean([row["native_hot_cell_is_in_hottest_regional_node"] for row in rows])
        ),
        "particle_diameter_parameter_id": "P048",
        "particle_diameter_m": diameter_m,
        "result_csv": csv_path.name,
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    metrics = summary["metrics"]
    lines = [
        "# P418原始单元场到区域图表示的温度保真度",
        "",
        f"本次比较使用{len(rows)}个已完成的真实OpenFOAM工况。它只量化约190万原始单元压缩到区域图节点时损失了多少温度细节，不是神经网络预测精度。",
        "",
        f"- 流体网格从{len(fluid_volume)}个单元压缩为{len(fluid_global)}个区域节点，平均每个节点代表{len(fluid_volume) / len(fluid_global):.2f}个原始单元。",
        f"- 颗粒网格从{len(solid_volume)}个单元压缩为{len(solid_global)}个区域节点，平均每个节点代表{len(solid_volume) / len(solid_global):.2f}个原始单元。",
        f"- 体积平均前后，全域流体和颗粒平均温度的最大绝对差分别为{summary['global_volume_mean_temperature_error_K']['fluid']['maximum_absolute']:.3g} K和{summary['global_volume_mean_temperature_error_K']['solid']['maximum_absolute']:.3g} K。",
        f"- 流体温度体积加权RMSE的工况平均值为{metrics['fluid_volume_weighted_rmse_K']['mean']:.4g} K，最大值为{metrics['fluid_volume_weighted_rmse_K']['maximum']:.4g} K。",
        f"- 颗粒温度体积加权RMSE的工况平均值为{metrics['solid_volume_weighted_rmse_K']['mean']:.4g} K，最大值为{metrics['solid_volume_weighted_rmse_K']['maximum']:.4g} K。",
        f"- 区域平均造成的颗粒最高温度损失平均为{metrics['solid_hotspot_temperature_loss_K']['mean']:.4g} K，最大为{metrics['solid_hotspot_temperature_loss_K']['maximum']:.4g} K。",
        f"- 原始最高温度单元到最高温区域所含最近原始单元的距离平均为{metrics['solid_hotspot_nearest_cell_distance_dp']['mean']:.4g}个颗粒直径，最大为{metrics['solid_hotspot_nearest_cell_distance_dp']['maximum']:.4g}个颗粒直径。",
        f"- 原始最高温度单元本身落在最高温区域内的工况比例为{summary['hottest_native_cell_region_match_fraction']:.1%}。区域中心距离另行保存在CSV中，不用它代替热点所属区域。",
        "",
        "完整60工况结束后，正式流程会用同一程序重新生成这些数值。最终模型误差必须与这里的区域化误差分开报告。",
    ]
    (output / "区域图温度保真度_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
