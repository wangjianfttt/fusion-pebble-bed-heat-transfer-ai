#!/usr/bin/env python3
"""Evaluate regional predictions after prolongation to native OpenFOAM cells."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from quantify_hccb_p418_regional_representation_fidelity import (
    error_metrics,
    particle_diameter_m,
    weighted_mean,
)
from compare_hccb_p418_native_reconstruction import (
    global_to_local,
    phase_reconstruction,
    reconstruction_operators,
    unique_neighbour_lists,
)


ROOT = Path(__file__).resolve().parents[1]


def phase_rmse(error: np.ndarray, volume: np.ndarray) -> float:
    return float(np.sqrt(weighted_mean(error * error, volume)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--subface-geometry", type=Path, required=True)
    parser.add_argument("--regional-state-targets", type=Path, required=True)
    parser.add_argument("--regional-predictions", type=Path, required=True)
    parser.add_argument("--training-statistics", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
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
    records = {row["condition_id"]: row for row in dataset["conditions"]}
    statistics = json.loads(args.training_statistics.resolve().read_text(encoding="utf-8"))
    split = statistics["splits"][args.split_name]
    target_stats = split["targets"]
    fluid_mean = float(target_stats["fluid_temperature_K"]["mean"][0])
    fluid_std = float(target_stats["fluid_temperature_K"]["standard_deviation"][0])
    solid_mean = float(target_stats["solid_temperature_K"]["mean"][0])
    solid_std = float(target_stats["solid_temperature_K"]["standard_deviation"][0])
    diameter_m = particle_diameter_m(args.parameter_source.resolve())

    with np.load(dataset_root / dataset["shared_topology_file"], allow_pickle=False) as topology:
        fluid_volume = topology["fluid_cell_volume_m3"].astype(np.float64)
        solid_volume = topology["solid_cell_volume_m3"].astype(np.float64)
        fluid_centroid = topology["fluid_cell_centroid_m"].astype(np.float64)
        solid_centroid = topology["solid_cell_centroid_m"].astype(np.float64)
    with np.load(args.subface_geometry.resolve(), allow_pickle=False) as geometry:
        parent = geometry["fine_to_regional_global"].astype(np.int64)
        fluid_global = geometry["fluid_global_region"].astype(np.int64)
        solid_global = geometry["solid_global_region"].astype(np.int64)
        regional_centroid = {
            phase: geometry[f"{phase}_cell_centroid_m"].astype(np.float64)
            for phase in ("fluid", "solid")
        }
        subface_owner = {
            phase: geometry[f"{phase}_internal_subface_owner"].astype(np.int64)
            for phase in ("fluid", "solid")
        }
        subface_neighbour = {
            phase: geometry[f"{phase}_internal_subface_neighbour"].astype(np.int64)
            for phase in ("fluid", "solid")
        }
    with np.load(args.regional_state_targets.resolve(), allow_pickle=False) as targets:
        target_ids = targets["condition_id"].astype(str)
        target_states = targets["state_physical"].astype(np.float64)
        target_by_id = {condition_id: index for index, condition_id in enumerate(target_ids)}
    with np.load(args.regional_predictions.resolve(), allow_pickle=False) as prediction_file:
        prediction_ids = prediction_file["condition_id"].astype(str)
        prediction_normalized = prediction_file["baseline_state_normalized"].astype(np.float64)
        node_type = prediction_file["node_type"].astype(np.int8)
        regional_volume = prediction_file["node_volume_m3"].astype(np.float64)

    fine_fluid_count = len(fluid_volume)
    if len(parent) != fine_fluid_count + len(solid_volume):
        raise ValueError("fine-to-regional map does not cover both native phases")
    if len(node_type) != len(regional_volume):
        raise ValueError("prediction node type and volume lengths differ")
    if not np.array_equal(np.flatnonzero(node_type == 0), fluid_global):
        raise ValueError("prediction fluid-node order differs from regional geometry")
    if not np.array_equal(np.flatnonzero(node_type == 1), solid_global):
        raise ValueError("prediction solid-node order differs from regional geometry")
    fluid_parent = parent[:fine_fluid_count]
    solid_parent = parent[fine_fluid_count:]
    selected_global = {"fluid": fluid_global, "solid": solid_global}
    phase_parent_global = {"fluid": fluid_parent, "solid": solid_parent}
    phase_parent_local = {
        phase: global_to_local(phase_parent_global[phase], selected_global[phase])
        for phase in ("fluid", "solid")
    }
    phase_neighbours = {
        phase: unique_neighbour_lists(
            len(selected_global[phase]), subface_owner[phase], subface_neighbour[phase]
        )
        for phase in ("fluid", "solid")
    }
    phase_operators = {
        phase: reconstruction_operators(
            regional_centroid[phase], phase_neighbours[phase]
        )[0]
        for phase in ("fluid", "solid")
    }

    rows: list[dict[str, object]] = []
    for local, condition_id in enumerate(prediction_ids):
        if condition_id not in records or condition_id not in target_by_id:
            raise ValueError(f"prediction condition is absent from the reference data: {condition_id}")
        record = records[condition_id]
        target_state = target_states[target_by_id[condition_id]]
        predicted_state = np.zeros_like(target_state)
        predicted_state[fluid_global, 4] = (
            prediction_normalized[local, fluid_global, 4] * fluid_std + fluid_mean
        )
        predicted_state[solid_global, 4] = (
            prediction_normalized[local, solid_global, 4] * solid_std + solid_mean
        )
        with np.load(dataset_root / record["field_file"], allow_pickle=False) as fields:
            native_fluid = fields["fluid_temperature_K"].astype(np.float64)
            native_solid = fields["solid_temperature_K"].astype(np.float64)

        row: dict[str, object] = {
            "condition_id": condition_id,
            "inlet_velocity_m_s": float(record["inlet_velocity_m_s"]),
            "inlet_temperature_K": float(record["inlet_temperature_K"]),
            "solid_heat_source_W_m3": float(record["solid_heat_source_W_m3"]),
        }
        limited_native: dict[str, np.ndarray] = {}
        for phase, native, centroid, volume, selected, fine_parent in (
            ("fluid", native_fluid, fluid_centroid, fluid_volume, fluid_global, fluid_parent),
            ("solid", native_solid, solid_centroid, solid_volume, solid_global, solid_parent),
        ):
            target_regional = target_state[selected, 4]
            predicted_regional = predicted_state[selected, 4]
            represented_native = target_state[fine_parent, 4]
            predicted_native = predicted_state[fine_parent, 4]
            representation_rmse = phase_rmse(represented_native - native, volume)
            regional_model_rmse = phase_rmse(
                predicted_regional - target_regional, regional_volume[selected]
            )
            native_total_rmse = phase_rmse(predicted_native - native, volume)
            _, _, limited, _, _ = phase_reconstruction(
                predicted_regional,
                regional_centroid[phase],
                centroid,
                volume,
                phase_parent_local[phase],
                phase_neighbours[phase],
                phase_operators[phase],
            )
            limited_native[phase] = limited
            limited_native_total_rmse = phase_rmse(limited - native, volume)
            identity_difference = (
                native_total_rmse**2
                - representation_rmse**2
                - regional_model_rmse**2
            )
            row.update(
                {
                    f"{phase}_representation_rmse_K": representation_rmse,
                    f"{phase}_regional_model_rmse_K": regional_model_rmse,
                    f"{phase}_native_total_rmse_K": native_total_rmse,
                    f"{phase}_limited_native_total_rmse_K": limited_native_total_rmse,
                    f"{phase}_squared_error_identity_difference_K2": identity_difference,
                }
            )

        native_hot_index = int(np.argmax(native_solid))
        predicted_hot_local = int(np.argmax(predicted_state[solid_global, 4]))
        predicted_hot_global = int(solid_global[predicted_hot_local])
        predicted_hot_cells = np.flatnonzero(solid_parent == predicted_hot_global)
        hotspot_distance_m = float(
            np.min(
                np.linalg.norm(
                    solid_centroid[predicted_hot_cells]
                    - solid_centroid[native_hot_index],
                    axis=1,
                )
            )
        )
        row.update(
            {
                "native_solid_max_temperature_K": float(native_solid[native_hot_index]),
                "predicted_solid_max_temperature_K": float(
                    predicted_state[predicted_hot_global, 4]
                ),
                "predicted_solid_max_temperature_error_K": float(
                    predicted_state[predicted_hot_global, 4]
                    - native_solid[native_hot_index]
                ),
                "predicted_hotspot_nearest_cell_distance_dp": hotspot_distance_m / diameter_m,
                "limited_predicted_solid_max_temperature_K": float(
                    np.max(limited_native["solid"])
                ),
                "limited_predicted_solid_max_temperature_error_K": float(
                    np.max(limited_native["solid"]) - native_solid[native_hot_index]
                ),
                "limited_predicted_hotspot_distance_dp": float(
                    np.linalg.norm(
                        solid_centroid[int(np.argmax(limited_native["solid"]))]
                        - solid_centroid[native_hot_index]
                    )
                    / diameter_m
                ),
            }
        )
        rows.append(row)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "native_cell_prediction_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [
        key
        for key in rows[0]
        if key.endswith(("_rmse_K", "_difference_K2", "_error_K", "_distance_dp"))
    ]
    aggregate: dict[str, dict[str, float | str]] = {}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows])
        maximum_index = int(np.argmax(np.abs(values)))
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "maximum_absolute": float(np.abs(values[maximum_index])),
            "maximum_absolute_condition_id": str(rows[maximum_index]["condition_id"]),
        }
    summary = {
        "status": "native_cell_prediction_metrics_ready",
        "scope": (
            "piecewise-constant and parameter-free limited-affine reconstruction "
            "of regional predictions on native OpenFOAM cells"
        ),
        "split_name": args.split_name,
        "prediction_file": str(args.regional_predictions.resolve()),
        "case_count": len(rows),
        "metrics": aggregate,
        "squared_error_identity": (
            "piecewise_constant_native_total_rmse^2 = representation_rmse^2 + "
            "regional_model_rmse^2"
        ),
        "particle_diameter_parameter_id": "P048",
        "native_reconstruction": {
            "method": "parameter-free Barth--Jespersen-style limited affine reconstruction",
            "doi": "10.2514/6.1989-366",
        },
        "new_physical_parameters": [],
        "result_csv": csv_path.name,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 区域预测还原到OpenFOAM原始单元后的温度误差",
        "",
        f"本结果包含{len(rows)}个{args.split_name}工况。总原始单元误差平方被分成区域化误差平方和网络区域预测误差平方。",
        "",
        f"- 流体：区域化RMSE平均{aggregate['fluid_representation_rmse_K']['mean']:.4g} K，网络区域RMSE平均{aggregate['fluid_regional_model_rmse_K']['mean']:.4g} K，还原到原始单元后的总RMSE平均{aggregate['fluid_native_total_rmse_K']['mean']:.4g} K。",
        f"- 颗粒：区域化RMSE平均{aggregate['solid_representation_rmse_K']['mean']:.4g} K，网络区域RMSE平均{aggregate['solid_regional_model_rmse_K']['mean']:.4g} K，还原到原始单元后的总RMSE平均{aggregate['solid_native_total_rmse_K']['mean']:.4g} K。",
        f"- 单调梯度重构后：流体原始单元总RMSE平均{aggregate['fluid_limited_native_total_rmse_K']['mean']:.4g} K，颗粒原始单元总RMSE平均{aggregate['solid_limited_native_total_rmse_K']['mean']:.4g} K。",
        f"- 误差平方分解的最大数值余差：流体{aggregate['fluid_squared_error_identity_difference_K2']['maximum_absolute']:.3g} K2，颗粒{aggregate['solid_squared_error_identity_difference_K2']['maximum_absolute']:.3g} K2。",
        "",
        "这三个误差必须分开报告。网络在区域节点上拟合得很好，并不自动表示原始单元温度场同样精确。",
    ]
    (output / "原始单元温度误差_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
