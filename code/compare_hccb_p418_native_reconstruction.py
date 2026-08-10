#!/usr/bin/env python3
"""Compare constant and parameter-free affine native-cell reconstructions."""

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


ROOT = Path(__file__).resolve().parents[1]


def unique_neighbour_lists(
    node_count: int, owner: np.ndarray, neighbour: np.ndarray
) -> list[np.ndarray]:
    pairs = np.column_stack((owner, neighbour)).astype(np.int64, copy=False)
    pairs.sort(axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for left, right in pairs:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    return [np.asarray(sorted(values), dtype=np.int64) for values in adjacency]


def reconstruction_operators(
    centroids: np.ndarray, neighbours: list[np.ndarray]
) -> tuple[list[np.ndarray | None], np.ndarray]:
    operators: list[np.ndarray | None] = []
    ranks = np.zeros(len(centroids), dtype=np.int8)
    for node, adjacent in enumerate(neighbours):
        if len(adjacent) == 0:
            operators.append(None)
            continue
        displacement = centroids[adjacent] - centroids[node]
        ranks[node] = int(np.linalg.matrix_rank(displacement))
        operators.append(np.linalg.pinv(displacement))
    return operators, ranks


def estimate_gradients(
    regional_temperature: np.ndarray,
    neighbours: list[np.ndarray],
    operators: list[np.ndarray | None],
) -> np.ndarray:
    gradient = np.zeros((len(regional_temperature), 3), dtype=np.float64)
    for node, (adjacent, operator) in enumerate(zip(neighbours, operators)):
        if operator is None:
            continue
        gradient[node] = operator @ (
            regional_temperature[adjacent] - regional_temperature[node]
        )
    return gradient


def global_to_local(parent_global: np.ndarray, selected_global: np.ndarray) -> np.ndarray:
    lookup = np.full(int(max(parent_global.max(), selected_global.max())) + 1, -1, dtype=np.int64)
    lookup[selected_global] = np.arange(len(selected_global), dtype=np.int64)
    local = lookup[parent_global]
    if np.any(local < 0):
        raise ValueError("native cells map outside the selected phase regions")
    return local


def phase_reconstruction(
    regional_temperature: np.ndarray,
    regional_centroid: np.ndarray,
    native_centroid: np.ndarray,
    native_volume: np.ndarray,
    parent_local: np.ndarray,
    neighbours: list[np.ndarray],
    operators: list[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    constant = regional_temperature[parent_local]
    gradient = estimate_gradients(regional_temperature, neighbours, operators)
    affine = constant + np.einsum(
        "ij,ij->i",
        gradient[parent_local],
        native_centroid - regional_centroid[parent_local],
    )
    # The regional centroid is a volume centroid. Remove only round-off drift so
    # the affine correction retains the exact regional volume mean.
    correction = affine - constant
    denominator = np.bincount(parent_local, weights=native_volume, minlength=len(regional_temperature))
    numerator = np.bincount(
        parent_local, weights=correction * native_volume, minlength=len(regional_temperature)
    )
    nonempty = denominator > 0.0
    mean_correction = np.zeros_like(regional_temperature)
    mean_correction[nonempty] = numerator[nonempty] / denominator[nonempty]
    affine -= mean_correction[parent_local]
    correction = affine - constant

    # Parameter-free Barth--Jespersen-style scalar limiting: native values in
    # each region are constrained to the extrema of that region and its unique
    # adjacent regional means. No empirical limiter coefficient is introduced.
    lower = regional_temperature.copy()
    upper = regional_temperature.copy()
    for node, adjacent in enumerate(neighbours):
        if len(adjacent):
            lower[node] = min(lower[node], float(np.min(regional_temperature[adjacent])))
            upper[node] = max(upper[node], float(np.max(regional_temperature[adjacent])))
    positive = np.full(len(regional_temperature), -np.inf)
    negative = np.full(len(regional_temperature), np.inf)
    np.maximum.at(positive, parent_local, correction)
    np.minimum.at(negative, parent_local, correction)
    alpha = np.ones(len(regional_temperature), dtype=np.float64)
    has_positive = positive > 0.0
    alpha[has_positive] = np.minimum(
        alpha[has_positive],
        (upper[has_positive] - regional_temperature[has_positive]) / positive[has_positive],
    )
    has_negative = negative < 0.0
    alpha[has_negative] = np.minimum(
        alpha[has_negative],
        (lower[has_negative] - regional_temperature[has_negative]) / negative[has_negative],
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    limited = constant + alpha[parent_local] * correction
    return constant, affine, limited, gradient, alpha


def nearest_hotspot_distance(
    native_centroid: np.ndarray, native_hot_index: int, reconstructed: np.ndarray
) -> float:
    reconstructed_hot_index = int(np.argmax(reconstructed))
    return float(
        np.linalg.norm(
            native_centroid[reconstructed_hot_index] - native_centroid[native_hot_index]
        )
    )


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
    diameter_m = particle_diameter_m(args.parameter_source.resolve())

    with np.load(dataset_root / dataset["shared_topology_file"], allow_pickle=False) as topology:
        native = {
            phase: {
                "centroid": topology[f"{phase}_cell_centroid_m"].astype(np.float64),
                "volume": topology[f"{phase}_cell_volume_m3"].astype(np.float64),
            }
            for phase in ("fluid", "solid")
        }
    with np.load(args.subface_geometry.resolve(), allow_pickle=False) as geometry:
        parent_global = geometry["fine_to_regional_global"].astype(np.int64)
        selected_global = {
            phase: geometry[f"{phase}_global_region"].astype(np.int64)
            for phase in ("fluid", "solid")
        }
        regional_centroid = {
            phase: geometry[f"{phase}_cell_centroid_m"].astype(np.float64)
            for phase in ("fluid", "solid")
        }
        owner = {
            phase: geometry[f"{phase}_internal_subface_owner"].astype(np.int64)
            for phase in ("fluid", "solid")
        }
        neighbour = {
            phase: geometry[f"{phase}_internal_subface_neighbour"].astype(np.int64)
            for phase in ("fluid", "solid")
        }
    with np.load(args.regional_state_targets.resolve(), allow_pickle=False) as targets:
        target_ids = targets["condition_id"].astype(str)
        states = targets["state_physical"].astype(np.float64)

    fluid_count = len(native["fluid"]["volume"])
    phase_parent_global = {
        "fluid": parent_global[:fluid_count],
        "solid": parent_global[fluid_count:],
    }
    parent_local = {
        phase: global_to_local(phase_parent_global[phase], selected_global[phase])
        for phase in ("fluid", "solid")
    }
    neighbours_by_phase: dict[str, list[np.ndarray]] = {}
    operators_by_phase: dict[str, list[np.ndarray | None]] = {}
    rank_by_phase: dict[str, np.ndarray] = {}
    for phase in ("fluid", "solid"):
        neighbours_by_phase[phase] = unique_neighbour_lists(
            len(selected_global[phase]), owner[phase], neighbour[phase]
        )
        operators_by_phase[phase], rank_by_phase[phase] = reconstruction_operators(
            regional_centroid[phase], neighbours_by_phase[phase]
        )

    dataset_ids = np.asarray([row["condition_id"] for row in dataset["conditions"]])
    if not np.array_equal(dataset_ids, target_ids):
        raise ValueError("dataset and regional target condition order differ")

    rows: list[dict[str, object]] = []
    for case_index, record in enumerate(dataset["conditions"]):
        with np.load(dataset_root / record["field_file"], allow_pickle=False) as fields:
            native_temperature = {
                phase: fields[f"{phase}_temperature_K"].astype(np.float64)
                for phase in ("fluid", "solid")
            }
        row: dict[str, object] = {
            "condition_id": record["condition_id"],
            "inlet_velocity_m_s": float(record["inlet_velocity_m_s"]),
            "inlet_temperature_K": float(record["inlet_temperature_K"]),
            "solid_heat_source_W_m3": float(record["solid_heat_source_W_m3"]),
        }
        reconstructed: dict[str, dict[str, np.ndarray]] = {}
        for phase in ("fluid", "solid"):
            regional_temperature = states[case_index, selected_global[phase], 4]
            constant, affine, limited, gradient, alpha = phase_reconstruction(
                regional_temperature,
                regional_centroid[phase],
                native[phase]["centroid"],
                native[phase]["volume"],
                parent_local[phase],
                neighbours_by_phase[phase],
                operators_by_phase[phase],
            )
            reconstructed[phase] = {
                "constant": constant,
                "affine": affine,
                "limited": limited,
            }
            for method, values in (
                ("constant", constant),
                ("affine", affine),
                ("limited", limited),
            ):
                metrics = error_metrics(
                    native_temperature[phase], values, native[phase]["volume"]
                )
                row.update(
                    {
                        f"{phase}_{method}_{key}": value
                        for key, value in metrics.items()
                    }
                )
            constant_mse = row[f"{phase}_constant_volume_weighted_rmse_K"] ** 2
            affine_mse = row[f"{phase}_affine_volume_weighted_rmse_K"] ** 2
            row[f"{phase}_affine_variance_reduction_percent"] = (
                100.0 * (1.0 - affine_mse / constant_mse)
                if constant_mse > 0.0
                else 0.0
            )
            row[f"{phase}_affine_max_gradient_K_m"] = float(
                np.max(np.linalg.norm(gradient, axis=1))
            )
            limited_mse = row[f"{phase}_limited_volume_weighted_rmse_K"] ** 2
            row[f"{phase}_limited_variance_reduction_percent"] = (
                100.0 * (1.0 - limited_mse / constant_mse)
                if constant_mse > 0.0
                else 0.0
            )
            row[f"{phase}_limited_region_fraction"] = float(np.mean(alpha < 1.0))

        native_solid = native_temperature["solid"]
        native_hot_index = int(np.argmax(native_solid))
        row["native_solid_max_temperature_K"] = float(native_solid[native_hot_index])
        for method in ("constant", "affine", "limited"):
            solid_prediction = reconstructed["solid"][method]
            row[f"solid_{method}_max_temperature_error_K"] = float(
                np.max(solid_prediction) - native_solid[native_hot_index]
            )
            row[f"solid_{method}_hotspot_distance_dp"] = (
                nearest_hotspot_distance(
                    native["solid"]["centroid"], native_hot_index, solid_prediction
                )
                / diameter_m
            )
        rows.append(row)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "native_reconstruction_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def aggregate(key: str) -> dict[str, float | str]:
        values = np.asarray([float(row[key]) for row in rows])
        maximum_index = int(np.argmax(np.abs(values)))
        return {
            "mean": float(np.mean(values)),
            "maximum_absolute": float(np.abs(values[maximum_index])),
            "maximum_absolute_condition_id": str(rows[maximum_index]["condition_id"]),
        }

    metric_keys = []
    for phase in ("fluid", "solid"):
        metric_keys.extend(
            (
                f"{phase}_constant_volume_weighted_rmse_K",
                f"{phase}_affine_volume_weighted_rmse_K",
                f"{phase}_limited_volume_weighted_rmse_K",
                f"{phase}_affine_variance_reduction_percent",
                f"{phase}_limited_variance_reduction_percent",
                f"{phase}_affine_volume_weighted_mean_error_K",
                f"{phase}_limited_volume_weighted_mean_error_K",
                f"{phase}_limited_region_fraction",
            )
        )
    metric_keys.extend(
        (
            "solid_constant_max_temperature_error_K",
            "solid_affine_max_temperature_error_K",
            "solid_limited_max_temperature_error_K",
            "solid_constant_hotspot_distance_dp",
            "solid_affine_hotspot_distance_dp",
            "solid_limited_hotspot_distance_dp",
        )
    )
    summary = {
        "status": "native_reconstruction_comparison_ready",
        "scope": "parameter-free geometric reconstruction of native-cell temperature",
        "case_count": len(rows),
        "method": (
            "piecewise constant regional mean versus unweighted least-squares "
            "affine reconstruction and parameter-free Barth--Jespersen-style "
            "neighbour-extrema limiting"
        ),
        "regional_geometry": {
            phase: {
                "nodes": len(selected_global[phase]),
                "full_rank_fraction": float(np.mean(rank_by_phase[phase] == 3)),
                "isolated_node_fraction": float(
                    np.mean([len(values) == 0 for values in neighbours_by_phase[phase]])
                ),
            }
            for phase in ("fluid", "solid")
        },
        "metrics": {key: aggregate(key) for key in metric_keys},
        "particle_diameter_parameter_id": "P048",
        "numerical_method_source": {
            "authors": "Barth and Jespersen",
            "year": 1989,
            "doi": "10.2514/6.1989-366",
        },
        "new_physical_parameters": [],
        "result_csv": csv_path.name,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 原始网格温度的常值与线性梯度重构比较",
        "",
        f"本结果使用{len(rows)}个真实OpenFOAM工况，不增加材料物性或运行参数。",
        "",
        f"- 流体温度RMSE：区域常值{summary['metrics']['fluid_constant_volume_weighted_rmse_K']['mean']:.4g} K，未限制线性梯度{summary['metrics']['fluid_affine_volume_weighted_rmse_K']['mean']:.4g} K，单调限制后{summary['metrics']['fluid_limited_volume_weighted_rmse_K']['mean']:.4g} K。",
        f"- 颗粒温度RMSE：区域常值{summary['metrics']['solid_constant_volume_weighted_rmse_K']['mean']:.4g} K，未限制线性梯度{summary['metrics']['solid_affine_volume_weighted_rmse_K']['mean']:.4g} K，单调限制后{summary['metrics']['solid_limited_volume_weighted_rmse_K']['mean']:.4g} K。",
        f"- 颗粒热点距离：区域常值{summary['metrics']['solid_constant_hotspot_distance_dp']['mean']:.4g} d_p，未限制线性梯度{summary['metrics']['solid_affine_hotspot_distance_dp']['mean']:.4g} d_p，单调限制后{summary['metrics']['solid_limited_hotspot_distance_dp']['mean']:.4g} d_p。",
        "",
        "线性重构只使用相邻区域温度和网格坐标，单调限制采用Barth--Jespersen邻域极值思想且不含经验系数。后续神经网络或扩散分支必须与这个确定性方法比较，而不能只与区域常值比较。",
    ]
    (output / "原始网格温度重构比较_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
