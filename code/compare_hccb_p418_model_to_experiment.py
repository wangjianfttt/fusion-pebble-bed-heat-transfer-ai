#!/usr/bin/env python3
"""Compare P418 regional fields with measurements at the actual sensor locations.

The script never creates measurements, sensor coordinates or uncertainty values.
It only joins a filled experimental package to an existing OpenFOAM or learned
regional state file. Point measurements use the sensor response approximation
declared in sensor_layout.csv and report that mapping distance. The current
program does not silently replace an explicit probe-body/contact/dynamic model
with a nearest-node value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


OUTPUT_COLUMNS = [
    "experiment_id",
    "model_condition_id",
    "sensor_id",
    "quantity",
    "model_observable",
    "sensor_response_model",
    "time_s",
    "measurement_value",
    "unit",
    "standard_uncertainty",
    "model_name",
    "predicted_value",
    "residual_model_minus_measurement",
    "residual_over_standard_uncertainty",
    "extraction_method",
    "model_x_m",
    "model_y_m",
    "model_z_m",
    "sensor_to_model_distance_m",
    "status",
    "notes",
]


EXPECTED_RESPONSE_MODELS = {
    "fluid_temperature": {"nearest_regional_phase_temperature", "explicit_sensor_body"},
    "solid_temperature": {"nearest_regional_phase_temperature", "explicit_sensor_body"},
    "fluid_absolute_pressure": {"nearest_regional_fluid_pressure", "explicit_sensor_body"},
    "outlet_fluid_temperature": {"direct_integral_or_boundary_quantity"},
    "solid_maximum_temperature": {"derived_model_quantity"},
    "pressure_drop": {"direct_integral_or_boundary_quantity"},
    "inlet_mass_flow_rate": {"direct_integral_or_boundary_quantity"},
    "outlet_mass_flow_rate": {"direct_integral_or_boundary_quantity"},
    "inlet_superficial_velocity": {"direct_integral_or_boundary_quantity"},
    "solid_volumetric_heating_power": {"direct_integral_or_boundary_quantity"},
    "cooling_wall_heat_into_fluid": {"direct_integral_or_boundary_quantity"},
    "fluid_cooling_wall_heat_flux_outward": {
        "nearest_regional_boundary_face",
        "explicit_sensor_body",
    },
    "solid_cooling_wall_heat_flux_outward": {
        "nearest_regional_boundary_face",
        "explicit_sensor_body",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def require_same_ids(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if [str(value) for value in left] != [str(value) for value in right]:
        raise ValueError(f"condition identifiers differ between state and {label}")


def field_scales(statistics_path: Path, split_name: str) -> dict[str, np.ndarray]:
    payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    try:
        targets = payload["splits"][split_name]["targets"]
    except KeyError as error:
        raise ValueError(f"training statistics do not contain split {split_name!r}") from error
    return {
        "velocity_mean": np.asarray(targets["fluid_velocity_m_s"]["mean"], dtype=float),
        "velocity_std": np.asarray(
            targets["fluid_velocity_m_s"]["standard_deviation"], dtype=float
        ),
        "pressure_mean": np.asarray(
            targets["fluid_gauge_pressure_Pa"]["mean"], dtype=float
        ),
        "pressure_std": np.asarray(
            targets["fluid_gauge_pressure_Pa"]["standard_deviation"], dtype=float
        ),
        "fluid_temperature_mean": np.asarray(
            targets["fluid_temperature_K"]["mean"], dtype=float
        ),
        "fluid_temperature_std": np.asarray(
            targets["fluid_temperature_K"]["standard_deviation"], dtype=float
        ),
        "solid_temperature_mean": np.asarray(
            targets["solid_temperature_K"]["mean"], dtype=float
        ),
        "solid_temperature_std": np.asarray(
            targets["solid_temperature_K"]["standard_deviation"], dtype=float
        ),
    }


def unnormalize_state(
    normalized: np.ndarray,
    condition: np.ndarray,
    node_type: np.ndarray,
    scales: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.zeros_like(normalized, dtype=np.float64)
    fluid = node_type == 0
    solid = node_type == 1
    output[fluid, :3] = (
        normalized[fluid, :3] * scales["velocity_std"] + scales["velocity_mean"]
    )
    output[fluid, 3] = (
        normalized[fluid, 3] * float(scales["pressure_std"][0])
        + float(scales["pressure_mean"][0])
        + float(condition[3])
    )
    output[fluid, 4] = (
        normalized[fluid, 4] * float(scales["fluid_temperature_std"][0])
        + float(scales["fluid_temperature_mean"][0])
    )
    output[solid, 4] = (
        normalized[solid, 4] * float(scales["solid_temperature_std"][0])
        + float(scales["solid_temperature_mean"][0])
    )
    return output


def load_states(args: argparse.Namespace) -> dict[str, Any]:
    state = load_npz(args.state_file)
    condition_ids = np.asarray(state["condition_id"]).astype(str)
    node_type = np.asarray(state["node_type"], dtype=np.int8)
    if "state_physical" in state:
        states = np.asarray(state["state_physical"], dtype=np.float64)
        conditions = np.asarray(state["condition_physical"], dtype=np.float64)
        source_kind = "openfoam_reference"
    elif "predicted_state_physical" in state:
        states = np.asarray(state["predicted_state_physical"], dtype=np.float64)
        if args.reference_state_targets is None:
            raise ValueError(
                "--reference-state-targets is required for predicted physical states"
            )
        reference = load_npz(args.reference_state_targets)
        reference_ids = np.asarray(reference["condition_id"]).astype(str)
        condition_by_id = {
            str(identifier): np.asarray(value, dtype=float)
            for identifier, value in zip(reference_ids, reference["condition_physical"])
        }
        conditions = np.stack([condition_by_id[identifier] for identifier in condition_ids])
        source_kind = "learned_prediction"
    elif "baseline_state_normalized" in state:
        if args.reference_state_targets is None or args.training_statistics is None:
            raise ValueError(
                "normalized predictions require --reference-state-targets and "
                "--training-statistics"
            )
        reference = load_npz(args.reference_state_targets)
        reference_ids = np.asarray(reference["condition_id"]).astype(str)
        condition_by_id = {
            str(identifier): np.asarray(value, dtype=float)
            for identifier, value in zip(reference_ids, reference["condition_physical"])
        }
        conditions = np.stack([condition_by_id[identifier] for identifier in condition_ids])
        scales = field_scales(args.training_statistics, args.split_name)
        states = np.stack(
            [
                unnormalize_state(value, condition, node_type, scales)
                for value, condition in zip(
                    state["baseline_state_normalized"], conditions
                )
            ]
        )
        source_kind = "learned_prediction"
    else:
        raise ValueError(
            "state file must contain state_physical, predicted_state_physical, "
            "or baseline_state_normalized"
        )
    if states.shape[:2] != (len(condition_ids), len(node_type)) or states.shape[2] != 5:
        raise ValueError("state array must have [condition,node,5] shape")
    return {
        "condition_ids": condition_ids,
        "conditions": conditions,
        "states": states,
        "node_type": node_type,
        "source_kind": source_kind,
    }


def load_engineering_predictions(path: Optional[Path]) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, float]] = {}
    for evaluation in payload.get("evaluations", {}).values():
        for case in evaluation.get("cases", []):
            values = case.get("predicted_engineering", {})
            if values:
                result[str(case["condition_id"])] = {
                    str(key): float(value) for key, value in values.items()
                }
    return result


def load_temporal_states(
    path: Optional[Path], source: str
) -> Optional[dict[str, np.ndarray]]:
    if path is None:
        return None
    loaded = load_npz(path)
    sequence_ids = np.asarray(loaded["sequence_id"]).astype(str)
    time_s = np.asarray(loaded["time_s"], dtype=float)
    node_type = np.asarray(loaded["node_type"], dtype=np.int8)
    if "temperature_physical" in loaded:
        temperature = np.asarray(loaded["temperature_physical"], dtype=float)
    else:
        key = {
            "prediction": "baseline_temperature_normalized",
            "reference": "target_temperature_normalized",
        }[source]
        if key not in loaded:
            raise ValueError(f"temporal state file lacks {key}")
        normalized = np.asarray(loaded[key], dtype=float)
        if normalized.shape[-1:] == (1,):
            normalized = normalized[..., 0]
        mean = np.asarray(loaded["temperature_mean_K_by_node_type"], dtype=float)
        std = np.asarray(loaded["temperature_std_K_by_node_type"], dtype=float)
        temperature = normalized * std[node_type][None, None, :] + mean[node_type][
            None, None, :
        ]
    if temperature.shape != (len(sequence_ids), time_s.shape[1], len(node_type)):
        raise ValueError("temporal temperature array must have [sequence,time,node] shape")
    return {
        "sequence_id": sequence_ids,
        "time_s": time_s,
        "temperature_K": temperature,
        "node_type": node_type,
    }


def nearest_value(
    state: np.ndarray,
    centroid: np.ndarray,
    node_type: np.ndarray,
    coordinate: np.ndarray,
    phase: int,
    channel: int,
) -> tuple[float, np.ndarray, float]:
    candidates = np.flatnonzero(node_type == phase)
    if not len(candidates):
        raise ValueError("regional state contains no nodes for the requested phase")
    distance = np.linalg.norm(centroid[candidates] - coordinate[None, :], axis=1)
    local = int(np.argmin(distance))
    node = int(candidates[local])
    return float(state[node, channel]), centroid[node], float(distance[local])


def condition_lookup(data: dict[str, Any], condition_id: str) -> tuple[np.ndarray, np.ndarray]:
    matches = np.flatnonzero(data["condition_ids"] == condition_id)
    if len(matches) != 1:
        raise KeyError(condition_id)
    index = int(matches[0])
    return data["states"][index], data["conditions"][index]


def condition_index(array: dict[str, np.ndarray], condition_id: str) -> int:
    matches = np.flatnonzero(np.asarray(array["condition_id"]).astype(str) == condition_id)
    if len(matches) != 1:
        raise KeyError(condition_id)
    return int(matches[0])


def interpolate_inside(time_s: np.ndarray, values: np.ndarray, query_s: float) -> float:
    finite = np.isfinite(time_s) & np.isfinite(values)
    time = np.asarray(time_s[finite], dtype=float)
    value = np.asarray(values[finite], dtype=float)
    if not len(time):
        raise KeyError("temporal model series contains no finite values")
    order = np.argsort(time)
    time = time[order]
    value = value[order]
    if query_s < time[0] or query_s > time[-1]:
        raise KeyError("experimental time lies outside the model time range")
    return float(np.interp(query_s, time, value))


def transient_value(
    temporal: dict[str, np.ndarray],
    sequence_id: str,
    observable: str,
    query_s: float,
    centroid: np.ndarray,
    coordinate: Optional[np.ndarray],
    mass: Optional[dict[str, np.ndarray]],
) -> tuple[float, str, Optional[np.ndarray], Optional[float]]:
    matches = np.flatnonzero(temporal["sequence_id"] == sequence_id)
    if len(matches) != 1:
        raise KeyError(sequence_id)
    index = int(matches[0])
    time_s = temporal["time_s"][index]
    temperature = temporal["temperature_K"][index]
    node_type = temporal["node_type"]
    if observable in {"fluid_temperature", "solid_temperature"}:
        if coordinate is None:
            raise KeyError("transient point-temperature comparison requires coordinates")
        phase = 0 if observable == "fluid_temperature" else 1
        candidates = np.flatnonzero(node_type == phase)
        distance = np.linalg.norm(centroid[candidates] - coordinate[None, :], axis=1)
        local = int(np.argmin(distance))
        node = int(candidates[local])
        return (
            interpolate_inside(time_s, temperature[:, node], query_s),
            (
                "linear interpolation in model time at nearest same-phase regional "
                "node; probe body, contact, local packing disturbance and response "
                "time are not resolved"
            ),
            centroid[node],
            float(distance[local]),
        )
    if observable == "solid_maximum_temperature":
        values = np.max(temperature[:, node_type == 1], axis=1)
        return (
            interpolate_inside(time_s, values, query_s),
            "linear interpolation of maximum regional solid temperature",
            None,
            None,
        )
    if observable == "outlet_fluid_temperature":
        if mass is None:
            raise KeyError("mass-target boundary geometry is required")
        owner = np.asarray(mass["boundary_owner"], dtype=np.int64)
        patch = np.asarray(mass["boundary_patch"], dtype=np.int64)
        area = np.asarray(mass["boundary_face_area_m2"], dtype=float)
        fluid_global = np.asarray(mass["fluid_global_region"], dtype=np.int64)
        outlet = patch == 1
        values = np.average(
            temperature[:, fluid_global[owner[outlet]]],
            axis=1,
            weights=area[outlet],
        )
        return (
            interpolate_inside(time_s, values, query_s),
            "linear interpolation of outlet-area weighted regional fluid temperature",
            None,
            None,
        )
    raise KeyError(f"transient model observable {observable!r} is not available")


def integrated_value(
    observable: str,
    condition_id: str,
    state: np.ndarray,
    condition: np.ndarray,
    node_type: np.ndarray,
    mass: Optional[dict[str, np.ndarray]],
    energy: Optional[dict[str, np.ndarray]],
    source_kind: str,
    engineering: dict[str, dict[str, float]],
    coordinate: Optional[np.ndarray],
) -> tuple[float, str, Optional[np.ndarray], Optional[float]]:
    summary_key = {
        "outlet_fluid_temperature": "outlet_temperature_K",
        "solid_maximum_temperature": "solid_maximum_temperature_K",
        "pressure_drop": "pressure_drop_Pa",
        "cooling_wall_heat_into_fluid": "cooling_wall_heat_into_fluid_W",
    }.get(observable)
    if summary_key and summary_key in engineering.get(condition_id, {}):
        return (
            engineering[condition_id][summary_key],
            "saved model engineering quantity",
            None,
            None,
        )

    if observable == "solid_maximum_temperature":
        return float(np.max(state[node_type == 1, 4])), "maximum regional solid temperature", None, None
    if observable == "inlet_superficial_velocity":
        return float(condition[0]), "declared inlet condition", None, None

    if observable in {
        "outlet_fluid_temperature",
        "pressure_drop",
        "inlet_mass_flow_rate",
        "outlet_mass_flow_rate",
    }:
        if mass is None:
            raise KeyError("mass targets are required")
        index = condition_index(mass, condition_id)
        owner_local = np.asarray(mass["boundary_owner"], dtype=np.int64)
        patch = np.asarray(mass["boundary_patch"], dtype=np.int64)
        area = np.asarray(mass["boundary_face_area_m2"], dtype=float)
        fluid_global = np.asarray(mass["fluid_global_region"], dtype=np.int64)
        inlet = patch == 0
        outlet = patch == 1
        if observable == "outlet_fluid_temperature":
            value = np.average(state[fluid_global[owner_local[outlet]], 4], weights=area[outlet])
            return float(value), "outlet-area weighted regional fluid temperature", None, None
        if observable == "pressure_drop":
            inlet_p = np.average(state[fluid_global[owner_local[inlet]], 3], weights=area[inlet])
            outlet_p = np.average(state[fluid_global[owner_local[outlet]], 3], weights=area[outlet])
            return float(inlet_p - outlet_p), "inlet minus outlet area-weighted pressure", None, None
        if source_kind != "openfoam_reference":
            raise KeyError("learned prediction file does not contain mass-flow predictions")
        flux = np.asarray(mass["boundary_mass_flow_kg_s"], dtype=float)[index]
        selected = inlet if observable == "inlet_mass_flow_rate" else outlet
        return abs(float(np.sum(flux[selected]))), "integrated OpenFOAM boundary mass flow", None, None

    if observable in {
        "solid_volumetric_heating_power",
        "cooling_wall_heat_into_fluid",
        "fluid_cooling_wall_heat_flux_outward",
        "solid_cooling_wall_heat_flux_outward",
    }:
        if energy is None or source_kind != "openfoam_reference":
            raise KeyError("OpenFOAM energy targets are required")
        index = condition_index(energy, condition_id)
        if observable == "solid_volumetric_heating_power":
            value = np.sum(np.asarray(energy["node_source_power_W"], dtype=float)[index])
            return float(value), "integrated OpenFOAM volumetric source power", None, None
        names = [str(value) for value in energy["boundary_kind_name"]]
        kind_name = {
            "cooling_wall_heat_into_fluid": "fluid:coolingWall",
            "fluid_cooling_wall_heat_flux_outward": "fluid:coolingWall",
            "solid_cooling_wall_heat_flux_outward": "solid:coolingWall",
        }[observable]
        selected = np.asarray(energy["boundary_kind"], dtype=int) == names.index(kind_name)
        flow = np.asarray(energy["boundary_energy_flow_W"], dtype=float)[index]
        if observable == "cooling_wall_heat_into_fluid":
            return float(-np.sum(flow[selected])), "negative outward fluid cooling-wall heat flow", None, None
        if coordinate is None:
            raise KeyError("wall heat-flux comparison requires sensor coordinates")
        face_centroid = np.asarray(energy["boundary_face_centroid_m"], dtype=float)[selected]
        face_area = np.asarray(energy["boundary_face_area_m2"], dtype=float)[selected]
        face_flow = flow[selected]
        distances = np.linalg.norm(face_centroid - coordinate[None, :], axis=1)
        nearest = int(np.argmin(distances))
        value = face_flow[nearest] / face_area[nearest]
        return float(value), "nearest regional cooling-wall outward heat flux", face_centroid[nearest], float(distances[nearest])

    raise KeyError(f"model observable {observable!r} is not available")


def aggregate(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, Union[float, int]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["status"] == "compared":
            groups.setdefault(row["quantity"], []).append(row)
    result: dict[str, dict[str, Union[float, int]]] = {}
    for quantity, members in sorted(groups.items()):
        residual = np.asarray(
            [float(row["residual_model_minus_measurement"]) for row in members]
        )
        normalized = [
            float(row["residual_over_standard_uncertainty"])
            for row in members
            if row["residual_over_standard_uncertainty"]
        ]
        result[quantity] = {
            "count": len(members),
            "mean_absolute_error": float(np.mean(np.abs(residual))),
            "root_mean_square_error": float(np.sqrt(np.mean(residual**2))),
            "mean_absolute_residual_over_standard_uncertainty": (
                float(np.mean(np.abs(normalized))) if normalized else math.nan
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--regional-level", type=int, default=5)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--reference-state-targets", type=Path)
    parser.add_argument("--training-statistics", type=Path)
    parser.add_argument("--split-name", default="completed_smoke")
    parser.add_argument("--mass-targets", type=Path)
    parser.add_argument("--energy-targets", type=Path)
    parser.add_argument("--model-summary", type=Path)
    parser.add_argument("--temporal-state-file", type=Path)
    parser.add_argument(
        "--temporal-source", choices=("prediction", "reference"), default="prediction"
    )
    parser.add_argument("--model-name", default="OpenFOAM regional reference")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    conditions = {
        row["experiment_id"]: row
        for row in read_csv(args.data_root / "experiment_conditions.csv")
    }
    sensors = {
        (row["experiment_id"], row["sensor_id"]): row
        for row in read_csv(args.data_root / "sensor_layout.csv")
    }
    measurements: list[tuple[str, dict[str, str]]] = []
    for name in ("steady_measurements.csv", "transient_measurements.csv"):
        measurements.extend((name, row) for row in read_csv(args.data_root / name))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "model_experiment_comparison.csv"
    if not measurements:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS).writeheader()
        summary = {
            "status": "no_experimental_measurements",
            "model_name": args.model_name,
            "measurement_count": 0,
            "compared_count": 0,
            "interpretation_cn": "实验表目前为空，没有生成或补填任何实验值。",
            "new_physical_parameters": [],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    data = load_states(args)
    topology = load_npz(args.regional_topology)
    centroid_key = f"level_{args.regional_level}_centroid_m"
    centroid = np.asarray(topology[centroid_key], dtype=float)
    if len(centroid) != len(data["node_type"]):
        raise ValueError("regional topology and state node counts differ")
    mass = load_npz(args.mass_targets) if args.mass_targets else None
    energy = load_npz(args.energy_targets) if args.energy_targets else None
    engineering = load_engineering_predictions(args.model_summary)
    temporal = load_temporal_states(args.temporal_state_file, args.temporal_source)

    rows: list[dict[str, str]] = []
    for table_name, measurement in measurements:
        experiment_id = measurement["experiment_id"]
        sensor = sensors[(experiment_id, measurement["sensor_id"])]
        experiment = conditions[experiment_id]
        model_condition_id = experiment.get("model_condition_id", "").strip()
        observable = sensor.get("model_observable", "").strip()
        response_model = sensor.get("sensor_response_model", "").strip()
        coordinate_values = [sensor[name].strip() for name in ("x_m", "y_m", "z_m")]
        coordinate = (
            np.asarray([float(value) for value in coordinate_values], dtype=float)
            if all(coordinate_values)
            else None
        )
        row = {
            "experiment_id": experiment_id,
            "model_condition_id": model_condition_id,
            "sensor_id": measurement["sensor_id"],
            "quantity": measurement["quantity"],
            "model_observable": observable,
            "sensor_response_model": response_model,
            "time_s": measurement.get("time_s", ""),
            "measurement_value": measurement["value"],
            "unit": measurement["unit"],
            "standard_uncertainty": measurement["standard_uncertainty"],
            "model_name": args.model_name,
            "predicted_value": "",
            "residual_model_minus_measurement": "",
            "residual_over_standard_uncertainty": "",
            "extraction_method": "",
            "model_x_m": "",
            "model_y_m": "",
            "model_z_m": "",
            "sensor_to_model_distance_m": "",
            "status": "",
            "notes": "",
        }
        if table_name == "transient_measurements.csv" and temporal is None:
            row["status"] = "transient_model_series_not_provided"
            row["notes"] = "稳态状态文件不能与热阶跃时序直接比较。"
        elif not model_condition_id:
            row["status"] = "missing_model_condition_id"
            row["notes"] = "该实验尚未指定同条件数值工况。"
        elif not observable:
            row["status"] = "missing_model_observable"
            row["notes"] = "该传感器尚未指定可直接比较的模型量。"
        elif not response_model:
            row["status"] = "missing_sensor_response_model"
            row["notes"] = "该传感器尚未说明模型量怎样对应真实仪器读数。"
        elif response_model not in EXPECTED_RESPONSE_MODELS.get(observable, set()):
            row["status"] = "incompatible_sensor_response_model"
            row["notes"] = (
                f"{response_model!r}不能用于模型量{observable!r}，"
                "请先通过实验数据结构检查。"
            )
        elif response_model == "explicit_sensor_body":
            row["status"] = "sensor_response_model_not_implemented"
            row["notes"] = (
                "已声明热电偶或压力接口实体、接触及动态响应模型，"
                "但当前比较程序尚未求解该仪器模型，因此不以最近区域值代替。"
            )
        else:
            try:
                if table_name == "transient_measurements.csv":
                    value, method, model_coordinate, distance = transient_value(
                        temporal,
                        model_condition_id,
                        observable,
                        float(measurement["time_s"]),
                        centroid,
                        coordinate,
                        mass,
                    )
                else:
                    state, condition = condition_lookup(data, model_condition_id)
                    if observable == "fluid_temperature":
                        if coordinate is None:
                            raise KeyError("fluid temperature comparison requires sensor coordinates")
                        value, model_coordinate, distance = nearest_value(
                            state, centroid, data["node_type"], coordinate, 0, 4
                        )
                        method = (
                            "nearest regional fluid node; probe body, contact, local "
                            "packing disturbance and response time are not resolved"
                        )
                    elif observable == "solid_temperature":
                        if coordinate is None:
                            raise KeyError("solid temperature comparison requires sensor coordinates")
                        value, model_coordinate, distance = nearest_value(
                            state, centroid, data["node_type"], coordinate, 1, 4
                        )
                        method = (
                            "nearest regional solid node; probe body, contact, local "
                            "packing disturbance and response time are not resolved"
                        )
                    elif observable == "fluid_absolute_pressure":
                        if coordinate is None:
                            raise KeyError("fluid pressure comparison requires sensor coordinates")
                        value, model_coordinate, distance = nearest_value(
                            state, centroid, data["node_type"], coordinate, 0, 3
                        )
                        method = (
                            "nearest regional fluid node; pressure-port volume, tubing "
                            "and transducer dynamics are not resolved"
                        )
                    else:
                        value, method, model_coordinate, distance = integrated_value(
                            observable,
                            model_condition_id,
                            state,
                            condition,
                            data["node_type"],
                            mass,
                            energy,
                            data["source_kind"],
                            engineering,
                            coordinate,
                        )
                measured = float(measurement["value"])
                uncertainty = float(measurement["standard_uncertainty"])
                residual = value - measured
                row["predicted_value"] = f"{value:.12g}"
                row["residual_model_minus_measurement"] = f"{residual:.12g}"
                if uncertainty > 0.0:
                    row["residual_over_standard_uncertainty"] = f"{residual / uncertainty:.12g}"
                row["extraction_method"] = method
                if model_coordinate is not None:
                    row["model_x_m"], row["model_y_m"], row["model_z_m"] = [
                        f"{value:.12g}" for value in model_coordinate
                    ]
                    row["sensor_to_model_distance_m"] = f"{distance:.12g}"
                row["status"] = "compared"
            except KeyError as error:
                row["status"] = "model_value_not_available"
                row["notes"] = str(error)
        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    compared = [row for row in rows if row["status"] == "compared"]
    nearest_phase_approximation_count = sum(
        row["status"] == "compared"
        and row["sensor_response_model"]
        in {"nearest_regional_phase_temperature", "nearest_regional_fluid_pressure"}
        for row in rows
    )
    summary = {
        "status": "model_experiment_comparison_ready",
        "model_name": args.model_name,
        "state_source_kind": data["source_kind"],
        "measurement_count": len(rows),
        "compared_count": len(compared),
        "uncompared_count": len(rows) - len(compared),
        "nearest_phase_approximation_count": nearest_phase_approximation_count,
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "error_summary_by_quantity": aggregate(rows),
        "comparison_file": csv_path.name,
        "method_cn": (
            "只有传感器表明确声明最近同相区域近似时，空间测点才取同相区域"
            "网格最近点并报告距离；该近似不包含探头实体、接触、局部装填扰动"
            "和动态响应。压降、出口温度、流量和热量采用声明的边界或积分量。"
        ),
        "interpretation_cn": (
            "残差为模型值减实验值；只有实验表给出非零标准不确定度时，"
            "才计算残差与该不确定度之比。实验不确定度不会作为精确温度"
            "强行写入计算场。"
        ),
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
