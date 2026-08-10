#!/usr/bin/env python3
"""Validate the long-form experimental data package used by P418 comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_table(path: Path, expected: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected:
            raise ValueError(f"{path.name}: columns differ from the declared schema")
        return list(reader)


def number(value: str, label: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{label}: expected a number, found {value!r}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("parameters/hccb_p418_experimental_data_schema.json"),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    if schema.get("new_physical_parameters") != []:
        raise ValueError("experimental schema must not define new physical parameters")
    tables: dict[str, list[dict[str, str]]] = {}
    for filename, definition in schema["tables"].items():
        rows = read_table(args.data_root / filename, definition["columns"])
        required = definition["required_when_row_exists"]
        for index, row in enumerate(rows, start=2):
            missing = [name for name in required if not row[name].strip()]
            if missing:
                raise ValueError(f"{filename}:{index}: missing {missing}")
        tables[filename] = rows

    conditions = tables["experiment_conditions.csv"]
    sensors = tables["sensor_layout.csv"]
    calibrations = tables["calibration_records.csv"]
    experiment_ids = {row["experiment_id"] for row in conditions}
    calibration_ids = {row["calibration_id"] for row in calibrations}
    if len(experiment_ids) != len(conditions):
        raise ValueError("experiment_id must be unique")
    if len(calibration_ids) != len(calibrations):
        raise ValueError("calibration_id must be unique")
    sensor_keys = {(row["experiment_id"], row["sensor_id"]) for row in sensors}
    if len(sensor_keys) != len(sensors):
        raise ValueError("sensor_id must be unique within each experiment")

    quantity_units: dict[str, list[str]] = schema["quantity_units"]
    model_observables: dict[str, dict[str, str]] = schema.get("model_observables", {})
    for index, row in enumerate(sensors, start=2):
        if row["experiment_id"] not in experiment_ids:
            raise ValueError(f"sensor_layout.csv:{index}: unknown experiment_id")
        if row["quantity"] not in quantity_units:
            raise ValueError(f"sensor_layout.csv:{index}: unknown quantity")
        if row["calibration_id"] not in calibration_ids:
            raise ValueError(f"sensor_layout.csv:{index}: unknown calibration_id")
        coordinates = [row[name].strip() for name in ("x_m", "y_m", "z_m")]
        if any(coordinates) and not all(coordinates):
            raise ValueError(f"sensor_layout.csv:{index}: provide all three coordinates or none")
        for coordinate in coordinates:
            if coordinate:
                number(coordinate, f"sensor_layout.csv:{index} coordinate")
        observable = row.get("model_observable", "").strip()
        response_model = row.get("sensor_response_model", "").strip()
        if observable:
            if observable not in model_observables:
                raise ValueError(
                    f"sensor_layout.csv:{index}: unknown model_observable {observable!r}"
                )
            definition = model_observables[observable]
            if row["quantity"] != definition["quantity"]:
                raise ValueError(
                    f"sensor_layout.csv:{index}: model_observable {observable!r} "
                    f"requires quantity {definition['quantity']!r}"
                )
            if definition["coordinates"] == "required" and not all(coordinates):
                raise ValueError(
                    f"sensor_layout.csv:{index}: model_observable {observable!r} "
                    "requires x_m, y_m and z_m"
                )
            if not response_model:
                raise ValueError(
                    f"sensor_layout.csv:{index}: model_observable {observable!r} "
                    "requires sensor_response_model"
                )
            allowed = definition.get("allowed_sensor_response_models", [])
            if response_model not in allowed:
                raise ValueError(
                    f"sensor_layout.csv:{index}: sensor_response_model "
                    f"{response_model!r} is not allowed for {observable!r}; "
                    f"choose one of {allowed}"
                )
        elif response_model:
            raise ValueError(
                f"sensor_layout.csv:{index}: sensor_response_model requires "
                "model_observable"
            )

    measurement_count = 0
    for filename in ("steady_measurements.csv", "transient_measurements.csv"):
        for index, row in enumerate(tables[filename], start=2):
            key = (row["experiment_id"], row["sensor_id"])
            if key not in sensor_keys:
                raise ValueError(f"{filename}:{index}: unknown experiment/sensor pair")
            if row["quantity"] not in quantity_units:
                raise ValueError(f"{filename}:{index}: unknown quantity")
            sensor_quantity = next(
                sensor["quantity"]
                for sensor in sensors
                if (sensor["experiment_id"], sensor["sensor_id"]) == key
            )
            if row["quantity"] != sensor_quantity:
                raise ValueError(f"{filename}:{index}: quantity differs from sensor layout")
            if row["unit"] not in quantity_units[row["quantity"]]:
                raise ValueError(f"{filename}:{index}: unit differs from schema")
            number(row["value"], f"{filename}:{index} value")
            uncertainty = number(
                row["standard_uncertainty"], f"{filename}:{index} uncertainty"
            )
            if uncertainty < 0:
                raise ValueError(f"{filename}:{index}: uncertainty must be nonnegative")
            if filename == "steady_measurements.csv":
                start = number(row["averaging_start_s"], f"{filename}:{index} start")
                end = number(row["averaging_end_s"], f"{filename}:{index} end")
                if end <= start:
                    raise ValueError(f"{filename}:{index}: averaging end must exceed start")
            else:
                if number(row["time_s"], f"{filename}:{index} time") < 0:
                    raise ValueError(f"{filename}:{index}: time must be nonnegative")
            measurement_count += 1

    payload = {
        "status": (
            "experimental_data_validated"
            if measurement_count
            else "empty_experimental_templates_ready"
        ),
        "experiment_count": len(conditions),
        "sensor_count": len(sensors),
        "calibration_count": len(calibrations),
        "steady_measurement_count": len(tables["steady_measurements.csv"]),
        "transient_measurement_count": len(tables["transient_measurements.csv"]),
        "new_physical_parameters": [],
        "interpretation_cn": (
            "表格结构已就绪，当前未填写实验数值。"
            if not measurement_count
            else (
                "实验编号、传感器、单位、标定、不确定度和传感器响应表示完整。"
                "实验不确定度只用于残差归一化，不会覆盖计算温度场。"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
