#!/usr/bin/env python3
"""Measure the final-window change of one formal P418 steady CHT case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from summarize_hccb_p418_steady_end_time import (
    FIELD_FILES,
    HISTORY_FILES,
    decomposed_field_change,
    read_history,
    value_at,
)


REQUIRED_ENGINEERING_CHANGES = (
    "outlet_temperature_K",
    "solid_maximum_temperature_K",
    "cooling_wall_power_W",
    "outlet_enthalpy_flow_W",
    "pressure_drop_Pa",
    "final_relative_mass_difference",
)


PRESSURE_HISTORIES = {
    "inlet_pressure_Pa": ("fluid", "inletPressure", "surfaceFieldValue.dat"),
    "outlet_pressure_Pa": ("fluid", "outletPressure", "surfaceFieldValue.dat"),
}


def relative_change(start: float, end: float) -> float:
    return abs(end - start) / max(abs(end), float.fromhex("0x1.0p-1022"))


def verified_recorded_summary(
    case: Path,
    marker: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    if marker.get("steady_final_window_status") != "formal_steady_final_window_measured":
        raise ValueError(f"{case.name}: formal steady final-window result is missing")
    path = Path(str(marker.get("steady_final_window_summary", "")))
    if not path.is_absolute():
        path = case / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != marker.get("steady_final_window_summary_sha256"):
        raise ValueError(f"{case.name}: steady final-window checksum differs")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("status") != "formal_steady_final_window_measured":
        raise ValueError(f"{case.name}: invalid steady final-window status")
    if document.get("condition_id") != case.name:
        raise ValueError(f"{case.name}: steady final-window condition identifier differs")
    window = document.get("window_iterations", document.get("window_s"))
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError(f"{case.name}: steady final-window interval is invalid")
    if not math.isclose(float(window[1]), float(marker.get("time")), abs_tol=1.0e-12):
        raise ValueError(f"{case.name}: steady final-window endpoint differs from completed time")
    changes = document.get("engineering_changes")
    if not isinstance(changes, dict):
        raise ValueError(f"{case.name}: steady final-window engineering changes are missing")
    for name in REQUIRED_ENGINEERING_CHANGES:
        if name not in changes:
            raise ValueError(f"{case.name}: steady final-window change is missing {name}")
    values: list[float] = []
    for value in changes.values():
        if isinstance(value, dict):
            values.extend(float(item) for item in value.values())
        else:
            values.append(float(value))
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{case.name}: steady final-window changes contain non-finite values")
    full_field_available = bool(document.get("full_field_available"))
    if bool(marker.get("steady_final_window_full_field_available")) != full_field_available:
        raise ValueError(f"{case.name}: steady final-window field-availability record differs")
    field_rows = document.get("full_field_changes")
    if not isinstance(field_rows, list):
        raise ValueError(f"{case.name}: steady final-window field changes are invalid")
    if full_field_available:
        by_name = {
            str(row.get("field_name")): row for row in field_rows if isinstance(row, dict)
        }
        if set(by_name) != set(FIELD_FILES):
            raise ValueError(f"{case.name}: steady final-window full fields are incomplete")
        field_values = [
            float(by_name[name][key])
            for name in FIELD_FILES
            for key in ("absolute_rms_change", "maximum_absolute_change", "relative_rms_change")
        ]
        if not all(math.isfinite(value) for value in field_values):
            raise ValueError(f"{case.name}: steady final-window field changes are non-finite")
    return path, document


def summarize_case(
    case: Path,
    start_iteration: float,
    end_iteration: float,
    allow_missing_fields: bool,
) -> dict[str, object]:
    histories = {
        name: read_history(case, specification)
        for name, specification in {**HISTORY_FILES, **PRESSURE_HISTORIES}.items()
    }
    start = {name: value_at(history, start_iteration) for name, history in histories.items()}
    end = {name: value_at(history, end_iteration) for name, history in histories.items()}
    engineering: dict[str, object] = {}
    for name in histories:
        engineering[name] = {
            "start": start[name],
            "end": end[name],
            "absolute_change": abs(end[name] - start[name]),
            "relative_change": relative_change(start[name], end[name]),
        }
    pressure_start = start["inlet_pressure_Pa"] - start["outlet_pressure_Pa"]
    pressure_end = end["inlet_pressure_Pa"] - end["outlet_pressure_Pa"]
    engineering["pressure_drop_Pa"] = {
        "start": pressure_start,
        "end": pressure_end,
        "absolute_change": abs(pressure_end - pressure_start),
        "relative_change": relative_change(pressure_start, pressure_end),
    }
    engineering["final_relative_mass_difference"] = abs(
        end["inlet_mass_flow_kg_s"] + end["outlet_mass_flow_kg_s"]
    ) / max(abs(end["inlet_mass_flow_kg_s"]), float.fromhex("0x1.0p-1022"))

    field_rows: list[dict[str, object]] = []
    missing_fields: list[str] = []
    for field_name, (field_path, unit) in FIELD_FILES.items():
        try:
            row = decomposed_field_change(
                case,
                field_path,
                f"{start_iteration:g}",
                f"{end_iteration:g}",
            )
        except FileNotFoundError:
            if not allow_missing_fields:
                raise
            missing_fields.append(field_name)
            continue
        except ValueError as error:
            if (
                not allow_missing_fields
                or not str(error).startswith("inconsistent decomposed field pair:")
            ):
                raise
            # Some compact cloud archives retain the complete final-time field
            # needed for the training sample but not a matching earlier field
            # in every MPI partition.  In the explicitly permissive recovery
            # mode, keep the scalar-history comparison and report the
            # decomposed full-field comparison as unavailable.
            missing_fields.append(field_name)
            continue
        row["field_name"] = field_name
        row["unit"] = unit
        field_rows.append(row)

    numeric_values: list[float] = []
    for value in engineering.values():
        if isinstance(value, dict):
            numeric_values.extend(float(item) for item in value.values())
        else:
            numeric_values.append(float(value))
    for row in field_rows:
        numeric_values.extend(
            float(row[key])
            for key in ("absolute_rms_change", "maximum_absolute_change", "relative_rms_change")
        )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError(f"non-finite final-window change in {case}")

    temperature_rows = [
        row
        for row in field_rows
        if row["field_name"] in {"fluid_temperature", "solid_temperature"}
    ]
    return {
        "status": "formal_steady_final_window_measured",
        "condition_id": case.name,
        "solver_time_semantics": "steady_iteration_index",
        "physical_time_s": None,
        "window_iterations": [start_iteration, end_iteration],
        "window_iteration_count": end_iteration - start_iteration,
        "engineering_changes": engineering,
        "full_field_available": len(field_rows) == len(FIELD_FILES),
        "full_field_changes": field_rows,
        "missing_full_fields": missing_fields,
        "maximum_temperature_field_rms_change_K": (
            max(float(row["absolute_rms_change"]) for row in temperature_rows)
            if temperature_rows
            else None
        ),
        "maximum_temperature_field_point_change_K": (
            max(float(row["maximum_absolute_change"]) for row in temperature_rows)
            if temperature_rows
            else None
        ),
        "interpretation": (
            "Measured numerical change over the final saved steady-iteration interval. "
            "The iteration labels are not physical seconds. No new physical parameter "
            "or fitted convergence threshold is introduced."
        ),
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument(
        "--start-iteration",
        "--start-time",
        dest="start_iteration",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--end-iteration",
        "--end-time",
        dest="end_iteration",
        type=float,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-fields", action="store_true")
    parser.add_argument("--update-completion-marker", type=Path)
    args = parser.parse_args()
    if args.end_iteration <= args.start_iteration:
        raise ValueError("end iteration must be greater than start iteration")
    payload = summarize_case(
        args.case.resolve(),
        args.start_iteration,
        args.end_iteration,
        args.allow_missing_fields,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.update_completion_marker:
        marker_path = args.update_completion_marker.resolve()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker.update(
            {
                "steady_final_window_status": payload["status"],
                "steady_final_window_iterations": payload["window_iterations"],
                "solver_time_semantics": "steady_iteration_index",
                "physical_time_s": None,
                "steady_final_window_full_field_available": payload["full_field_available"],
                "steady_final_window_summary": str(output),
                "steady_final_window_summary_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )
        marker_path.write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
