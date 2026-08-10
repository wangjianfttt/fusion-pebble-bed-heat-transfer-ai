#!/usr/bin/env python3
"""Build restart cases for physical steps between published P418 endpoints."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from build_hccb_li4sio4_transient_solid_thermo import (
    PARAMETER_IDS as TRANSIENT_THERMO_PARAMETER_IDS,
    write_transient_solid_physical_properties,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "parameters/hccb_p418_transient_step_plan.json"


def replace_control_value(text: str, keyword: str, value: object) -> str:
    pattern = rf"(?m)^(\s*{re.escape(keyword)}\s+)[^;]+;"
    updated, count = re.subn(pattern, rf"\g<1>{value};", text)
    if count != 1:
        raise ValueError(f"expected one {keyword} entry, found {count}")
    return updated


def replace_ddt_scheme(text: str, scheme: str) -> str:
    """Replace the single default time scheme in an OpenFOAM fvSchemes file."""
    pattern = r"(ddtSchemes\s*\{\s*default\s+)[^;]+;"
    updated, count = re.subn(pattern, rf"\g<1>{scheme};", text, flags=re.S)
    if count != 1:
        raise ValueError(f"expected one default ddt scheme, found {count}")
    return updated


def integer_write_interval(write_interval_s: float, delta_t_s: float) -> int:
    """Convert a physical output interval to an exact solver-step count."""
    count = int(round(write_interval_s / delta_t_s))
    if count <= 0 or abs(count * delta_t_s - write_interval_s) > 1.0e-10:
        raise ValueError(
            f"write interval {write_interval_s} s is not an integer multiple "
            f"of deltaT={delta_t_s} s"
        )
    return count


def ensure_transient_density_solver(text: str) -> str:
    """Add the OpenFOAM-13 CHT density update used in transient mode."""
    if re.search(r'"\(rho\|rhoFinal\)"', text):
        return text
    pattern = r"(?m)^(solvers\s*\{\s*)"
    block = '''solvers
{
    "(rho|rhoFinal)"
    {
        solver diagonal;
    }

'''
    updated, count = re.subn(pattern, block, text, count=1)
    if count != 1:
        raise ValueError(f"expected one solvers dictionary, found {count}")
    return updated


def configure_thermal_transient(case: Path) -> None:
    """Enable transient energy storage while keeping target hydrodynamics fixed."""
    for region in ("fluid", "solid"):
        path = case / f"system/{region}/fvSchemes"
        text = replace_ddt_scheme(path.read_text(encoding="utf-8"), "Euler")
        path.write_text(text, encoding="utf-8")

    solution_path = case / "system/fluid/fvSolution"
    solution = solution_path.read_text(encoding="utf-8")
    solution = ensure_transient_density_solver(solution)
    solution = replace_control_value(solution, "flow", "no")
    solution = replace_control_value(solution, "momentumPredictor", "no")
    solution_path.write_text(solution, encoding="utf-8")


def copy_case_template(target: Path, destination: Path) -> None:
    """Copy editable dictionaries and hard-link the immutable shared mesh."""
    destination.mkdir(parents=True)
    shutil.copytree(target / "0", destination / "0")
    shutil.copytree(target / "system", destination / "system")
    shutil.copytree(
        target / "constant",
        destination / "constant",
        copy_function=os.link,
    )
    # decomposePar rewrites cellProc for the active MPI partition count.
    # Remove the hard-linked endpoint copies so transient decomposition cannot
    # modify the packaged steady endpoint.
    for region in ("fluid", "solid"):
        (destination / f"constant/{region}/polyMesh/cellProc").unlink(
            missing_ok=True
        )
    # The transient solid heat-capacity model rewrites this dictionary below.
    # Break its hard link so the converged steady endpoint remains immutable.
    solid_properties = destination / "constant/solid/physicalProperties"
    solid_properties.unlink()
    shutil.copy2(
        target / "constant/solid/physicalProperties",
        solid_properties,
    )
    shutil.copy2(target / "cht_smoke_metadata.json", destination / "cht_smoke_metadata.json")


def load_conditions(matrix_root: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    manifest = json.loads((matrix_root / "matrix_manifest.json").read_text(encoding="utf-8"))
    conditions = {str(row["condition_id"]): row for row in manifest["published_conditions"]}
    return conditions, manifest


def field_write_stages(numerical_time: dict[str, object]) -> tuple[list[dict[str, float]], list[float]]:
    """Return a contiguous full-field output schedule and its unique times."""
    duration = float(numerical_time["duration_s"])
    configured = numerical_time.get("field_write_schedule")
    if configured is None:
        configured = [
            {
                "start_s": 0.0,
                "end_s": duration,
                "interval_s": float(numerical_time["field_write_interval_s"]),
            }
        ]
    stages = [
        {
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "interval_s": float(row["interval_s"]),
        }
        for row in configured
    ]
    if not stages or abs(stages[0]["start_s"]) > 1.0e-12:
        raise ValueError("field-write schedule must start at 0 s")
    if abs(stages[-1]["end_s"] - duration) > 1.0e-12:
        raise ValueError("field-write schedule must end at the response duration")
    times: set[float] = set()
    previous_end = 0.0
    for stage in stages:
        start, end, interval = (stage[key] for key in ("start_s", "end_s", "interval_s"))
        if abs(start - previous_end) > 1.0e-12 or end <= start or interval <= 0.0:
            raise ValueError(f"invalid or non-contiguous field-write stage: {stage}")
        count = int(round((end - start) / interval))
        if abs(start + count * interval - end) > 1.0e-10:
            raise ValueError(f"stage duration must be an integer multiple of interval: {stage}")
        times.update(round(start + index * interval, 12) for index in range(count + 1))
        previous_end = end
    return stages, sorted(times)


def time_step_stages(numerical_time: dict[str, object]) -> list[dict[str, float]]:
    """Return a contiguous staged numerical time-step schedule."""
    duration = float(numerical_time["duration_s"])
    configured = numerical_time.get("time_step_schedule")
    if configured is None:
        configured = [
            {
                "start_s": 0.0,
                "end_s": duration,
                "delta_t_s": float(numerical_time["delta_t_s"]),
            }
        ]
    stages = [
        {
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "delta_t_s": float(row["delta_t_s"]),
        }
        for row in configured
    ]
    if not stages or abs(stages[0]["start_s"]) > 1.0e-12:
        raise ValueError("time-step schedule must start at 0 s")
    if abs(stages[-1]["end_s"] - duration) > 1.0e-12:
        raise ValueError("time-step schedule must end at the response duration")
    previous_end = 0.0
    for stage in stages:
        start, end, delta_t = (stage[key] for key in ("start_s", "end_s", "delta_t_s"))
        if abs(start - previous_end) > 1.0e-12 or end <= start or delta_t <= 0.0:
            raise ValueError(f"invalid or non-contiguous time-step stage: {stage}")
        count = int(round((end - start) / delta_t))
        if abs(start + count * delta_t - end) > 1.0e-10:
            raise ValueError(f"stage duration must be an integer multiple of deltaT: {stage}")
        previous_end = end
    return stages


def execution_stages(
    time_steps: list[dict[str, float]],
    field_writes: list[dict[str, float]],
) -> list[dict[str, float]]:
    """Split execution at every requested full-field output time."""
    write_times: set[float] = set()
    for row in field_writes:
        start = row["start_s"]
        end = row["end_s"]
        interval = row["interval_s"]
        count = int(round((end - start) / interval))
        write_times.update(round(start + index * interval, 12) for index in range(count + 1))
    boundaries = sorted(
        {
            *(row[key] for row in time_steps for key in ("start_s", "end_s")),
            *write_times,
        }
    )

    def containing(rows: list[dict[str, float]], start: float, end: float) -> dict[str, float]:
        for row in rows:
            if row["start_s"] <= start + 1.0e-12 and row["end_s"] >= end - 1.0e-12:
                return row
        raise ValueError(f"no staged setting covers {start}--{end} s")

    combined: list[dict[str, float]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        time_row = containing(time_steps, start, end)
        write_row = containing(field_writes, start, end)
        duration = end - start
        delta_t = time_row["delta_t_s"]
        write_interval = write_row["interval_s"]
        delta_count = int(round(duration / delta_t))
        if abs(delta_count * delta_t - duration) > 1.0e-10:
            raise ValueError(
                f"execution stage {start}--{end} s is not divisible by deltaT={delta_t}"
            )
        if abs(duration - write_interval) > 1.0e-10:
            raise ValueError(
                f"execution stage {start}--{end} s must end at the next "
                f"full-field output separated by {write_interval} s"
            )
        combined.append(
            {
                "start_s": start,
                "end_s": end,
                "delta_t_s": delta_t,
                "write_interval_s": write_interval,
            }
        )
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--sequence-id", action="append", default=[])
    parser.add_argument("--require-all-ready", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    matrix_root = args.matrix_root.resolve()
    output_root = args.output_root.resolve()
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    conditions, matrix_manifest = load_conditions(matrix_root)
    requested = set(args.sequence_id)
    sequences = [row for row in plan["sequences"] if not requested or row["sequence_id"] in requested]
    missing_requested = requested.difference(row["sequence_id"] for row in sequences)
    if missing_requested:
        raise ValueError(f"unknown sequence ids: {sorted(missing_requested)}")

    output_root.mkdir(parents=True, exist_ok=True)
    built: list[dict[str, object]] = []
    waiting: list[dict[str, object]] = []
    numerical_time = plan["numerical_time_design"]
    duration = float(numerical_time["duration_s"])
    delta_t = float(numerical_time["delta_t_s"])
    if duration <= 0.0 or delta_t <= 0.0:
        raise ValueError("duration and initial deltaT must be positive")
    time_steps = time_step_stages(numerical_time)
    write_stages, snapshot_times = field_write_stages(numerical_time)
    staged_execution = execution_stages(time_steps, write_stages)
    first_stage = staged_execution[0]
    if abs(delta_t - first_stage["delta_t_s"]) > 1.0e-15:
        raise ValueError("delta_t_s must equal the first staged numerical time step")

    readiness = []
    for sequence in sequences:
        source_id = str(sequence["source_condition_id"])
        target_id = str(sequence["target_condition_id"])
        readiness.append(
            {
                "sequence_id": str(sequence["sequence_id"]),
                "source_condition_id": source_id,
                "source_ready": (matrix_root / source_id / "formal_sample_complete.json").is_file(),
                "target_condition_id": target_id,
                "target_ready": (matrix_root / target_id / "formal_sample_complete.json").is_file(),
            }
        )
    not_ready = [row for row in readiness if not (row["source_ready"] and row["target_ready"])]
    if args.require_all_ready and not_ready:
        raise RuntimeError(f"{len(not_ready)} step sequences are waiting for converged endpoints")

    for sequence in sequences:
        sequence_id = str(sequence["sequence_id"])
        source_id = str(sequence["source_condition_id"])
        target_id = str(sequence["target_condition_id"])
        if source_id not in conditions or target_id not in conditions:
            raise ValueError(f"{sequence_id} uses a condition outside the P418 matrix")
        source_case = matrix_root / source_id
        target_case = matrix_root / target_id
        source_ready = (source_case / "formal_sample_complete.json").is_file()
        target_ready = (target_case / "formal_sample_complete.json").is_file()
        if not (source_ready and target_ready):
            waiting.append(
                {
                    "sequence_id": sequence_id,
                    "source_condition_id": source_id,
                    "source_ready": source_ready,
                    "target_condition_id": target_id,
                    "target_ready": target_ready,
                }
            )
            continue
        source_marker = json.loads(
            (source_case / "formal_sample_complete.json").read_text(encoding="utf-8")
        )
        target_marker = json.loads(
            (target_case / "formal_sample_complete.json").read_text(encoding="utf-8")
        )

        case = output_root / sequence_id
        if case.exists():
            if args.overwrite:
                shutil.rmtree(case)
            elif (case / "step_case_metadata.json").is_file():
                built.append(json.loads((case / "step_case_metadata.json").read_text(encoding="utf-8")))
                continue
            else:
                raise FileExistsError(case)
        copy_case_template(target_case, case)
        transient_thermo = write_transient_solid_physical_properties(
            case / "constant/solid/physicalProperties",
            metadata_path=case / "transient_solid_thermo.json",
        )
        configure_thermal_transient(case)

        control_path = case / "system/controlDict"
        control = control_path.read_text(encoding="utf-8")
        for keyword, value in (
            ("startTime", 0),
            ("endTime", first_stage["end_s"]),
            ("deltaT", first_stage["delta_t_s"]),
            ("writeControl", "timeStep"),
            (
                "writeInterval",
                integer_write_interval(
                    first_stage["write_interval_s"],
                    first_stage["delta_t_s"],
                ),
            ),
            ("purgeWrite", 0),
        ):
            control = replace_control_value(control, keyword, value)
        control_path.write_text(control, encoding="utf-8")

        source = conditions[source_id]
        target = conditions[target_id]
        source_case_metadata = json.loads(
            (source_case / "cht_smoke_metadata.json").read_text(encoding="utf-8")
        )
        target_case_metadata = json.loads(
            (target_case / "cht_smoke_metadata.json").read_text(encoding="utf-8")
        )
        source_with_boundary = {
            **source,
            "pore_opening_boundary_velocity_m_s": source_case_metadata[
                "pore_opening_boundary_velocity_m_s"
            ],
            "inlet_open_area_fraction": source_case_metadata["inlet_open_area_fraction"],
        }
        target_with_boundary = {
            **target,
            "pore_opening_boundary_velocity_m_s": target_case_metadata[
                "pore_opening_boundary_velocity_m_s"
            ],
            "inlet_open_area_fraction": target_case_metadata["inlet_open_area_fraction"],
        }
        changed = {
            key: {"source": source[key], "target": target[key]}
            for key in ("inlet_velocity_m_s", "inlet_temperature_K", "solid_heat_source_MW_m3")
            if float(source[key]) != float(target[key])
        }
        if len(changed) != 1:
            raise ValueError(f"{sequence_id} must change exactly one physical input, found {changed}")
        metadata = {
            "status": "p418_quasi_steady_flow_thermal_step_case_built_waiting_for_field_map",
            "sequence_id": sequence_id,
            "family": sequence["family"],
            "source_condition_id": source_id,
            "target_condition_id": target_id,
            "source_case": str(source_case),
            "target_case": str(target_case),
            "source_final_time_s": source_marker["time"],
            "target_final_time_s": target_marker["time"],
            "duration_s": duration,
            "delta_t_s": delta_t,
            "time_step_schedule": time_steps,
            "write_interval_s": float(numerical_time["field_write_interval_s"]),
            "field_write_schedule": write_stages,
            "execution_schedule": staged_execution,
            "snapshot_times_s": snapshot_times,
            "transient_model": "thermal_step_with_quasi_steady_target_hydrodynamics",
            "hydrodynamic_initialization": "target converged U, p and p_rgh fields",
            "thermal_initialization": "source converged fluid and solid temperature fields",
            "time_discretization": "first-order implicit Euler for fluid and solid energy equations",
            "flow_treatment": "target U, p and solved conservative face mass flux phi are frozen while energy equations evolve",
            "source_parameters": source_with_boundary,
            "target_parameters": target_with_boundary,
            "changed_physical_input": changed,
            "source_parameter_id": plan["source_parameter_id"],
            "source_title": plan["source_title"],
            "source_doi": plan["source_doi"],
            "new_physical_parameters": [],
            "transient_thermo_parameter_ids": list(TRANSIENT_THERMO_PARAMETER_IDS),
            "transient_solid_thermo": transient_thermo,
            "scientific_scope": plan["scientific_scope"],
        }
        (case / "step_case_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (case / "transient_snapshot_plan.json").write_text(
            json.dumps(
                {
                    "status": "preserve_physical_step_response_fields",
                    "sequence_id": sequence_id,
                    "snapshot_times_s": snapshot_times,
                    "new_physical_parameters": [],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        target_metadata = json.loads((case / "cht_smoke_metadata.json").read_text(encoding="utf-8"))
        target_metadata.update(
            {
                "status": "p418_quasi_steady_flow_thermal_step_response_case",
                "condition_id": sequence_id,
                "source_condition_id": source_id,
                "target_condition_id": target_id,
                "purpose": (
                    "three-dimensional thermal response after one published P418 operating input is stepped; "
                    "the converged target U, p and conservative face mass flux phi are held fixed"
                ),
                "sample_use": "physical step-response learning and independent sequence prediction",
                "new_physical_parameters": [],
                "transient_thermo_parameter_ids": list(TRANSIENT_THERMO_PARAMETER_IDS),
            }
        )
        (case / "cht_smoke_metadata.json").write_text(
            json.dumps(target_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        built.append(metadata)

    step_conditions = []
    for row in built:
        target = row["target_parameters"]
        step_conditions.append(
            {
                "condition_id": row["sequence_id"],
                "inlet_velocity_m_s": target["inlet_velocity_m_s"],
                "inlet_temperature_K": target["inlet_temperature_K"],
                "solid_heat_source_MW_m3": target["solid_heat_source_MW_m3"],
            }
        )
    output_manifest = {
        "status": "p418_quasi_steady_flow_thermal_step_response_case_set_built",
        "history_kind": "physical_step_response",
        "transient_model": "thermal_step_with_quasi_steady_target_hydrodynamics",
        "source_title": matrix_manifest["source_title"],
        "source_doi": matrix_manifest["source_doi"],
        "source_matrix": str(matrix_root),
        "selected_case_count": len(built),
        "waiting_case_count": len(waiting),
        "published_conditions": step_conditions,
        "cases": built,
        "waiting": waiting,
        "new_physical_parameters": [],
    }
    (output_root / "matrix_manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
