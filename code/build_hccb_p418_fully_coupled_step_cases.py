#!/usr/bin/env python3
"""Prepare fully coupled P418 flow--heat step cases without running OpenFOAM."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from build_hccb_li4sio4_transient_solid_thermo import (
    PARAMETER_IDS as TRANSIENT_THERMO_PARAMETER_IDS,
    write_transient_solid_physical_properties,
)
from build_hccb_p418_step_response_cases import (
    copy_case_template,
    ensure_transient_density_solver,
    execution_stages,
    field_write_stages,
    load_conditions,
    replace_control_value,
    replace_ddt_scheme,
    time_step_stages,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "parameters/hccb_p418_fully_coupled_step_plan.json"


def configure_fully_coupled_transient(case: Path) -> None:
    """Advance momentum and both energy regions with first-order implicit time terms."""
    for region in ("fluid", "solid"):
        path = case / f"system/{region}/fvSchemes"
        path.write_text(
            replace_ddt_scheme(path.read_text(encoding="utf-8"), "Euler"),
            encoding="utf-8",
        )
    solution_path = case / "system/fluid/fvSolution"
    solution = ensure_transient_density_solver(solution_path.read_text(encoding="utf-8"))
    solution = replace_control_value(solution, "flow", "yes")
    solution = replace_control_value(solution, "momentumPredictor", "yes")
    solution_path.write_text(solution, encoding="utf-8")


def changed_input(source: dict, target: dict) -> dict[str, dict[str, object]]:
    changed = {
        key: {"source": source[key], "target": target[key]}
        for key in (
            "inlet_velocity_m_s",
            "inlet_temperature_K",
            "solid_heat_source_MW_m3",
        )
        if float(source[key]) != float(target[key])
    }
    if len(changed) != 1:
        raise ValueError(f"a physical step must change exactly one input, found {changed}")
    return changed


def build(args: argparse.Namespace) -> dict[str, object]:
    matrix_root = args.matrix_root.resolve()
    output_root = args.output_root.resolve()
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    base_plan = json.loads((ROOT / plan["base_thermal_step_plan"]).read_text(encoding="utf-8"))
    base_sequences = {
        str(row["sequence_id"]): row for row in base_plan["sequences"]
    }
    for row in plan["sequences"]:
        sequence_id = str(row["sequence_id"])
        if sequence_id not in base_sequences or row != base_sequences[sequence_id]:
            raise ValueError(
                "fully coupled endpoint pairs differ from the main thermal-step plan"
            )
    analysis_kind = plan.get("analysis_kind", "formal_fully_coupled_steps")
    if analysis_kind == "formal_fully_coupled_steps":
        if plan["sequences"] != base_plan["sequences"]:
            raise ValueError("formal fully coupled calculation must retain all twelve curves")
    elif analysis_kind == "fully_coupled_time_step_sensitivity":
        if len(plan["sequences"]) != 1:
            raise ValueError("time-step sensitivity must contain one declared curve")
    elif analysis_kind == "independent_high_re_test":
        if plan["sequences"] != base_plan["sequences"] or len(plan["sequences"]) != 6:
            raise ValueError(
                "high-Re fully coupled test must retain the six declared independent curves"
            )
        if plan.get("data_role") != "frozen_model_independent_test_only":
            raise ValueError("high-Re curves must remain frozen-model independent tests")
        if any(bool(value) for value in plan.get("model_use", {}).values() if isinstance(value, bool)):
            raise ValueError("high-Re independent curves cannot be used for model fitting")
    else:
        raise ValueError(f"unsupported fully coupled analysis kind: {analysis_kind}")
    if plan.get("new_physical_parameters") != []:
        raise ValueError("fully coupled plan introduces new physical parameters")

    conditions, matrix_manifest = load_conditions(matrix_root)
    requested = set(args.sequence_id)
    sequences = [
        row for row in plan["sequences"]
        if not requested or str(row["sequence_id"]) in requested
    ]
    unknown = requested.difference(str(row["sequence_id"]) for row in sequences)
    if unknown:
        raise ValueError(f"unknown sequence ids: {sorted(unknown)}")

    numerical_time = plan.get(
        "numerical_time_design", base_plan["numerical_time_design"]
    )
    time_steps = time_step_stages(numerical_time)
    write_stages, snapshot_times = field_write_stages(numerical_time)
    stages = execution_stages(time_steps, write_stages)
    first_stage = stages[0]
    output_root.mkdir(parents=True, exist_ok=True)
    built: list[dict[str, object]] = []
    waiting: list[dict[str, object]] = []

    for sequence in sequences:
        sequence_id = str(sequence["sequence_id"])
        source_id = str(sequence["source_condition_id"])
        target_id = str(sequence["target_condition_id"])
        if source_id not in conditions or target_id not in conditions:
            raise ValueError(f"{sequence_id} uses a condition outside the P418 matrix")
        source_case = matrix_root / source_id
        target_case = matrix_root / target_id
        source_marker = source_case / "formal_sample_complete.json"
        target_marker = target_case / "formal_sample_complete.json"
        if not source_marker.is_file() or not target_marker.is_file():
            waiting.append(
                {
                    "sequence_id": sequence_id,
                    "source_ready": source_marker.is_file(),
                    "target_ready": target_marker.is_file(),
                }
            )
            continue

        case = output_root / sequence_id
        if case.exists():
            if args.overwrite:
                shutil.rmtree(case)
            elif (case / "fully_coupled_step_metadata.json").is_file():
                built.append(
                    json.loads(
                        (case / "fully_coupled_step_metadata.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
                continue
            else:
                raise FileExistsError(case)
        copy_case_template(target_case, case)
        configure_fully_coupled_transient(case)
        transient_thermo = write_transient_solid_physical_properties(
            case / "constant/solid/physicalProperties",
            metadata_path=case / "transient_solid_thermo.json",
        )

        control_path = case / "system/controlDict"
        control = control_path.read_text(encoding="utf-8")
        for keyword, value in (
            ("startTime", 0),
            ("endTime", first_stage["end_s"]),
            ("deltaT", first_stage["delta_t_s"]),
            ("writeControl", "runTime"),
            ("writeInterval", first_stage["write_interval_s"]),
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
        source_record = json.loads(source_marker.read_text(encoding="utf-8"))
        target_record = json.loads(target_marker.read_text(encoding="utf-8"))
        metadata = {
            "status": "p418_fully_coupled_step_input_prepared_not_run",
            "transient_model": "fully_coupled_flow_heat_step_response",
            "sequence_id": sequence_id,
            "family": sequence["family"],
            "source_condition_id": source_id,
            "target_condition_id": target_id,
            "source_case": str(source_case),
            "target_case": str(target_case),
            "source_final_time_s": source_record["time"],
            "target_final_time_s": target_record["time"],
            "initial_field_rule": "copy source endpoint U, p, p_rgh, phi, fluid T and solid T into time 0 before execution",
            "target_boundary_rule": plan["target_rule"],
            "flow_treatment": "time-dependent momentum, continuity, fluid energy and solid energy are solved together",
            "duration_s": float(numerical_time["duration_s"]),
            "time_step_schedule": time_steps,
            "field_write_schedule": write_stages,
            "execution_schedule": stages,
            "snapshot_times_s": snapshot_times,
            "source_parameters": source_with_boundary,
            "target_parameters": target_with_boundary,
            "changed_physical_input": changed_input(source, target),
            "source_parameter_id": plan["source_parameter_id"],
            "source_doi": plan["source_doi"],
            "transient_thermo_parameter_ids": list(TRANSIENT_THERMO_PARAMETER_IDS),
            "transient_solid_thermo": transient_thermo,
            "new_physical_parameters": [],
            "scientific_scope": plan["scientific_scope"],
        }
        (case / "fully_coupled_step_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        built.append(metadata)

    if args.require_all_ready and waiting:
        raise RuntimeError(f"{len(waiting)} fully coupled curves wait for steady endpoints")
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
    manifest = {
        "status": "p418_fully_coupled_step_inputs_prepared_not_run",
        "history_kind": "fully_coupled_flow_heat_step_response",
        "analysis_kind": analysis_kind,
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
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--sequence-id", action="append", default=[])
    parser.add_argument("--require-all-ready", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
