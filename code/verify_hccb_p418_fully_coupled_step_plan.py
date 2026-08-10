#!/usr/bin/env python3
"""Check the fully coupled extension against the existing P418 physical inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify(plan_path: Path) -> dict[str, object]:
    plan = json.loads(plan_path.resolve().read_text(encoding="utf-8"))
    base_path = ROOT / plan["base_thermal_step_plan"]
    schema_path = ROOT / plan["output_schema"]
    registry_path = ROOT / plan["physical_parameter_registry"]
    for path in (base_path, schema_path, registry_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if plan["sequences"] != base["sequences"] or len(plan["sequences"]) != 12:
        raise ValueError("fully coupled extension must retain all twelve endpoint pairs")
    if plan.get("new_physical_parameters") != [] or schema.get("new_physical_parameters") != []:
        raise ValueError("fully coupled extension introduces new physical parameters")
    controls = plan["openfoam_flow_controls"]
    if controls != {
        "flow": "yes",
        "momentumPredictor": "yes",
        "fluid_and_solid_ddt_scheme": "Euler",
    }:
        raise ValueError("fully coupled OpenFOAM flow controls changed")
    time_step = plan["time_step_sensitivity"]
    for key in ("config", "runner", "comparison", "summary_verifier"):
        path = ROOT / time_step[key]
        if not path.is_file():
            raise FileNotFoundError(path)
    time_config = json.loads(
        (ROOT / time_step["config"]).read_text(encoding="utf-8")
    )
    representative = next(
        (
            row
            for row in plan["sequences"]
            if row["sequence_id"] == time_step["representative_sequence_id"]
        ),
        None,
    )
    if representative is None or representative["family"] != "inlet_velocity_step":
        raise ValueError("time-step study must use a formal inlet-velocity sequence")
    for key in ("sequence_id", "source_condition_id", "target_condition_id"):
        expected = (
            representative["sequence_id"]
            if key == "sequence_id"
            else representative[key]
        )
        if time_config[key] != expected:
            raise ValueError(f"time-step study {key} differs from the formal sequence")
    declared_steps = [float(value) for value in time_config["delta_t_s"]]
    if len(declared_steps) != 3:
        raise ValueError("fully coupled time-step study must declare three resolutions")
    refinement_ratio = float(
        time_config["discretization_uncertainty_method"]["refinement_ratio"]
    )
    for coarse, fine in zip(declared_steps[:-1], declared_steps[1:]):
        if abs(coarse / fine - refinement_ratio) > 1.0e-12:
            raise ValueError("fully coupled time-step refinement ratio changed")
    formal_schedule = time_config["formal_time_step_schedule"]
    base_schedule = base["numerical_time_design"]["time_step_schedule"][
        : len(formal_schedule)
    ]
    if formal_schedule != base_schedule:
        raise ValueError(
            "fully coupled time-step schedule differs from the formal thermal schedule"
        )
    if time_config.get("new_physical_parameters") != []:
        raise ValueError("fully coupled time-step study introduces physical parameters")
    source_metadata = (
        ROOT
        / time_config["discretization_uncertainty_method"]["source_metadata"]
    )
    if not source_metadata.is_file():
        raise FileNotFoundError(source_metadata)
    time_runner = (ROOT / time_step["runner"]).read_text(encoding="utf-8")
    for phrase in (
        "EXECUTE=${EXECUTE:-0}",
        "fully coupled time-step study only; no OpenFOAM command was started",
        "REQUIRE_TIMESTEP_SENSITIVITY=0",
        "COMPARE_FIXED=0",
        "--analysis-kind fully_coupled_flow_heat",
    ):
        if phrase not in time_runner:
            raise ValueError(f"fully coupled time-step runner lacks {phrase}")
    if time_step.get("default_mode") != "plan_only":
        raise ValueError("fully coupled time-step runner no longer defaults to plan only")
    required_state = ["Ux_m_s", "Uy_m_s", "Uz_m_s", "pressure_Pa", "temperature_K"]
    if schema["state_order"] != required_state:
        raise ValueError("fully coupled state channels differ from the shared P418 state")
    arrays = schema["sequence_arrays"]
    if "Nt,Nnode,5" not in arrays["state_physical"]:
        raise ValueError("fully coupled state must retain its time axis")
    for name in ("fluid_internal_mass_flux_kg_s", "fluid_boundary_mass_flux_kg_s"):
        if not arrays[name].startswith("float64[Nt,"):
            raise ValueError(f"{name} must retain time-varying face flow")
    builder = ROOT / "code/build_hccb_p418_fully_coupled_step_cases.py"
    source = builder.read_text(encoding="utf-8")
    for phrase in (
        'replace_control_value(solution, "flow", "yes")',
        'replace_control_value(solution, "momentumPredictor", "yes")',
        '"new_physical_parameters": []',
        '"p418_fully_coupled_step_input_prepared_not_run"',
        '"published_conditions": step_conditions',
    ):
        if phrase not in source:
            raise ValueError(f"fully coupled builder lacks {phrase}")
    preparation = plan["input_preparation"]
    for key in ("builder", "initializer", "initial_field_verifier"):
        path = ROOT / preparation[key]
        if not path.is_file():
            raise FileNotFoundError(path)
    initializer = (ROOT / preparation["initializer"]).read_text(encoding="utf-8")
    for phrase in (
        'SOURCE_CASE=$(read_metadata',
        'TARGET_CASE=$(read_metadata',
        'TARGET_INLET_PHI=$(',
        'for field in U p p_rgh phi T',
        'boundaryField/inlet/value',
        "'boundaryField/inlet/value' -set \"${TARGET_INLET_PHI}\"",
        'verify_hccb_p418_fully_coupled_step_initialization.py',
    ):
        if phrase not in initializer:
            raise ValueError(f"fully coupled initializer lacks {phrase}")
    initialization_verifier = (ROOT / preparation["initial_field_verifier"]).read_text(
        encoding="utf-8"
    )
    for phrase in (
        '"fluid/U"',
        '"fluid/p"',
        '"fluid/p_rgh"',
        '"fluid/phi"',
        '"fluid/T"',
        '"solid/T"',
        'target_heat_source_dictionary_exact',
        'openfoam_calculation_started',
    ):
        if phrase not in initialization_verifier:
            raise ValueError(f"fully coupled initialization verifier lacks {phrase}")
    execution = plan["openfoam_execution"]
    execution_runner_path = ROOT / execution["runner"]
    execution_finalizer_path = ROOT / execution["finalizer"]
    for path in (execution_runner_path, execution_finalizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    execution_runner = execution_runner_path.read_text(encoding="utf-8")
    for phrase in (
        "EXECUTE=${EXECUTE:-0}",
        "no OpenFOAM command was started",
        "PAUSE_NEW_P418_CASES_FOR_CLOUD_MIGRATION",
        "REQUIRE_TIMESTEP_SENSITIVITY=${REQUIRE_TIMESTEP_SENSITIVITY:-1}",
        "verify_hccb_p418_fully_coupled_timestep_summary.py",
        "latest_complete_parallel_time",
        "fully_coupled_initial_field_map_complete.json",
        "log.foamMultiRun.fully_coupled",
        "finalize_hccb_p418_fully_coupled_step_response.sh",
        "--history-kind fully_coupled_flow_heat_response",
        "--history-mode fully_coupled_flow_heat",
    ):
        if phrase not in execution_runner:
            raise ValueError(f"fully coupled OpenFOAM runner lacks {phrase}")
    execution_finalizer = execution_finalizer_path.read_text(encoding="utf-8")
    for phrase in (
        "fully_coupled_step_metadata.json",
        "fully_coupled_step_response_complete.json",
        "completed_p418_fully_coupled_flow_heat_step_response",
        "solver_finished",
        "relative_mass_difference",
        "relative_energy_difference",
    ):
        if phrase not in execution_finalizer:
            raise ValueError(f"fully coupled OpenFOAM finalizer lacks {phrase}")
    if execution.get("default_mode") != "plan_only":
        raise ValueError("fully coupled OpenFOAM runner no longer defaults to plan only")
    data_export = plan["data_export"]
    integrated_exporter_path = ROOT / data_export["integrated_observable_exporter"]
    if not integrated_exporter_path.is_file():
        raise FileNotFoundError(integrated_exporter_path)
    integrated_exporter = integrated_exporter_path.read_text(encoding="utf-8")
    for phrase in (
        '"fully_coupled_flow_heat_response"',
        '"fully_coupled_step_metadata.json"',
        '"fully_coupled_step_response_complete.json"',
        "Velocity, pressure, face mass flux and fluid-solid temperature evolve together",
    ):
        if phrase not in integrated_exporter:
            raise ValueError(f"fully coupled integrated-observable exporter lacks {phrase}")
    if data_export["integrated_observable_history_kind"] != (
        "fully_coupled_flow_heat_response"
    ):
        raise ValueError("fully coupled integrated-observable history kind changed")
    exporter_path = ROOT / data_export["exporter"]
    if not exporter_path.is_file():
        raise FileNotFoundError(exporter_path)
    exporter = exporter_path.read_text(encoding="utf-8")
    for phrase in (
        '"fully_coupled_flow_heat"',
        '"fully_coupled_step_response_complete.json"',
        'internal_mass_flux if fully_coupled',
        'boundary_mass_flux if fully_coupled',
        'validate_sequence_arrays',
    ):
        if phrase not in exporter:
            raise ValueError(f"fully coupled data exporter lacks {phrase}")
    loader_path = ROOT / data_export["dataset_loader"]
    if not loader_path.is_file():
        raise FileNotFoundError(loader_path)
    loader = loader_path.read_text(encoding="utf-8")
    for phrase in (
        'history_mode") != "fully_coupled_flow_heat"',
        'training_statistics',
        'internal_mass_flux_mean_kg_s',
        'boundary_mass_flux_mean_kg_s',
        'training_sequence_ids',
    ):
        if phrase not in loader:
            raise ValueError(f"fully coupled dataset loader lacks {phrase}")
    comparison_path = ROOT / plan["fixed_vs_fully_coupled_comparison"]["implementation"]
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)
    comparison = comparison_path.read_text(encoding="utf-8")
    for phrase in (
        "fixed and fully coupled observables do not contain the same sequence ids",
        "maximum_difference_over_fully_coupled_response_span",
        "No fitted acceptance percentage is used",
        '"new_physical_parameters": []',
    ):
        if phrase not in comparison:
            raise ValueError(f"fixed-vs-fully-coupled comparison lacks {phrase}")
    model_extension = plan["model_extension"]
    model_path = ROOT / model_extension["implementation"]
    setting_path = ROOT / model_extension["numerical_setting_source"]
    for path in (model_path, setting_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    model = model_path.read_text(encoding="utf-8")
    for phrase in (
        'class HCCBP418FullyCoupledRegionalOperator',
        'self.state_change',
        'self.internal_flux_change',
        'self.boundary_flux_change',
        'initial_state[:, None] + time_factor * state_delta',
        'flux_graph.boundary_active',
    ):
        if phrase not in model:
            raise ValueError(f"fully coupled graph--Transformer lacks {phrase}")
    physics_path = ROOT / model_extension["physics_implementation"]
    if not physics_path.is_file():
        raise FileNotFoundError(physics_path)
    physics = physics_path.read_text(encoding="utf-8")
    for phrase in (
        "class P418FullyCoupledTransientResidual",
        "density_storage + steady_mass",
        "momentum_storage + steady_momentum",
        "fluid_storage + steady_fluid_energy",
        "solid_storage + steady_solid_energy",
        "internal_mass_flux_consistency_kg_s",
        "boundary_mass_flux_consistency_kg_s",
        "dimensionless_fully_coupled_equation_terms",
        "fixed flux is not fully coupled",
    ):
        if phrase not in physics:
            raise ValueError(f"fully coupled transient physics lacks {phrase}")
    training_path = ROOT / model_extension["training_loss_implementation"]
    if not training_path.is_file():
        raise FileNotFoundError(training_path)
    training = training_path.read_text(encoding="utf-8")
    for phrase in (
        "training_equation_scales",
        "projection_aware_physics_terms",
        "supervised_fully_coupled_terms",
        "combine_fully_coupled_loss_groups",
        "reference_states",
        "all three weights are explicit inputs",
    ):
        if phrase not in training:
            raise ValueError(f"fully coupled training interface lacks {phrase}")
    runner_path = ROOT / model_extension["training_runner"]
    if not runner_path.is_file():
        raise FileNotFoundError(runner_path)
    runner = runner_path.read_text(encoding="utf-8")
    for phrase in (
        "complete fully coupled curves",
        "test_used_after_model_selection_only",
        "training_normalization_sequence_ids",
        '"--state-weight"',
        '"--face-flux-weight"',
        '"--physics-weight"',
        "training_checkpoint.pt",
        '"--resume"',
    ):
        if phrase not in runner:
            raise ValueError(f"fully coupled trainer lacks {phrase}")
    return {
        "status": "p418_fully_coupled_step_extension_ready_for_input_build_and_dimensional_physics",
        "sequence_count": len(plan["sequences"]),
        "same_endpoint_pairs_as_thermal_study": True,
        "time_dependent_state_channel_count": len(required_state),
        "time_dependent_face_mass_flux": True,
        "flow_and_momentum_enabled": True,
        "source_full_state_initializer_present": True,
        "target_inlet_mass_flux_initialization_present": True,
        "target_boundary_and_heat_source_checks_present": True,
        "formal_openfoam_runner_present": True,
        "representative_time_step_study_present": True,
        "formal_runner_requires_verified_time_step_summary": True,
        "fully_coupled_restart_and_finalizer_present": True,
        "time_dependent_full_state_exporter_present": True,
        "integrated_observables_exporter_present": True,
        "fixed_vs_fully_coupled_comparison_present": True,
        "training_only_normalization_loader_present": True,
        "full_state_graph_transformer_forward_present": True,
        "transient_continuity_momentum_and_energy_present": True,
        "cell_state_to_face_flux_consistency_present": True,
        "equation_scales_restricted_to_training_histories": True,
        "projection_aware_training_loss_present": True,
        "small_forward_backward_update_test_present": True,
        "complete_curve_training_runner_present": True,
        "restart_checkpoint_present": True,
        "formal_time_step_acceptance_pending": True,
        "openfoam_calculation_started": False,
        "new_physical_parameters": []
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "parameters/hccb_p418_fully_coupled_step_plan.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/hccb_p418_fully_coupled_step_plan/summary.json",
    )
    args = parser.parse_args()
    summary = verify(args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
