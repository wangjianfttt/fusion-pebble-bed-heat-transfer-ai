#!/usr/bin/env python3
"""Verify the shared physical interface of the P418 fused model chain."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

from verify_hccb_p418_parameter_evidence_files import verify as verify_evidence_files
from verify_hccb_p418_model_comparison_protocol import verify as verify_model_comparison
from verify_hccb_p418_fully_coupled_step_plan import verify as verify_fully_coupled


ROOT = Path(__file__).resolve().parents[1]


def literal_assignment(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if name in targets:
                value = ast.literal_eval(node.value)
                return [str(item) for item in value]
    raise ValueError(f"{path} does not define {name}")


def assert_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: expected {expected!r}, found {actual!r}")


def verify_parameter_sources(
    contract: dict[str, object], require_local_evidence_files: bool = True
) -> dict[str, object]:
    source_path = ROOT / contract["physical_parameter_sources"]["registry"]
    evidence_path = ROOT / contract["physical_parameter_sources"]["evidence_registry"]
    equation_path = ROOT / contract["physical_parameter_sources"]["equation_map"]
    with source_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    with equation_path.open(newline="", encoding="utf-8") as handle:
        equation_rows = list(csv.DictReader(handle))
    required_source_columns = (
        "parameter_id",
        "物理量",
        "采用值或关系式",
        "单位",
        "文献",
        "链接或DOI",
        "原文位置说明",
    )
    for row in source_rows:
        missing = [name for name in required_source_columns if not row.get(name, "").strip()]
        if missing:
            raise ValueError(f"physical parameter {row.get('parameter_id')} lacks {missing}")
    source_ids = {row["parameter_id"] for row in source_rows}
    if len(source_ids) != len(source_rows):
        raise ValueError("physical parameter identifiers are duplicated")
    used_ids: set[str] = set()
    for row in equation_rows:
        identifiers = [item.strip() for item in row["文献参数编号"].replace(";", ",").split(",")]
        used_ids.update(item for item in identifiers if item.startswith("P"))
    missing_sources = sorted(used_ids - source_ids)
    if missing_sources:
        raise ValueError(f"equation map refers to unregistered physical parameters: {missing_sources}")
    unused_sources = sorted(source_ids - used_ids)
    if unused_sources:
        raise ValueError(f"registered physical parameters are absent from the equation map: {unused_sources}")
    evidence_summary = verify_evidence_files(
        source_path,
        evidence_path,
        equation_path,
        root=ROOT,
        require_local_files=require_local_evidence_files,
    )
    return {
        "physical_parameter_count": len(source_rows),
        "equation_map_row_count": len(equation_rows),
        "local_evidence_reference_count": evidence_summary[
            "local_evidence_reference_count"
        ],
        "parameter_evidence_status_counts": evidence_summary[
            "evidence_status_counts"
        ],
        "parameter_evidence_verification_mode": evidence_summary[
            "evidence_verification_mode"
        ],
        "p429_derivative_check": evidence_summary["p429_derivative_check"],
        "p430_molar_mass_g_mol": evidence_summary["p430_molar_mass_g_mol"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "parameters/hccb_p418_fused_model_contract.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--metadata-only-evidence",
        action="store_true",
        help="Do not require copyrighted local evidence files in a public source package.",
    )
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    export_path = ROOT / "code/export_hccb_p418_step_regional_sequences.py"
    steady_target_path = ROOT / "code/build_hccb_p418_regional_state_targets.py"
    state_names = literal_assignment(export_path, "STATE_NAMES")
    transient_conditions = literal_assignment(export_path, "CONDITION_NAMES")
    steady_conditions = literal_assignment(steady_target_path, "CONDITION_KEYS")
    shared_state = contract["shared_state"]["channel_order"]
    assert_equal(state_names, shared_state, "exported transient state order differs from the contract")
    assert_equal(
        contract["steady_stage"]["state_output_order"],
        shared_state,
        "steady state order differs from the shared state",
    )
    assert_equal(
        contract["physical_transient_stage"]["state_output_order"],
        shared_state,
        "transient state order differs from the shared state",
    )
    assert_equal(
        steady_conditions,
        contract["steady_stage"]["condition_order"],
        "steady condition order differs from the contract",
    )
    assert_equal(
        transient_conditions,
        contract["physical_transient_stage"]["condition_order"],
        "transient condition order differs from the contract",
    )

    plan_path = ROOT / contract["physical_transient_stage"]["plan_file"]
    split_path = ROOT / contract["physical_transient_stage"]["split_file"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))["splits"]
    sequence_ids = {str(row["sequence_id"]) for row in plan["sequences"]}
    assert_equal(
        len(sequence_ids),
        contract["physical_transient_stage"]["required_sequence_count"],
        "physical step count differs from the contract",
    )
    for split_name, split in splits.items():
        roles = {role: [str(item) for item in split[role]] for role in ("train", "validation", "test")}
        role_sets = {role: set(values) for role, values in roles.items()}
        if any(len(role_sets[role]) != len(roles[role]) for role in roles):
            raise ValueError(f"{split_name} contains a duplicate curve")
        if any(
            role_sets[left] & role_sets[right]
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise ValueError(f"{split_name} shares a complete curve across roles")
        assert_equal(set().union(*role_sets.values()), sequence_ids, f"{split_name} does not cover all curves")

    step_runner = (ROOT / "code/run_hccb_p418_step_responses.sh").read_text(encoding="utf-8")
    if "parameters/hccb_p418_step_response_splits.json" not in step_runner:
        raise ValueError("physical-step runner does not use the physical-step split file")
    if "--history-kind physical_step_response" not in step_runner:
        raise ValueError("physical-step runner does not identify physical step histories")
    relaxation_runner = (ROOT / "code/run_hccb_p418_60_transient_models.sh").read_text(encoding="utf-8")
    if "--history-kind physical_step_response" in relaxation_runner:
        raise ValueError("steady solver-relaxation runner is mislabeled as physical transient")

    transient_source = (ROOT / "code/train_hccb_p418_spatiotemporal_regional_operator.py").read_text(encoding="utf-8")
    for phrase in (
        "Velocity and pressure remain exact inputs",
        "baseline_temperature_normalized",
        "fixed_hydrodynamics_physical",
        "fluid_internal_mass_flux_kg_s",
        "fluid_boundary_mass_flux_kg_s",
    ):
        if phrase not in transient_source:
            raise ValueError(f"transient implementation lacks required interface marker: {phrase}")
    diffusion_source = (ROOT / "code/train_hccb_p418_temporal_temperature_diffusion.py").read_text(encoding="utf-8")
    for phrase in (
        "Combine one temperature history with the fixed target velocity/pressure field",
        "baseline_temperature_normalized",
        "target_temperature_normalized",
        "fluid_internal_mass_flux_kg_s",
        "fluid_boundary_mass_flux_kg_s",
    ):
        if phrase not in diffusion_source:
            raise ValueError(f"diffusion implementation lacks required interface marker: {phrase}")

    inference = contract["complete_inference_entry"]
    inference_paths = [
        "runner",
        "deterministic_evaluator",
        "diffusion_evaluator",
        "common_energy_evaluator",
        "summary_builder",
        "table_builder",
    ]
    for key in inference_paths:
        path = ROOT / inference[key]
        if not path.is_file():
            raise FileNotFoundError(path)
    chain_roles = (ROOT / "code/hccb_p418_chain_roles.py").read_text(encoding="utf-8")
    deterministic_evaluator = (ROOT / inference["deterministic_evaluator"]).read_text(
        encoding="utf-8"
    )
    diffusion_evaluator = (ROOT / inference["diffusion_evaluator"]).read_text(
        encoding="utf-8"
    )
    runner = (ROOT / inference["runner"]).read_text(encoding="utf-8")
    legacy_names = (
        "completed_p418_steady_PINN_to_transient_PINN_chain",
        "completed_p418_steady_PINN_transient_PINN_diffusion_chain",
    )
    if any(name in deterministic_evaluator or name in diffusion_evaluator for name in legacy_names):
        raise ValueError("complete chain still uses the old transient-PINN status name")
    for name in (
        "completed_p418_steady_PINN_to_graph_transformer_chain",
        "completed_p418_steady_PINN_graph_transformer_diffusion_chain",
    ):
        if name not in chain_roles:
            raise ValueError(f"shared chain status is missing: {name}")
    for relative in (
        inference["deterministic_evaluator"],
        inference["diffusion_evaluator"],
        inference["common_energy_evaluator"],
        inference["summary_builder"],
        inference["table_builder"],
    ):
        if Path(relative).name not in runner:
            raise ValueError(f"complete inference runner does not call {relative}")
    runner_splits = set(inference["required_split_names"])
    if not all(name in runner for name in runner_splits):
        raise ValueError("complete inference runner does not cover every declared split")

    parameter_counts = verify_parameter_sources(
        contract, require_local_evidence_files=not args.metadata_only_evidence
    )
    comparison_path = ROOT / contract["model_comparison_protocol"]
    comparison_summary = verify_model_comparison(comparison_path)
    data_preparation = contract["model_data_preparation"]
    data_preparation_config_path = ROOT / data_preparation["config"]
    data_preparation_runner_path = ROOT / data_preparation["runner"]
    if not data_preparation_config_path.is_file():
        raise FileNotFoundError(data_preparation_config_path)
    if not data_preparation_runner_path.is_file():
        raise FileNotFoundError(data_preparation_runner_path)
    data_preparation_config = json.loads(
        data_preparation_config_path.read_text(encoding="utf-8")
    )
    data_preparation_source = data_preparation_runner_path.read_text(encoding="utf-8")
    if data_preparation.get("starts_model_training") is not False:
        raise ValueError("model data preparation is not declared as data-only")
    if data_preparation.get("new_physical_parameters") != []:
        raise ValueError("model data preparation introduces new physical parameters")
    if data_preparation_config.get("new_physical_parameters") != []:
        raise ValueError("model data preparation config introduces new physical parameters")
    for phrase in (
        "starts_model_training",
        "training command entered data-only pipeline",
        "--execute-stage",
    ):
        if phrase not in data_preparation_source:
            raise ValueError(
                f"model data preparation lacks required no-training marker: {phrase}"
            )
    if (
        data_preparation_config["integrated_observable_exporter"]
        != "code/export_hccb_p418_transient_observables.py"
    ):
        raise ValueError("model data preparation uses the wrong integrated exporter")
    coupled_extension = contract["fully_coupled_transient_extension"]
    coupled_plan_path = ROOT / coupled_extension["plan_file"]
    coupled_summary = verify_fully_coupled(
        coupled_plan_path,
        require_local_source_metadata=not args.metadata_only_evidence,
    )
    assert_equal(
        coupled_extension["predicted_channels"],
        shared_state,
        "fully coupled state channels differ from the shared state",
    )
    if coupled_extension.get("new_physical_parameters") != []:
        raise ValueError("fully coupled extension introduces new physical parameters")
    implementation_paths = []
    for stage in (
        "steady_stage",
        "physical_transient_stage",
        "fully_coupled_transient_extension",
        "diffusion_stage",
    ):
        for relative in contract[stage]["implementation"]:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            implementation_paths.append(relative)
    implementation_paths.append(data_preparation["runner"])
    if contract.get("new_physical_parameters") != []:
        raise ValueError("fused model contract introduces unregistered physical parameters")

    result = {
        "status": "p418_fused_model_contract_verified",
        "contract": str(contract_path),
        "shared_state_channels": shared_state,
        "steady_condition_count": contract["steady_stage"]["required_condition_count"],
        "physical_transient_sequence_count": len(sequence_ids),
        "physical_transient_split_count": len(splits),
        "physical_transient_output_time_count": contract["physical_transient_stage"]["required_output_time_count"],
        "diffusion_corrected_channels": contract["diffusion_stage"]["corrected_channels"],
        "implementation_file_count": len(implementation_paths),
        "complete_inference_entry": inference["runner"],
        "complete_inference_split_count": len(inference["required_split_names"]),
        "model_comparison_protocol": str(comparison_path),
        "model_data_preparation_config": str(data_preparation_config_path),
        "model_data_preparation_runner": data_preparation["runner"],
        "model_data_preparation_starts_training": False,
        "steady_model_comparison_count": comparison_summary["steady_model_count"],
        "transient_model_comparison_count": comparison_summary[
            "transient_model_count"
        ],
        "packing_generalization_seeds": comparison_summary["packing_seeds"],
        "fully_coupled_extension_sequence_count": coupled_summary["sequence_count"],
        "fully_coupled_extension_input_preparation_ready": coupled_summary[
            "source_full_state_initializer_present"
        ],
        "fully_coupled_openfoam_calculation_started": coupled_summary[
            "openfoam_calculation_started"
        ],
        **parameter_counts,
        "new_physical_parameters": [],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
