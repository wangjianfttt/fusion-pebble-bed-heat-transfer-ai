#!/usr/bin/env python3
"""Check that the standalone CPU migration package keeps runtime inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_standalone_runtime_inputs_exist_and_match() -> None:
    packing_root = ROOT / "data/apd006_hccb_source_sequence_target_packings"
    records = json.loads(
        (packing_root / "packing_set_summary.json").read_text(encoding="utf-8")
    )
    assert [int(row["seed"]) for row in records] == [101, 202, 303]
    for row in records:
        seed = int(row["seed"])
        path = packing_root / f"seed{seed}_s80_xlo_ycentre/packing.npz"
        assert path.is_file()
        assert sha256(path) == row["packing_npz_sha256"]

    assert (
        ROOT / "literature/raw/numerical_methods/Celik_2008_GCI_crossref.json"
    ).is_file()
    assert (
        ROOT
        / "results/hccb_p418_inlet_dimensionless_envelope/inlet_dimensionless_conditions.csv"
    ).is_file()
    assert (
        ROOT / "results/hccb_p418_step_split_coverage/summary.json"
    ).is_file()
    assert (
        ROOT / "results/hccb_p418_high_re_independent_plan/summary.json"
    ).is_file()
    assert (
        ROOT / "results/hccb_p418_parameter_evidence/summary.json"
    ).is_file()
    parameter_use = ROOT / "results/hccb_p418_parameter_use/summary.json"
    assert parameter_use.is_file()
    parameter_use_summary = json.loads(parameter_use.read_text(encoding="utf-8"))
    assert parameter_use_summary["physical_parameter_count"] == 22
    assert parameter_use_summary["physical_parameters_used_by_equations"] == 22
    assert parameter_use_summary["model_numerical_setting_count"] == 78
    assert parameter_use_summary["experimental_templates_contain_no_measurements"]
    assert (
        ROOT
        / "results/hccb_p418_parameter_use/P418_参数怎样进入研究_CN.md"
    ).is_file()
    route_check = (
        ROOT / "results/hccb_p418_research_route_completeness/summary.json"
    )
    assert route_check.is_file()
    route_summary = json.loads(route_check.read_text(encoding="utf-8"))
    assert route_summary["scheme_complete"]
    assert not route_summary["formal_calculation_complete"]
    assert (
        ROOT
        / "results/hccb_p418_research_route_completeness/P418_研究方案完成情况_CN.md"
    ).is_file()
    assert (ROOT / "实验实施步骤_简明_CN.md").is_file()
    assert (
        ROOT / "results/hccb_p418_end_to_end_model_interface/summary.json"
    ).is_file()
    assert (
        ROOT / "results/hccb_p418_model_data_preparation/summary.json"
    ).is_file()
    assert (
        ROOT / "results/hccb_p418_end_to_end_plan/summary.json"
    ).is_file()
    assert (
        ROOT / "results/hccb_p418_sourceflow_partial_relations/summary.json"
    ).is_file()
    assert (
        ROOT
        / "results/hccb_p418_sourceflow_partial_relations/hccb_p418_partial_physics_relations.pdf"
    ).is_file()
    assert (
        ROOT / "results/hccb_p418_local_transport_model_support/summary.json"
    ).is_file()
    assert (
        ROOT
        / "results/hccb_p418_local_transport_model_support/local_transport_model_support.csv"
    ).is_file()
    sensitivity = (
        ROOT / "results/hccb_p418_local_transport_model_sensitivity/summary.json"
    )
    assert sensitivity.is_file()
    assert json.loads(sensitivity.read_text(encoding="utf-8"))["status"] == (
        "p418_local_transport_input_paths_confirmed"
    )


def test_package_builder_copies_standalone_runtime_inputs() -> None:
    source = (ROOT / "code/build_hccb_p418_cpu_code_package.sh").read_text(
        encoding="utf-8"
    )
    assert "data/apd006_hccb_source_sequence_target_packings" in source
    assert "literature/raw/numerical_methods/Celik_2008_GCI_crossref.json" in source
    assert "results/hccb_p418_inlet_dimensionless_envelope" in source
    assert "results/hccb_p418_step_split_coverage" in source
    assert "results/hccb_p418_high_re_independent_plan" in source
    assert "results/hccb_p418_parameter_evidence" in source
    assert "results/hccb_p418_parameter_use" in source
    assert "results/hccb_p418_research_route_completeness" in source
    assert "results/hccb_p418_end_to_end_model_interface" in source
    assert "results/hccb_p418_model_data_preparation" in source
    assert "results/hccb_p418_end_to_end_plan" in source
    assert "results/hccb_p418_sourceflow_partial_pressure_correlation" in source
    assert "results/hccb_p418_sourceflow_partial_boundary_heat" in source
    assert "results/hccb_p418_sourceflow_partial_dimensionless_heat_transfer_with_flux" in source
    assert "results/hccb_p418_sourceflow_partial_relations" in source
    assert "results/hccb_p418_local_transport_model_support" in source
    assert "results/hccb_p418_local_transport_model_sensitivity" in source
    assert '"${ROOT}/manuscript"' in source
    assert "CURRENT_STATUS_CN.md" in source
    assert "PROCESS_LOG_CN.md" in source
    assert "实验实施步骤_简明_CN.md" in source
    assert "研究主线_简明版_CN.md" in source
