#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_fully_coupled_step_cases import (  # noqa: E402
    configure_fully_coupled_transient,
)
from verify_hccb_p418_fully_coupled_step_initialization import (  # noqa: E402
    boundary_scalar_values,
    inlet_mass_flow_consistency,
    internal_field_values,
)
from verify_hccb_p418_fully_coupled_step_plan import verify  # noqa: E402


def test_fully_coupled_plan_reuses_the_twelve_literature_endpoint_pairs() -> None:
    summary = verify(ROOT / "parameters/hccb_p418_fully_coupled_step_plan.json")
    assert summary["sequence_count"] == 12
    assert summary["same_endpoint_pairs_as_thermal_study"] is True
    assert summary["flow_and_momentum_enabled"] is True
    assert summary["source_full_state_initializer_present"] is True
    assert summary["target_inlet_mass_flux_initialization_present"] is True
    assert summary["target_boundary_and_heat_source_checks_present"] is True
    assert summary["formal_openfoam_runner_present"] is True
    assert summary["fully_coupled_restart_and_finalizer_present"] is True
    assert summary["time_dependent_full_state_exporter_present"] is True
    assert summary["integrated_observables_exporter_present"] is True
    assert summary["fixed_vs_fully_coupled_comparison_present"] is True
    assert summary["training_only_normalization_loader_present"] is True
    assert summary["full_state_graph_transformer_forward_present"] is True
    assert summary["new_physical_parameters"] == []


def test_fully_coupled_case_switches_on_flow_and_momentum(tmp_path: Path) -> None:
    for region in ("fluid", "solid"):
        path = tmp_path / "system" / region
        path.mkdir(parents=True, exist_ok=True)
        (path / "fvSchemes").write_text(
            "ddtSchemes\n{\n    default steadyState;\n}\n", encoding="utf-8"
        )
    (tmp_path / "system/fluid/fvSolution").write_text(
        "solvers\n{\n}\nPIMPLE\n{\n    flow no;\n    momentumPredictor no;\n}\n",
        encoding="utf-8",
    )
    configure_fully_coupled_transient(tmp_path)
    solution = (tmp_path / "system/fluid/fvSolution").read_text(encoding="utf-8")
    assert "flow yes;" in solution
    assert "momentumPredictor yes;" in solution
    for region in ("fluid", "solid"):
        assert "default Euler;" in (
            tmp_path / "system" / region / "fvSchemes"
        ).read_text(encoding="utf-8")


def test_fully_coupled_field_check_reads_scalar_and_vector_source_fields(
    tmp_path: Path,
) -> None:
    scalar = tmp_path / "T"
    scalar.write_text(
        "FoamFile{}\ninternalField nonuniform List<scalar> 3(300 301 302);\n"
        "boundaryField{}\n",
        encoding="utf-8",
    )
    vector = tmp_path / "U"
    vector.write_text(
        "FoamFile{}\ninternalField nonuniform List<vector> 2((0 0 0.1)(0 0 0.2));\n"
        "boundaryField{}\n",
        encoding="utf-8",
    )
    assert internal_field_values(scalar).tolist() == [300.0, 301.0, 302.0]
    assert internal_field_values(vector).tolist() == [0.0, 0.0, 0.1, 0.0, 0.0, 0.2]


def write_phi(path: Path, inlet_values: list[float]) -> None:
    values = "\n".join(f"        {value:.16e}" for value in inlet_values)
    path.write_text(
        "FoamFile{}\n"
        "internalField uniform 0;\n"
        "boundaryField\n"
        "{\n"
        "    inlet\n"
        "    {\n"
        "        type calculated;\n"
        f"        value nonuniform List<scalar> {len(inlet_values)}\n"
        "        (\n"
        f"{values}\n"
        "        );\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )


def test_fully_coupled_inlet_phi_check_detects_source_target_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_phi"
    target = tmp_path / "target_phi"
    inconsistent = tmp_path / "inconsistent_phi"
    consistent = tmp_path / "consistent_phi"
    write_phi(source, [-2.0e-8, -3.0e-8])
    write_phi(target, [-1.0e-7, -1.5e-7])
    write_phi(inconsistent, [-2.0e-8, -3.0e-8])
    write_phi(consistent, [-1.0e-7, -1.5e-7])

    assert boundary_scalar_values(target, "inlet").tolist() == [
        -1.0e-7,
        -1.5e-7,
    ]
    failed = inlet_mass_flow_consistency(inconsistent, source, target)
    assert failed["target_to_source_ratio"] == 5.0
    assert failed["initial_inlet_phi_matches_target_boundary"] is False
    passed = inlet_mass_flow_consistency(consistent, source, target)
    assert passed["case_to_target_relative_difference"] == 0.0
    assert passed["initial_inlet_phi_matches_target_boundary"] is True
