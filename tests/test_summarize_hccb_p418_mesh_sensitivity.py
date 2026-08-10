from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/summarize_hccb_p418_mesh_sensitivity.py"
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_mesh_sensitivity import generalized_gci_triplet  # noqa: E402


def mesh(path: Path, cells: int, volume: float) -> None:
    payload = {
        "cell_volume_porosity": 0.3868,
        "fluid": {"cells": cells, "volume_m3": volume, "mesh_ok": True},
        "solid": {"cells": cells + 10, "volume_m3": 2.0 * volume, "mesh_ok": True},
        "checks": {"fluid_mesh_passes": True, "solid_mesh_passes": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def result(path: Path, pressure: float, outlet: float, maximum: float, wall: float) -> None:
    payload = {
        "physical_conditions": {
            "inlet_temperature_K": 700.0,
            "inlet_velocity_m_s": 0.2,
            "cooling_wall_temperature_K": 635.0,
            "solid_heat_source_W_m3": 6.85e6,
        },
        "flow": {"pressure_drop_Pa": pressure, "relative_mass_difference": 1e-8},
        "temperature": {"outlet_average_K": outlet, "solid_maximum_K": maximum},
        "heat_balance": {
            "cooling_wall_heat_flow_W": wall,
            "solid_generated_power_W": 0.24,
            "relative_energy_difference": 1e-5,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_three_mesh_summary_uses_temperature_changes_and_fine_reference(tmp_path: Path) -> None:
    paths = {}
    for index, level in enumerate(("coarse", "medium", "fine"), start=1):
        mesh_path = tmp_path / f"{level}_mesh.json"
        result_path = tmp_path / f"{level}_result.json"
        mesh(mesh_path, index * 100, 1e-8)
        result(result_path, 10.0 + index, 670.0 + index, 698.0 + index / 2, -0.08)
        paths[level] = (mesh_path, result_path)
    output = tmp_path / "out"
    command = [sys.executable, str(SCRIPT)]
    for level in ("coarse", "medium", "fine"):
        command.extend([f"--{level}-mesh", str(paths[level][0])])
        command.extend([f"--{level}-result", str(paths[level][1])])
    command.extend(["--output-dir", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed_three_mesh_p418_cht_comparison"
    assert summary["new_physical_parameters"] == []
    assert summary["mesh_levels"][2]["outlet_temperature_change_K"] == -27.0
    assert summary["mesh_levels"][2][
        "pressure_drop_Pa_relative_difference_from_fine"
    ] == 0.0
    assert (output / "engineering_observables.csv").is_file()
    assert (output / "mesh_gci.csv").is_file()
    assert len(summary["grid_convergence"]) == 4


def test_unequal_refinement_gci_recovers_known_second_order_sequence() -> None:
    h_fine = 1.0
    h_medium = 1.25
    h_coarse = 1.75
    exact = 10.0
    coefficient = 0.5
    order = 2.0
    result = generalized_gci_triplet(
        exact + coefficient * h_coarse**order,
        exact + coefficient * h_medium**order,
        exact + coefficient * h_fine**order,
        h_coarse,
        h_medium,
        h_fine,
    )
    assert result["convergence_status"] == "monotonic_gci_reported"
    assert result["observed_order"] == pytest.approx(2.0, rel=1.0e-8)
    assert result["richardson_extrapolated_value"] == pytest.approx(exact)


def test_oscillatory_mesh_sequence_does_not_force_gci() -> None:
    result = generalized_gci_triplet(10.0, 11.0, 10.5, 1.75, 1.25, 1.0)
    assert result["convergence_status"] == "oscillatory_no_gci_reported"
    assert result["fine_gci_fraction"] is None


def test_zero_crossing_mesh_sequence_does_not_report_percentage_gci() -> None:
    result = generalized_gci_triplet(-0.14, 0.05, 0.16, 1.75, 1.25, 1.0)
    assert result["convergence_status"] == "zero_crossing_no_gci_reported"
    assert result["observed_order"] is None
    assert result["fine_gci_fraction"] is None
    assert result["fine_gci_absolute"] is None
