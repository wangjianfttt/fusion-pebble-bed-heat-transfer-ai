from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/compare_hccb_p418_cloud_reference.py"


def summary(*, pressure: float, outlet: float, inlet_temperature: float = 300.0) -> dict:
    return {
        "solver_finished": True,
        "physical_conditions": {
            "inlet_temperature_K": inlet_temperature,
            "inlet_velocity_m_s": 0.05,
            "cooling_wall_temperature_K": 635.0,
            "solid_heat_source_W_m3": 4.85e6,
        },
        "flow": {
            "pressure_drop_Pa": pressure,
            "relative_mass_difference": 6.9e-9,
        },
        "temperature": {
            "outlet_average_K": outlet,
            "solid_maximum_K": 634.93,
        },
        "heat_balance": {
            "cooling_wall_heat_flow_W": 0.12,
            "solid_generated_power_W": 0.31,
            "relative_energy_difference": 1.2e-5,
        },
    }


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reports_physical_differences_without_inventing_acceptance_limit(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.json"
    cloud = tmp_path / "cloud.json"
    output = tmp_path / "comparison.json"
    write(reference, summary(pressure=8.70, outlet=540.35))
    write(cloud, summary(pressure=8.72, outlet=540.31))
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--cloud",
            str(cloud),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "cloud_reference_result_comparison_complete"
    assert result["same_physical_inputs"] is True
    by_name = {row["quantity"]: row for row in result["result_comparison"]}
    assert abs(by_name["pressure_drop_Pa"]["signed_difference"] - 0.02) < 1.0e-12
    assert abs(
        by_name["outlet_average_temperature_K"]["signed_difference"] + 0.04
    ) < 1.0e-12
    assert "acceptance" not in result
    assert result["new_physical_parameters"] == []


def test_rejects_comparison_between_different_physical_conditions(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    cloud = tmp_path / "cloud.json"
    output = tmp_path / "comparison.json"
    write(reference, summary(pressure=8.70, outlet=540.35))
    write(
        cloud,
        summary(pressure=8.70, outlet=540.35, inlet_temperature=500.0),
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--cloud",
            str(cloud),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["same_physical_inputs"] is False
    assert result["status"] == "cloud_reference_result_incomplete_or_input_mismatch"
