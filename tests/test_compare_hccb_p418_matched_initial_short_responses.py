from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/compare_hccb_p418_matched_initial_short_responses.py"
SIGNALS = (
    "inlet_temperature_K",
    "outlet_temperature_K",
    "inlet_pressure_Pa",
    "outlet_pressure_Pa",
    "inlet_mass_flow_kg_s",
    "outlet_mass_flow_kg_s",
    "inlet_enthalpy_flow_W",
    "outlet_enthalpy_flow_W",
    "cooling_wall_power_W",
    "maximum_solid_temperature_K",
    "volume_average_fluid_temperature_K",
    "volume_average_solid_temperature_K",
    "pressure_drop_Pa",
    "signed_mass_residual_kg_s",
    "net_outward_enthalpy_flow_W",
)


def write_response(path: Path, scale: float) -> None:
    rows = []
    for index, time_s in enumerate((0.0, 0.005, 0.01)):
        row = {"time_s": time_s}
        for signal_index, name in enumerate(SIGNALS, start=1):
            row[name] = 100.0 * signal_index + scale * index
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("time_s", *SIGNALS))
        writer.writeheader()
        writer.writerows(rows)


def test_matched_initial_short_comparison(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.csv"
    coupled = tmp_path / "coupled.csv"
    output = tmp_path / "comparison.json"
    write_response(fixed, 1.0)
    write_response(coupled, 1.5)
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--fixed-flow-csv",
            str(fixed),
            "--fully-coupled-csv",
            str(coupled),
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text())
    assert payload["status"].endswith("short_comparison_complete")
    assert payload["signal_count"] == 15
    assert payload["fully_coupled_time_point_count"] == 3
    outlet = payload["signals"]["outlet_temperature_K"]
    assert outlet["initial_difference"] == 0.0
    assert outlet["fixed_change"] == 2.0
    assert outlet["coupled_change"] == 3.0
    assert outlet["change_relative_difference"] == 0.5
    assert payload["openfoam_solver_started_by_this_comparison"] is False
    assert payload["new_physical_parameters"] == []
