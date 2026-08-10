from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/plot_hccb_p418_physical_response.py"


def write_matrix(path: Path, count: int = 60) -> None:
    rows = []
    for velocity in (0.05, 0.10, 0.15, 0.20, 0.25):
        for temperature in (300.0, 500.0, 700.0, 900.0):
            for source in (4.85, 6.85, 8.85):
                rows.append(
                    {
                        "condition_id": f"u{velocity}_T{temperature}_q{source}",
                        "inlet_velocity_m_s": velocity,
                        "inlet_temperature_K": temperature,
                        "solid_heat_source_MW_m3": source,
                        "pressure_drop_Pa": 20.0 * velocity + 0.001 * temperature,
                        "outlet_temperature_K": temperature + source / velocity,
                        "solid_maximum_temperature_K": temperature + 2.0 * source / velocity,
                        "cooling_wall_heat_over_generated": (635.0 - temperature) / 1000.0,
                        "relative_mass_difference": 1.0e-8,
                        "relative_energy_difference": 2.0e-6,
                    }
                )
    rows = rows[:count]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_plot_requires_and_renders_complete_matrix(tmp_path: Path):
    source = tmp_path / "physics.csv"
    output = tmp_path / "figures"
    write_matrix(source)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--physical-csv", str(source), "--output-dir", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "hccb_p418_physical_response.pdf").is_file()
    assert (output / "hccb_p418_physical_response.png").is_file()
    assert (output / "hccb_p418_physical_response.json").is_file()
    summary = json.loads((output / "hccb_p418_physical_response.json").read_text())
    assert summary["figure_size_inch"] == [5.4, 6.7]
    assert all(
        abs(actual - expected) < 1.0e-9
        for actual, expected in zip(summary["figure_size_mm"], [137.16, 170.18])
    )
    assert 1.15 <= summary["panel_width_to_height_ratio"] <= 1.35


def test_plot_rejects_partial_matrix(tmp_path: Path):
    source = tmp_path / "partial.csv"
    write_matrix(source, count=17)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--physical-csv", str(source), "--output-dir", str(tmp_path / "out")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires all 60 conditions" in result.stderr
