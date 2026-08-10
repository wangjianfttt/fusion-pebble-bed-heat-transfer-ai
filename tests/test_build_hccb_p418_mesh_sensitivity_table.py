from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_mesh_sensitivity_table.py"


def test_mesh_table_reports_values_and_does_not_force_oscillatory_gci(
    tmp_path: Path,
) -> None:
    levels = []
    for index, level in enumerate(("coarse", "medium", "fine")):
        levels.append(
            {
                "mesh_level": level,
                "total_cells": (index + 1) * 100000,
                "equivalent_cell_size_m": [5.4e-5, 4.0e-5, 3.1e-5][index],
                "cell_volume_porosity": 0.386 + 0.0004 * index,
                "pressure_drop_Pa": [12.0, 11.0, 10.5][index],
                "outlet_temperature_change_K": [30.0, 29.0, 28.5][index],
                "solid_maximum_temperature_change_K": [40.0, 41.0, 40.5][index],
                "cooling_wall_heat_fraction": [-0.4, -0.35, -0.33][index],
            }
        )
    convergence = []
    for metric in (
        "pressure_drop_Pa",
        "outlet_temperature_change_K",
        "solid_maximum_temperature_change_K",
        "cooling_wall_heat_fraction",
    ):
        convergence.append(
            {
                "metric": metric,
                "convergence_status": (
                    "oscillatory_no_gci_reported"
                    if metric == "solid_maximum_temperature_change_K"
                    else "monotonic_gci_reported"
                ),
                "observed_order": (
                    None if metric == "solid_maximum_temperature_change_K" else 1.5
                ),
                "fine_gci_fraction": (
                    None if metric == "solid_maximum_temperature_change_K" else 0.012
                ),
            }
        )
    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "status": "completed_three_mesh_p418_cht_comparison",
                "mesh_levels": levels,
                "grid_convergence": convergence,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "mesh.tex"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-summary",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
    )
    text = output.read_text(encoding="utf-8")
    assert "Three-mesh sensitivity" in text
    assert "300000" in text
    assert "31.00" in text
    assert "1.200\\%" in text
    assert "oscillatory" in text
    assert "no percentage GCI is reported" in text
    assert "when the response crosses zero" in text
