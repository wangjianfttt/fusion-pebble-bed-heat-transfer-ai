from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/plot_hccb_p418_cross_packing_results.py"
ARCHITECTURES = ("pinn_data_only", "pinn", "graph", "transolver")
METRICS = (
    "fluid_temperature_volume_weighted_rmse_K",
    "solid_temperature_volume_weighted_rmse_K",
    "solid_hotspot_location_error_m",
    "engineering_absolute_errors.pressure_drop_Pa",
    "engineering_absolute_errors.cooling_wall_heat_into_fluid_W",
    "local_mass_l1_over_two_inlet",
    "local_energy_l1_over_two_generated_power",
)


def metric_rows(scale: float) -> dict:
    return {
        name: {"mean": scale, "median": scale, "p95": scale, "maximum": scale}
        for name in METRICS
    }


def run(seed: int, architecture: str, scale: float) -> dict:
    return {
        "source_file": f"seed{seed}_{architecture}.json",
        "source_sha256": f"sha_{seed}_{architecture}",
        "packing_seed": seed,
        "packing_role": "development_packing" if seed == 202 else "final_zero_shot_packing",
        "architecture": architecture,
        "condition_ids": [f"case_{index}" for index in range(9)],
        "metrics": metric_rows(scale),
    }


def write_fixture(root: Path, *, selected: str = "graph") -> tuple[Path, Path, Path]:
    development = root / "development.json"
    selection = root / "selection.json"
    final = root / "final.json"
    development_runs = [run(202, architecture, index + 1.0) for index, architecture in enumerate(ARCHITECTURES)]
    development.write_text(
        json.dumps(
            {
                "status": "cross_packing_model_summary_complete",
                "runs": development_runs,
            }
        ),
        encoding="utf-8",
    )
    selection.write_text(
        json.dumps(
            {
                "status": "seed202_architecture_fixed_before_seed303",
                "selected_architecture": selected,
                "seed303_fields_read": False,
                "composite_score_used": False,
            }
        ),
        encoding="utf-8",
    )
    selected_development = next(row for row in development_runs if row["architecture"] == selected)
    final.write_text(
        json.dumps(
            {
                "status": "cross_packing_model_summary_complete",
                "runs": [selected_development, run(303, selected, 1.5 * selected_development["metrics"][METRICS[0]]["p95"])],
            }
        ),
        encoding="utf-8",
    )
    return development, selection, final


def test_renders_complete_cross_packing_figure(tmp_path: Path) -> None:
    development, selection, final = write_fixture(tmp_path)
    output = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--development-summary",
            str(development),
            "--selection",
            str(selection),
            "--final-summary",
            str(final),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "hccb_p418_cross_packing_results.pdf").is_file()
    assert (output / "hccb_p418_cross_packing_results.svg").is_file()
    assert (output / "hccb_p418_cross_packing_results.png").is_file()
    svg_text = (
        output / "hccb_p418_cross_packing_results.svg"
    ).read_text(encoding="utf-8")
    assert "<text" in svg_text
    assert "<image" not in svg_text
    summary = json.loads(
        (output / "hccb_p418_cross_packing_results.json").read_text(encoding="utf-8")
    )
    assert summary["selected_architecture"] == "graph"
    assert summary["seed303_fields_read_during_selection"] is False
    assert summary["new_physical_parameter_values_added"] == []


def test_rejects_final_architecture_changed_after_seed202(tmp_path: Path) -> None:
    development, selection, final = write_fixture(tmp_path)
    payload = json.loads(final.read_text(encoding="utf-8"))
    payload["runs"][1]["architecture"] = "pinn"
    final.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--development-summary",
            str(development),
            "--selection",
            str(selection),
            "--final-summary",
            str(final),
            "--output-dir",
            str(tmp_path / "figures"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "does not use the architecture fixed on seed202" in result.stderr
