import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_cross_packing_result_text.py"
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


def run(seed: int, architecture: str, scale: float, source: str) -> dict:
    values = {
        "fluid_temperature_volume_weighted_rmse_K": 4.0,
        "solid_temperature_volume_weighted_rmse_K": 3.0,
        "solid_hotspot_location_error_m": 0.001,
        "engineering_absolute_errors.pressure_drop_Pa": 2.0,
        "engineering_absolute_errors.cooling_wall_heat_into_fluid_W": 0.3,
        "local_mass_l1_over_two_inlet": 0.02,
        "local_energy_l1_over_two_generated_power": 0.03,
    }
    return {
        "packing_seed": seed,
        "architecture": architecture,
        "source_sha256": source,
        "metrics": {
            metric: {"p95": values[metric] * scale} for metric in METRICS
        },
    }


def test_cross_packing_text_reports_frozen_model_transfer(tmp_path: Path) -> None:
    development_runs = [
        run(202, architecture, 1.0 + 0.1 * index, f"source-{architecture}")
        for index, architecture in enumerate(ARCHITECTURES)
    ]
    development = tmp_path / "development.json"
    development.write_text(
        json.dumps(
            {
                "status": "cross_packing_model_summary_complete",
                "runs": development_runs,
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "seed202_architecture_fixed_before_seed303",
                "selected_architecture": "pinn",
                "pareto_architectures": ["pinn", "graph"],
                "seed303_fields_read": False,
                "composite_score_used": False,
            }
        ),
        encoding="utf-8",
    )
    selected_202 = next(row for row in development_runs if row["architecture"] == "pinn")
    selected_303 = run(303, "pinn", 1.65, "source-seed303")
    final = tmp_path / "final.json"
    final.write_text(
        json.dumps(
            {
                "status": "cross_packing_model_summary_complete",
                "runs": [selected_202, selected_303],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "text.tex"
    summary = tmp_path / "text.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--development-summary",
            str(development),
            "--selection",
            str(selection),
            "--final-summary",
            str(final),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text = output.read_text(encoding="utf-8")
    assert "physics-informed PINN" in text
    assert "no composite" in text
    assert "seed303-to-seed202" in text
    assert "not a blanket-module prediction" in text
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["selected_architecture"] == "pinn"
    assert len(payload["seed303_to_seed202_p95_ratios"]) == 7
    assert payload["new_physical_parameters"] == []


def test_formal_route_generates_and_requires_cross_packing_text() -> None:
    stage = (ROOT / "code/run_hccb_p418_cross_packing_model_stage.sh").read_text(
        encoding="utf-8"
    )
    formal = (ROOT / "code/run_hccb_p418_formal_calculations.sh").read_text(
        encoding="utf-8"
    )
    refresh = (ROOT / "code/run_hccb_p418_manuscript_refresh.sh").read_text(
        encoding="utf-8"
    )
    manuscript = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    assert "build_hccb_p418_cross_packing_result_text.py" in stage
    assert "cross_packing_manuscript_text_sha256" in formal
    assert "generated_cross_packing_result_text.tex" in refresh
    assert "generated_cross_packing_result_text.tex" in manuscript
