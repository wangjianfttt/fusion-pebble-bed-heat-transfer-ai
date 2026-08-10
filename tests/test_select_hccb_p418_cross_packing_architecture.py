from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/select_hccb_p418_cross_packing_architecture.py"
METRICS = (
    "fluid_temperature_volume_weighted_rmse_K",
    "solid_temperature_volume_weighted_rmse_K",
    "solid_hotspot_location_error_m",
    "engineering_absolute_errors.pressure_drop_Pa",
    "engineering_absolute_errors.cooling_wall_heat_into_fluid_W",
    "local_mass_l1_over_two_inlet",
    "local_energy_l1_over_two_generated_power",
)


def run(architecture: str, values: dict[str, float]) -> dict[str, object]:
    return {
        "packing_seed": 202,
        "architecture": architecture,
        "metrics": {
            name: {"mean": value * 0.8, "p95": value, "maximum": value * 1.1}
            for name, value in values.items()
        },
    }


def base(value: float) -> dict[str, float]:
    return {name: value for name in METRICS}


def execute(tmp_path: Path, runs: list[dict[str, object]]):
    source = tmp_path / "summary.json"
    output = tmp_path / "selection.json"
    chinese = tmp_path / "selection_CN.md"
    source.write_text(
        json.dumps(
            {"status": "cross_packing_model_summary_complete", "runs": runs}
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(source),
            "--output",
            str(output),
            "--chinese-output",
            str(chinese),
        ],
        capture_output=True,
        text=True,
    )
    return result, output, chinese


def test_selects_non_dominated_model_with_best_solid_temperature(tmp_path: Path):
    first = base(2.0)
    first["solid_temperature_volume_weighted_rmse_K"] = 1.0
    second = base(1.0)
    second["solid_temperature_volume_weighted_rmse_K"] = 1.5
    dominated = base(3.0)
    result, output, chinese = execute(
        tmp_path,
        [run("pinn", first), run("graph", second), run("transolver", dominated)],
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_architecture"] == "pinn"
    assert payload["pareto_architectures"] == ["graph", "pinn"]
    assert payload["seed303_fields_read"] is False
    assert payload["composite_score_used"] is False
    assert "尚未读取seed303" in chinese.read_text(encoding="utf-8")


def test_rejects_seed303_input(tmp_path: Path):
    record = run("pinn", base(1.0))
    record["packing_seed"] = 303
    result, _, _ = execute(tmp_path, [record])
    assert result.returncode != 0
    assert "seed202 results only" in result.stderr
