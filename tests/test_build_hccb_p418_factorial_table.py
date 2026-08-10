from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_factorial_table.py"
OBSERVABLES = (
    "pressure_drop_Pa",
    "outlet_temperature_K",
    "solid_maximum_temperature_K",
    "net_outward_enthalpy_flow_W",
    "cooling_wall_heat_into_fluid_W",
)
EFFECTS = {
    "inlet_velocity": 35.0,
    "inlet_temperature": 40.0,
    "solid_heat_source": 15.0,
    "velocity_x_temperature": 4.0,
    "velocity_x_heat_source": 3.0,
    "temperature_x_heat_source": 2.0,
    "velocity_x_temperature_x_heat_source": 1.0,
}


def write_physics_summary(path: Path, completed: int = 60) -> None:
    records = []
    for observable in OBSERVABLES:
        for effect, fraction in EFFECTS.items():
            records.append(
                {
                    "observable": observable,
                    "effect": effect,
                    "variance_fraction_percent": fraction,
                }
            )
    path.write_text(
        json.dumps(
            {
                "status": "completed_p418_case_physics_summarized",
                "completed_case_count": completed,
                "complete_factorial_decomposition_available": completed == 60,
                "factorial_variance_decomposition": records,
            }
        ),
        encoding="utf-8",
    )


def test_builds_complete_factorial_table(tmp_path: Path) -> None:
    source = tmp_path / "physics.json"
    output = tmp_path / "table.tex"
    summary = tmp_path / "table.json"
    write_physics_summary(source)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--physics-summary",
            str(source),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text = output.read_text(encoding="utf-8")
    assert "Full-factorial variation" in text
    assert "Maximum solid temperature & 35.00 & 40.00 & 15.00 & 10.00" in text
    assert "not statistical uncertainty estimates" in text
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["completed_case_count"] == 60
    assert len(payload["rows"]) == 5
    assert payload["new_physical_parameters"] == []


def test_rejects_partial_matrix(tmp_path: Path) -> None:
    source = tmp_path / "physics.json"
    write_physics_summary(source, completed=59)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--physics-summary",
            str(source),
            "--output",
            str(tmp_path / "table.tex"),
            "--summary-output",
            str(tmp_path / "table.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires all 60 steady conditions" in result.stderr


def test_formal_route_includes_factorial_table() -> None:
    postprocess = (ROOT / "code/run_hccb_p418_60_postprocess.sh").read_text(
        encoding="utf-8"
    )
    poststeady = (ROOT / "code/run_hccb_p418_poststeady_pipeline.sh").read_text(
        encoding="utf-8"
    )
    refresh = (ROOT / "code/run_hccb_p418_manuscript_refresh.sh").read_text(
        encoding="utf-8"
    )
    manuscript = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    assert "build_hccb_p418_factorial_table.py" in postprocess
    assert "generated_steady_factorial_effects.tex" in poststeady
    assert "generated_steady_factorial_effects.tex" in refresh
    assert "generated_steady_factorial_effects.tex" in manuscript
