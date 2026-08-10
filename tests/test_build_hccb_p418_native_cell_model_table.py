import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_native_cell_model_table.py"
MODELS = ("response_surface", "pinn_data_only", "pinn", "graph_operator", "transolver")


def test_native_cell_table_contains_all_formal_models(tmp_path: Path) -> None:
    rows = []
    for index, model in enumerate(MODELS, start=1):
        rows.append(
            {
                "model": model,
                "fluid_limited_native_total_rmse_K_mean": 10.0 + index,
                "solid_limited_native_total_rmse_K_mean": 5.0 + index,
                "limited_predicted_solid_max_temperature_error_K_mean": 2.0 + index,
                "limited_predicted_hotspot_distance_dp_mean": 0.1 * index,
            }
        )
    source = tmp_path / "native.json"
    source.write_text(
        json.dumps(
            {
                "status": "native_cell_model_comparison_ready",
                "rows": rows,
                "new_physical_parameters": [],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "table.tex"
    summary = tmp_path / "table.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--comparison-summary",
            str(source),
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
    assert "Data-only PINN" in text
    assert "Physics-informed PINN" in text
    assert "Physics-Attention" in text
    assert "Hotspot shift" in text
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 5
    assert payload["new_physical_parameters"] == []


def test_formal_routes_require_native_cell_table() -> None:
    comparison = (ROOT / "code/run_hccb_p418_60_model_comparison.sh").read_text(
        encoding="utf-8"
    )
    refresh = (ROOT / "code/run_hccb_p418_manuscript_refresh.sh").read_text(
        encoding="utf-8"
    )
    manuscript = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    assert "build_hccb_p418_native_cell_model_table.py" in comparison
    assert "generated_native_cell_performance.tex" in refresh
    assert "generated_native_cell_performance.tex" in manuscript
