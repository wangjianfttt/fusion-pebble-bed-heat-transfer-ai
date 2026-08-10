import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_steady_result_text.py"
METHODS = ("response_surface", "pinn_data_only", "pinn", "graph", "transolver")
SPLITS = (
    "interleaved_all_ranges",
    "temperature_extrapolation",
    "velocity_extrapolation",
    "heat_source_interpolation",
    "heat_source_extrapolation",
)


def test_steady_text_uses_worst_complete_split_and_no_composite_score(tmp_path: Path) -> None:
    rows = []
    for method_index, method in enumerate(METHODS, start=1):
        for split_index, split in enumerate(SPLITS, start=1):
            scale = float(method_index * split_index)
            rows.append(
                {
                    "architecture": method,
                    "split": split,
                    "test_fluid_temperature_normalized_rmse": 0.01 * scale,
                    "test_solid_temperature_normalized_rmse": 0.02 * scale,
                    "test_pressure_drop_p95_Pa": 2.0 * scale,
                    "test_solid_maximum_temperature_p95_K": 3.0 * scale,
                    "test_cooling_wall_heat_over_generated_p95_percent": 0.5 * scale,
                    "test_local_energy_l1_over_two_generated_power_mean": 0.001 * scale,
                }
            )
    source = tmp_path / "comparison.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    coverage = tmp_path / "thermal_regime_split_coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "status": "thermal_regime_split_coverage_complete",
                "coverage_complete": True,
                "rows": [
                    {
                        "split": "temperature_extrapolation",
                        "role": "train",
                        "case_count": 30,
                        "coverage_complete": True,
                        "unknown_case_count": 0,
                        "wall_to_fluid_count": 30,
                        "fluid_to_wall_count": 0,
                        "zero_wall_heat_count": 0,
                    },
                    {
                        "split": "temperature_extrapolation",
                        "role": "validation",
                        "case_count": 15,
                        "coverage_complete": True,
                        "unknown_case_count": 0,
                        "wall_to_fluid_count": 0,
                        "fluid_to_wall_count": 15,
                        "zero_wall_heat_count": 0,
                    },
                    {
                        "split": "temperature_extrapolation",
                        "role": "test",
                        "case_count": 15,
                        "coverage_complete": True,
                        "unknown_case_count": 0,
                        "wall_to_fluid_count": 0,
                        "fluid_to_wall_count": 15,
                        "zero_wall_heat_count": 0,
                    },
                    {
                        "split": "interleaved_all_ranges",
                        "role": "test",
                        "case_count": 12,
                        "coverage_complete": True,
                        "unknown_case_count": 0,
                        "wall_to_fluid_count": 6,
                        "fluid_to_wall_count": 6,
                        "zero_wall_heat_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.tex"
    summary = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--comparison-csv",
            str(source),
            "--thermal-regime-coverage",
            str(coverage),
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
    assert "five complete-condition splits" in text
    assert "not combined into a scalar model score" in text
    assert "data-only and physics-informed PINNs" in text
    assert "temperature-extrapolation split" in text
    assert "training 30 conditions (30 wall-to-fluid, 0 fluid-to-wall)" in text
    assert "independent prediction 15 conditions (0 wall-to-fluid, 15 fluid-to-wall)" in text
    assert "not only interpolation between nearby inlet temperatures" in text
    assert "contain one training source level" in text
    assert "not learned source sensitivity" in text
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["split_count"] == 5
    assert payload["thermal_regime_coverage"] == str(coverage.resolve())
    assert len(payload["leaders_by_metric"]) == 6
    assert payload["new_physical_parameters"] == []


def test_formal_routes_generate_and_require_steady_text() -> None:
    comparison = (ROOT / "code/run_hccb_p418_60_model_comparison.sh").read_text(
        encoding="utf-8"
    )
    poststeady = (ROOT / "code/run_hccb_p418_poststeady_pipeline.sh").read_text(
        encoding="utf-8"
    )
    refresh = (ROOT / "code/run_hccb_p418_manuscript_refresh.sh").read_text(
        encoding="utf-8"
    )
    manuscript = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    results = (ROOT / "manuscript/results_condensed.tex").read_text(encoding="utf-8")
    assert "build_hccb_p418_steady_result_text.py" in comparison
    assert "--thermal-regime-coverage" in comparison
    assert "steady_result_text=" in poststeady
    assert "generated_steady_result_text.tex" in refresh
    assert "\\input{results_condensed}" in manuscript
    assert "generated_steady_result_text.tex" in results
    assert "generated_steady_model_comparison_validated.tex" in results
    assert "generated_steady_model_comparison_validated.tex" in (
        ROOT / "code/run_hccb_p418_60_model_postprocess_only.sh"
    ).read_text(encoding="utf-8")
    assert "generated_steady_model_comparison_validated.tex" in comparison
