import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_regional_fidelity_text.py"


def test_completed_summaries_generate_manuscript_text(tmp_path: Path) -> None:
    representation = {
        "status": "regional_representation_fidelity_ready",
        "counts": {
            "cases": 60,
            "native_fluid_cells": 1000,
            "native_solid_cells": 1200,
            "regional_fluid_nodes": 100,
            "regional_solid_nodes": 120,
        },
        "compression_ratio": {"total_native_cells_per_regional_node": 10.0},
        "exact_saved_regional_state_matches_direct_volume_average": True,
        "metrics": {
            "fluid_volume_weighted_rmse_K": {"mean": 12.0},
            "solid_volume_weighted_rmse_K": {"mean": 8.0},
            "fluid_rmse_over_native_range_percent": {"mean": 5.0},
            "solid_rmse_over_native_range_percent": {"mean": 3.0},
            "solid_hotspot_temperature_loss_K": {"mean": 0.4},
            "solid_hotspot_nearest_cell_distance_dp": {"mean": 0.8},
        },
        "hottest_native_cell_region_match_fraction": 0.5,
        "new_physical_parameters": [],
    }
    reconstruction = {
        "status": "native_reconstruction_comparison_ready",
        "case_count": 60,
        "metrics": {
            "fluid_affine_volume_weighted_rmse_K": {"mean": 4.0},
            "solid_affine_volume_weighted_rmse_K": {"mean": 2.0},
            "solid_affine_max_temperature_error_K": {"mean": 500.0},
            "solid_affine_hotspot_distance_dp": {"mean": 2.0},
            "fluid_limited_volume_weighted_rmse_K": {"mean": 4.5},
            "solid_limited_volume_weighted_rmse_K": {"mean": 2.2},
            "solid_constant_hotspot_distance_dp": {"mean": 1.0},
            "solid_limited_hotspot_distance_dp": {"mean": 0.6},
            "fluid_limited_variance_reduction_percent": {"mean": 80.0},
            "solid_limited_variance_reduction_percent": {"mean": 90.0},
        },
        "new_physical_parameters": [],
    }
    rep_path = tmp_path / "representation.json"
    rec_path = tmp_path / "reconstruction.json"
    output = tmp_path / "result.tex"
    summary = tmp_path / "summary.json"
    rep_path.write_text(json.dumps(representation), encoding="utf-8")
    rec_path.write_text(json.dumps(reconstruction), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--representation-summary",
            str(rep_path),
            "--reconstruction-summary",
            str(rec_path),
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
    assert "Across the 60 completed fields" in text
    assert "hotspot magnitude" in text
    assert "\\SI{500}{K}" in text
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["case_count"] == 60
    assert payload["new_physical_parameters"] == []


def test_postprocess_generates_the_text() -> None:
    runner = (ROOT / "code/run_hccb_p418_60_postprocess.sh").read_text(
        encoding="utf-8"
    )
    assert "build_hccb_p418_regional_fidelity_text.py" in runner
    manuscript = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    assert "generated_regional_fidelity_formal.tex" in manuscript


def test_postprocess_recomputes_time_scales_from_corrected_fast_field() -> None:
    runner = (ROOT / "code/run_hccb_p418_60_postprocess.sh").read_text(
        encoding="utf-8"
    )
    assert "analyze_hccb_p418_velocity_step_time_scales.py" in runner
    assert "fields/u0p25_T300_q4p85.npz" in runner
    assert '--sourceflow-input-summary "${INPUT_CHECK_DIR}/summary.json"' in runner
