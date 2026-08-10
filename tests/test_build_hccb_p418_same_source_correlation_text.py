import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_same_source_correlation_text.py"


def test_same_source_text_keeps_whole_bed_and_local_crop_separate(tmp_path: Path) -> None:
    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "status": "p418_dimensionless_heat_transfer_comparison_complete",
                "case_count": 60,
                "p419_positive_phase_difference_case_count": 36,
                "p419_nonpositive_phase_difference_case_count": 24,
                "p417_p419_in_range_comparable_case_count": 35,
                "p417_reference_within_source_30_percent_case_count": 8,
                "p417_reference_within_source_30_percent_fraction": 8 / 35,
                "maximum_absolute_in_range_correlation_difference_percent": 91.2,
                "reynolds_axial_throughflow_range": [0.09, 1.93],
                "prandtl_mean_properties_range": [0.662, 0.669],
                "openfoam_interface_flux_nusselt_range": [0.4, 8.7],
                "openfoam_interface_flux_case_count": 60,
                "openfoam_interface_flux_sign_consistent_case_count": 58,
                "maximum_absolute_openfoam_solid_energy_partition_error_over_generated": 0.0004,
                "parameter_ids": ["P048", "P417", "P419"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.tex"
    summary = tmp_path / "result.json"
    pressure = tmp_path / "pressure.json"
    pressure.write_text(
        json.dumps(
            {
                "status": "p418_local_crop_pressure_correlation_complete",
                "case_count": 60,
                "median_absolute_difference_percent": 1.2,
                "maximum_absolute_difference_percent": 16.4,
                "inside_source_P422_case_count": 44,
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-summary",
            str(source),
            "--pressure-summary",
            str(pressure),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        check=True,
    )
    text = output.read_text(encoding="utf-8")
    assert "60 local-crop fields" in text
    assert "whole-bed P417 Nusselt correlation" in text
    assert "not removed by refitting" in text
    assert "rather than a local-field label" in text
    assert "Direct integration of the finite-volume fluid--solid interface" in text
    assert "58 of 60 cases" in text
    assert "supplies the local finite-volume heat-transfer target" in text
    assert "superficial" in text
    record = json.loads(summary.read_text(encoding="utf-8"))
    assert record["comparable_case_count"] == 35
    assert record["new_physical_parameters"] == []


def test_same_source_text_rejects_partial_matrix(tmp_path: Path) -> None:
    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "status": "p418_dimensionless_heat_transfer_comparison_complete",
                "case_count": 32,
                "parameter_ids": ["P417", "P419"],
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-summary",
            str(source),
            "--pressure-summary",
            str(tmp_path / "missing-pressure.json"),
            "--output",
            str(tmp_path / "result.tex"),
            "--summary",
            str(tmp_path / "result.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires all 60" in result.stderr


def test_formal_routes_generate_and_require_same_source_text() -> None:
    postprocess = (ROOT / "code/run_hccb_p418_60_postprocess.sh").read_text(
        encoding="utf-8"
    )
    refresh = (ROOT / "code/run_hccb_p418_manuscript_refresh.sh").read_text(
        encoding="utf-8"
    )
    manuscript = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    assert "build_hccb_p418_same_source_correlation_text.py" in postprocess
    assert "generated_same_source_correlation.tex" in refresh
    assert "generated_same_source_correlation.tex" in manuscript
