from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "run_hccb_p418_sourceflow_after_preflight.sh"


def test_sourceflow_continuation_has_valid_shell_syntax() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_sourceflow_continuation_covers_the_declared_calculation_route() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "source_channel_volume_flow_preserved",
        "maximum_relative_mass_difference",
        "maximum_relative_energy_difference",
        "analyze_hccb_p418_pressure_correlation.py",
        "maximum_superficial_vs_source_channel_velocity_difference_fraction",
        "maximum_boundary_vs_reported_pressure_difference_fraction",
        "published_pressure_relation_absolute_difference_percent",
        "record_exit_status",
        "automatic continuation stopped with return code",
        "formal_input_check_is_current",
        "reusing unchanged verified formal inputs",
        "all_openfoam_dictionary_values_match_registered_sources",
        "verify_hccb_p418_actual_case_inputs.py",
        "run_hccb_dense_cht_p418_matrix_parallel.sh",
        "run_hccb_p418_formal_calculations.sh",
        'P418_PYTHON="${PYTHON}"',
        "EXECUTE=1",
    ):
        assert required in text


def test_sourceflow_continuation_uses_the_corrected_matrix() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "hccb_dense_cht_p418_60_sourceflow_r3" in text
    assert "hccb_dense_cht_p418_60_r2" not in text


def test_preflight_summary_does_not_reload_the_openfoam_shell_environment() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "source /opt/openfoam13/etc/bashrc" not in text
    assert "boundary_pressure_error > 1.0e-4" in text


def test_cached_input_check_ignores_solver_outputs_but_tracks_actual_inputs() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "written time directories do not invalidate the input check" in text
    for required_input in (
        "cht_smoke_metadata.json",
        "*/0/fluid/U",
        "*/0/fluid/T",
        "*/0/fluid/p",
        "*/0/solid/T",
        "*/constant/fluid/physicalProperties",
        "*/constant/solid/physicalProperties",
        "*/constant/solid/fvModels",
        "hccb_p418_physical_parameter_sources.csv",
    ):
        assert required_input in text
