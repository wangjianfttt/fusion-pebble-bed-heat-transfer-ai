from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_formal_calculations.sh"


def test_formal_runner_is_dry_by_default(tmp_path: Path):
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={"ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "dry run only" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_formal_runner_preserves_physical_sequence():
    text = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    seed101 = text.index("run_hccb_p418_poststeady_pipeline.sh")
    full_timestep = text.index(
        "run_hccb_p418_fully_coupled_timestep_sensitivity.sh"
    )
    full_steps = text.index("run_hccb_p418_fully_coupled_step_responses.sh")
    fixed_full_comparison = text.index(
        "compare_hccb_p418_fixed_and_fully_coupled_steps.py"
    )
    full_model = text.index("run_hccb_p418_fully_coupled_model_stage.sh")
    setup = text.index("run_hccb_p418_cross_packing_setup.sh")
    seed202 = text.index('SEED=202 EXECUTE=1')
    development = text.index("STAGE=development EXECUTE=1")
    seed303 = text.index('SEED=303 EXECUTE=1')
    final = text.index("STAGE=final EXECUTE=1")
    high_re_fixed = text.index("MODE=fixed EXECUTE=1")
    high_re_full = text.index("MODE=fully_coupled EXECUTE=1")
    completion_record = text.index(
        '"${RESULT_ROOT}/hccb_p418_formal_calculations_complete.json"'
    )
    refresh = text.index("run_hccb_p418_manuscript_refresh.sh")
    assert (
        seed101
        < full_timestep
        < full_steps
        < fixed_full_comparison
        < full_model
        < setup
        < seed202
        < development
        < seed303
        < final
        < high_re_fixed
        < high_re_full
        < completion_record
        < refresh
    )
    assert "seed101 must contain 60 completed steady cases" in text
    assert "build_hccb_pore_resolved_cht_case_matrix.py" in text
    assert "audit_hccb_pore_resolved_cht_case_matrix.py" in text
    assert "seed101 60 + seed202 9 + seed303 9" in text
    assert "hccb_p418_cross_packing_seed101_model_sources.json" in text
    assert "verify_hccb_p418_poststeady_completion.py" in text
    assert "hccb_p418_poststeady_pipeline_current.json" in text
    assert "seed101_model_sources_sha256" in text
    assert "new_physical_parameter_values_added" in text
    assert "P418_PYTHON" in text
    assert "torch.cuda.is_available" in text
    assert 'export PATH="$(dirname "${P418_PYTHON}"):${PATH}"' in text
    assert "capture_hccb_p418_runtime_environment.py" in text
    assert "runtime_environment.json" in text
    assert "runtime_environment_sha256" in text
    assert "fully_coupled_final_model_summary_sha256" in text
    assert "fixed_vs_fully_coupled_comparison_sha256" in text
    assert "fixed_vs_fully_coupled_manuscript_table_sha256" in text
    assert "fixed_vs_fully_coupled_manuscript_text_sha256" in text
    assert "high_re_fixed_frozen_summary_sha256" in text
    assert "high_re_fully_coupled_frozen_summary_sha256" in text
    assert "high_re_joint_comparison_sha256" in text
    assert "high_re_manuscript_table_sha256" in text
    assert "OPENFOAM_BASHRC" in text
    assert 'source "${OPENFOAM_BASHRC}"' in text
    assert (
        'FORMAL_LOCK=${FORMAL_LOCK:-${RESULT_ROOT}/hccb_p418_formal_calculations.lock}'
        in text
    )
    assert "flock -n 7" in text


def test_cross_packing_geometry_is_regenerated_and_recorded():
    text = SCRIPT.read_text(encoding="utf-8")
    mesh_setup = text.index("run_hccb_p418_cross_packing_setup.sh")
    geometry = text.index("summarize_hccb_p418_cross_packing_geometry.py")
    seed202_solver = text.index("SEED=202 EXECUTE=1")
    assert mesh_setup < geometry < seed202_solver
    assert "cross_packing_geometry_summary_sha256" in text
    assert "cross_packing_geometry_table_sha256" in text
    assert "cross_packing_manuscript_text_sha256" in text
    assert "cross_packing_manuscript_text_summary_sha256" in text
    assert "generated_cross_packing_geometry.tex" in text
