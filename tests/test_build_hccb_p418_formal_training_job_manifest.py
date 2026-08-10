import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_formal_training_job_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("training_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_command_text_quotes_paths():
    module = load_module()
    text = module.command_text(["python3", Path("/tmp/path with space/a.py"), "--x", "1"])
    assert "'/tmp/path with space/a.py'" in text


def test_valid_completion_rejects_incomplete_status(tmp_path):
    module = load_module()
    complete = tmp_path / "complete.json"
    incomplete = tmp_path / "incomplete.json"
    failed = tmp_path / "failed.json"
    complete.write_text(
        json.dumps({"status": "completed_p418_model_training"}),
        encoding="utf-8",
    )
    incomplete.write_text(
        json.dumps({"status": "incomplete_p418_model_comparison"}),
        encoding="utf-8",
    )
    failed.write_text(
        json.dumps({"status": "failed_p418_model_training"}),
        encoding="utf-8",
    )

    assert module.valid_completion(complete) is True
    assert module.valid_completion(incomplete) is False
    assert module.valid_completion(failed) is False


def test_manifest_has_unique_jobs_and_valid_dependencies(tmp_path):
    output = tmp_path / "manifest.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        check=True,
        cwd=ROOT,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    jobs = payload["jobs"]
    ids = [job["job_id"] for job in jobs]
    assert len(ids) == len(set(ids)) == payload["job_count"]
    id_set = set(ids)
    for job in jobs:
        assert set(job["depends_on"]) <= id_set
        assert job["new_physical_parameters"] == []
        assert job["completion_file"]
    assert payload["execution_state"] in {
        "prepared_not_started",
        "partially_completed_on_registered_local_or_workstation_results",
        "all_registered_jobs_complete",
    }
    assert (
        payload["completed_job_count"] + payload["remaining_job_count"]
        == payload["job_count"]
    )


def test_manifest_preserves_formal_model_families(tmp_path):
    output = tmp_path / "manifest.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        check=True,
        cwd=ROOT,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    families = {job["model_family"] for job in payload["jobs"]}
    assert {
        "observable_transformer",
        "observable_dmdc",
        "observable_model_comparison",
        "regional_model_comparison",
        "initial_temperature_persistence",
        "regional_dmdc",
        "graph_transformer_data_only",
        "graph_transformer_energy_flux",
        "graph_transformer_factorized_energy_flux",
        "low_rank_temperature_residual",
        "temporal_diffusion_residual",
        "common_energy_balance",
        "seed_robustness_summary",
        "formal_model_comparison",
        "paper_performance_table",
        "paper_cost_table",
        "paper_generated_result_text",
        "paper_model_comparison_figure",
        "paper_openfoam_model_field_figure",
    } <= families
    assert payload["formal_training_settings"]["graph_epochs"] == 500
    assert payload["formal_training_settings"]["complete_curve_splits_only"] is True
    assert (
        payload["formal_training_settings"]["temperature_output_mode"]
        == "literature_bounded_residual"
    )
    assert payload["formal_training_settings"]["fluid_temperature_range_K"] == [
        300.0,
        1000.0,
    ]
    assert payload["formal_training_settings"]["solid_temperature_range_K"] == [
        298.0,
        1300.0,
    ]
    graph_jobs = [
        job
        for job in payload["jobs"]
        if job["model_family"].startswith("graph_transformer")
    ]
    assert graph_jobs
    for job in graph_jobs:
        assert (
            "--temperature-output-mode literature_bounded_residual"
            in job["command"]
        )
        assert "regional_graph_transformer_bounded_" in job["output_dir"]
        if job["model_family"] in {
            "graph_transformer_energy_flux",
            "graph_transformer_factorized_energy_flux",
        }:
            assert "--physics-device cuda" in job["command"]
        else:
            assert "--physics-device" not in job["command"]
    comparison = next(
        job
        for job in payload["jobs"]
        if job["model_family"] == "observable_model_comparison"
    )
    assert "generated_observable_dynamics.tex" in comparison["command"]
    assert "generated_observable_dynamics_text.tex" in comparison["command"]
    regional = next(
        job
        for job in payload["jobs"]
        if job["model_family"] == "regional_model_comparison"
    )
    assert regional["split_name"] == "pair_disjoint_stress_test"
    assert "generated_regional_dynamics.tex" in regional["command"]
    assert "generated_regional_dynamics_text.tex" in regional["command"]
    assert {
        "persistence__pair_disjoint_stress_test",
        "dmdc__pair_disjoint_stress_test",
        "graph_data_only__pair_disjoint_stress_test__seed20260717",
        "graph_physics__pair_disjoint_stress_test__seed20260717",
        "graph_factorized__pair_disjoint_stress_test__seed20260717",
    } == set(regional["depends_on"])
    field_figure = next(
        job
        for job in payload["jobs"]
        if job["model_family"] == "paper_openfoam_model_field_figure"
    )
    assert "build_hccb_p418_selected_field_figure.sh" in field_figure["command"]
    assert field_figure["depends_on"] == ["summarize_model_comparison"]
    assert field_figure["seed"] is None
    assert field_figure["completion_file"].endswith(
        "generated_openfoam_model_field_comparison_validated.tex"
    )
    transient_figure = next(
        job
        for job in payload["jobs"]
        if job["model_family"] == "paper_model_comparison_figure"
    )
    assert transient_figure["completion_file"].endswith(
        "generated_transient_model_comparison_validated.tex"
    )
    assert {
        "plot_transient_model_comparison",
        "plot_openfoam_model_field_comparison",
    } <= set(payload["final_summary_dependencies"])
