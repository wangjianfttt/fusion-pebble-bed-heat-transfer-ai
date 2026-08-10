from pathlib import Path
import json
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "run_hccb_p418_cross_packing_model_stage.sh"


def run_dry(tmp_path: Path, **environment: str):
    env = {"ROOT": str(tmp_path), "EXECUTE": "0", **environment}
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )


def test_development_stage_lists_four_models_and_reads_no_field(tmp_path: Path):
    result = run_dry(tmp_path, STAGE="development")
    assert result.returncode == 0
    assert "seed202" in result.stdout
    assert "pinn_data_only pinn graph transolver" in result.stdout
    assert "no independent packing field was loaded" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_final_stage_requires_seed202_selection(tmp_path: Path):
    missing = run_dry(tmp_path, STAGE="final")
    assert missing.returncode != 0
    assert "requires the seed202 selection file" in missing.stderr

    selection = (
        tmp_path
        / "results/hccb_p418_cross_packing_seed202_model_comparison"
        / "architecture_selection.json"
    )
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps(
            {
                "status": "seed202_architecture_fixed_before_seed303",
                "selected_architecture": "transolver",
                "seed303_fields_read": False,
            }
        ),
        encoding="utf-8",
    )
    selected = run_dry(tmp_path, STAGE="final")
    assert selected.returncode == 0
    assert "seed303" in selected.stdout
    assert "models: transolver" in selected.stdout

    disagreement = run_dry(
        tmp_path, STAGE="final", SELECTED_ARCHITECTURE="graph"
    )
    assert disagreement.returncode != 0
    assert "disagrees with frozen seed202 selection" in disagreement.stderr


def test_script_preserves_first_pass_results_and_uses_physical_summary():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "refusing to replace existing first-pass result" in text
    assert "evaluate_hccb_p418_cross_packing_conservative_operator.py" in text
    assert "summarize_hccb_p418_cross_packing_models.py" in text
    assert "select_hccb_p418_cross_packing_architecture.py" in text
    assert "architecture_selection.json" in text
    assert "seed101 normalization and validation-selected checkpoints remain fixed" in text
    assert "hccb_p418_cross_packing_seed101_model_sources.json" in text
    assert "independent_test_used_for_selection" in text
    assert "verify_hccb_p418_cross_packing_fixed_model.py" in text
    assert "hccb_p418_seed303_fixed_model_check.json" in text
