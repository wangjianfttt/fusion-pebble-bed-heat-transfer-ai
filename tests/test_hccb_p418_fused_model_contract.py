import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fused_model_contract_verifies(tmp_path: Path) -> None:
    output = tmp_path / "verified.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/verify_hccb_p418_fused_model_contract.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "p418_fused_model_contract_verified"
    assert result["shared_state_channels"] == [
        "Ux_m_s",
        "Uy_m_s",
        "Uz_m_s",
        "pressure_Pa",
        "temperature_K",
    ]
    assert result["steady_condition_count"] == 60
    assert result["physical_transient_sequence_count"] == 12
    assert result["physical_transient_output_time_count"] == 56
    assert result["diffusion_corrected_channels"] == ["temperature_K"]
    assert result["complete_inference_entry"] == "code/run_hccb_p418_chained_initial_state_evaluation.sh"
    assert result["complete_inference_split_count"] == 3
    assert result["new_physical_parameters"] == []


def test_contract_excludes_solver_relaxation_from_physical_transients() -> None:
    contract = json.loads(
        (ROOT / "parameters/hccb_p418_fused_model_contract.json").read_text(
            encoding="utf-8"
        )
    )
    excluded = {
        row["path"]: row["reason"]
        for row in contract["excluded_from_physical_transient_results"]
    }
    assert "code/run_hccb_p418_60_transient_models.sh" in excluded
    assert "solver-relaxation" in excluded["code/run_hccb_p418_60_transient_models.sh"]
    assert contract["diffusion_stage"]["unchanged_channels"] == [
        "Ux_m_s",
        "Uy_m_s",
        "Uz_m_s",
        "pressure_Pa",
    ]
