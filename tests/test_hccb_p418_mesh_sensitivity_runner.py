import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "code/run_hccb_p418_mesh_sensitivity.sh"
FINALIZER = ROOT / "code/finalize_hccb_p418_mesh_sensitivity_case.sh"
FINE_REFERENCE = ROOT / "code/verify_hccb_p418_mesh_fine_reference.py"


def test_mesh_sensitivity_runner_has_power_loss_resume_and_no_new_parameters() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    fine_reference = FINE_REFERENCE.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(FINALIZER)], check=True)
    assert "resumed from complete parallel time" in text
    assert "fluid/T fluid/U fluid/p fluid/p_rgh solid/T uniform/time" in text
    assert '-entry runTimeModifiable -set false' in text
    assert '--condition-id "${CONDITION_ID}"' in text
    assert "u0p20_T700_q6p85" in text
    assert "summarize_hccb_p418_mesh_sensitivity.py" in text
    assert 'mesh_summaries/${level}.log" || true' not in text
    assert "verify_hccb_p418_mesh_fine_reference.py" in text
    assert "mesh_source_packing_sha256" in fine_reference
    assert "source_packing_sha256" in fine_reference
    assert 'metadata.get("mesh_resolution_label") == "fine"' in fine_reference
    assert "formal_sample_complete.json" in text
    assert "fine_reference_check.json" in text
    summary = text.index("summarize_hccb_p418_mesh_sensitivity.py")
    table = text.index("build_hccb_p418_mesh_sensitivity_table.py")
    assert summary < table
    assert 'generated_mesh_sensitivity.tex"' in text
