from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/finalize_hccb_p418_recovered_steady_case.sh"


def test_recovered_case_finalizer_preserves_physics_and_requires_full_tail() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "summarize_hccb_p418_formal_steady_tail.py" in text
    assert "full_field_available" in text
    assert "export_hccb_cht_training_sample.py" in text
    assert "training_sample_schema_version" in text
    assert "actual_mpi_process_count" in text
    assert "ALLOW_MISSING_TAIL_FIELDS" in text
    assert "--allow-missing-fields" in text
    assert "new_physical_parameters" not in text
    assert "mpirun" not in text
    assert "foamMultiRun -case" not in text
