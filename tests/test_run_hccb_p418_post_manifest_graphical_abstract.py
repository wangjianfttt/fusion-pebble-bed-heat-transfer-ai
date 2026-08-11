from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_post_manifest_graphical_abstract.sh"


def test_graphical_abstract_waiter_is_validation_and_test_strict() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "WAIT_PID" in text
    assert "generated_openfoam_model_field_comparison_validated.tex" in text
    assert "complete_same_scale_openfoam_model_field_comparison" in text
    assert 'selection_data_role") != "validation"' in text
    assert 'display_data_role") != "test"' in text
    assert "plot_hccb_p418_graphical_abstract.py" in text
    assert "foamMultiRun" not in text
    assert "mpirun" not in text
    assert "torch" not in text
