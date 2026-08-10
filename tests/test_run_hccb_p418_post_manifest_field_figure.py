from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_post_manifest_field_figure.sh"


def test_post_manifest_field_figure_waits_and_uses_selected_formal_prediction() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert "while kill -0" in text
    assert "WAIT_VALIDATION_PID" in text
    assert "hccb_p418_openfoam_model_field_selection.json" in text
    assert "build_hccb_p418_selected_field_figure.sh" in text
    assert "selected_model" in text
    assert "prediction_file_sha256" in text
    assert "selection_data_role" in text
    assert "display_data_role" in text
    assert "regional_graph_transformer_bounded_physics_pair_disjoint_stress_test" not in text
    assert "complete_same_scale_openfoam_model_field_comparison" in text
    assert "new_physical_parameters" in text
    assert "foamMultiRun" not in text
    assert "sbatch" not in text


def test_selected_field_figure_uses_a_registered_test_trajectory() -> None:
    text = (ROOT / "code/build_hccb_p418_selected_field_figure.sh").read_text(
        encoding="utf-8"
    )
    split = json.loads(
        (ROOT / "parameters/hccb_p418_step_response_splits.json").read_text(
            encoding="utf-8"
        )
    )["splits"]["pair_disjoint_stress_test"]
    match = re.search(r"^SEQUENCE_ID=(\S+)$", text, flags=re.MULTILINE)
    assert match is not None
    sequence_id = match.group(1)
    assert sequence_id in split["test"]
    assert sequence_id not in split["train"]
    assert sequence_id not in split["validation"]
