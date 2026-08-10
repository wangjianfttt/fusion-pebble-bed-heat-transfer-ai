from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/smoke_hccb_p418_actual_spatiotemporal_operator.py"


def test_actual_spatiotemporal_resource_script_records_matching_inputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    assert "validate_input_provenance" in source
    assert '"state source subface"' in source
    assert '"state source dataset"' in source
    assert '"input_provenance": input_provenance' in source
    assert '"step_plan_sha256": sha256' in source
    assert "loss.backward()" in source
