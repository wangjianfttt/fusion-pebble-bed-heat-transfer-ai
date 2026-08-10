from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_supplement_inputs import build  # noqa: E402


def test_supplement_inputs_are_complete_and_source_backed() -> None:
    summary, outputs = build(ROOT)
    assert summary["status"] == "completed_p418_supplement_inputs"
    assert summary["physical_parameter_count"] == 22
    assert summary["equation_input_count"] == 31
    assert summary["model_numerical_setting_count"] == 78
    assert all(
        content.isascii()
        for content in outputs.values()
    )
    assert summary["result_source_map_count"] >= 40
    assert summary["all_physical_parameters_mapped"]
    assert summary["all_model_settings_nonphysical"]
    assert len(outputs) == 4
    assert "generated_supp_physical_parameters.tex" in outputs
    assert "P048" in outputs["generated_supp_physical_parameters.tex"]
    assert "OpenFOAM/Python implementation" in outputs[
        "generated_supp_equation_input_map.tex"
    ]
    assert "Graph--Transformer" in outputs["generated_supp_model_settings.tex"]
    assert "formal manuscript PDF" in outputs["generated_supp_result_source_map.tex"]


def test_supplement_tex_uses_the_condensed_body() -> None:
    text = (ROOT / "manuscript/supplement.tex").read_text(encoding="utf-8")
    body = (ROOT / "manuscript/supplement_condensed_body.tex").read_text(
        encoding="utf-8"
    )
    assert r"\input{supplement_condensed_body}" in text
    assert r"\input{generated_data_splits}" in body
    assert "No cell from an independent steady condition" in body
    assert "rather than repeated here as multi-page tables" in body
    for name in (
        "generated_supp_physical_parameters",
        "generated_supp_equation_input_map",
        "generated_supp_model_settings",
        "generated_supp_result_source_map",
    ):
        assert rf"\input{{{name}}}" not in text
