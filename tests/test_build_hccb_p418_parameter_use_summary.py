from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_parameter_use_summary import build  # noqa: E402


def test_parameter_and_model_input_summary() -> None:
    summary, document = build(ROOT)
    assert summary["physical_parameter_count"] == 22
    assert summary["physical_parameters_used_by_equations"] == 22
    assert summary["unused_physical_parameter_ids"] == []
    assert summary["unknown_equation_parameter_ids"] == []
    assert summary["model_numerical_setting_count"] == 78
    assert summary["literature_or_official_model_setting_count"] == 55
    assert summary["case_or_data_derived_model_setting_count"] == 14
    assert summary["predeclared_project_comparison_setting_count"] == 9
    assert summary["all_model_settings_are_nonphysical"]
    assert summary["all_model_setting_source_paths_exist"]
    assert summary["experimental_observable_count"] == 12
    assert summary["experimental_observation_source_count"] == 17
    assert summary["experimental_templates_contain_no_measurements"]
    assert "为什么不能把所有数字都叫“文献参数”" in document
    assert "78项模型设置" in document
