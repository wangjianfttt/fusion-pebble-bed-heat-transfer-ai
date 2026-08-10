import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from preflight_hccb_p418_fused_research import chinese_summary, count_completion


def test_count_completion_keeps_fully_coupled_marker_separate(tmp_path: Path) -> None:
    for identifier in ("step_a", "step_b"):
        case = tmp_path / identifier
        case.mkdir()
    (tmp_path / "step_a" / "fully_coupled_step_response_complete.json").write_text(
        "{}\n", encoding="utf-8"
    )

    result = count_completion(
        tmp_path, "fully_coupled_step_response_complete.json", expected=2
    )

    assert result["completed"] == 1
    assert result["required"] == 2
    assert result["complete"] is False


def test_chinese_summary_reports_all_three_data_families() -> None:
    payload = {
        "current_data": {
            "steady": {"completed": 14, "required": 60},
            "physical_transient": {"completed": 0, "required": 12},
            "fully_coupled_transient": {"completed": 0, "required": 12},
        },
        "full_training_can_start": False,
        "fused_contract": {
            "physical_parameter_count": 22,
            "equation_map_row_count": 31,
            "local_evidence_reference_count": 24,
        },
        "model_settings": {"verified_setting_count": 71, "setting_count": 71},
        "algorithm_sources": {"architecture_count": 9},
    }

    text = chinese_summary(payload)

    assert "稳态三维工况：`14/60`" in text
    assert "固定流场物理热阶跃：`0/12`" in text
    assert "速度、压力和温度全耦合阶跃：`0/12`" in text


def test_count_completion_uses_verified_sequence_index(tmp_path: Path) -> None:
    missing_raw_root = tmp_path / "raw_steps"
    summary = tmp_path / "dataset_index.json"
    summary.write_text('{"sequence_count": 12}\n', encoding="utf-8")

    result = count_completion(
        missing_raw_root,
        "step_response_complete.json",
        expected=12,
        fallback_summary=summary,
        fallback_completed_key="sequence_count",
        fallback_required_key=None,
    )

    assert result["completed"] == 12
    assert result["required"] == 12
    assert result["complete"] is True
    assert result["progress_source"] == "verified_summary"


def test_chinese_summary_treats_coupled_failure_as_scope_limit() -> None:
    payload = {
        "current_data": {
            "steady": {"completed": 60, "required": 60},
            "physical_transient": {"completed": 12, "required": 12},
            "fully_coupled_transient": {"completed": 0, "required": 12},
        },
        "full_training_can_start": True,
        "fused_contract": {
            "physical_parameter_count": 22,
            "equation_map_row_count": 31,
            "local_evidence_reference_count": 24,
        },
        "model_settings": {"verified_setting_count": 71, "setting_count": 71},
        "algorithm_sources": {"architecture_count": 9},
    }

    text = chinese_summary(payload)

    assert "稳态和固定流场瞬态训练输入已经齐全" in text
    assert "只用来界定本文的适用范围" in text
    assert "生成瞬态误差图" in text
