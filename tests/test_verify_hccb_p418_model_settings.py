from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/verify_hccb_p418_model_settings.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_hccb_p418_model_settings", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_current_p418_model_settings_match_code_and_sources():
    module = load_module()
    payload = module.verify(
        ROOT,
        ROOT / "parameters/hccb_p418_model_numerical_settings.csv",
        ROOT / "parameters/hccb_p418_ai_architecture_sources.json",
    )
    assert payload["status"] == "current_p418_model_settings_match_code_and_sources"
    assert payload["setting_count"] == 78
    assert payload["verified_setting_count"] == 78
    assert payload["all_settings_are_nonphysical_numerical_choices"]
    assert payload["architecture_registry_check_count"] == 19
    assert all(row["passed"] for row in payload["architecture_registry_checks"])
    assert payload["failures"] == []


def test_key_fused_architectures_are_checked_separately():
    module = load_module()
    payload = module.verify(
        ROOT,
        ROOT / "parameters/hccb_p418_model_numerical_settings.csv",
        ROOT / "parameters/hccb_p418_ai_architecture_sources.json",
    )
    models = {row["model"] for row in payload["results"]}
    assert {
        "工程量时间Transformer",
        "图-Transformer",
        "全耦合图-Transformer",
        "扩散剩余误差修正",
    } <= models
    fully_coupled = [
        row for row in payload["results"] if row["model"] == "全耦合图-Transformer"
    ]
    assert len(fully_coupled) == 4
    assert all(row["passed"] for row in fully_coupled)
    diffusion = [row for row in payload["results"] if row["model"] == "扩散剩余误差修正"]
    assert len(diffusion) == 17
    assert all(row["passed"] for row in diffusion)
