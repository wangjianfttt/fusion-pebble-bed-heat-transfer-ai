#!/usr/bin/env python3
"""Checks for the unified P418 research-route status."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from check_hccb_p418_research_route_completeness import build  # noqa: E402


def test_research_route_recognizes_verified_formal_training_data() -> None:
    payload, document = build(ROOT)
    coverage = json.loads(
        (
            ROOT
            / "results/hccb_p418_training_data_coverage_partial/summary.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["scheme_complete"]
    assert payload["formal_calculation_complete"]
    assert payload["status"] == "research_route_and_formal_calculation_complete"
    assert payload["formal_data_progress"] == {
        "steady": (
            f"{coverage['completed_case_count']}/"
            f"{coverage['expected_case_count']}"
        ),
        "fixed_hydrodynamics_steps": "12/12",
        "fully_coupled_steps": "0/12",
    }
    assert (
        payload["fully_coupled_scope_status"]
        == "property_range_limited_not_part_of_formal_training_data"
    )
    assert payload["physical_and_model_inputs"]["new_physical_parameters"] == []
    assert all(payload["current_summary_count_checks"].values())
    assert "稳态OpenFOAM矩阵和固定流场瞬态数据已经完成" in document
    assert "仅用于说明本研究的适用范围" in document
