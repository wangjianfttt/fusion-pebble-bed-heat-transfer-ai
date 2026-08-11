#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from verify_hccb_p418_model_comparison_protocol import verify  # noqa: E402


def test_common_model_comparison_protocol_matches_current_programs() -> None:
    summary = verify(ROOT / "parameters/hccb_p418_model_comparison_protocol.json")
    assert summary["status"] == "p418_common_model_comparison_protocol_verified"
    assert summary["physical_parameter_count"] == 22
    assert summary["steady_condition_count"] == 60
    assert summary["steady_model_count"] == 5
    assert summary["transient_sequence_count"] == 12
    assert summary["transient_output_time_count"] == 56
    assert summary["transient_model_count"] == 6
    assert summary["packing_seeds"] == [101, 202, 303]
    assert summary["formal_manifest_job_count"] == 75
    assert summary["direct_common_split_job_count"] == 27
    assert summary["upstream_inherited_split_job_count"] == 10
    assert summary["common_energy_evaluation_job_count"] == 29
    assert summary["train_only_source_program_count"] == 6
    assert summary["same_physical_inputs_for_all_models"] is True
    assert summary["train_only_normalization"] is True
    assert summary["complete_curve_splitting"] is True
    assert summary["new_physical_parameters"] == []


def test_protocol_declares_no_weighted_overall_model_score() -> None:
    protocol = json.loads(
        (ROOT / "parameters/hccb_p418_model_comparison_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert "Do not collapse" in protocol["steady_comparison"]["ranking_rule"]
    assert "joint improvement" in protocol["physical_transient_comparison"][
        "diffusion_interpretation_rule"
    ]
