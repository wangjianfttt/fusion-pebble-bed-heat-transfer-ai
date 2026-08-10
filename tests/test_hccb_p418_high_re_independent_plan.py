#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from verify_hccb_p418_high_re_independent_plan import verify  # noqa: E402


def test_high_re_plan_is_separate_and_test_only() -> None:
    summary = verify(
        ROOT / "parameters/hccb_p418_high_re_independent_step_plan.json",
        ROOT
        / "parameters/hccb_p418_high_re_independent_fully_coupled_step_plan.json",
        ROOT / "parameters/hccb_p418_transient_step_plan.json",
        ROOT
        / "results/hccb_p418_inlet_dimensionless_envelope/inlet_dimensionless_conditions.csv",
    )
    assert summary["sequence_count"] == 6
    assert summary["pair_disjoint_from_main_twelve"] is True
    assert summary["frozen_model_test_only"] is True
    assert summary["new_physical_parameters"] == []
    assert summary["inlet_dimensionless_range"]["particle_reynolds_max"] > 2.4
    assert summary["inlet_dimensionless_range"]["particle_peclet_max"] > 1.6
    assert summary["openfoam_calculation_started"] is False
    assert summary["model_training_started"] is False


def test_high_re_runner_defaults_to_plan_only() -> None:
    runner = ROOT / "code/run_hccb_p418_high_re_independent_steps.sh"
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no OpenFOAM or model training was started" in result.stdout
    assert "frozen_model_independent_test_only" in result.stdout
    subprocess.run(["bash", "-n", str(runner)], check=True)


def test_fully_coupled_builder_allows_only_declared_test_role() -> None:
    source = (
        ROOT / "code/build_hccb_p418_fully_coupled_step_cases.py"
    ).read_text(encoding="utf-8")
    assert 'analysis_kind == "independent_high_re_test"' in source
    assert "high-Re curves must remain frozen-model independent tests" in source
    assert "high-Re independent curves cannot be used for model fitting" in source
