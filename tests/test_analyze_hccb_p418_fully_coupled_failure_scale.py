from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from analyze_hccb_p418_fully_coupled_failure_scale import analyze  # noqa: E402


def test_failed_coupled_temperature_is_far_above_physical_heating_scale() -> None:
    payload = analyze(
        ROOT / "results/hccb_p418_public_figure_data/direct_transport_scope_limit.json",
        ROOT
        / "results/hccb_p418_fully_coupled_failure_scale_20260812/source_initial_temperature_extrema.json",
        ROOT / "results/hccb_p418_sourceflow_complete_physics_60/completed_case_physics.csv",
    )
    assert payload["status"] == "fully_coupled_failure_temperature_scale_quantified"
    assert payload["new_physical_parameters"] == []
    assert payload["source_only_temperature_rise_scale_K"] < 0.01
    assert payload["observed_excursion_over_source_only_scale"] > 1.0e4
    assert payload["failed_query_above_generous_upper_K"] > 500.0
