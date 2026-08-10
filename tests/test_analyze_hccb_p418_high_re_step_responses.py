from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from analyze_hccb_p418_high_re_step_responses import build_summary


def make_case(root: Path, index: int) -> None:
    case = root / "by_sequence" / f"case_{index}"
    results = case / "results"
    results.mkdir(parents=True)
    (case / "cloud_sequence_complete.json").write_text(
        '{"solver_finished":true}\n', encoding="utf-8"
    )
    (results / "summary.json").write_text(
        '{"completed_case_count":1,"maximum_time_points":3}\n',
        encoding="utf-8",
    )
    fields = [
        "condition_id",
        "source_inlet_velocity_m_s",
        "source_inlet_temperature_K",
        "source_solid_heat_source_MW_m3",
        "target_inlet_velocity_m_s",
        "target_inlet_temperature_K",
        "target_solid_heat_source_MW_m3",
        "time_s",
        "outlet_temperature_K",
        "maximum_solid_temperature_K",
        "volume_average_fluid_temperature_K",
        "volume_average_solid_temperature_K",
        "cooling_wall_power_W",
        "signed_mass_residual_kg_s",
    ]
    with (results / "hccb_p418_transient_observables_long.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for time_s, fraction in ((0.0, 0.0), (150.0, 0.5), (300.0, 1.0)):
            writer.writerow(
                {
                    "condition_id": f"case_{index}",
                    "source_inlet_velocity_m_s": 0.05,
                    "source_inlet_temperature_K": 300.0,
                    "source_solid_heat_source_MW_m3": 4.85,
                    "target_inlet_velocity_m_s": 0.25,
                    "target_inlet_temperature_K": 900.0,
                    "target_solid_heat_source_MW_m3": 8.85,
                    "time_s": time_s,
                    "outlet_temperature_K": 300.0 + 100.0 * fraction,
                    "maximum_solid_temperature_K": 400.0 + 100.0 * fraction,
                    "volume_average_fluid_temperature_K": 350.0
                    + 100.0 * fraction,
                    "volume_average_solid_temperature_K": 375.0
                    + 100.0 * fraction,
                    "cooling_wall_power_W": 0.3 - 0.1 * fraction,
                    "signed_mass_residual_kg_s": 1.0e-15,
                }
            )


def test_build_summary_requires_six_complete_curves(tmp_path: Path) -> None:
    for index in range(6):
        make_case(tmp_path, index)
    summary = build_summary(tmp_path, expected_points=3)
    assert summary["status"] == "completed_p418_high_re_step_response_analysis"
    assert summary["curve_count"] == 6
    assert summary["points_per_curve"] == 3
    assert summary["cases"][0]["responses"]["outlet_temperature_K"]["t50_s"] == 150.0
    assert summary["cases"][0]["responses"]["outlet_temperature_K"]["t90_s"] == 300.0
    assert summary["new_physical_parameters"] == []
