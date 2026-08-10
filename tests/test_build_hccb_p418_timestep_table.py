from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_timestep_table import LABELS, build_table  # noqa: E402


def test_table_reports_values_and_nonmonotonic_outcomes() -> None:
    comparisons = []
    gci = []
    for index, signal in enumerate(LABELS):
        comparisons.extend(
            [
                {
                    "signal": signal,
                    "coarse_delta_t_s": 1.0,
                    "fine_delta_t_s": 0.5,
                    "maximum_difference_over_response_span": 0.02,
                },
                {
                    "signal": signal,
                    "coarse_delta_t_s": 0.5,
                    "fine_delta_t_s": 0.25,
                    "maximum_difference_over_response_span": 0.01 + index * 0.001,
                },
            ]
        )
        for quantity in ("endpoint", "curve_maximum"):
            gci.append(
                {
                    "signal": signal,
                    "quantity": quantity,
                    "convergence_status": (
                        "oscillatory_no_gci_reported" if index == 0 else "monotonic_gci_reported"
                    ),
                    "fine_gci_fraction": None if index == 0 else 0.005,
                }
            )
    text = build_table(
        {
            "status": "completed_p418_thermal_timestep_sensitivity",
            "selected_delta_t_s": 0.25,
            "delta_t_s": [1.0, 0.5, 0.25],
            "selected_time_step_schedule": [
                {"start_s": 0.0, "end_s": 2.0, "delta_t_s": 0.25}
            ],
            "comparisons": comparisons,
            "gci_results": gci,
        }
    )
    assert "Outlet temperature & 1\\% & oscillatory" in text
    assert "Cooling-wall power & 1.1\\% & 0.5\\%" in text
    assert "computed trajectories use the finest" in text
