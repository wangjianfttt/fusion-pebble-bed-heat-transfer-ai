#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_thermal_regime_model_errors import (  # noqa: E402
    load_physical_conditions,
    summarize_payload,
)


def case(condition_id: str, scale: float) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "generated_power_W": 2.0,
        "engineering_absolute_errors": {
            "pressure_drop_Pa": scale,
            "outlet_temperature_K": 2.0 * scale,
            "solid_maximum_temperature_K": 3.0 * scale,
            "cooling_wall_heat_into_fluid_W": 0.1 * scale,
            "solid_to_fluid_interphase_net_W": 0.2 * scale,
        },
        "global_mass_imbalance_over_inlet": 0.01 * scale,
        "global_energy_imbalance_over_generated_power": 0.02 * scale,
    }


class ThermalRegimeModelErrorsTest(unittest.TestCase):
    def test_test_cases_are_split_by_heat_direction_and_solid_wall_relation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            physical_csv = Path(temporary) / "physical.csv"
            with physical_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "condition_id",
                        "cooling_wall_heat_direction",
                        "solid_maximum_minus_cooling_wall_K",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "condition_id": "cold",
                            "cooling_wall_heat_direction": "wall_to_fluid",
                            "solid_maximum_minus_cooling_wall_K": -0.1,
                        },
                        {
                            "condition_id": "hot_a",
                            "cooling_wall_heat_direction": "fluid_to_wall",
                            "solid_maximum_minus_cooling_wall_K": 20.0,
                        },
                        {
                            "condition_id": "hot_b",
                            "cooling_wall_heat_direction": "fluid_to_wall",
                            "solid_maximum_minus_cooling_wall_K": 40.0,
                        },
                    ]
                )
            physical = load_physical_conditions(physical_csv)
            payload = {
                "architecture": "graph",
                "split_name": "physical_holdout",
                "evaluations": {
                    "test": {
                        "cases": [case("cold", 1.0), case("hot_a", 2.0), case("hot_b", 4.0)]
                    }
                },
            }
            rows = summarize_payload(payload, physical)
            self.assertEqual(len(rows), 4)
            heat_rows = {
                row["thermal_regime"]: row
                for row in rows
                if row["classification_axis"] == "cooling_wall_heat_direction"
            }
            self.assertEqual(heat_rows["wall_to_fluid"]["case_count"], 1)
            self.assertEqual(heat_rows["fluid_to_wall"]["case_count"], 2)
            self.assertAlmostEqual(
                heat_rows["fluid_to_wall"]["outlet_temperature_mae_K"], 6.0
            )
            self.assertAlmostEqual(
                heat_rows["wall_to_fluid"][
                    "cooling_wall_heat_error_mean_percent_generated"
                ],
                5.0,
            )
            solid_rows = {
                row["thermal_regime"]: row
                for row in rows
                if row["classification_axis"] == "solid_temperature_relation"
            }
            self.assertEqual(
                solid_rows["solid_maximum_at_or_below_wall"]["condition_ids"],
                "cold",
            )
            self.assertEqual(
                solid_rows["solid_maximum_above_wall"]["condition_ids"],
                "hot_a;hot_b",
            )

    def test_missing_physical_case_is_rejected(self) -> None:
        payload = {
            "architecture": "pinn",
            "split_name": "holdout",
            "evaluations": {"test": {"cases": [case("missing", 1.0)]}},
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            summarize_payload(payload, {})


if __name__ == "__main__":
    unittest.main()
