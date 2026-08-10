#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_completed_matrix_physics import (  # noqa: E402
    completed_rows,
    factorial_variance_decomposition,
    heat_source_pairs,
    physical_trend_checks,
    single_factor_linearity,
    summarize_single_factor_linearity,
    steady_transition_proximity,
    temperature_pairs,
    thermal_regime_summary,
    velocity_pairs,
    wall_heat_zero_crossings,
    wall_heat_zero_crossing_responses,
    write_optional_csv,
)


class CompletedMatrixPhysicsTest(unittest.TestCase):
    def test_temperature_pair_reports_signed_wall_heat_reversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            matrix = Path(temporary)
            for inlet_temperature, wall_heat in ((300.0, 0.2), (900.0, -0.3)):
                name = f"T{int(inlet_temperature)}"
                case = matrix / name
                case.mkdir()
                completion_time = "200" if inlet_temperature == 300.0 else "300"
                (case / "formal_sample_complete.json").write_text(
                    json.dumps({"time": completion_time})
                )
                payload = {
                    "solver_finished": True,
                    "all_reported_values_are_finite": True,
                    "physical_conditions": {
                        "inlet_temperature_K": inlet_temperature,
                        "inlet_velocity_m_s": 0.05,
                        "cooling_wall_temperature_K": 635.0,
                        "solid_heat_source_W_m3": 8.85e6,
                    },
                    "flow": {
                        "pressure_drop_Pa": 4.0 if inlet_temperature == 300 else 3.0,
                        "relative_mass_difference": 1.0e-9,
                    },
                    "temperature": {
                        "outlet_average_K": 500.0 if inlet_temperature == 300 else 740.0,
                        "solid_maximum_K": 635.0 if inlet_temperature == 300 else 890.0,
                    },
                    "heat_balance": {
                        "solid_generated_power_W": 0.3,
                        "net_outward_enthalpy_flow_W": 0.1,
                        "cooling_wall_heat_flow_W": wall_heat,
                        "external_fluid_conductive_heat_flow_W": wall_heat,
                        "external_solid_conductive_heat_flow_W": (
                            -0.2 - wall_heat
                        ),
                        "relative_energy_difference": 1.0e-6,
                    },
                }
                (case / f"cht_result_summary_{completion_time}.json").write_text(
                    json.dumps(payload)
                )
            rows = completed_rows(
                matrix, None, time_from_completion_marker=True
            )
            pairs = temperature_pairs(rows)
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(
                rows[0]["solid_maximum_minus_cooling_wall_K"], 0.0
            )
            self.assertAlmostEqual(
                rows[0]["outlet_minus_inlet_temperature_K"], 200.0
            )
            self.assertAlmostEqual(
                rows[0]["net_outward_enthalpy_over_generated"], 1.0 / 3.0
            )
            self.assertAlmostEqual(rows[0]["total_external_boundary_heat_W"], -0.2)
            self.assertAlmostEqual(rows[0]["reconstructed_energy_difference_W"], 0.0)
            self.assertFalse(rows[0]["boundary_heat_fallback_used"])
            self.assertEqual(rows[0]["cooling_wall_heat_direction"], "wall_to_fluid")
            self.assertEqual(rows[1]["cooling_wall_heat_direction"], "fluid_to_wall")
            self.assertEqual(
                {str(row["completion_time_s"]) for row in rows}, {"200", "300"}
            )
            self.assertEqual(len(pairs), 1)
            self.assertTrue(pairs[0]["wall_heat_changes_sign"])
            self.assertAlmostEqual(pairs[0]["pressure_drop_change_percent"], -25.0)
            self.assertAlmostEqual(
                pairs[0]["outlet_temperature_response_K_per_K"], 0.4
            )
            crossings = wall_heat_zero_crossings(rows)
            self.assertEqual(len(crossings), 1)
            self.assertAlmostEqual(
                crossings[0]["interpolated_zero_wall_heat_inlet_temperature_K"],
                540.0,
            )
            self.assertAlmostEqual(crossings[0]["temperature_bracket_width_K"], 600.0)
            regimes = thermal_regime_summary(rows)
            self.assertEqual(
                regimes["cooling_wall_heat_direction_counts"],
                {"fluid_to_wall": 1, "wall_to_fluid": 1},
            )
            self.assertEqual(regimes["solid_maximum_above_cooling_wall_count"], 1)
            self.assertEqual(
                regimes["solid_maximum_at_or_below_cooling_wall_count"], 1
            )
            self.assertEqual(
                regimes["solid_maximum_temperature_range_K"], [635.0, 890.0]
            )

    def test_heat_source_and_velocity_pairs_hold_other_inputs_fixed(self) -> None:
        rows = [
            {
                "condition_id": "low_q_low_u",
                "inlet_velocity_m_s": 0.05,
                "inlet_temperature_K": 500.0,
                "solid_heat_source_MW_m3": 4.85,
                "pressure_drop_Pa": 3.0,
                "outlet_temperature_K": 600.0,
                "solid_maximum_temperature_K": 640.0,
                "cooling_wall_temperature_K": 635.0,
                "cooling_wall_heat_into_fluid_W": 0.10,
            },
            {
                "condition_id": "high_q_low_u",
                "inlet_velocity_m_s": 0.05,
                "inlet_temperature_K": 500.0,
                "solid_heat_source_MW_m3": 8.85,
                "pressure_drop_Pa": 3.3,
                "outlet_temperature_K": 620.0,
                "solid_maximum_temperature_K": 648.0,
                "cooling_wall_temperature_K": 635.0,
                "cooling_wall_heat_into_fluid_W": 0.06,
            },
            {
                "condition_id": "low_q_high_u",
                "inlet_velocity_m_s": 0.25,
                "inlet_temperature_K": 500.0,
                "solid_heat_source_MW_m3": 4.85,
                "pressure_drop_Pa": 15.0,
                "outlet_temperature_K": 580.0,
                "solid_maximum_temperature_K": 638.0,
                "cooling_wall_temperature_K": 635.0,
                "cooling_wall_heat_into_fluid_W": 0.14,
            },
        ]
        source = heat_source_pairs(rows)
        flow = velocity_pairs(rows)
        self.assertEqual(len(source), 1)
        self.assertEqual(len(flow), 1)
        self.assertAlmostEqual(
            source[0]["outlet_temperature_response_K_per_MW_m3"], 5.0
        )
        self.assertAlmostEqual(source[0]["solid_maximum_response_K_per_MW_m3"], 2.0)
        self.assertAlmostEqual(
            source[0]["low_solid_maximum_minus_cooling_wall_K"], 5.0
        )
        self.assertAlmostEqual(
            source[0]["high_solid_maximum_minus_cooling_wall_K"], 13.0
        )
        self.assertAlmostEqual(flow[0]["pressure_drop_response_Pa_per_m_s"], 60.0)
        self.assertAlmostEqual(flow[0]["outlet_temperature_response_K_per_m_s"], -100.0)

    def test_physical_trend_checks_report_failing_pair(self) -> None:
        rows = [{"condition_id": "a", "pressure_drop_Pa": 1.0}]
        temperature = [
            {
                "low_condition_id": "a",
                "high_condition_id": "b",
                "pressure_drop_change_percent": -10.0,
                "outlet_temperature_response_K_per_K": 0.5,
                "solid_maximum_response_K_per_K": 0.4,
            }
        ]
        source = [
            {
                "low_condition_id": "a",
                "high_condition_id": "c",
                "outlet_temperature_response_K_per_MW_m3": 2.0,
                "solid_maximum_response_K_per_MW_m3": -0.1,
            }
        ]
        velocity = [
            {
                "low_condition_id": "a",
                "high_condition_id": "d",
                "pressure_drop_response_Pa_per_m_s": 40.0,
            }
        ]
        checks = physical_trend_checks(rows, temperature, source, velocity)
        self.assertFalse(checks["all_evaluated_checks_passed"])
        failed = checks["checks"][
            "solid_maximum_does_not_decrease_with_heat_source_at_fixed_u_T"
        ]
        self.assertEqual(failed["failing_condition_pairs"], [["a", "c"]])

    def test_single_factor_linearity_reports_middle_point_departure(self) -> None:
        rows = []
        for source, outlet, solid_maximum in (
            (4.85, 500.0, 630.0),
            (6.85, 506.0, 634.0),
            (8.85, 510.0, 638.0),
        ):
            rows.append(
                {
                    "condition_id": f"q{source}",
                    "inlet_velocity_m_s": 0.05,
                    "inlet_temperature_K": 300.0,
                    "solid_heat_source_MW_m3": source,
                    "pressure_drop_Pa": 3.0 + source,
                    "outlet_temperature_K": outlet,
                    "solid_maximum_temperature_K": solid_maximum,
                    "net_outward_enthalpy_flow_W": 2.0 * source,
                    "cooling_wall_heat_into_fluid_W": 1.0 - source,
                }
            )
        records = single_factor_linearity(rows)
        self.assertEqual(len(records), 5)
        outlet_record = next(
            row for row in records if row["observable"] == "outlet_temperature_K"
        )
        self.assertAlmostEqual(outlet_record["endpoint_linear_prediction"], 505.0)
        self.assertAlmostEqual(outlet_record["signed_deviation"], 1.0)
        self.assertAlmostEqual(
            outlet_record["deviation_over_observed_range_percent"], 10.0
        )
        maximum_record = next(
            row
            for row in records
            if row["observable"] == "solid_maximum_temperature_K"
        )
        self.assertAlmostEqual(maximum_record["signed_deviation"], 0.0)
        summary = summarize_single_factor_linearity(records)
        self.assertEqual(len(summary), 5)
        outlet_summary = next(
            row for row in summary if row["observable"] == "outlet_temperature_K"
        )
        self.assertEqual(outlet_summary["worst_interior_condition_id"], "q6.85")

    def test_single_factor_linearity_requires_three_levels(self) -> None:
        rows = [
            {
                "condition_id": f"q{source}",
                "inlet_velocity_m_s": 0.05,
                "inlet_temperature_K": 300.0,
                "solid_heat_source_MW_m3": source,
            }
            for source in (4.85, 8.85)
        ]
        self.assertEqual(single_factor_linearity(rows), [])

    def test_steady_transition_proximity_uses_exact_published_temperatures(self) -> None:
        rows = [
            {
                "condition_id": "below",
                "inlet_velocity_m_s": 0.05,
                "inlet_temperature_K": 900.0,
                "solid_heat_source_MW_m3": 4.85,
                "solid_maximum_temperature_K": 930.0,
            },
            {
                "condition_id": "above_first",
                "inlet_velocity_m_s": 0.05,
                "inlet_temperature_K": 900.0,
                "solid_heat_source_MW_m3": 8.85,
                "solid_maximum_temperature_K": 950.0,
            },
        ]
        records = steady_transition_proximity(rows, (938.0, 996.0))
        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(
            records[0]["transition_1_minus_solid_maximum_K"], 8.0
        )
        self.assertFalse(records[0]["transition_1_reached_by_solid_maximum"])
        self.assertAlmostEqual(
            records[1]["transition_1_minus_solid_maximum_K"], -12.0
        )
        self.assertTrue(records[1]["transition_1_reached_by_solid_maximum"])
        self.assertFalse(records[1]["transition_2_reached_by_solid_maximum"])

    def test_zero_wall_heat_response_uses_computed_crossing_extrema(self) -> None:
        crossings = [
            {
                "inlet_velocity_m_s": 0.05,
                "solid_heat_source_MW_m3": 4.85,
                "interpolated_zero_wall_heat_inlet_temperature_K": 620.0,
                "temperature_bracket_width_K": 200.0,
            },
            {
                "inlet_velocity_m_s": 0.05,
                "solid_heat_source_MW_m3": 6.85,
                "interpolated_zero_wall_heat_inlet_temperature_K": 615.0,
                "temperature_bracket_width_K": 200.0,
            },
            {
                "inlet_velocity_m_s": 0.05,
                "solid_heat_source_MW_m3": 8.85,
                "interpolated_zero_wall_heat_inlet_temperature_K": 610.0,
                "temperature_bracket_width_K": 200.0,
            },
            {
                "inlet_velocity_m_s": 0.25,
                "solid_heat_source_MW_m3": 4.85,
                "interpolated_zero_wall_heat_inlet_temperature_K": 640.0,
                "temperature_bracket_width_K": 400.0,
            },
        ]
        responses = wall_heat_zero_crossing_responses(crossings)
        source = next(
            row
            for row in responses
            if row["varied_factor"] == "solid_heat_source_MW_m3"
        )
        velocity = next(
            row
            for row in responses
            if row["varied_factor"] == "inlet_velocity_m_s"
        )
        self.assertEqual(source["source_crossing_count"], 3)
        self.assertAlmostEqual(source["zero_wall_heat_temperature_change_K"], -10.0)
        self.assertAlmostEqual(source["zero_wall_heat_temperature_response"], -2.5)
        self.assertAlmostEqual(velocity["zero_wall_heat_temperature_change_K"], 20.0)
        self.assertAlmostEqual(velocity["zero_wall_heat_temperature_response"], 100.0)
        self.assertAlmostEqual(velocity["maximum_temperature_bracket_width_K"], 400.0)

    def test_complete_factorial_decomposition_recovers_additive_effects(self) -> None:
        rows = []
        for velocity in (0.05, 0.25):
            for temperature in (300.0, 900.0):
                for source in (4.85, 8.85):
                    additive = 2.0 * velocity + 3.0 * temperature + 5.0 * source
                    rows.append(
                        {
                            "inlet_velocity_m_s": velocity,
                            "inlet_temperature_K": temperature,
                            "solid_heat_source_MW_m3": source,
                            "pressure_drop_Pa": additive,
                            "outlet_temperature_K": additive,
                            "solid_maximum_temperature_K": additive,
                            "net_outward_enthalpy_flow_W": additive,
                            "cooling_wall_heat_into_fluid_W": additive,
                        }
                    )
        decomposition = factorial_variance_decomposition(rows)
        self.assertEqual(len(decomposition), 35)
        for observable in {str(row["observable"]) for row in decomposition}:
            subset = [row for row in decomposition if row["observable"] == observable]
            self.assertAlmostEqual(
                sum(float(row["variance_fraction_percent"]) for row in subset),
                100.0,
            )
            interactions = [
                row for row in subset if "_x_" in str(row["effect"])
            ]
            self.assertTrue(
                all(abs(float(row["variance_fraction_percent"])) < 1.0e-20 for row in interactions)
            )

    def test_incomplete_factorial_matrix_is_not_decomposed(self) -> None:
        rows = [
            {
                "inlet_velocity_m_s": 0.05,
                "inlet_temperature_K": 300.0,
                "solid_heat_source_MW_m3": 4.85,
            },
            {
                "inlet_velocity_m_s": 0.25,
                "inlet_temperature_K": 900.0,
                "solid_heat_source_MW_m3": 8.85,
            },
        ]
        self.assertEqual(factorial_variance_decomposition(rows), [])

    def test_single_factor_slice_is_not_called_factorial_decomposition(self) -> None:
        rows = [
            {
                "inlet_velocity_m_s": 0.05,
                "inlet_temperature_K": 300.0,
                "solid_heat_source_MW_m3": source,
            }
            for source in (4.85, 6.85, 8.85)
        ]
        self.assertEqual(factorial_variance_decomposition(rows), [])

    def test_optional_result_removes_stale_file_when_no_rows_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "conditional.csv"
            output.write_text("stale result\n", encoding="utf-8")
            write_optional_csv(output, [])
            self.assertFalse(output.exists())

            write_optional_csv(output, [{"value": 3.0}])
            self.assertEqual(output.read_text(encoding="utf-8"), "value\n3.0\n")


if __name__ == "__main__":
    unittest.main()
