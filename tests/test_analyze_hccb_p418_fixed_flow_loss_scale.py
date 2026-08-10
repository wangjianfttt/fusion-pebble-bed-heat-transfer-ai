#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


class FixedFlowLossScaleAnalysisTest(unittest.TestCase):
    def test_registered_weights_are_applied_before_fractions(self) -> None:
        from analyze_hccb_p418_fixed_flow_loss_scale import contribution_record

        record = contribution_record(
            {
                "epoch": 7,
                "temperature_data_loss": 2.0,
                "reference_edge_flux_loss": 10.0,
                "projection_aware_energy_loss": 80.0,
                "validation_solid_temperature_RMSE_K": 12.0,
                "validation_projection_aware_energy_normalized_RMSE": 3.0,
                "validation_selection_score": 4.0,
            }
        )
        self.assertEqual(
            record["weighted_contributions"],
            {
                "temperature_data_loss": 10.0,
                "reference_edge_flux_loss": 10.0,
                "projection_aware_energy_loss": 80.0,
            },
        )
        self.assertAlmostEqual(
            record["weighted_fraction"]["temperature_data_loss"], 0.1
        )
        self.assertEqual(record["weighted_dynamic_range"], 8.0)

    def test_recent_summary_uses_only_requested_tail(self) -> None:
        from analyze_hccb_p418_fixed_flow_loss_scale import analyze_history

        history = []
        for epoch, energy in enumerate((1.0, 10.0, 100.0), start=1):
            history.append(
                {
                    "epoch": epoch,
                    "temperature_data_loss": 1.0,
                    "reference_edge_flux_loss": 1.0,
                    "projection_aware_energy_loss": energy,
                    "validation_solid_temperature_RMSE_K": 10.0,
                    "validation_projection_aware_energy_normalized_RMSE": 2.0,
                    "validation_selection_score": 3.0,
                }
            )
        summary = analyze_history(history, recent_epochs=2)
        self.assertEqual(summary["completed_epochs"], 3)
        self.assertEqual(summary["recent_epoch_count"], 2)
        self.assertEqual(summary["latest"]["epoch"], 3)
        expected = sorted((10.0 / 16.0, 100.0 / 106.0))
        self.assertAlmostEqual(
            summary["recent_weighted_fraction_median"][
                "projection_aware_energy_loss"
            ],
            sum(expected) / 2.0,
        )

    def test_invalid_history_is_rejected(self) -> None:
        from analyze_hccb_p418_fixed_flow_loss_scale import analyze_history

        with self.assertRaisesRegex(ValueError, "no history"):
            analyze_history([], recent_epochs=10)
        with self.assertRaisesRegex(ValueError, "positive"):
            analyze_history(
                [
                    {
                        "epoch": 1,
                        "temperature_data_loss": 0.0,
                        "reference_edge_flux_loss": 0.0,
                        "projection_aware_energy_loss": 0.0,
                        "validation_solid_temperature_RMSE_K": 0.0,
                        "validation_projection_aware_energy_normalized_RMSE": 0.0,
                        "validation_selection_score": 0.0,
                    }
                ],
                recent_epochs=1,
            )


if __name__ == "__main__":
    unittest.main()
