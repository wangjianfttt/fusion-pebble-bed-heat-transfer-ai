#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required")
class EvaluateP418ChainedDiffusionTest(unittest.TestCase):
    def test_ensemble_metrics_use_regional_volume_and_material_type(self) -> None:
        from evaluate_hccb_p418_chained_diffusion import (
            complete_chain_timing,
            ensemble_temperature_metrics,
        )

        target = np.zeros((1, 2, 2, 1), dtype=np.float32)
        members = np.stack((target, np.full_like(target, 2.0)), axis=0)
        metrics = ensemble_temperature_metrics(
            members=members,
            target=target,
            node_type=np.asarray([0, 1]),
            node_volume=np.asarray([1.0, 1.0]),
            temperature_std_by_type=np.asarray([10.0, 20.0]),
        )
        self.assertAlmostEqual(
            metrics["ensemble_mean_temperature_RMSE_K"], np.sqrt(250.0)
        )
        self.assertAlmostEqual(
            metrics["ensemble_mean_fluid_temperature_RMSE_K"], 10.0
        )
        self.assertAlmostEqual(
            metrics["ensemble_mean_solid_temperature_RMSE_K"], 20.0
        )
        self.assertEqual(metrics["interval_90pct_coverage_fraction"], 0.0)
        self.assertAlmostEqual(
            metrics["interval_90pct_mean_width_K"], 27.0, places=5
        )
        self.assertEqual(
            metrics[
                "unobserved_dynamic_90pct_interval_coverage_fraction"
            ],
            0.0,
        )
        self.assertAlmostEqual(
            metrics["unobserved_dynamic_90pct_interval_mean_width_K"],
            27.0,
            places=5,
        )
        self.assertAlmostEqual(
            metrics[
                "unobserved_dynamic_fluid_90pct_interval_mean_width_K"
            ],
            18.0,
            places=5,
        )
        self.assertAlmostEqual(
            metrics[
                "unobserved_dynamic_solid_90pct_interval_mean_width_K"
            ],
            36.0,
            places=5,
        )
        self.assertAlmostEqual(metrics["unobserved_dynamic_CRPS_K"], 7.5)
        self.assertAlmostEqual(metrics["unobserved_dynamic_fluid_CRPS_K"], 5.0)
        self.assertAlmostEqual(metrics["unobserved_dynamic_solid_CRPS_K"], 10.0)

        timing = complete_chain_timing(
            chained_summary={
                "timing": {
                    "graph_transformer_inference_seconds": 4.0,
                    "warm_start_deterministic_chain_inference_seconds": 4.0,
                    "cold_start_deterministic_chain_inference_seconds": 6.0,
                }
            },
            diffusion_seconds=8.0,
            curve_count=4,
        )
        self.assertEqual(
            timing["warm_start_complete_chain_inference_seconds_per_curve"], 3.0
        )
        self.assertEqual(
            timing["cold_start_complete_chain_inference_seconds_per_curve"], 3.5
        )


if __name__ == "__main__":
    unittest.main()
