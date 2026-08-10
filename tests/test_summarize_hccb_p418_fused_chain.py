#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_fused_chain import ENERGY_METRIC, build_payload  # noqa: E402


class FusedChainSummaryTest(unittest.TestCase):
    def inputs(self) -> dict[str, dict[str, object]]:
        interval = {
            "unobserved_dynamic_90pct_interval_coverage_fraction": 0.87,
            "unobserved_dynamic_90pct_interval_mean_width_K": 12.0,
            "unobserved_dynamic_fluid_90pct_interval_coverage_fraction": 0.91,
            "unobserved_dynamic_fluid_90pct_interval_mean_width_K": 8.0,
            "unobserved_dynamic_solid_90pct_interval_coverage_fraction": 0.82,
            "unobserved_dynamic_solid_90pct_interval_mean_width_K": 16.0,
            "unobserved_dynamic_CRPS_K": 4.0,
            "unobserved_dynamic_fluid_CRPS_K": 3.0,
            "unobserved_dynamic_solid_CRPS_K": 5.0,
        }
        return {
            "chained": {
                "transient_split_name": "pair_disjoint_stress_test",
                "curve_count": 4,
                "exact_initial_mean_solid_temperature_RMSE_K": 2.0,
                "steady_PINN_initial_mean_solid_temperature_RMSE_K": 5.0,
                "endpoint_novelty_groups": {
                    "both_steady_endpoints_unseen": {
                        "curve_count": 2,
                        "mean_source_initial_temperature_RMSE_K": 3.0,
                        "exact_initial_mean_solid_temperature_RMSE_K": 4.0,
                        "steady_PINN_initial_mean_solid_temperature_RMSE_K": 6.0,
                        "mean_error_amplification": 1.5,
                    }
                },
                "timing": {
                    "registered_steady_PINN_endpoint_timing": {
                        "unique_endpoint_count": 4,
                        "unique_endpoint_condition_ids": ["a", "b", "c", "d"],
                    }
                },
            },
            "chained_energy": {
                "role_metrics": {"test": {ENERGY_METRIC: 0.4}},
                "endpoint_novelty_metrics": {
                    "test": {
                        "both_steady_endpoints_unseen": {
                            "curve_count": 2,
                            ENERGY_METRIC: 0.5,
                        }
                    }
                },
            },
            "diffusion": {
                "metrics": {
                    "test": {
                        "ensemble_mean_solid_temperature_RMSE_K": 4.0,
                        **interval,
                    }
                },
                "endpoint_novelty_metrics": {
                    "both_steady_endpoints_unseen": {
                        "ensemble_mean_solid_temperature_RMSE_K": 5.0
                    }
                },
                "complete_chain_timing": {
                    "warm_start_complete_chain_inference_seconds_per_curve": 2.0,
                    "cold_start_complete_chain_inference_seconds_per_curve": 2.5,
                },
                "complete_chain_model_cost": {
                    "complete_chain_model_parameter_count": 505,
                    "complete_chain_training_seconds": 100.0,
                },
            },
            "diffusion_energy": {
                "role_metrics": {"test": {ENERGY_METRIC: 0.3}},
                "endpoint_novelty_metrics": {
                    "test": {
                        "both_steady_endpoints_unseen": {
                            "curve_count": 2,
                            ENERGY_METRIC: 0.4,
                        }
                    }
                },
            },
        }

    def test_interval_results_are_preserved_in_fused_summary(self) -> None:
        payload = build_payload(**self.inputs())
        self.assertTrue(payload["diffusion_improves_both"])
        self.assertEqual(
            payload[
                "fused_diffusion_unobserved_dynamic_solid_90pct_interval_coverage_fraction"
            ],
            0.82,
        )
        self.assertEqual(
            payload[
                "fused_diffusion_unobserved_dynamic_fluid_90pct_interval_mean_width_K"
            ],
            8.0,
        )
        self.assertEqual(
            payload["fused_diffusion_unobserved_dynamic_solid_CRPS_K"], 5.0
        )
        self.assertEqual(
            payload["strict_end_to_end_group"][
                "fused_diffusion_solid_temperature_RMSE_K"
            ],
            5.0,
        )
        self.assertTrue(
            payload["strict_end_to_end_group"][
                "diffusion_improves_solid_temperature"
            ]
        )
        self.assertEqual(payload["registered_steady_PINN_unique_endpoint_count"], 4)
        self.assertEqual(
            payload["complete_chain_model_cost"]["complete_chain_model_parameter_count"],
            505,
        )
        self.assertTrue(
            payload["strict_end_to_end_group"][
                "diffusion_improves_temperature_and_energy"
            ]
        )

    def test_missing_dynamic_interval_results_are_rejected(self) -> None:
        values = self.inputs()
        del values["diffusion"]["metrics"]["test"][
            "unobserved_dynamic_solid_90pct_interval_mean_width_K"
        ]
        with self.assertRaisesRegex(ValueError, "unobserved dynamic"):
            build_payload(**values)


if __name__ == "__main__":
    unittest.main()
