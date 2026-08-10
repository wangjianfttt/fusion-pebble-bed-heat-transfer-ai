#!/usr/bin/env python3

from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from analyze_hccb_p418_transient_time_resolution import analyze  # noqa: E402


class P418TransientTimeResolutionTest(unittest.TestCase):
    def test_uses_literature_properties_and_resolves_early_response(self) -> None:
        result = analyze(ROOT / "parameters/hccb_p418_physical_parameter_sources.csv")
        self.assertEqual(result["new_physical_parameters"], [])
        self.assertEqual(result["parameter_ids"], ["P048", "P092", "P403", "P418", "P429", "P430"])
        self.assertGreater(result["maximum_particle_radial_diffusion_scale_s"], 0.1)
        self.assertLess(result["maximum_particle_radial_diffusion_scale_s"], 2.0)
        self.assertGreater(result["previous_25_s_output_interval_to_maximum_scale_ratio"], 10.0)
        self.assertEqual(result["selected_output_schedule"][0]["interval_s"], 0.005)
        self.assertEqual(result["selected_time_step_schedule"][0]["delta_t_s"], 1.0e-5)
        self.assertGreater(result["steps_per_fastest_crossing"], 100.0)
        self.assertLess(
            result["projected_high_velocity_p95_Courant_at_initial_step"],
            2.0,
        )
        self.assertGreater(
            result["particle_scale_steps_in_1_to_25_s_stage"]["minimum"],
            40.0,
        )


if __name__ == "__main__":
    unittest.main()
