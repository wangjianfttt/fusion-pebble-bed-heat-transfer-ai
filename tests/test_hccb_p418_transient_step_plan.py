#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class P418TransientStepPlanTest(unittest.TestCase):
    def test_plan_uses_published_endpoints_and_one_change_per_sequence(self) -> None:
        result = subprocess.run(
            ["python3", str(ROOT / "code/validate_hccb_p418_transient_step_plan.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads(result.stdout)
        self.assertEqual(summary["sequence_count"], 12)
        self.assertEqual(summary["new_physical_parameters"], [])
        self.assertEqual(set(summary["family_counts"].values()), {4})
        plan = json.loads((ROOT / "parameters/hccb_p418_transient_step_plan.json").read_text())
        self.assertEqual(
            plan["transient_model"],
            "thermal_step_with_quasi_steady_target_hydrodynamics",
        )
        self.assertIn(
            "the label 200 is an iteration index rather than 200 s",
            plan["initial_state_rule"],
        )
        self.assertNotIn("Five early endpoints", plan["initial_state_rule"])
        self.assertEqual(plan["numerical_time_design"]["ddt_scheme"], "Euler")
        self.assertEqual(plan["numerical_time_design"]["delta_t_s"], 1.0e-5)
        time_steps = plan["numerical_time_design"]["time_step_schedule"]
        self.assertEqual(
            time_steps,
            [
                {"start_s": 0, "end_s": 0.1, "delta_t_s": 1.0e-5},
                {"start_s": 0.1, "end_s": 1, "delta_t_s": 5.0e-4},
                {"start_s": 1, "end_s": 25, "delta_t_s": 1.0e-2},
                {"start_s": 25, "end_s": 300, "delta_t_s": 1.25e-1},
            ],
        )
        schedule = plan["numerical_time_design"]["field_write_schedule"]
        self.assertEqual(schedule[0], {"start_s": 0, "end_s": 0.1, "interval_s": 0.005})
        self.assertEqual(schedule[-1], {"start_s": 25, "end_s": 300, "interval_s": 25})

    def test_time_step_sensitivity_changes_no_physical_parameter(self) -> None:
        payload = json.loads(
            (ROOT / "parameters/hccb_p418_thermal_step_timestep_sensitivity.json").read_text()
        )
        self.assertEqual(payload["new_physical_parameters"], [])
        self.assertEqual(payload["delta_t_s"], [4.0e-5, 2.0e-5, 1.0e-5])
        self.assertEqual(payload["duration_s"], 25)


if __name__ == "__main__":
    unittest.main()
