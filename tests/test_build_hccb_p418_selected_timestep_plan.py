import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_selected_timestep_plan.py"


class SelectedTimestepPlanTest(unittest.TestCase):
    def test_cloud_config_path_falls_back_to_declared_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parameters = root / "parameters"
            parameters.mkdir()
            config = parameters / "config.json"
            config.write_text(json.dumps({"delta_t_s": [1.0, 0.5, 0.25]}))
            base = parameters / "plan.json"
            base.write_text(
                json.dumps(
                    {
                        "numerical_time_design": {
                            "delta_t_s": 1.0,
                            "time_step_schedule": [
                                {"start_s": 0.0, "end_s": 1.0, "delta_t_s": 1.0}
                            ],
                        },
                        "time_step_sensitivity_file": "parameters/config.json",
                        "new_physical_parameters": [],
                    }
                )
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "completed_p418_thermal_timestep_sensitivity",
                        "config": "/cloud/path/that/is/not-mounted/config.json",
                        "selected_delta_t_s": 0.25,
                        "selected_time_step_schedule": [
                            {"start_s": 0.0, "end_s": 1.0, "delta_t_s": 0.25}
                        ],
                        "formal_selection_rule": "finest_completed_predeclared_step",
                        "selection_scope": "numerical time resolution only",
                    }
                )
            )
            output = root / "selected.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base-plan",
                    str(base),
                    "--sensitivity-summary",
                    str(summary),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                json.loads(output.read_text())["numerical_time_design"]["delta_t_s"],
                0.25,
            )

    def test_selected_numerical_step_changes_no_physical_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps({"delta_t_s": [1.0, 0.5, 0.25]}))
            base = root / "plan.json"
            base.write_text(
                json.dumps(
                    {
                        "numerical_time_design": {
                            "delta_t_s": 1.0,
                            "time_step_schedule": [
                                {"start_s": 0.0, "end_s": 1.0, "delta_t_s": 1.0}
                            ],
                        },
                        "time_step_sensitivity_file": str(config),
                        "new_physical_parameters": [],
                    }
                )
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "completed_p418_thermal_timestep_sensitivity",
                        "config": str(config),
                        "selected_delta_t_s": 0.25,
                        "selected_time_step_schedule": [
                            {"start_s": 0.0, "end_s": 1.0, "delta_t_s": 0.25}
                        ],
                        "formal_selection_rule": "finest_completed_predeclared_step",
                        "selection_scope": "numerical time resolution only",
                    }
                )
            )
            output = root / "selected.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base-plan",
                    str(base),
                    "--sensitivity-summary",
                    str(summary),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            selected = json.loads(output.read_text())
            self.assertEqual(selected["numerical_time_design"]["delta_t_s"], 0.25)
            self.assertEqual(
                selected["numerical_time_design"]["time_step_schedule"][0]["delta_t_s"],
                0.25,
            )
            self.assertEqual(selected["new_physical_parameters"], [])

    def test_coarser_selected_step_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps({"delta_t_s": [1.0, 0.5, 0.25]}))
            base = root / "plan.json"
            base.write_text(
                json.dumps(
                    {
                        "numerical_time_design": {"delta_t_s": 1.0},
                        "time_step_sensitivity_file": "unused.json",
                    }
                )
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "completed_p418_thermal_timestep_sensitivity",
                        "config": str(config),
                        "selected_delta_t_s": 0.5,
                        "formal_selection_rule": "finest_completed_predeclared_step",
                        "selection_scope": "numerical time resolution only",
                    }
                )
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base-plan",
                    str(base),
                    "--sensitivity-summary",
                    str(summary),
                    "--output",
                    str(root / "selected.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("finest predeclared time step", result.stderr)


if __name__ == "__main__":
    unittest.main()
