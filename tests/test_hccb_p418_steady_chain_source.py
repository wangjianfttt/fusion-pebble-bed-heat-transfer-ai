import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from select_hccb_p418_steady_chain_source import choose  # noqa: E402


class P418SteadyChainSourceTest(unittest.TestCase):
    def write_summary(
        self,
        path: Path,
        *,
        validation_loss: float,
        epochs: int,
        fingerprint: str = "same-physical-data",
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "architecture": "pinn",
                    "split_name": "interleaved_all_ranges",
                    "split_case_ids": {
                        "train": ["train-a", "train-b"],
                        "validation": ["validation-a"],
                        "test": ["test-a"],
                    },
                    "epochs": epochs,
                    "best_epoch": epochs - 1,
                    "best_validation_total_loss": validation_loss,
                    "run_provenance": {
                        "common_comparison_fingerprint": fingerprint,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def write_followup_plan(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "architecture": "pinn",
                            "split": "interleaved_all_ranges",
                            "followup_result_directory": "followup",
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_lower_followup_validation_loss_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            base_summary = project / "base" / "summary.json"
            followup_summary = project / "followup" / "summary.json"
            plan = project / "plan.json"
            self.write_summary(base_summary, validation_loss=0.20, epochs=100)
            self.write_summary(followup_summary, validation_loss=0.08, epochs=500)
            self.write_followup_plan(plan)

            result = choose(
                base_summary=base_summary,
                followup_plan=plan,
                project_root=project,
                architecture="pinn",
                split_name="interleaved_all_ranges",
            )

            self.assertEqual(
                Path(result["selected_summary"]).resolve(),
                followup_summary.resolve(),
            )
            self.assertEqual(result["selected_epochs"], 500)
            self.assertFalse(result["independent_test_used_for_selection"])
            self.assertEqual(result["selection_data"], "validation conditions only")

    def test_initial_model_is_kept_when_its_validation_loss_is_lower(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            base_summary = project / "base" / "summary.json"
            followup_summary = project / "followup" / "summary.json"
            plan = project / "plan.json"
            self.write_summary(base_summary, validation_loss=0.05, epochs=100)
            self.write_summary(followup_summary, validation_loss=0.09, epochs=500)
            self.write_followup_plan(plan)

            result = choose(
                base_summary=base_summary,
                followup_plan=plan,
                project_root=project,
                architecture="pinn",
                split_name="interleaved_all_ranges",
            )

            self.assertEqual(
                Path(result["selected_summary"]).resolve(),
                base_summary.resolve(),
            )
            self.assertEqual(result["selected_epochs"], 100)

    def test_different_physical_data_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            base_summary = project / "base" / "summary.json"
            followup_summary = project / "followup" / "summary.json"
            plan = project / "plan.json"
            self.write_summary(base_summary, validation_loss=0.20, epochs=100)
            self.write_summary(
                followup_summary,
                validation_loss=0.08,
                epochs=500,
                fingerprint="different-physical-data",
            )
            self.write_followup_plan(plan)

            with self.assertRaisesRegex(ValueError, "different physical data"):
                choose(
                    base_summary=base_summary,
                    followup_plan=plan,
                    project_root=project,
                    architecture="pinn",
                    split_name="interleaved_all_ranges",
                )


if __name__ == "__main__":
    unittest.main()
