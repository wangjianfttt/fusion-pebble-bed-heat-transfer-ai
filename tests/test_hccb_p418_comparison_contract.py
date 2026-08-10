#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from check_hccb_p418_steady_result_current import implementation_files  # noqa: E402
from hccb_p418_comparison_contract import (  # noqa: E402
    STEADY_METRIC_CONTRACT,
    integrated_heat_transfer_metrics,
    run_provenance,
    sha256_file,
    validate_split_and_statistics,
)


class P418ComparisonContractTest(unittest.TestCase):
    def test_integrated_heat_transfer_uses_declared_flow_orientation(self) -> None:
        metrics = integrated_heat_transfer_metrics(
            internal_energy_flow_w=np.asarray([2.0, -3.0, -4.0, 1.0]),
            boundary_energy_flow_w=np.asarray([-5.0, 2.0, -7.0]),
            internal_kind=np.asarray([0, 2, 2, 1]),
            internal_kind_name=np.asarray(["fluid", "solid", "fluid_to_solid"]),
            boundary_kind=np.asarray([2, 0, 2]),
            boundary_kind_name=np.asarray(
                ["fluid:inlet", "fluid:outlet", "fluid:coolingWall"]
            ),
        )
        self.assertAlmostEqual(metrics["cooling_wall_heat_into_fluid_W"], 12.0)
        self.assertAlmostEqual(metrics["solid_to_fluid_interphase_net_W"], 7.0)
        self.assertAlmostEqual(
            metrics["fluid_solid_interphase_absolute_flow_W"], 7.0
        )

    def make_split_and_statistics(self, root: Path) -> tuple[Path, Path, np.ndarray]:
        identifiers = np.asarray(["a", "b", "c", "d"])
        split = root / "splits.json"
        split.write_text(
            json.dumps(
                {
                    "splits": {
                        "interleaved_all_ranges": {
                            "train": ["a", "b"],
                            "validation": ["c"],
                            "test": ["d"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        statistics = root / "statistics.json"
        statistics.write_text(
            json.dumps(
                {
                    "splits": {
                        "interleaved_all_ranges": {
                            "train_conditions": ["a", "b"],
                            "validation_conditions": ["c"],
                            "test_conditions": ["d"],
                        }
                    },
                    "source": {"split_file_sha256": sha256_file(split)},
                }
            ),
            encoding="utf-8",
        )
        return split, statistics, identifiers

    def test_statistics_must_match_exact_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, statistics, identifiers = self.make_split_and_statistics(root)
            exact, _ = validate_split_and_statistics(
                split_file=split,
                training_statistics=statistics,
                split_name="interleaved_all_ranges",
                condition_ids=identifiers,
            )
            self.assertEqual(exact["test"], ["d"])
            payload = json.loads(statistics.read_text(encoding="utf-8"))
            payload["splits"]["interleaved_all_ranges"]["train_conditions"] = ["b", "a"]
            statistics.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training-statistics train conditions differ"):
                validate_split_and_statistics(
                    split_file=split,
                    training_statistics=statistics,
                    split_name="interleaved_all_ranges",
                    condition_ids=identifiers,
                )

    def test_ordinary_split_still_requires_every_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, statistics, identifiers = self.make_split_and_statistics(root)
            payload = json.loads(split.read_text(encoding="utf-8"))
            payload["splits"]["interleaved_all_ranges"]["train"] = ["a"]
            split.write_text(json.dumps(payload), encoding="utf-8")
            statistics_payload = json.loads(statistics.read_text(encoding="utf-8"))
            statistics_payload["source"]["split_file_sha256"] = sha256_file(split)
            statistics_payload["splits"]["interleaved_all_ranges"][
                "train_conditions"
            ] = ["a"]
            statistics.write_text(json.dumps(statistics_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not partition all target"):
                validate_split_and_statistics(
                    split_file=split,
                    training_statistics=statistics,
                    split_name="interleaved_all_ranges",
                    condition_ids=identifiers,
                )

    def test_explicit_unused_conditions_allow_nested_training_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, statistics, identifiers = self.make_split_and_statistics(root)
            split_payload = json.loads(split.read_text(encoding="utf-8"))
            split_payload["splits"]["interleaved_all_ranges"]["train"] = ["a"]
            split_payload["splits"]["interleaved_all_ranges"]["unused"] = ["b"]
            split.write_text(json.dumps(split_payload), encoding="utf-8")
            statistics_payload = json.loads(statistics.read_text(encoding="utf-8"))
            statistics_payload["source"]["split_file_sha256"] = sha256_file(split)
            recorded = statistics_payload["splits"]["interleaved_all_ranges"]
            recorded["train_conditions"] = ["a"]
            recorded["unused_conditions"] = ["b"]
            statistics.write_text(json.dumps(statistics_payload), encoding="utf-8")
            exact, _ = validate_split_and_statistics(
                split_file=split,
                training_statistics=statistics,
                split_name="interleaved_all_ranges",
                condition_ids=identifiers,
            )
            self.assertEqual(exact["train"], ["a"])
            self.assertNotIn("unused", exact)

    def test_unused_conditions_cannot_overlap_or_leave_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, statistics, identifiers = self.make_split_and_statistics(root)
            split_payload = json.loads(split.read_text(encoding="utf-8"))
            split_payload["splits"]["interleaved_all_ranges"]["unused"] = ["b"]
            split.write_text(json.dumps(split_payload), encoding="utf-8")
            statistics_payload = json.loads(statistics.read_text(encoding="utf-8"))
            statistics_payload["source"]["split_file_sha256"] = sha256_file(split)
            statistics_payload["splits"]["interleaved_all_ranges"][
                "unused_conditions"
            ] = ["b"]
            statistics.write_text(json.dumps(statistics_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model roles and unused"):
                validate_split_and_statistics(
                    split_file=split,
                    training_statistics=statistics,
                    split_name="interleaved_all_ranges",
                    condition_ids=identifiers,
                )

            split_payload["splits"]["interleaved_all_ranges"]["train"] = ["a"]
            split_payload["splits"]["interleaved_all_ranges"]["unused"] = []
            split.write_text(json.dumps(split_payload), encoding="utf-8")
            statistics_payload["source"]["split_file_sha256"] = sha256_file(split)
            statistics_payload["splits"]["interleaved_all_ranges"][
                "train_conditions"
            ] = ["a"]
            statistics_payload["splits"]["interleaved_all_ranges"][
                "unused_conditions"
            ] = []
            statistics.write_text(json.dumps(statistics_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not partition all target"):
                validate_split_and_statistics(
                    split_file=split,
                    training_statistics=statistics,
                    split_name="interleaved_all_ranges",
                    condition_ids=identifiers,
                )

    def test_saved_result_is_rejected_after_an_input_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, statistics, identifiers = self.make_split_and_statistics(root)
            state = root / "state.npz"
            np.savez(state, condition_id=identifiers)
            mass = root / "mass.npz"
            energy = root / "energy.npz"
            np.savez(mass, value=np.asarray([1.0]))
            np.savez(energy, value=np.asarray([2.0]))
            split_ids, _ = validate_split_and_statistics(
                split_file=split,
                training_statistics=statistics,
                split_name="interleaved_all_ranges",
                condition_ids=identifiers,
            )
            provenance = run_provenance(
                architecture="response_surface",
                comparison_epochs=100,
                split_name="interleaved_all_ranges",
                split_case_ids=split_ids,
                common_inputs={
                    "state_targets": state,
                    "mass_targets": mass,
                    "energy_targets": energy,
                    "split_file": split,
                    "training_statistics": statistics,
                },
                implementation_files=implementation_files(CODE, "response_surface"),
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "architecture": "response_surface",
                        "split_name": "interleaved_all_ranges",
                        "comparison_requested_epochs": 100,
                        "run_provenance": provenance,
                    }
                ),
                encoding="utf-8",
            )
            command = [
                "python3",
                str(CODE / "check_hccb_p418_steady_result_current.py"),
                "--summary",
                str(summary),
                "--architecture",
                "response_surface",
                "--epochs",
                "100",
                "--split-name",
                "interleaved_all_ranges",
                "--state-targets",
                str(state),
                "--mass-targets",
                str(mass),
                "--energy-targets",
                str(energy),
                "--split-file",
                str(split),
                "--training-statistics",
                str(statistics),
            ]
            subprocess.run(command, check=True)
            np.savez(mass, value=np.asarray([9.0]))
            changed = subprocess.run(command, check=False)
            self.assertNotEqual(changed.returncode, 0)

    def test_saved_neural_result_is_rejected_for_a_different_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, statistics, identifiers = self.make_split_and_statistics(root)
            state = root / "state.npz"
            mass = root / "mass.npz"
            energy = root / "energy.npz"
            np.savez(state, condition_id=identifiers)
            np.savez(mass, value=np.asarray([1.0]))
            np.savez(energy, value=np.asarray([2.0]))
            split_ids, _ = validate_split_and_statistics(
                split_file=split,
                training_statistics=statistics,
                split_name="interleaved_all_ranges",
                condition_ids=identifiers,
            )
            provenance = run_provenance(
                architecture="pinn",
                comparison_epochs=100,
                split_name="interleaved_all_ranges",
                split_case_ids=split_ids,
                common_inputs={
                    "state_targets": state,
                    "mass_targets": mass,
                    "energy_targets": energy,
                    "split_file": split,
                    "training_statistics": statistics,
                },
                implementation_files=implementation_files(CODE, "pinn"),
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "architecture": "pinn",
                        "split_name": "interleaved_all_ranges",
                        "epochs": 100,
                        "training_seed": 20260717,
                        "run_provenance": provenance,
                    }
                ),
                encoding="utf-8",
            )
            base = [
                "python3",
                str(CODE / "check_hccb_p418_steady_result_current.py"),
                "--summary",
                str(summary),
                "--architecture",
                "pinn",
                "--epochs",
                "100",
                "--split-name",
                "interleaved_all_ranges",
                "--state-targets",
                str(state),
                "--mass-targets",
                str(mass),
                "--energy-targets",
                str(energy),
                "--split-file",
                str(split),
                "--training-statistics",
                str(statistics),
            ]
            subprocess.run(base + ["--training-seed", "20260717"], check=True)
            wrong = subprocess.run(
                base + ["--training-seed", "20260718"], check=False
            )
            self.assertNotEqual(wrong.returncode, 0)


if __name__ == "__main__":
    unittest.main()
