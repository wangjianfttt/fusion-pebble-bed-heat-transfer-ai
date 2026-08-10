#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "code/run_hccb_dense_cht_p418_matrix_parallel.sh"


class HCCBP418ParallelRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = RUNNER.read_text(encoding="utf-8")

    def test_default_parallel_use_fits_112_core_machine(self) -> None:
        self.assertIn("NP_PER_CASE=${NP_PER_CASE:-32}", self.text)
        self.assertIn("CONCURRENT_CASES=${CONCURRENT_CASES:-3}", self.text)

    def test_only_one_matrix_runner_can_write_the_case_set(self) -> None:
        self.assertIn("LOCK_FILE=${LOCK_FILE:-${MATRIX_ROOT}.run.lock}", self.text)
        self.assertIn("flock -n 9", self.text)

    def test_cloud_migration_marker_stops_before_an_unfinished_case(self) -> None:
        self.assertIn("PAUSE_NEW_P418_CASES_FOR_CLOUD_MIGRATION", self.text)
        complete = self.text.index('if [[ -f ${case_dir}/formal_sample_complete.json ]]')
        pause = self.text.index('if [[ -f ${PAUSE_NEW_CASES_FILE} ]]')
        dictionary_write = self.text.index('local property_tmp=')
        self.assertLess(complete, pause)
        self.assertLess(pause, dictionary_write)
        self.assertIn("return 75", self.text)
        self.assertIn("export PAUSE_NEW_CASES_FILE", self.text)

    def test_completed_pilot_cases_are_reused(self) -> None:
        self.assertIn("formal_sample_complete.json", self.text)
        self.assertIn('cp -al "${pilot_case}" "${target}"', self.text)

    def test_processor_directories_are_removed_by_finalizer(self) -> None:
        self.assertIn("finalize_hccb_cht_case.sh", self.text)

    def test_parallel_cases_do_not_share_a_writable_property_dictionary(self) -> None:
        self.assertIn("physicalProperties.tmp.$$", self.text)
        self.assertIn('mv -f "${property_tmp}"', self.text)
        self.assertIn("-entry runTimeModifiable -set false", self.text)

    def test_solver_and_finalizer_failures_stop_the_case(self) -> None:
        self.assertIn('echo "solver failed ${condition}"', self.text)
        self.assertIn('echo "finalization failed ${condition}"', self.text)
        self.assertIn("return 1", self.text)

    def test_completed_cases_update_a_serialized_partial_physics_summary(self) -> None:
        for required in (
            "PARTIAL_PHYSICS_DIR=",
            "PARTIAL_PHYSICS_LOCK=",
            "update_partial_physics_summary",
            "summarize_hccb_p418_completed_matrix_physics.py",
            "--time-from-completion-marker",
            'flock -x 8',
            '8>"${PARTIAL_PHYSICS_LOCK}"',
            'mv -f "${stdout_tmp}"',
        ):
            self.assertIn(required, self.text)

    def test_partial_summary_failure_does_not_discard_a_completed_case(self) -> None:
        self.assertIn(
            'echo "warning: partial physics summary could not be updated"',
            self.text,
        )
        self.assertIn(
            'echo "warning: partial physics summary file update failed"',
            self.text,
        )
        self.assertIn("return 0", self.text)

    def test_runner_has_valid_shell_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_interrupted_case_resumes_only_from_complete_cht_fields(self) -> None:
        for field in (
            "fluid/T",
            "fluid/U",
            "fluid/p",
            "fluid/p_rgh",
            "solid/T",
            "uniform/time",
        ):
            self.assertIn(field, self.text)
        self.assertIn('resume ${condition} from ${restart_time} s', self.text)
        self.assertIn('-entry startTime -set "${restart_time}"', self.text)
        self.assertIn('if [[ -n ${restart_time} && ${restart_time} != 0 ]]', self.text)


if __name__ == "__main__":
    unittest.main()
