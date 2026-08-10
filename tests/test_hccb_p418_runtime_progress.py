#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p418_progress",
    ROOT / "code/report_hccb_p418_runtime_progress.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HCCBP418RuntimeProgressTest(unittest.TestCase):
    def test_progress_uses_completed_case_clock_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, seconds in enumerate((100.0, 200.0, None)):
                case = root / f"u0p0{index + 1}_T300_q4p85"
                case.mkdir()
                if seconds is not None:
                    (case / "formal_sample_complete.json").write_text("{}\n", encoding="utf-8")
                    (case / "log.foamMultiRun.formal").write_text(
                        f"ExecutionTime = 1 s ClockTime = {seconds:g} s\n",
                        encoding="utf-8",
                    )
            result = MODULE.progress(root, concurrent_cases=2)
            self.assertEqual(result["total_cases"], 3)
            self.assertEqual(result["completed_cases"], 2)
            self.assertEqual(result["median_completed_case_accumulated_clock_seconds"], 150.0)
            self.assertAlmostEqual(result["simple_remaining_walltime_estimate_hours"], 150.0 / 2 / 3600)

    def test_accumulated_clock_time_sums_restart_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.foamMultiRun.formal"
            log.write_text(
                "ExecutionTime = 1 s ClockTime = 10 s\n"
                "ExecutionTime = 2 s ClockTime = 20 s\n"
                "===== resumed from complete parallel time 25 at "
                "2026-07-19T18:00:00+08:00 =====\n"
                "ExecutionTime = 1 s ClockTime = 5 s\n"
                "ExecutionTime = 2 s ClockTime = 12 s\n"
                "ExecutionTime = 0 s ClockTime = 0 s\n",
                encoding="utf-8",
            )
            self.assertEqual(MODULE.accumulated_solver_clock_time(log), 32.0)

    def test_progress_includes_completed_case_postprocess_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = root / "u0p05_T300_q4p85"
            remaining = root / "u0p10_T300_q4p85"
            completed.mkdir()
            remaining.mkdir()
            marker = completed / "formal_sample_complete.json"
            marker.write_text(
                json.dumps({"time": "200", "formal_finalization_seconds": 40.0}),
                encoding="utf-8",
            )
            (completed / "log.foamMultiRun.formal").write_text(
                "ClockTime = 100 s\n", encoding="utf-8"
            )
            result = MODULE.progress(root, concurrent_cases=1)
            self.assertEqual(result["mean_completed_case_postprocess_seconds"], 40.0)
            self.assertEqual(result["mean_completed_case_total_seconds"], 140.0)
            self.assertAlmostEqual(
                result["simple_remaining_walltime_estimate_hours"], 140.0 / 3600.0
            )

    def test_copied_marker_without_explicit_finalization_time_is_not_timed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            case.mkdir()
            (case / "formal_sample_complete.json").write_text(
                '{"time": "200"}\n', encoding="utf-8"
            )
            (case / "cht_result_summary_200.json").write_text("{}\n", encoding="utf-8")
            self.assertIsNone(MODULE.completed_case_postprocess_seconds(case))

    def test_restart_segment_ignores_old_later_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "u0p05_T500_q6p85"
            case.mkdir()
            (case / "cht_smoke_metadata.json").write_text(
                '{"end_time": 200}\n', encoding="utf-8"
            )
            log = case / "log.foamMultiRun.formal"
            log.write_text(
                "Time = 37s\n"
                "===== resumed from complete parallel time 25 at "
                "2026-07-19T18:00:00+08:00 =====\n"
                "Time = 26s\n"
                "Time = 28s\n",
                encoding="utf-8",
            )
            now = dt.datetime.fromisoformat("2026-07-19T19:00:00+08:00")
            result = MODULE.progress(
                root,
                concurrent_cases=3,
                now=now,
                active_log_max_age_seconds=1.0e12,
                parallel_ranks=2,
            )
            self.assertEqual(result["active_case_count"], 1)
            active = result["active_cases"][0]
            self.assertEqual(active["restart_iteration"], 25.0)
            self.assertEqual(active["current_iteration"], 28.0)
            self.assertAlmostEqual(active["steady_iterations_per_wall_hour"], 3.0)
            self.assertEqual(active["solver_time_semantics"], "steady_iteration_index")
            self.assertIsNone(active["physical_time_s"])
            self.assertAlmostEqual(active["estimated_hours_to_case_end"], 172.0 / 3.0)

    def test_parallel_checkpoint_requires_all_new_rank_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "u0p05_T500_q6p85"
            case.mkdir()
            restart_text = "2026-07-19T18:00:00+08:00"
            restart_epoch = dt.datetime.fromisoformat(restart_text).timestamp()
            for rank in range(2):
                for field in MODULE.RESTART_FIELDS:
                    path = case / f"processor{rank}" / "26" / field
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("field\n", encoding="utf-8")
                    os.utime(path, (restart_epoch + 60.0, restart_epoch + 60.0))
            self.assertEqual(
                MODULE.latest_new_complete_parallel_iteration(
                    case,
                    restart_iteration=25.0,
                    restart_wall_time=restart_text,
                    parallel_ranks=2,
                ),
                26.0,
            )
            stale = case / "processor1" / "26" / "solid/T"
            os.utime(stale, (restart_epoch - 60.0, restart_epoch - 60.0))
            self.assertEqual(
                MODULE.latest_new_complete_parallel_iteration(
                    case,
                    restart_iteration=25.0,
                    restart_wall_time=restart_text,
                    parallel_ranks=2,
                ),
                25.0,
            )

    def test_fresh_run_uses_latest_complete_parallel_fields_without_resume_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "u0p10_T500_q4p85"
            case.mkdir()
            (case / "cht_smoke_metadata.json").write_text(
                '{"end_time": 200}\n', encoding="utf-8"
            )
            log = case / "log.foamMultiRun.formal"
            log.write_text("solver output is buffered\n", encoding="utf-8")
            for rank in range(2):
                for field in MODULE.RESTART_FIELDS:
                    path = case / f"processor{rank}" / "9" / field
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("field\n", encoding="utf-8")
            result = MODULE.progress(
                root,
                concurrent_cases=3,
                active_log_max_age_seconds=1.0e12,
                parallel_ranks=2,
            )
            self.assertEqual(result["active_case_count"], 1)
            active = result["active_cases"][0]
            self.assertIsNone(active["restart_iteration"])
            self.assertEqual(active["current_iteration"], 9.0)
            self.assertEqual(active["latest_new_complete_parallel_iteration"], 9.0)
            self.assertEqual(active["phase"], "solving")

    def test_observed_time_accepts_partial_new_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "u0p05_T500_q6p85"
            restart_text = "2026-07-19T18:00:00+08:00"
            restart_epoch = dt.datetime.fromisoformat(restart_text).timestamp()
            old = case / "processor0" / "37" / "uniform" / "probe.dat"
            old.parent.mkdir(parents=True)
            old.write_text("old\n", encoding="utf-8")
            os.utime(old, (restart_epoch - 60.0, restart_epoch - 60.0))
            new = case / "processor0" / "36" / "uniform" / "probe.dat"
            new.parent.mkdir(parents=True)
            new.write_text("new\n", encoding="utf-8")
            os.utime(new, (restart_epoch + 60.0, restart_epoch + 60.0))
            self.assertEqual(
                MODULE.latest_new_observed_iteration(
                    case,
                    restart_iteration=25.0,
                    restart_wall_time=restart_text,
                ),
                36.0,
            )


if __name__ == "__main__":
    unittest.main()
