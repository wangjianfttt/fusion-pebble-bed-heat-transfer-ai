#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_source_summary.py"


class P418SourceSummaryTest(unittest.TestCase):
    def test_current_physical_and_model_sources_form_one_complete_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            json_output = output / "summary.json"
            markdown_output = output / "summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--physical-sources",
                    str(ROOT / "parameters/hccb_p418_physical_parameter_sources.csv"),
                    "--equation-map",
                    str(ROOT / "parameters/hccb_p418_equation_input_map.csv"),
                    "--actual-cases",
                    str(ROOT / "results/hccb_p418_60_actual_case_input_check/summary.json"),
                    "--step-plan",
                    str(ROOT / "parameters/hccb_p418_transient_step_plan.json"),
                    "--architecture-sources",
                    str(ROOT / "parameters/hccb_p418_ai_architecture_sources.json"),
                    "--numerical-settings",
                    str(ROOT / "parameters/hccb_p418_model_numerical_settings.csv"),
                    "--json-output",
                    str(json_output),
                    "--markdown-output",
                    str(markdown_output),
                ],
                check=True,
            )
            summary = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(summary["actual_openfoam_case_count"], 60)
            self.assertEqual(summary["transient_sequence_count"], 12)
            self.assertEqual(summary["new_physical_parameters"], [])
            self.assertTrue(summary["all_physical_parameter_statuses_extracted"])
            self.assertTrue(summary["literature_manifest_records_match_physical_table"])
            self.assertTrue(summary["all_physical_parameters_are_used_by_equations"])
            self.assertTrue(summary["actual_operating_points_match_P418"])
            self.assertTrue(summary["actual_cases_share_one_fixed_mesh"])
            self.assertTrue(summary["packing_and_meshing_inputs_match_sources"])
            self.assertTrue(summary["fine_local_crop_is_a_computed_geometry_result"])
            self.assertEqual(summary["fine_local_retained_particle_fragments"], 125)
            self.assertAlmostEqual(summary["fine_local_triangulated_porosity"], 0.38679132863021193)
            self.assertGreaterEqual(summary["physical_parameter_count"], 18)
            self.assertGreaterEqual(summary["architecture_count"], 9)
            self.assertTrue(
                summary["all_model_setting_source_and_implementation_files_exist"]
            )
            text = markdown_output.read_text(encoding="utf-8")
            self.assertIn("P418", text)
            self.assertIn("扩散", text)
            self.assertIn("OpenFOAM", text)
            self.assertIn("结论先说", text)
            self.assertIn("颗粒片段", text)

    def test_geometry_source_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            actual_path = output / "actual.json"
            actual = json.loads(
                (ROOT / "results/hccb_p418_60_actual_case_input_check/summary.json").read_text(
                    encoding="utf-8"
                )
            )
            actual["geometry_sources"][
                "all_published_geometry_and_meshing_inputs_match"
            ] = False
            actual_path.write_text(
                json.dumps(actual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--actual-cases",
                    str(actual_path),
                    "--json-output",
                    str(output / "summary.json"),
                    "--markdown-output",
                    str(output / "summary.md"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("packing or meshing", result.stderr)

    def test_non_extracted_physical_parameter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = output / "literature.csv"
            with (ROOT / "parameters/literature_parameter_manifest.csv").open(
                newline="", encoding="utf-8-sig"
            ) as stream:
                rows = list(csv.DictReader(stream))
                fieldnames = stream.seek(0) or next(csv.reader(stream))
            for row in rows:
                if row["parameter_id"] == "P048":
                    row["status"] = "blocked_missing_value"
                    break
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--literature-manifest",
                    str(manifest),
                    "--json-output",
                    str(output / "summary.json"),
                    "--markdown-output",
                    str(output / "summary.md"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("P048", result.stderr)

    def test_physical_parameter_missing_from_equation_map_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            equation_map = output / "equations.csv"
            with (ROOT / "parameters/hccb_p418_equation_input_map.csv").open(
                newline="", encoding="utf-8-sig"
            ) as stream:
                rows = list(csv.DictReader(stream))
                fieldnames = stream.seek(0) or next(csv.reader(stream))
            rows = [row for row in rows if "P424" not in row["文献参数编号"]]
            with equation_map.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--equation-map",
                    str(equation_map),
                    "--json-output",
                    str(output / "summary.json"),
                    "--markdown-output",
                    str(output / "summary.md"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("P424", result.stderr)


if __name__ == "__main__":
    unittest.main()
