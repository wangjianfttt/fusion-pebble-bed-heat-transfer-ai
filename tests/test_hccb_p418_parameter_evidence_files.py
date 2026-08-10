#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_parameter_evidence_summary import build  # noqa: E402
from verify_hccb_p418_parameter_evidence_files import verify  # noqa: E402


class P418ParameterEvidenceFilesTest(unittest.TestCase):
    def test_all_physical_parameters_have_source_files(self) -> None:
        summary = verify(
            ROOT / "parameters/hccb_p418_physical_parameter_sources.csv",
            ROOT / "parameters/hccb_p418_physical_parameter_evidence_files.csv",
            ROOT / "parameters/hccb_p418_equation_input_map.csv",
        )
        self.assertEqual(summary["physical_parameter_count"], 22)
        self.assertEqual(summary["equation_map_row_count"], 31)
        self.assertTrue(summary["p429_derivative_check"])
        self.assertAlmostEqual(summary["p430_molar_mass_g_mol"], 119.841)
        self.assertEqual(summary["new_physical_parameters"], [])

    def test_chinese_cross_reference_contains_all_ids(self) -> None:
        text = build(
            ROOT / "parameters/hccb_p418_physical_parameter_sources.csv",
            ROOT / "parameters/hccb_p418_physical_parameter_evidence_files.csv",
            ROOT / "parameters/hccb_p418_equation_input_map.csv",
        )
        for parameter_id in (
            "P048", "P049", "P050", "P070", "P071", "P092", "P388",
            "P389", "P390", "P403", "P404", "P406", "P418", "P423",
            "P424", "P425", "P426", "P427", "P428", "P429", "P430", "P431",
        ):
            self.assertIn(f"|{parameter_id}|", text)
        self.assertIn("没有把元数据文件写成论文全文", text)
        self.assertIn("项参数的来源文件", text)


if __name__ == "__main__":
    unittest.main()
