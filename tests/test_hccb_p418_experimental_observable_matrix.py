#!/usr/bin/env python3

from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "parameters/hccb_p418_experimental_observable_matrix.csv"
MANIFEST = ROOT / "parameters/literature_parameter_manifest.csv"


class P418ExperimentalObservableMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
            cls.source_ids = {
                row["parameter_id"].strip() for row in csv.DictReader(stream)
            }
        with MATRIX.open(newline="", encoding="utf-8-sig") as stream:
            cls.rows = list(csv.DictReader(stream))

    def test_every_row_has_traceable_sources(self) -> None:
        self.assertGreaterEqual(len(self.rows), 10)
        for row in self.rows:
            ids = {
                item.strip()
                for item in row["文献参数编号"].split(";")
                if item.strip()
            }
            self.assertTrue(ids, row["观测量"])
            self.assertTrue(ids.issubset(self.source_ids), (row["观测量"], ids))
            self.assertTrue(row["文献依据"].strip(), row["观测量"])
            self.assertTrue(row["限制"].strip(), row["观测量"])

    def test_interphase_heat_is_not_called_direct_measurement(self) -> None:
        rows = [row for row in self.rows if row["观测量"] == "气固界面净换热"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["获得方式"], "由稳态热量收支反推")
        self.assertIn("不能", row["限制"])
        self.assertIn("储热项", row["限制"])

    def test_core_experimental_quantities_are_present(self) -> None:
        names = {row["观测量"] for row in self.rows}
        required = {
            "入口质量流量或表观速度",
            "入口与出口气体温度",
            "轴向压降",
            "冷却侧带走的热功率",
            "颗粒床内部温度",
            "输入加热功率",
            "气固界面净换热",
            "热阶跃响应时间",
        }
        self.assertTrue(required.issubset(names), sorted(required - names))


if __name__ == "__main__":
    unittest.main()
