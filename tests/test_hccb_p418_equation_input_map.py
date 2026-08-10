#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_source_contract import ALL_STEADY_PHYSICAL_PARAMETER_IDS  # noqa: E402
from hccb_p418_transient_regional_physics import (  # noqa: E402
    TRANSIENT_STORAGE_PARAMETER_IDS,
)


class P418EquationInputMapTest(unittest.TestCase):
    def setUp(self) -> None:
        with (ROOT / "parameters/hccb_p418_physical_parameter_sources.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            self.source_ids = {row["parameter_id"] for row in csv.DictReader(stream)}
        with (ROOT / "parameters/hccb_p418_equation_input_map.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            self.rows = list(csv.DictReader(stream))

    def test_every_mapped_source_id_exists(self) -> None:
        used: set[str] = set()
        for row in self.rows:
            ids = {item.strip() for item in row["文献参数编号"].split(";") if item.strip()}
            self.assertTrue(ids, row["物理量或方程"])
            self.assertTrue(ids.issubset(self.source_ids), (row["物理量或方程"], ids))
            used.update(ids)
        required = set(ALL_STEADY_PHYSICAL_PARAMETER_IDS) | set(
            TRANSIENT_STORAGE_PARAMETER_IDS
        )
        self.assertTrue(required.issubset(used), sorted(required - used))

    def test_every_python_implementation_exists(self) -> None:
        for row in self.rows:
            for item in row["Python实现"].split(";"):
                path = ROOT / item.strip()
                self.assertTrue(path.is_file(), path)

    def test_required_equation_families_are_present(self) -> None:
        names = {row["物理量或方程"] for row in self.rows}
        required = {
            "质量守恒",
            "流体能量守恒",
            "固体能量守恒",
            "流固界面温度与热流连续",
            "流体储热",
            "固体储热",
        }
        self.assertTrue(required.issubset(names), sorted(required - names))


if __name__ == "__main__":
    unittest.main()
