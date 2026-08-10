import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_high_re_reference_table import build_table, load_summary


SUMMARY = ROOT / "results" / "hccb_p418_high_re_step_response_analysis" / "summary.json"


def test_formal_summary_builds_six_row_table() -> None:
    payload = load_summary(SUMMARY)
    table = build_table(payload)
    assert table.count(r"\\") == 7
    assert r"$300\rightarrow900$ K" in table
    assert r"$0.05\rightarrow0.25$ m s$^{-1}$" in table
    assert r"$4.85\rightarrow8.85$ MW m$^{-3}$" in table
    assert "+231.85" in table
    assert "-29.30" in table
    assert "11.41" in table


def test_incomplete_summary_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    payload["curve_count"] = 5
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="six complete"):
        load_summary(path)


def test_large_mass_residual_is_rejected() -> None:
    payload = load_summary(SUMMARY)
    payload["cases"][0]["maximum_absolute_mass_residual_kg_s"] = 1.0e-6
    with pytest.raises(ValueError, match="mass residual"):
        build_table(payload)
