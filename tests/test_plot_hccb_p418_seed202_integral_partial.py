from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from plot_hccb_p418_seed202_integral_partial import parse_condition, read_rows  # noqa: E402


def test_condition_parser_uses_registered_factor_levels() -> None:
    assert parse_condition("u0p15_T700_q6p85") == (0.15, 700, 6.85)


def test_public_seed202_table_contains_nine_unique_finite_cases() -> None:
    rows = read_rows(
        ROOT / "results/hccb_p418_public_figure_data/seed202_integral_comparison_9.csv"
    )
    assert len(rows) == 9
    assert len({row["condition_id"] for row in rows}) == 9
    assert {row["velocity"] for row in rows} == {0.05, 0.15, 0.25}
    assert {row["temperature"] for row in rows} == {300.0, 700.0, 900.0}


def test_public_data_regenerates_registered_two_by_two_figure(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/plot_hccb_p418_seed202_integral_partial.py"),
            "--comparison-csv",
            str(ROOT / "results/hccb_p418_public_figure_data/seed202_integral_comparison_9.csv"),
            "--summary-json",
            str(ROOT / "results/hccb_p418_public_figure_data/seed202_integral_summary.json"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    metadata = json.loads(
        (tmp_path / "hccb_p418_seed202_integral_9.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "complete_p418_seed202_integral_9_figure"
    assert metadata["panel_d_xscale"] == {
        "type": "symlog",
        "linear_threshold_percent": 1.0,
        "purpose": (
            "show sub-percent thermal changes and 14.7--18.0 percent "
            "pressure-drop changes on one axis"
        ),
    }
    assert metadata["condition_count"] == 9
    assert metadata["figure_size_inch"] == [5.4, 4.04]
    assert metadata["layout"] == "two columns by two rows"
    assert metadata["uniform_panel_dimensions"] is True
    bounds = metadata["panel_bounds_figure_fraction"]
    assert len(bounds) == 4
    assert len({round(item[2], 12) for item in bounds}) == 1
    assert len({round(item[3], 12) for item in bounds}) == 1
    for suffix in ("pdf", "svg", "png"):
        assert (tmp_path / f"hccb_p418_seed202_integral_9.{suffix}").stat().st_size > 0
