from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from check_hccb_p418_parallel_time_fields import REQUIRED_FIELDS, check_case  # noqa: E402


def write_time(case: Path, rank: int, time_name: str, *, omit: str | None = None) -> None:
    time_dir = case / f"processor{rank}" / time_name
    for field in REQUIRED_FIELDS:
        if field == omit:
            continue
        path = time_dir / field
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("field\n", encoding="utf-8")


def test_reports_latest_complete_and_first_incomplete(tmp_path: Path) -> None:
    for rank in range(2):
        write_time(tmp_path, rank, "1")
        write_time(tmp_path, rank, "5", omit="fluid/T" if rank == 1 else None)

    result = check_case(tmp_path, 2)

    assert result["latest_complete_time_name"] == "1"
    assert result["first_incomplete_time_name"] == "5"
    assert result["complete_time_count"] == 1
    assert result["incomplete_time_count"] == 1
    assert result["times"][1]["missing"] == [{"rank": 1, "fields": ["fluid/T"]}]


def test_rejects_missing_processor_directory(tmp_path: Path) -> None:
    (tmp_path / "processor0").mkdir()
    with pytest.raises(FileNotFoundError, match="processor directories"):
        check_case(tmp_path, 2)
