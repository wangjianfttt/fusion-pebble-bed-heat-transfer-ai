from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from verify_hccb_p418_seed202_terminal_status import verify  # noqa: E402


SOURCE = (
    ROOT
    / "results/hccb_p418_cross_packing_seed202_cloud_20260729"
    / "seed202_terminal_status.json"
)


def test_verifies_current_partial_seed202_matrix() -> None:
    result = verify(SOURCE)
    assert result["accepted_case_count"] == 7
    assert result["failed_case_count"] == 2
    assert not result["complete_nine_case_comparison"]


def test_rejects_failed_case_relabelled_as_accepted(tmp_path: Path) -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    altered = copy.deepcopy(payload)
    altered["accepted_cases"].append(altered["failed_cases"].pop())
    altered["accepted_case_count"] = 8
    altered["failed_case_count"] = 1
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly seven"):
        verify(path)
