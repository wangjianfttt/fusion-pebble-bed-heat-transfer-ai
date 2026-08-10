from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from rerun_hccb_p418_post_selection_outputs import FINAL_JOB_IDS, load_jobs


def test_final_output_jobs_are_rebuilt_after_validation_selection(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "jobs": [
                    {"job_id": "earlier_training", "command": "true"},
                    *[
                        {"job_id": job_id, "command": "true"}
                        for job_id in FINAL_JOB_IDS
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    assert tuple(job["job_id"] for job in load_jobs(path)) == FINAL_JOB_IDS


def test_final_output_jobs_reject_changed_order(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "jobs": [
                    {"job_id": job_id, "command": "true"}
                    for job_id in reversed(FINAL_JOB_IDS)
                ]
            }
        ),
        encoding="utf-8",
    )
    try:
        load_jobs(path)
    except ValueError as error:
        assert "registered order" in str(error)
    else:
        raise AssertionError("changed final-output order was accepted")
