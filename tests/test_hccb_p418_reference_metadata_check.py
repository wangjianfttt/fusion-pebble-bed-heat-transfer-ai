from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from check_hccb_p418_reference_metadata import (  # noqa: E402
    apply_source_decisions,
    classify,
    parse_bib,
)


def cache_path(cache_dir: Path, doi: str) -> Path:
    return cache_dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", doi) + ".json")


def test_all_doi_entries_have_cached_metadata_and_no_unresolved_difference() -> None:
    entries = parse_bib(ROOT / "manuscript/references.bib")
    doi_entries = [entry for entry in entries if entry.get("doi")]
    cache_dir = ROOT / "results/hccb_p418_reference_metadata_cache"

    records = []
    for entry in doi_entries:
        path = cache_path(cache_dir, entry["doi"])
        assert path.is_file(), f"missing DOI metadata cache for {entry['key']}"
        records.append(classify(entry, json.loads(path.read_text(encoding="utf-8"))))

    apply_source_decisions(
        records, ROOT / "manuscript/reference_metadata_decisions.json"
    )
    assert len(entries) == 44
    assert len(doi_entries) == 40
    assert all(record["status"] != "review" for record in records)
    assert sum(
        record["status"] == "accepted_after_source_check" for record in records
    ) == 3


def test_reference_metadata_report_matches_current_bibliography() -> None:
    summary = json.loads(
        (
            ROOT
            / "results/hccb_p418_reference_metadata_check_20260812/summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["entry_count"] == 44
    assert summary["doi_entry_count"] == 40
    assert summary["doi_unresolved_review_count"] == 0
    assert summary["doi_fetch_failure_count"] == 0
    assert summary["no_doi_count"] == 4
