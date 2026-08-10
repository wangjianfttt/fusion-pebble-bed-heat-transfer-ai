#!/usr/bin/env python3
"""Compare manuscript BibTeX records with DOI registration metadata."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


FIELD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*\{(.*)\},?\s*$")
ENTRY_RE = re.compile(r"^@([A-Za-z]+)\{([^,]+),")


def plain_text(value: str) -> str:
    value = re.sub(r"\\['\"`^~=.uvHckbdtr]\{?([A-Za-z])\}?", r"\1", value)
    value = value.replace("{", "").replace("}", "")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def parse_bib(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        start = ENTRY_RE.match(raw)
        if start:
            current = {"entry_type": start.group(1).lower(), "key": start.group(2)}
            entries.append(current)
            continue
        if current is None:
            continue
        field = FIELD_RE.match(raw)
        if field:
            current[field.group(1).lower()] = field.group(2)
    return entries


def fetch_doi_metadata(doi: str, cache_dir: Path, pause_s: float) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", doi) + ".json")
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    request = urllib.request.Request(
        "https://doi.org/" + urllib.parse.quote(doi, safe="/():.-_"),
        headers={
            "Accept": "application/vnd.citationstyles.csl+json",
            "User-Agent": "P418-reference-check/1.0 (mailto:wjfttt@mail.ustc.edu.cn)",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.load(response)
    cache_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    time.sleep(pause_s)
    return metadata


def first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def issued_year(metadata: dict[str, Any]) -> str:
    parts = metadata.get("issued", {}).get("date-parts", [])
    if parts and parts[0]:
        return str(parts[0][0])
    return ""


def classify(entry: dict[str, str], metadata: dict[str, Any]) -> dict[str, Any]:
    title_registered = first_text(metadata.get("title"))
    journal_registered = first_text(metadata.get("container-title"))
    page_registered = first_text(metadata.get("page") or metadata.get("article-number"))
    title_similarity = difflib.SequenceMatcher(
        None, plain_text(entry.get("title", "")), plain_text(title_registered)
    ).ratio()
    checks = {
        "title": title_similarity >= 0.94,
        "year": not issued_year(metadata) or entry.get("year", "") == issued_year(metadata),
        "journal": (
            entry.get("entry_type") in {"phdthesis", "mastersthesis"}
            or not journal_registered
            or difflib.SequenceMatcher(
                None,
                plain_text(entry.get("journal", entry.get("booktitle", ""))),
                plain_text(journal_registered),
            ).ratio()
            >= 0.90
        ),
        "volume": not first_text(metadata.get("volume"))
        or entry.get("volume", "") == first_text(metadata.get("volume")),
        "pages": not page_registered
        or plain_text(entry.get("pages", "")).replace(" ", "")
        == plain_text(page_registered).replace(" ", ""),
    }
    return {
        "key": entry["key"],
        "doi": entry.get("doi", ""),
        "status": "match" if all(checks.values()) else "review",
        "title_similarity": round(title_similarity, 4),
        "bib_title": plain_text(entry.get("title", "")),
        "registered_title": title_registered,
        "bib_journal": entry.get("journal", entry.get("booktitle", "")),
        "registered_journal": journal_registered,
        "bib_year": entry.get("year", ""),
        "registered_year": issued_year(metadata),
        "bib_volume": entry.get("volume", ""),
        "registered_volume": first_text(metadata.get("volume")),
        "bib_pages": entry.get("pages", ""),
        "registered_pages": page_registered,
        "checks": checks,
        "metadata_type": first_text(metadata.get("type")),
        "metadata_url": first_text(metadata.get("URL")),
    }


def apply_source_decisions(
    records: list[dict[str, Any]], decisions_path: Path | None
) -> None:
    if decisions_path is None:
        return
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    for record in records:
        decision = decisions.get(record["key"])
        if not decision:
            continue
        failed = {name for name, ok in record["checks"].items() if not ok}
        accepted = set(decision.get("accepted_checks", []))
        if failed and failed <= accepted:
            record["status"] = "accepted_after_source_check"
            record["source_decision"] = decision


def write_report(
    entries: list[dict[str, str]],
    records: list[dict[str, Any]],
    failures: list[dict[str, str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    no_doi = [
        {
            "key": entry["key"],
            "entry_type": entry["entry_type"],
            "title": entry.get("title", ""),
            "year": entry.get("year", ""),
            "venue": entry.get("journal", entry.get("booktitle", "")),
            "url": entry.get("url", ""),
        }
        for entry in entries
        if not entry.get("doi")
    ]
    unresolved = [record for record in records if record["status"] == "review"]
    accepted = [
        record
        for record in records
        if record["status"] == "accepted_after_source_check"
    ]
    summary = {
        "status": "p418_reference_metadata_check_complete",
        "entry_count": len(entries),
        "doi_entry_count": len(records) + len(failures),
        "doi_metadata_match_count": sum(record["status"] == "match" for record in records),
        "doi_source_checked_exception_count": len(accepted),
        "doi_unresolved_review_count": len(unresolved),
        "doi_fetch_failure_count": len(failures),
        "no_doi_count": len(no_doi),
        "records": records,
        "fetch_failures": failures,
        "entries_without_doi": no_doi,
        "new_physical_parameters": [],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    columns = [
        "key",
        "doi",
        "status",
        "title_similarity",
        "bib_year",
        "registered_year",
        "bib_journal",
        "registered_journal",
        "bib_volume",
        "registered_volume",
        "bib_pages",
        "registered_pages",
        "registered_title",
        "metadata_url",
    ]
    with (output_dir / "doi_metadata_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# P418 reference metadata check",
        "",
        f"- BibTeX entries: {len(entries)}",
        f"- Entries with DOI: {len(records) + len(failures)}",
        f"- DOI records matching registered metadata: "
        f"{sum(record['status'] == 'match' for record in records)}",
        f"- DOI records accepted after publisher/source check: {len(accepted)}",
        f"- DOI records still requiring review: {len(unresolved)}",
        f"- DOI metadata fetch failures: {len(failures)}",
        f"- Entries without DOI: {len(no_doi)}",
        "",
        "## DOI records requiring review",
        "",
    ]
    if not unresolved:
        lines.append("- None.")
    for record in unresolved:
        failed = ", ".join(name for name, ok in record["checks"].items() if not ok)
        lines.append(f"- `{record['key']}`: {failed}; DOI `{record['doi']}`.")
    lines.extend(["", "## Source-checked metadata differences", ""])
    if not accepted:
        lines.append("- None.")
    for record in accepted:
        decision = record["source_decision"]
        lines.append(
            f"- `{record['key']}`: {decision['reason']} "
            f"Source: {decision['source_url']}"
        )
    lines.extend(["", "## Entries without DOI", ""])
    for record in no_doi:
        lines.append(
            f"- `{record['key']}` ({record['year']}): {record['title']} "
            f"[{record['venue']}]"
        )
    (output_dir / "REFERENCE_METADATA_CHECK.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--pause-s", type=float, default=0.15)
    args = parser.parse_args()
    entries = parse_bib(args.bib)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for entry in entries:
        doi = entry.get("doi")
        if not doi:
            continue
        try:
            records.append(
                classify(
                    entry,
                    fetch_doi_metadata(doi, args.cache_dir, args.pause_s),
                )
            )
        except Exception as error:  # Keep the complete DOI failure list.
            failures.append({"key": entry["key"], "doi": doi, "error": str(error)})
    apply_source_decisions(records, args.decisions)
    write_report(entries, records, failures, args.output_dir)
    if failures:
        raise SystemExit(f"{len(failures)} DOI metadata requests failed")


if __name__ == "__main__":
    main()
