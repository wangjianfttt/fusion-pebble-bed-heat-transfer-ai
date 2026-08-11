#!/usr/bin/env python3
"""Write the assigned P418 Zenodo DOI into the final submission files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from build_hccb_p418_cover_letter_pdf import build_pdf


DOI_PATTERN = re.compile(r"10\.5281/zenodo\.\d+", re.IGNORECASE)
MAIN_PENDING = (
    "A versioned DOI will be added to\n"
    "the repository record before submission."
)
COVER_PENDING = (
    "The final validation-selected predictions and figure records will be added "
    "to the same repository, and a versioned Zenodo DOI will be included before "
    "submission."
)


def normalized_doi(value: str) -> str:
    match = DOI_PATTERN.search(value.strip())
    if match is None:
        raise ValueError("DOI must have the form 10.5281/zenodo.<record>")
    return match.group(0).lower()


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one pending DOI sentence in {path}")
    return text.replace(old, new, 1)


def apply(project_root: Path, doi_value: str) -> dict[str, object]:
    root = project_root.resolve()
    doi = normalized_doi(doi_value)
    doi_url = f"https://doi.org/{doi}"
    release_summary = root / "results/hccb_p418_public_data_release_preflight/summary.json"
    if not release_summary.is_file():
        raise FileNotFoundError(release_summary)
    release = json.loads(release_summary.read_text(encoding="utf-8"))
    if release.get("status") != "p418_public_data_release_ready":
        raise RuntimeError("processed data release is not complete")

    main_path = root / "manuscript/main.tex"
    cover_path = root / "submission/cover_letter_IJHMT.md"
    cover_pdf_path = root / "submission/cover_letter_IJHMT.pdf"
    record_path = root / "submission/data_release_repository_record.json"
    main_text = replace_once(
        main_path.read_text(encoding="utf-8"),
        MAIN_PENDING,
        "The processed data and figure records are archived at\n"
        f"\\url{{{doi_url}}}.",
        main_path,
    )
    cover_text = replace_once(
        cover_path.read_text(encoding="utf-8"),
        COVER_PENDING,
        "The final validation-selected predictions and figure records are "
        f"included in the same repository and archived at Zenodo DOI {doi}.",
        cover_path,
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    existing = str(record.get("repository_doi") or "").strip()
    if existing and existing.lower() != doi:
        raise RuntimeError(f"repository record already contains another DOI: {existing}")
    record["repository_doi"] = doi
    record["dataset_doi_url"] = doi_url
    record["status"] = "public_code_and_processed_data_archived"

    main_path.write_text(main_text, encoding="utf-8")
    cover_path.write_text(cover_text, encoding="utf-8")
    build_pdf(cover_path, cover_pdf_path)
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "p418_repository_doi_written_to_submission_files",
        "repository_doi": doi,
        "dataset_doi_url": doi_url,
        "updated_files": [
            main_path.relative_to(root).as_posix(),
            cover_path.relative_to(root).as_posix(),
            cover_pdf_path.relative_to(root).as_posix(),
            record_path.relative_to(root).as_posix(),
        ],
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = apply(args.project_root, args.doi)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
