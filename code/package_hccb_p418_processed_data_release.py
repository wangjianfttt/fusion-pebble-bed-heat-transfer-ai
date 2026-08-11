#!/usr/bin/env python3
"""Build the citable processed-data archive after formal model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from build_hccb_p418_public_data_release import file_format_valid, sha256_and_read_size
from package_hccb_p418_ijhmt_submission import sha256, write_deterministic_zip


ARCHIVE_NAME = "p418_processed_data_release.zip"
PRIVATE_TEXT = (
    "/" + "Users/",
    "/" + "data2/",
    "/" + "n96pfs/",
    "192" + ".168.",
    "BEGIN OPENSSH " + "PRIVATE KEY",
    "BEGIN RSA " + "PRIVATE KEY",
)
ROOT_FILES = ("LICENSE", "DATA_LICENSE.md", "CITATION.cff")
RELEASE_FILES = ("README.md", "summary.json", "zenodo_metadata_draft.json")


def verified_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rows = list(summary.get("compact_files", [])) + list(
        summary.get("final_processed_files", [])
    )
    if not rows or not all(bool(row.get("present")) for row in rows):
        raise RuntimeError("processed-data file list is incomplete")
    return rows


def scan_private_text(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix.lower() not in {".md", ".txt", ".json", ".csv", ".cff"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = [token for token in PRIVATE_TEXT if token in text]
        if found:
            raise RuntimeError(f"private machine text in {path}: {found}")


def verify_readable_file(path: Path, expected_sha256: str | None = None) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    logical_size = path.stat().st_size
    digest, read_size = sha256_and_read_size(path)
    if read_size != logical_size:
        raise RuntimeError(f"release file is an unreadable cloud placeholder: {path}")
    if not file_format_valid(path):
        raise RuntimeError(f"release file format is invalid: {path}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"release SHA mismatch for {path}")
    return digest


def build_archive(
    project_root: Path,
    release_dir: Path,
    output: Path,
) -> dict[str, object]:
    project_root = project_root.resolve()
    release_dir = release_dir.resolve()
    summary_path = release_dir / "summary.json"
    metadata_path = release_dir / "zenodo_metadata_draft.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if summary.get("status") != "p418_public_data_release_ready":
        raise RuntimeError("public data release is not marked ready")
    if not summary.get("final_processed_archive_ready"):
        raise RuntimeError("final processed files are not ready")
    if metadata.get("status") != "p418_repository_metadata_ready" or not metadata.get(
        "ready_for_deposition"
    ):
        raise RuntimeError("Zenodo metadata is not ready")

    entries: list[tuple[Path, str]] = []
    seen_names: set[str] = set()
    source_paths: list[Path] = []
    for row in verified_rows(summary):
        relative = str(row["path"])
        source = (project_root / relative).resolve()
        try:
            archive_name = source.relative_to(project_root).as_posix()
        except ValueError as error:
            raise RuntimeError(f"release file is outside the project root: {source}") from error
        expected = str(row.get("sha256") or "")
        if not expected:
            raise RuntimeError(f"release SHA is missing for {relative}")
        verify_readable_file(source, expected)
        if archive_name in seen_names:
            continue
        seen_names.add(archive_name)
        entries.append((source, archive_name))
        source_paths.append(source)

    for relative in ROOT_FILES:
        source = project_root / relative
        verify_readable_file(source)
        entries.append((source, relative))
        source_paths.append(source)
    for name in RELEASE_FILES:
        source = release_dir / name
        verify_readable_file(source)
        entries.append((source, f"release/{name}"))
        source_paths.append(source)

    scan_private_text(source_paths)
    write_deterministic_zip(output, entries)
    expected_sizes = {name: source.stat().st_size for source, name in entries}
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("processed-data archive failed its CRC check")
        actual_sizes = {info.filename: info.file_size for info in archive.infolist()}
    if actual_sizes != expected_sizes:
        raise RuntimeError("processed-data archive member sizes do not match the sources")
    return {
        "status": "completed_p418_processed_data_release_archive",
        "archive": output.name,
        "archive_size_bytes": output.stat().st_size,
        "archive_sha256": sha256(output),
        "member_count": len(entries),
        "metadata_status": metadata["status"],
        "license": metadata["metadata"].get("license"),
        "new_physical_parameters": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    output = args.output or args.release_dir / ARCHIVE_NAME
    payload = build_archive(args.project_root, args.release_dir, output.resolve())
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
