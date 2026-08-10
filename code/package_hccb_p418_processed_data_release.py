#!/usr/bin/env python3
"""Create the final deterministic processed-data archive for the P418 paper."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath

from build_hccb_p418_public_data_release import (
    COMPACT_FILES,
    FINAL_PROCESSED_FILES,
    PRIVATE_TEXT,
    build,
)


ARCHIVE_ROOT = "p418_pebble_heat_processed_data"
GENERATED_RELEASE_FILES = (
    "README.md",
    "summary.json",
    "zenodo_metadata_draft.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe release path: {value}")
    return path.as_posix()


def collect_sources(
    project_root: Path,
    release_dir: Path,
) -> tuple[list[tuple[Path, str]], dict[str, object]]:
    payload = build(project_root, release_dir)
    missing = [
        str(row["path"])
        for row in payload["final_processed_files"]
        if not row["present"]
    ]
    metadata = json.loads(
        (release_dir / "zenodo_metadata_draft.json").read_text(encoding="utf-8")
    )
    if missing or not metadata["ready_for_deposition"]:
        pending = list(metadata["pending_fields"])
        raise RuntimeError(
            "processed-data release is incomplete; "
            f"missing_files={missing}; pending_metadata={pending}"
        )

    sources: dict[str, Path] = {}
    for relative in (*COMPACT_FILES, *FINAL_PROCESSED_FILES):
        relative = safe_relative(relative)
        path = project_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        sources[relative] = path
    for name in GENERATED_RELEASE_FILES:
        path = release_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        sources[f"release/{name}"] = path

    ordered = sorted(
        ((path, relative) for relative, path in sources.items()),
        key=lambda item: item[1],
    )
    for path, relative in ordered:
        if path.suffix.lower() not in {".json", ".csv", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = [token for token in PRIVATE_TEXT if token in text]
        if found:
            raise RuntimeError(
                f"private machine text in processed release {relative}: {found}"
            )
    return ordered, payload


def tar_info(path: Path, name: str) -> tarfile.TarInfo:
    data = path.read_bytes()
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
    return info


def build_archive(
    project_root: Path,
    release_dir: Path,
    output: Path,
) -> dict[str, object]:
    sources, release_payload = collect_sources(project_root, release_dir)
    checksums = "".join(
        f"{sha256(path)}  {relative}\n" for path, relative in sources
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path, relative in sources:
                    name = f"{ARCHIVE_ROOT}/{relative}"
                    tar.addfile(tar_info(path, name), io.BytesIO(path.read_bytes()))
                checksum_name = f"{ARCHIVE_ROOT}/SHA256SUMS.txt"
                checksum_info = tarfile.TarInfo(checksum_name)
                checksum_info.size = len(checksums)
                checksum_info.mtime = 0
                checksum_info.uid = checksum_info.gid = 0
                checksum_info.uname = checksum_info.gname = ""
                checksum_info.mode = 0o644
                tar.addfile(checksum_info, io.BytesIO(checksums))
    temporary.replace(output)

    with tarfile.open(output, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("processed-data archive has duplicate members")
        if any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise RuntimeError("processed-data archive has an unsafe path")
        if any(not member.isfile() for member in members):
            raise RuntimeError("processed-data archive contains a non-file member")
        for member in members:
            handle = tar.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            while handle.read(1024 * 1024):
                pass
    return {
        "status": "p418_processed_data_release_archive_ready",
        "archive_path": output.name,
        "archive_sha256": sha256(output),
        "archive_size_bytes": output.stat().st_size,
        "archive_root": ARCHIVE_ROOT,
        "member_count": len(sources) + 1,
        "compact_file_count": len(COMPACT_FILES),
        "final_processed_file_count": len(FINAL_PROCESSED_FILES),
        "repository_metadata_ready": release_payload["repository_metadata_ready"],
        "private_machine_text_scan_passed": True,
        "new_physical_parameters": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    payload = build_archive(
        args.project_root.resolve(),
        args.release_dir.resolve(),
        args.output.resolve(),
    )
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
