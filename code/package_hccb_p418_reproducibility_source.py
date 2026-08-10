#!/usr/bin/env python3
"""Create a deterministic small-source archive for the P418 paper."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath


ARCHIVE_ROOT = "p418_pebble_heat_reproduction"
MANIFEST_FILES = (
    "results/hccb_p418_reproducibility_manifest/manifest.json",
    "results/hccb_p418_reproducibility_manifest/manifest.csv",
    "results/hccb_p418_reproducibility_manifest/P418_复现文件说明_CN.md",
)
RECORD_ALIASES = (
    "archive_record.json",
    "package_record.json",
    "p418_reproduction_source_record.json",
    "source_archive_record.json",
    "source_package_record.json",
)
FORBIDDEN_PUBLIC_TEXT = (
    "/" + "Users/" + "wangjian",
    "/" + "data2/" + "CodexWork",
    "/" + "n96pfs/" + "home/",
    "192" + ".168.",
    "ysn" + "96pc",
    "ssh" + "pass",
    "BEGIN OPENSSH " + "PRIVATE KEY",
    "BEGIN RSA " + "PRIVATE KEY",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe package path: {value}")
    return path


def load_sources(project_root: Path, manifest_path: Path) -> list[tuple[Path, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("source_package_ready"):
        raise RuntimeError("source manifest is not complete")
    sources: dict[str, Path] = {}
    for row in manifest["files"]:
        if not row.get("present"):
            continue
        relative = safe_relative(str(row["path"])).as_posix()
        source = project_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = sha256(source)
        if actual != row["sha256"]:
            raise RuntimeError(f"SHA mismatch for {relative}: {actual}")
        sources[relative] = source
    for relative in MANIFEST_FILES:
        safe_relative(relative)
        source = project_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        sources[relative] = source
    public_readme = sources.get("reproducibility/README.md")
    if public_readme is not None:
        # The project-root README is an internal working document.  A public
        # archive should open with the concise, manuscript-specific guide.
        sources["README.md"] = public_readme
    ordered = sorted(
        ((source, relative) for relative, source in sources.items()),
        key=lambda item: item[1],
    )
    for source, relative in ordered:
        if source.stat().st_size > 10 * 1024 * 1024:
            continue
        text = source.read_text(encoding="utf-8", errors="ignore")
        found = [token for token in FORBIDDEN_PUBLIC_TEXT if token in text]
        if found:
            raise RuntimeError(
                f"public source contains private machine text in {relative}: {found}"
            )
    return ordered


def tar_info(source: Path, archive_name: str) -> tarfile.TarInfo:
    data = source.read_bytes()
    info = tarfile.TarInfo(archive_name)
    info.size = len(data)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o755 if os.access(source, os.X_OK) else 0o644
    return info


def build_archive(
    project_root: Path, manifest_path: Path, output: Path
) -> dict[str, object]:
    sources = load_sources(project_root, manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for source, relative in sources:
                    archive_name = f"{ARCHIVE_ROOT}/{relative}"
                    info = tar_info(source, archive_name)
                    tar.addfile(info, io.BytesIO(source.read_bytes()))
    temporary.replace(output)
    with tarfile.open(output, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate member names")
        if any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise RuntimeError("archive contains an unsafe member path")
        if any(member.issym() or member.islnk() for member in members):
            raise RuntimeError("source archive must not contain links")
        if any(not member.isfile() for member in members):
            raise RuntimeError("source archive contains a non-file member")
    return {
        "status": "p418_reproducibility_source_archive_ready",
        "archive_path": output.name,
        "archive_sha256": sha256(output),
        "archive_size_bytes": output.stat().st_size,
        "member_count": len(sources),
        "archive_root": ARCHIVE_ROOT,
        "source_manifest_path": manifest_path.relative_to(project_root).as_posix(),
        "source_manifest_sha256": sha256(manifest_path),
        "raw_openfoam_fields_included": False,
        "model_checkpoints_included": False,
        "private_machine_text_scan_passed": True,
        "public_readme_source": "reproducibility/README.md",
    }


def write_record(payload: dict[str, object], record: Path) -> None:
    """Write the canonical record and keep historical aliases consistent."""
    record.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    record.write_text(text, encoding="utf-8")
    if record.parent.name == "hccb_p418_reproducibility_manifest":
        for name in RECORD_ALIASES:
            (record.parent / name).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    payload = build_archive(
        args.project_root.resolve(),
        args.manifest.resolve(),
        args.output.resolve(),
    )
    write_record(payload, args.record.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
