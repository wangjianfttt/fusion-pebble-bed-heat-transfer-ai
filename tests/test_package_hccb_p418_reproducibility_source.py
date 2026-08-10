from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from package_hccb_p418_reproducibility_source import (
    ARCHIVE_ROOT,
    RECORD_ALIASES,
    build_archive,
    write_record,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "code" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('example')\n", encoding="utf-8")
    public_readme = tmp_path / "reproducibility" / "README.md"
    public_readme.parent.mkdir(parents=True)
    public_readme.write_text("# Public reproduction guide\n", encoding="utf-8")
    manifest_dir = tmp_path / "results" / "hccb_p418_reproducibility_manifest"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "source_package_ready": True,
        "files": [
            {
                "path": "code/example.py",
                "present": True,
                "sha256": digest(source),
            },
            {
                "path": "reproducibility/README.md",
                "present": True,
                "sha256": digest(public_readme),
            },
        ],
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (manifest_dir / "manifest.csv").write_text("path\ncode/example.py\n")
    (manifest_dir / "P418_复现文件说明_CN.md").write_text("复现说明\n")
    return tmp_path, manifest_path


def test_archive_is_safe_and_contains_manifest(tmp_path: Path) -> None:
    root, manifest = fixture(tmp_path)
    output = tmp_path / "archive.tar.gz"
    payload = build_archive(root, manifest, output)
    assert payload["status"] == "p418_reproducibility_source_archive_ready"
    with tarfile.open(output, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
    assert f"{ARCHIVE_ROOT}/code/example.py" in names
    assert f"{ARCHIVE_ROOT}/README.md" in names
    assert f"{ARCHIVE_ROOT}/reproducibility/README.md" in names
    assert f"{ARCHIVE_ROOT}/results/hccb_p418_reproducibility_manifest/manifest.json" in names
    assert all(member.isfile() for member in members)
    assert all(not member.issym() and not member.islnk() for member in members)
    with tarfile.open(output, "r:gz") as tar:
        root_readme = tar.extractfile(f"{ARCHIVE_ROOT}/README.md")
        nested_readme = tar.extractfile(
            f"{ARCHIVE_ROOT}/reproducibility/README.md"
        )
        assert root_readme is not None
        assert nested_readme is not None
        assert root_readme.read() == nested_readme.read()


def test_archive_is_byte_reproducible(tmp_path: Path) -> None:
    root, manifest = fixture(tmp_path)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    build_archive(root, manifest, first)
    build_archive(root, manifest, second)
    assert first.read_bytes() == second.read_bytes()


def test_record_aliases_stay_consistent(tmp_path: Path) -> None:
    root, manifest = fixture(tmp_path)
    output = tmp_path / "archive.tar.gz"
    payload = build_archive(root, manifest, output)
    record = manifest.parent / "source_archive_record.json"
    write_record(payload, record)
    expected = record.read_bytes()
    assert all((record.parent / name).read_bytes() == expected for name in RECORD_ALIASES)


def test_private_machine_path_is_rejected(tmp_path: Path) -> None:
    root, manifest = fixture(tmp_path)
    source = root / "code" / "example.py"
    private_root = "/" + "Users/" + "wangjian/private"
    source.write_text(f"root = '{private_root}'\n", encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = digest(source)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    try:
        build_archive(root, manifest, tmp_path / "private.tar.gz")
    except RuntimeError as error:
        assert "private machine text" in str(error)
    else:
        raise AssertionError("private machine path was not rejected")
