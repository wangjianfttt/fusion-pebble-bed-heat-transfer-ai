#!/usr/bin/env python3
"""Tests for the unified P418 cloud handoff sheet."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_cloud_submission_manifest.py"
SPEC = ROOT / "cloud_migration/cloud_package_specs.json"
TASKS = ROOT / "cloud_migration/cpu_task_candidates.csv"


def prepare_fake_packages(root: Path) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for index, package in enumerate(spec["packages"]):
        path = root / package["filename"]
        path.write_bytes(f"package-{index}".encode("ascii"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / f"{path.name}.sha256").write_text(
            f"{digest}  {path.name}\n", encoding="utf-8"
        )


def run_builder(package_dir: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--spec",
            str(SPEC),
            "--tasks",
            str(TASKS),
            "--package-dir",
            str(package_dir),
            "--output-json",
            str(output_dir / "manifest.json"),
            "--output-cn",
            str(output_dir / "manifest.md"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_all_four_packages_produce_one_ready_manifest(tmp_path: Path) -> None:
    packages = tmp_path / "packages"
    output = tmp_path / "output"
    packages.mkdir()
    prepare_fake_packages(packages)
    completed = run_builder(packages, output)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "all_four_cloud_packages_verified"
    assert payload["package_count"] == 4
    assert all(row["ready_for_transfer"] for row in payload["packages"])
    note = (output / "manifest.md").read_text(encoding="utf-8")
    assert "项目根目录若有早期同名副本" in note
    assert "/home" in note


def test_changed_archive_is_reported(tmp_path: Path) -> None:
    packages = tmp_path / "packages"
    output = tmp_path / "output"
    packages.mkdir()
    prepare_fake_packages(packages)
    target = packages / "p418_openfoam13_pending_46.tar.zst"
    target.write_bytes(target.read_bytes() + b"-changed")
    completed = run_builder(packages, output)
    assert completed.returncode == 1
    payload = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "cloud_package_problem_found"
    failed = next(
        row
        for row in payload["packages"]
        if row["filename"] == target.name
    )
    assert failed["checks"]["checksum_matches"] is False
    assert failed["ready_for_transfer"] is False
