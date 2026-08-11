from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from package_hccb_p418_processed_data_release import build_archive


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path, *, ready: bool = True) -> tuple[Path, Path]:
    project = tmp_path / "project"
    release = project / "release"
    release.mkdir(parents=True)
    for name in ("LICENSE", "DATA_LICENSE.md", "CITATION.cff"):
        (project / name).write_text(name + "\n", encoding="utf-8")
    compact = project / "results/compact.csv"
    prediction = project / "results/selected_prediction.npz"
    compact.parent.mkdir(parents=True)
    compact.write_text("x,y\n1,2\n", encoding="utf-8")
    prediction.write_bytes(b"selected-test-prediction")
    rows = [
        {"path": "results/compact.csv", "present": True, "sha256": digest(compact)},
        {
            "path": "results/selected_prediction.npz",
            "present": ready,
            "sha256": digest(prediction),
        },
    ]
    (release / "summary.json").write_text(
        json.dumps(
            {
                "status": (
                    "p418_public_data_release_ready"
                    if ready
                    else "p418_public_data_release_preflight"
                ),
                "compact_files": rows[:1],
                "final_processed_files": rows[1:],
                "final_processed_archive_ready": ready,
            }
        ),
        encoding="utf-8",
    )
    (release / "zenodo_metadata_draft.json").write_text(
        json.dumps(
            {
                "status": (
                    "p418_repository_metadata_ready"
                    if ready
                    else "p418_repository_metadata_draft"
                ),
                "ready_for_deposition": ready,
                "metadata": {"license": "cc-by-4.0"},
            }
        ),
        encoding="utf-8",
    )
    (release / "README.md").write_text("release\n", encoding="utf-8")
    return project, release


def test_processed_archive_is_complete_and_reproducible(tmp_path: Path) -> None:
    project, release = fixture(tmp_path)
    first = release / "first.zip"
    second = release / "second.zip"
    payload = build_archive(project, release, first)
    build_archive(project, release, second)
    assert payload["status"] == "completed_p418_processed_data_release_archive"
    assert payload["license"] == "cc-by-4.0"
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
    assert "results/selected_prediction.npz" in names
    assert "release/zenodo_metadata_draft.json" in names
    assert "DATA_LICENSE.md" in names


def test_processed_archive_rejects_incomplete_release(tmp_path: Path) -> None:
    project, release = fixture(tmp_path, ready=False)
    try:
        build_archive(project, release, release / "output.zip")
    except RuntimeError as error:
        assert "not marked ready" in str(error)
    else:
        raise AssertionError("an incomplete processed-data release was accepted")


def test_processed_archive_rejects_private_machine_text(tmp_path: Path) -> None:
    project, release = fixture(tmp_path)
    (release / "README.md").write_text("source: /data2/private\n", encoding="utf-8")
    try:
        build_archive(project, release, release / "output.zip")
    except RuntimeError as error:
        assert "private machine text" in str(error)
    else:
        raise AssertionError("private machine text entered the release archive")
