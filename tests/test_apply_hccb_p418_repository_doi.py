from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from apply_hccb_p418_repository_doi import (  # noqa: E402
    COVER_PENDING,
    MAIN_PENDING,
    apply,
    normalized_doi,
)


def make_project(tmp_path: Path, *, ready: bool = True) -> Path:
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "submission").mkdir()
    summary = tmp_path / "results/hccb_p418_public_data_release_preflight/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "status": (
                    "p418_public_data_release_ready"
                    if ready
                    else "p418_public_data_release_preflight"
                )
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manuscript/main.tex").write_text(
        "Data availability. " + MAIN_PENDING + "\n", encoding="utf-8"
    )
    (tmp_path / "submission/cover_letter_IJHMT.md").write_text(
        COVER_PENDING + "\n", encoding="utf-8"
    )
    (tmp_path / "submission/data_release_repository_record.json").write_text(
        '{"status":"processed_data_pending","repository_doi":null}\n',
        encoding="utf-8",
    )
    return tmp_path


def test_apply_writes_one_doi_to_all_submission_files(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    payload = apply(root, "https://doi.org/10.5281/zenodo.1234567")
    assert payload["repository_doi"] == "10.5281/zenodo.1234567"
    main = (root / "manuscript/main.tex").read_text(encoding="utf-8")
    cover = (root / "submission/cover_letter_IJHMT.md").read_text(encoding="utf-8")
    record = json.loads(
        (root / "submission/data_release_repository_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert r"\url{https://doi.org/10.5281/zenodo.1234567}" in main
    assert "will be added" not in main
    assert "Zenodo DOI 10.5281/zenodo.1234567" in cover
    assert "will be included" not in cover
    assert (root / "submission/cover_letter_IJHMT.pdf").is_file()
    assert "submission/cover_letter_IJHMT.pdf" in payload["updated_files"]
    assert record["repository_doi"] == "10.5281/zenodo.1234567"
    assert record["dataset_doi_url"] == "https://doi.org/10.5281/zenodo.1234567"


def test_apply_refuses_incomplete_release(tmp_path: Path) -> None:
    root = make_project(tmp_path, ready=False)
    with pytest.raises(RuntimeError, match="processed data release is not complete"):
        apply(root, "10.5281/zenodo.1234567")


def test_normalized_doi_rejects_another_project_or_invalid_value() -> None:
    with pytest.raises(ValueError):
        normalized_doi("10.1000/example")


def test_make_target_and_submission_readme_document_guarded_doi_update() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (ROOT / "submission/README_CN.md").read_text(encoding="utf-8")
    assert "p418-apply-doi:" in makefile
    assert "--doi \"$(P418_DOI)\"" in makefile
    assert "make p418-apply-doi P418_DOI=10.5281/zenodo.<record>" in readme
