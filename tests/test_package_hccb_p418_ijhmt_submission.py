from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from package_hccb_p418_ijhmt_submission import build_bundle, collect_manuscript_files


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> Path:
    manuscript = tmp_path / "manuscript"
    figures = tmp_path / "figures"
    submission = tmp_path / "submission"
    results = tmp_path / "results"
    manuscript.mkdir()
    figures.mkdir()
    submission.mkdir()
    for index in range(1, 8):
        (figures / f"figure_{index}.pdf").write_bytes(f"pdf{index}".encode())
    ga_pdf = figures / "hccb_p418_graphical_abstract.pdf"
    ga_png = figures / "hccb_p418_graphical_abstract.png"
    ga_pdf.write_bytes(b"graphical-abstract-pdf")
    ga_png.write_bytes(b"graphical-abstract-png")
    (figures / "hccb_p418_graphical_abstract.json").write_text(
        json.dumps(
            {
                "status": "p418_ijhmt_graphical_abstract_ready",
                "outputs": {
                    "pdf": {"sha256": digest(ga_pdf)},
                    "png": {"sha256": digest(ga_png)},
                },
            }
        ),
        encoding="utf-8",
    )
    (manuscript / "main.tex").write_text(
        "\\documentclass{elsarticle}\n"
        "\\input{body}\n"
        "\\bibliographystyle{elsarticle-num}\n"
        "\\bibliography{references}\n",
        encoding="utf-8",
    )
    (manuscript / "body.tex").write_text(
        "\\IfFileExists{marker.tex}{marker-present}{}\n"
        + "\n".join(
            f"\\begin{{figure}}"
            f"\\includegraphics{{../figures/figure_{index}.pdf}}"
            f"\\caption{{Caption {index}.}}"
            f"\\end{{figure}}"
            for index in range(1, 8)
        )
        + "\n",
        encoding="utf-8",
    )
    (manuscript / "marker.tex").write_text("marker\n", encoding="utf-8")
    for name in ("elsarticle.cls", "elsarticle-num.bst", "references.bib", "main.bbl"):
        (manuscript / name).write_text(name + "\n", encoding="utf-8")
    (manuscript / "main.pdf").write_bytes(b"manuscript-pdf")
    (submission / "title_page.txt").write_text(
        "title page\n", encoding="utf-8"
    )
    (submission / "cover_letter_IJHMT.md").write_text("cover\n", encoding="utf-8")
    (submission / "cover_letter_IJHMT.pdf").write_bytes(b"cover-letter-pdf")
    (submission / "highlights.txt").write_text("highlight\n", encoding="utf-8")
    (submission / "CRediT_author_statement.md").write_text(
        "credit\n", encoding="utf-8"
    )
    (submission / "declaration_of_competing_interest.md").write_text(
        "conflict\n", encoding="utf-8"
    )
    (submission / "acknowledgements.md").write_text(
        "funding\n", encoding="utf-8"
    )
    (submission / "declaration_of_generative_ai_use.md").write_text(
        "AI-assisted preparation declaration\n", encoding="utf-8"
    )

    check_dir = results / "hccb_p418_ijhmt_submission_check"
    check_dir.mkdir(parents=True)
    (check_dir / "summary.json").write_text(
        json.dumps({"status": "completed_p418_ijhmt_submission_check"}),
        encoding="utf-8",
    )
    repro_dir = results / "hccb_p418_reproducibility_manifest"
    repro_dir.mkdir(parents=True)
    archive = repro_dir / "p418_reproduction_source.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("source\n", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="p418_pebble_heat_reproduction/payload.txt")
    (repro_dir / "source_archive_record.json").write_text(
        json.dumps(
            {
                "status": "p418_reproducibility_source_archive_ready",
                "archive_sha256": digest(archive),
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_submission_bundle_contains_exact_main_figure_set(tmp_path: Path) -> None:
    root = fixture(tmp_path)
    output = root / "bundle"
    payload = build_bundle(root, output, require_complete=True)
    assert payload["status"] == "completed_p418_ijhmt_submission_bundle"
    assert payload["main_figure_count"] == 7
    assert payload["supplement_included"] is False
    assert len(list((output / "upload").glob("Figure_*.pdf"))) == 7
    assert (output / "upload/CRediT_author_statement.md").is_file()
    assert (output / "upload/Title_page.txt").is_file()
    assert (output / "upload/Cover_letter.pdf").is_file()
    assert (output / "upload/Cover_letter.md").is_file()
    assert payload["cover_letter"]["pdf_sha256"] == digest(
        output / "upload/Cover_letter.pdf"
    )
    assert (output / "upload/Declaration_of_competing_interest.md").is_file()
    assert (output / "upload/Acknowledgements.md").is_file()
    assert (output / "upload/Declaration_of_generative_AI_use.md").is_file()
    assert payload["graphical_abstract_included"] is True
    assert (output / "upload/Graphical_abstract.pdf").is_file()
    assert (output / "upload/Graphical_abstract.png").is_file()
    captions = (output / "upload/Figure_captions.tex").read_text(encoding="utf-8")
    assert captions.count("Figure ") == 7
    assert "Figure 7. Caption 7." in captions
    assert not any("supplement" in path.name for path in output.rglob("*"))
    with zipfile.ZipFile(output / "upload/p418_manuscript_source.zip") as archive:
        names = archive.namelist()
    assert "manuscript/main.tex" in names
    assert "manuscript/body.tex" in names
    assert "manuscript/marker.tex" in names
    assert "figures/figure_1.pdf" in names


def test_submission_bundle_is_byte_reproducible(tmp_path: Path) -> None:
    root = fixture(tmp_path)
    first = root / "first"
    second = root / "second"
    build_bundle(root, first, require_complete=True)
    build_bundle(root, second, require_complete=True)
    assert (first / "p418_ijhmt_upload_bundle.zip").read_bytes() == (
        second / "p418_ijhmt_upload_bundle.zip"
    ).read_bytes()


def test_incomplete_submission_check_is_rejected(tmp_path: Path) -> None:
    root = fixture(tmp_path)
    summary = root / "results/hccb_p418_ijhmt_submission_check/summary.json"
    summary.write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
    try:
        build_bundle(root, root / "bundle", require_complete=True)
    except RuntimeError as error:
        assert "submission check is not complete" in str(error)
    else:
        raise AssertionError("incomplete submission check was accepted")


def test_non_strict_bundle_is_labeled_as_preflight(tmp_path: Path) -> None:
    root = fixture(tmp_path)
    summary = root / "results/hccb_p418_ijhmt_submission_check/summary.json"
    summary.write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
    payload = build_bundle(root, root / "bundle", require_complete=False)
    assert payload["status"] == "p418_ijhmt_submission_bundle_preflight"


def test_unvalidated_final_figures_are_not_packaged(tmp_path: Path) -> None:
    root = tmp_path
    manuscript = root / "manuscript"
    figures = root / "figures"
    manuscript.mkdir()
    figures.mkdir()
    (manuscript / "main.tex").write_text(
        "\\documentclass{article}\n\\input{body}\n", encoding="utf-8"
    )
    (manuscript / "body.tex").write_text(
        "\\IfFileExists{generated_transient_model_comparison_validated.tex}{%\n"
        "\\includegraphics{../figures/hccb_p418_transient_model_comparison.pdf}}{}\n"
        "\\IfFileExists{generated_openfoam_model_field_comparison_validated.tex}{%\n"
        "\\includegraphics{../figures/hccb_p418_openfoam_model_field_comparison.pdf}}{}\n",
        encoding="utf-8",
    )
    for name in ("article.cls", "elsarticle.cls", "elsarticle-num.bst", "main.bbl"):
        (manuscript / name).write_text(name + "\n", encoding="utf-8")
    (figures / "hccb_p418_transient_model_comparison.pdf").write_bytes(b"old")
    (figures / "hccb_p418_openfoam_model_field_comparison.pdf").write_bytes(b"old")
    _, collected = collect_manuscript_files(root)
    assert collected == []
