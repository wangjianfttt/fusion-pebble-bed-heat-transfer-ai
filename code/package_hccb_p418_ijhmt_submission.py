#!/usr/bin/env python3
"""Build a compact, directly usable IJHMT submission bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath


BUNDLE_NAME = "p418_ijhmt_upload_bundle.zip"
SOURCE_NAME = "p418_manuscript_source.zip"
REPRODUCTION_NAME = "p418_reproduction_source.tar.gz"
PRIVATE_TEXT = (
    "/" + "Users/" + "wangjian",
    "/" + "data2/" + "CodexWork",
    "/" + "n96pfs/" + "home/",
    "192" + "." + "168.",
    "ysn" + "96pc",
    "BEGIN OPENSSH " + "PRIVATE KEY",
    "BEGIN RSA " + "PRIVATE KEY",
)
INPUT_PATTERN = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
GRAPHIC_PATTERN = re.compile(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}")
BIB_PATTERN = re.compile(r"\\bibliography\s*\{([^}]+)\}")
BST_PATTERN = re.compile(r"\\bibliographystyle\s*\{([^}]+)\}")

FINAL_FIGURE_MARKERS = {
    "hccb_p418_transient_model_comparison": (
        "manuscript/generated_transient_model_comparison_validated.tex"
    ),
    "hccb_p418_openfoam_model_field_comparison": (
        "manuscript/generated_openfoam_model_field_comparison_validated.tex"
    ),
}
CLASS_PATTERN = re.compile(r"\\documentclass(?:\[[^]]*\])?\s*\{([^}]+)\}")
IF_FILE_PATTERN = re.compile(r"\\IfFileExists\s*\{([^}]+)\}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def resolve_local(base: Path, value: str, suffix: str) -> Path:
    candidate = (base / value).resolve()
    if not candidate.suffix:
        candidate = candidate.with_suffix(suffix)
    return candidate


def figure_is_enabled(project_root: Path, figure: Path) -> bool:
    marker = FINAL_FIGURE_MARKERS.get(figure.stem)
    return marker is None or (project_root / marker).is_file()


def collect_manuscript_files(project_root: Path) -> tuple[list[Path], list[Path]]:
    manuscript = project_root / "manuscript"
    main = manuscript / "main.tex"
    if not main.is_file():
        raise FileNotFoundError(main)
    tex_files: list[Path] = []
    figures: list[Path] = []
    pending = [main.resolve()]
    seen: set[Path] = set()
    while pending:
        source = pending.pop(0)
        if source in seen:
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        seen.add(source)
        tex_files.append(source)
        text = source.read_text(encoding="utf-8")
        for value in IF_FILE_PATTERN.findall(text):
            conditional = (source.parent / value.strip()).resolve()
            if not conditional.is_file() or conditional.stat().st_size == 0:
                continue
            if conditional.suffix.lower() == ".tex":
                pending.append(conditional)
            elif (
                conditional.suffix.lower() == ".pdf"
                and figure_is_enabled(project_root, conditional)
                and conditional not in figures
            ):
                figures.append(conditional)
        for value in INPUT_PATTERN.findall(text):
            child = resolve_local(source.parent, value.strip(), ".tex")
            if not child.is_file():
                literal = value.strip()
                optional_names = {literal, f"{literal}.tex"}
                if any(
                    re.search(
                        rf"\\IfFileExists\s*\{{{re.escape(name)}\}}",
                        text,
                    )
                    for name in optional_names
                ):
                    continue
                raise FileNotFoundError(
                    f"referenced TeX input is missing: {child}"
                )
            pending.append(child)
        for value in GRAPHIC_PATTERN.findall(text):
            figure = resolve_local(source.parent, value.strip(), ".pdf")
            if not figure.is_file() or figure.stat().st_size == 0:
                literal = value.strip()
                optional_names = {literal, f"{literal}.pdf"}
                if any(
                    re.search(
                        rf"\\IfFileExists\s*\{{{re.escape(name)}\}}",
                        text,
                    )
                    for name in optional_names
                ):
                    continue
                raise FileNotFoundError(f"referenced figure is missing: {figure}")
            if figure_is_enabled(project_root, figure) and figure not in figures:
                figures.append(figure)

    extras: list[Path] = []
    main_text = main.read_text(encoding="utf-8")
    for values in BIB_PATTERN.findall(main_text):
        for value in values.split(","):
            extras.append(resolve_local(manuscript, value.strip(), ".bib"))
    for value in BST_PATTERN.findall(main_text):
        extras.append(resolve_local(manuscript, value.strip(), ".bst"))
    for value in CLASS_PATTERN.findall(main_text):
        extras.append(resolve_local(manuscript, value.strip(), ".cls"))
    bbl = manuscript / "main.bbl"
    if bbl.is_file() and bbl.stat().st_size > 0:
        extras.append(bbl.resolve())
    for path in extras:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"manuscript dependency is missing: {path}")
        if path.resolve() not in tex_files:
            tex_files.append(path.resolve())
    return tex_files, figures


def write_deterministic_zip(
    output: Path, entries: list[tuple[Path, str]]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source, name in sorted(entries, key=lambda item: item[1]):
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe archive path: {name}")
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o644 & 0xFFFF) << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    temporary.replace(output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate archive members in {output}")
        if any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise RuntimeError(f"unsafe archive member in {output}")
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"corrupt archive member in {output}: {bad}")


def copy_required(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def extract_balanced_arguments(text: str, command: str) -> list[str]:
    """Return brace-balanced arguments for a LaTeX command."""
    values: list[str] = []
    start = 0
    token = command + "{"
    while True:
        marker = text.find(token, start)
        if marker < 0:
            return values
        index = marker + len(token)
        depth = 1
        while index < len(text) and depth:
            if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
            index += 1
        if depth:
            raise RuntimeError(f"unbalanced {command} argument")
        values.append(text[marker + len(token) : index - 1].strip())
        start = index


def manuscript_figure_captions(project_root: Path) -> list[str]:
    captions: list[str] = []
    source_files, _ = collect_manuscript_files(project_root)
    for source in source_files:
        if source.suffix.lower() != ".tex":
            continue
        text = source.read_text(encoding="utf-8")
        for block in re.findall(
            r"\\begin\{figure\}(.*?)\\end\{figure\}",
            text,
            flags=re.DOTALL,
        ):
            captions.extend(extract_balanced_arguments(block, "\\caption"))
    return captions


def write_figure_captions(path: Path, captions: list[str]) -> None:
    path.write_text(
        "% Editable figure captions extracted from the manuscript source.\n\n"
        + "\n\n".join(
            f"Figure {index}. {caption}" for index, caption in enumerate(captions, 1)
        )
        + "\n",
        encoding="utf-8",
    )


def scan_public_text(paths: list[Path]) -> None:
    for path in paths:
        if path.stat().st_size > 10 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = [token for token in PRIVATE_TEXT if token in text]
        if found:
            raise RuntimeError(f"private machine text in {path.name}: {found}")


def build_bundle(
    project_root: Path,
    output_dir: Path,
    *,
    require_complete: bool = False,
) -> dict[str, object]:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    submission_check = (
        project_root
        / "results/hccb_p418_ijhmt_submission_check/summary.json"
    )
    if not submission_check.is_file():
        raise FileNotFoundError(submission_check)
    check_payload = json.loads(submission_check.read_text(encoding="utf-8"))
    if require_complete and check_payload.get("status") != (
        "completed_p418_ijhmt_submission_check"
    ):
        raise RuntimeError("IJHMT submission check is not complete")

    source_archive = (
        project_root
        / "results/hccb_p418_reproducibility_manifest/p418_reproduction_source.tar.gz"
    )
    source_record = (
        project_root
        / "results/hccb_p418_reproducibility_manifest/source_archive_record.json"
    )
    source_payload = json.loads(source_record.read_text(encoding="utf-8"))
    if source_payload.get("status") != "p418_reproducibility_source_archive_ready":
        raise RuntimeError("reproducibility source archive is not ready")
    if sha256(source_archive) != source_payload.get("archive_sha256"):
        raise RuntimeError("reproducibility source archive SHA mismatch")

    tex_files, figures = collect_manuscript_files(project_root)
    if require_complete and len(figures) != 7:
        raise RuntimeError(f"expected 7 main figures, found {len(figures)}")
    scan_public_text(tex_files)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    upload = output_dir / "upload"
    upload.mkdir(parents=True)

    main_pdf = project_root / "manuscript/main.pdf"
    title_page = project_root / "submission/title_page.txt"
    cover_letter = project_root / "submission/cover_letter_IJHMT.md"
    cover_letter_pdf = project_root / "submission/cover_letter_IJHMT.pdf"
    highlights = project_root / "submission/highlights.txt"
    credit = project_root / "submission/CRediT_author_statement.md"
    competing_interest = project_root / "submission/declaration_of_competing_interest.md"
    acknowledgements = project_root / "submission/acknowledgements.md"
    ai_declaration = project_root / "submission/declaration_of_generative_ai_use.md"
    copy_required(main_pdf, upload / "Manuscript.pdf")
    copy_required(title_page, upload / "Title_page.txt")
    copy_required(cover_letter, upload / "Cover_letter.md")
    copy_required(cover_letter_pdf, upload / "Cover_letter.pdf")
    copy_required(highlights, upload / "Highlights.txt")
    copy_required(credit, upload / "CRediT_author_statement.md")
    copy_required(
        competing_interest, upload / "Declaration_of_competing_interest.md"
    )
    copy_required(acknowledgements, upload / "Acknowledgements.md")
    copy_required(ai_declaration, upload / "Declaration_of_generative_AI_use.md")
    copy_required(source_archive, upload / REPRODUCTION_NAME)

    captions = manuscript_figure_captions(project_root)
    if len(captions) != 7:
        raise RuntimeError(f"expected 7 figure captions, found {len(captions)}")
    write_figure_captions(upload / "Figure_captions.tex", captions)

    graphical_abstract_included = False
    graphical_abstract_pdf = project_root / "figures/hccb_p418_graphical_abstract.pdf"
    graphical_abstract_png = project_root / "figures/hccb_p418_graphical_abstract.png"
    graphical_abstract_record = project_root / "figures/hccb_p418_graphical_abstract.json"
    if any(path.exists() for path in (graphical_abstract_pdf, graphical_abstract_png, graphical_abstract_record)):
        for path in (graphical_abstract_pdf, graphical_abstract_png, graphical_abstract_record):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"incomplete graphical abstract output: {path}")
        ga_payload = json.loads(graphical_abstract_record.read_text(encoding="utf-8"))
        if ga_payload.get("status") != "p418_ijhmt_graphical_abstract_ready":
            raise RuntimeError("graphical abstract is not validated")
        expected_pdf_sha = ga_payload.get("outputs", {}).get("pdf", {}).get("sha256")
        expected_png_sha = ga_payload.get("outputs", {}).get("png", {}).get("sha256")
        if sha256(graphical_abstract_pdf) != expected_pdf_sha or sha256(graphical_abstract_png) != expected_png_sha:
            raise RuntimeError("graphical abstract SHA mismatch")
        copy_required(graphical_abstract_pdf, upload / "Graphical_abstract.pdf")
        copy_required(graphical_abstract_png, upload / "Graphical_abstract.png")
        graphical_abstract_included = True

    manuscript_entries: list[tuple[Path, str]] = []
    for path in tex_files + figures:
        manuscript_entries.append((path, project_relative(project_root, path)))
    manuscript_source = upload / SOURCE_NAME
    write_deterministic_zip(manuscript_source, manuscript_entries)

    figure_records = []
    for index, figure in enumerate(figures, start=1):
        target_name = f"Figure_{index:02d}_{figure.stem}.pdf"
        target = upload / target_name
        copy_required(figure, target)
        figure_records.append(
            {
                "figure_number": index,
                "source": project_relative(project_root, figure),
                "upload_name": target_name,
                "sha256": sha256(target),
            }
        )

    upload_files = sorted(path for path in upload.iterdir() if path.is_file())
    scan_public_text(
        [path for path in upload_files if path.suffix.lower() in {".md", ".txt"}]
    )
    checksum_lines = [f"{sha256(path)}  {path.name}" for path in upload_files]
    checksums = upload / "SHA256SUMS.txt"
    checksums.write_text("\n".join(checksum_lines) + "\n", encoding="ascii")

    guide = output_dir / "UPLOAD_GUIDE_CN.md"
    guide.write_text(
        "# IJHMT投稿文件说明\n\n"
        "`upload/`中是可直接用于投稿的文件：\n\n"
        "- `Manuscript.pdf`：主文PDF；\n"
        "- `p418_manuscript_source.zip`：LaTeX源文件和7张矢量主图；\n"
        "- `Figure_*.pdf`：按正文出现顺序编号的独立图件；\n"
        "- `Figure_captions.tex`：从正文源文件自动提取的7条可编辑图注；\n"
        + (
            "- `Graphical_abstract.pdf/png`：由正式计算图自动生成的图文摘要；\n"
            if graphical_abstract_included
            else ""
        )
        + "- `Cover_letter.pdf`：可直接上传的投稿信；`Cover_letter.md`为可编辑源文；\n"
        "- `Highlights.txt`：论文要点；\n"
        "- `CRediT_author_statement.md`、`Declaration_of_competing_interest.md`"
        "和`Acknowledgements.md`：作者贡献、利益冲突和基金致谢；\n"
        "- `Declaration_of_generative_AI_use.md`：AI辅助使用声明；\n"
        "- `p418_reproduction_source.tar.gz`：可公开的复现代码和参数说明；\n"
        "- `SHA256SUMS.txt`：文件完整性校验值。\n\n"
        "默认不包含英文补充材料，关键方法、结果和局限均放在正文。\n",
        encoding="utf-8",
    )

    bundle_entries = [
        (path, f"upload/{path.name}")
        for path in sorted(upload.iterdir())
        if path.is_file()
    ] + [(guide, guide.name)]
    bundle_zip = output_dir / BUNDLE_NAME
    write_deterministic_zip(bundle_zip, bundle_entries)

    bundle_complete = (
        check_payload.get("status") == "completed_p418_ijhmt_submission_check"
        and len(figures) == 7
    )
    return {
        "status": (
            "completed_p418_ijhmt_submission_bundle"
            if bundle_complete
            else "p418_ijhmt_submission_bundle_preflight"
        ),
        "submission_check_status": check_payload.get("status"),
        "main_pdf": {
            "path": "upload/Manuscript.pdf",
            "sha256": sha256(upload / "Manuscript.pdf"),
        },
        "cover_letter": {
            "pdf_path": "upload/Cover_letter.pdf",
            "pdf_sha256": sha256(upload / "Cover_letter.pdf"),
            "source_path": "upload/Cover_letter.md",
            "source_sha256": sha256(upload / "Cover_letter.md"),
        },
        "main_figure_count": len(figures),
        "figures": figure_records,
        "supplement_included": False,
        "graphical_abstract_included": graphical_abstract_included,
        "manuscript_source": {
            "path": f"upload/{SOURCE_NAME}",
            "sha256": sha256(manuscript_source),
        },
        "reproduction_source": {
            "path": f"upload/{REPRODUCTION_NAME}",
            "sha256": sha256(upload / REPRODUCTION_NAME),
        },
        "bundle_zip": {
            "path": bundle_zip.name,
            "sha256": sha256(bundle_zip),
            "size_bytes": bundle_zip.stat().st_size,
        },
        "upload_file_count": len(list(upload.iterdir())),
        "private_machine_text_scan_passed": True,
        "new_physical_parameters": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    payload = build_bundle(
        args.project_root,
        args.output_dir,
        require_complete=args.require_complete,
    )
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
