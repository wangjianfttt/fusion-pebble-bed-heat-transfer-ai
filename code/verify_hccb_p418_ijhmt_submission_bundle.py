#!/usr/bin/env python3
"""Verify that the IJHMT upload bundle is complete and self-compiling."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"required source-verification tool is missing: {name}")
    return path


def safe_zip(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate archive members in {path}")
        if any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise RuntimeError(f"unsafe archive member in {path}")
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"corrupt archive member in {path}: {bad}")
    return names


def pdf_pages(pdfinfo: str, path: Path) -> int:
    output = subprocess.check_output([pdfinfo, str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)", output, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"cannot read page count from {path}")
    return int(match.group(1))


def normalized_pdf_text(pdftotext: str, path: Path) -> str:
    output = subprocess.check_output(
        [pdftotext, str(path), "-"], text=True, errors="replace"
    )
    return " ".join(output.split())


def resolve_record_path(bundle_dir: Path, record_path: Path | None) -> Path:
    if record_path is not None:
        resolved = record_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
    for name in ("record.json", "package_record.json"):
        candidate = bundle_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"submission bundle record is missing from {bundle_dir}: "
        "expected record.json or package_record.json"
    )


def verify_bundle(
    bundle_dir: Path,
    *,
    record_path: Path | None = None,
    require_complete: bool = False,
    log_path: Path | None = None,
    texinputs: Path | None = None,
) -> dict[str, object]:
    bundle_dir = bundle_dir.resolve()
    upload = bundle_dir / "upload"
    record = resolve_record_path(bundle_dir, record_path)
    payload = json.loads(record.read_text(encoding="utf-8"))
    if require_complete and payload.get("status") != (
        "completed_p418_ijhmt_submission_bundle"
    ):
        raise RuntimeError("submission bundle is not marked complete")

    checksums_path = upload / "SHA256SUMS.txt"
    expected: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if name in expected:
            raise RuntimeError(f"duplicate checksum entry: {name}")
        expected[name] = digest
    actual_files = {
        path.name for path in upload.iterdir() if path.is_file() and path != checksums_path
    }
    if actual_files != set(expected):
        raise RuntimeError(
            f"upload/checksum file mismatch: actual={sorted(actual_files)}, "
            f"expected={sorted(expected)}"
        )
    for name, digest in expected.items():
        actual = sha256(upload / name)
        if actual != digest:
            raise RuntimeError(f"SHA mismatch for upload file {name}: {actual}")

    bundle_zip = bundle_dir / "p418_ijhmt_upload_bundle.zip"
    bundle_members = safe_zip(bundle_zip)
    source_zip = upload / "p418_manuscript_source.zip"
    source_members = safe_zip(source_zip)
    figure_members = [
        name
        for name in source_members
        if name.startswith("figures/") and name.lower().endswith(".pdf")
    ]
    if require_complete and len(figure_members) != 7:
        raise RuntimeError(f"expected 7 source figures, found {len(figure_members)}")
    if any("supplement" in name.lower() for name in source_members):
        raise RuntimeError("supplement material is unexpectedly present")

    pdfinfo = require_tool("pdfinfo")
    pdftotext = require_tool("pdftotext")
    latexmk = shutil.which("latexmk")
    if latexmk:
        compiler = "latexmk"
        compile_commands = [
            [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        ]
    else:
        compiler = "pdflatex+bibtex"
        pdflatex = require_tool("pdflatex")
        bibtex = require_tool("bibtex")
        compile_commands = [
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            [bibtex, "main"],
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ]
    with tempfile.TemporaryDirectory(prefix="p418_ijhmt_source_compile_") as temp:
        root = Path(temp)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(root)
        manuscript = root / "manuscript"
        compile_parts = []
        process = None
        compile_env = os.environ.copy()
        if texinputs is not None:
            texinputs = texinputs.resolve()
            if not texinputs.is_dir():
                raise FileNotFoundError(texinputs)
            existing = compile_env.get("TEXINPUTS", "")
            compile_env["TEXINPUTS"] = f"{texinputs}//:{existing}"
        for command in compile_commands:
            process = subprocess.run(
                command,
                cwd=manuscript,
                text=True,
                capture_output=True,
                timeout=300,
                env=compile_env,
            )
            compile_parts.append(
                f"$ {' '.join(command)}\n{process.stdout}{process.stderr}"
            )
            if process.returncode != 0:
                break
        assert process is not None
        compile_text = "\n".join(compile_parts)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(compile_text, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(
                "standalone manuscript source compilation failed:\n"
                + compile_text[-4000:]
            )
        rebuilt = manuscript / "main.pdf"
        reference = upload / "Manuscript.pdf"
        if not rebuilt.is_file() or rebuilt.stat().st_size == 0:
            raise RuntimeError("standalone source did not produce main.pdf")
        reference_pages = pdf_pages(pdfinfo, reference)
        rebuilt_pages = pdf_pages(pdfinfo, rebuilt)
        reference_text = normalized_pdf_text(pdftotext, reference)
        rebuilt_text = normalized_pdf_text(pdftotext, rebuilt)
        if reference_pages != rebuilt_pages:
            raise RuntimeError(
                f"source rebuild page mismatch: {reference_pages} != {rebuilt_pages}"
            )
        reference_tokens = reference_text.split()
        rebuilt_tokens = rebuilt_text.split()
        text_match_ratio = difflib.SequenceMatcher(
            None, reference_tokens, rebuilt_tokens, autojunk=False
        ).ratio()
        if text_match_ratio < 0.999:
            raise RuntimeError(
                "source rebuild text differs materially from submitted Manuscript.pdf: "
                f"token sequence ratio={text_match_ratio:.6f}"
            )

    complete = (
        payload.get("status") == "completed_p418_ijhmt_submission_bundle"
        and len(figure_members) == 7
    )
    return {
        "status": (
            "completed_p418_ijhmt_submission_bundle_verification"
            if complete
            else "p418_ijhmt_submission_bundle_preflight_verified"
        ),
        "submission_bundle_status": payload.get("status"),
        "upload_checksum_count": len(expected),
        "bundle_member_count": len(bundle_members),
        "source_member_count": len(source_members),
        "source_figure_count": len(figure_members),
        "supplement_included": False,
        "reference_page_count": reference_pages,
        "rebuilt_page_count": rebuilt_pages,
        "normalized_pdf_text_sha256": hashlib.sha256(
            reference_text.encode("utf-8")
        ).hexdigest(),
        "rebuilt_normalized_pdf_text_sha256": hashlib.sha256(
            rebuilt_text.encode("utf-8")
        ).hexdigest(),
        "pdf_text_exact_match": reference_text == rebuilt_text,
        "pdf_token_sequence_match_ratio": text_match_ratio,
        "minimum_pdf_token_sequence_match_ratio": 0.999,
        "standalone_source_compiler": compiler,
        "standard_tex_runtime_supplied": texinputs is not None,
        "standalone_source_compilation_passed": True,
        "standalone_source_text_matches_manuscript_pdf": text_match_ratio >= 0.999,
        "new_physical_parameters": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--texinputs", type=Path)
    args = parser.parse_args()
    log_path = args.output.with_name("source_compile.log")
    payload = verify_bundle(
        args.bundle_dir,
        record_path=args.record,
        require_complete=args.require_complete,
        log_path=log_path,
        texinputs=args.texinputs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
