#!/usr/bin/env python3
"""Build the one-page IJHMT cover letter PDF from its editable Markdown source."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


TOKEN = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|https?://\S+)")


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "–": "--",
        "—": "---",
    }
    return "".join(replacements.get(character, character) for character in text)


def inline_markdown(text: str) -> str:
    chunks: list[str] = []
    cursor = 0
    for match in TOKEN.finditer(text):
        chunks.append(escape_latex(text[cursor : match.start()]))
        token = match.group(0)
        if token.startswith("**"):
            chunks.append(r"\textbf{" + escape_latex(token[2:-2]) + "}")
        elif token.startswith("*"):
            chunks.append(r"\textit{" + escape_latex(token[1:-1]) + "}")
        else:
            chunks.append(r"\url{" + token + "}")
        cursor = match.end()
    chunks.append(escape_latex(text[cursor:]))
    return "".join(chunks)


def markdown_to_latex_source(markdown: str) -> str:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]
    body: list[str] = []
    for paragraph in paragraphs:
        lines = [inline_markdown(line.rstrip()) for line in paragraph.splitlines()]
        body.append(r"\\".join(lines))
    return (
        r"\documentclass[11pt,letterpaper]{article}" "\n"
        r"\usepackage[margin=25mm]{geometry}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage[hidelinks]{hyperref}" "\n"
        r"\urlstyle{same}" "\n"
        r"\pagestyle{empty}" "\n"
        r"\setlength{\parindent}{0pt}" "\n"
        r"\setlength{\parskip}{0.72em}" "\n"
        r"\pdfinfoomitdate=1" "\n"
        r"\pdftrailerid{}" "\n"
        r"\pdfsuppressptexinfo=15" "\n"
        r"\begin{document}" "\n"
        + "\n\n".join(body)
        + "\n"
        r"\end{document}" "\n"
    )


def build_pdf(source: Path, output: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError("pdflatex is required to build the cover letter PDF")
    output.parent.mkdir(parents=True, exist_ok=True)
    latex = markdown_to_latex_source(source.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="p418_cover_letter_") as temporary:
        work = Path(temporary)
        tex = work / "cover_letter.tex"
        tex.write_text(latex, encoding="utf-8")
        environment = dict(os.environ)
        environment["SOURCE_DATE_EPOCH"] = "1786406400"
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=work,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        built = work / "cover_letter.pdf"
        if result.returncode != 0 or not built.is_file() or built.stat().st_size == 0:
            raise RuntimeError("cover letter PDF build failed:\n" + result.stdout[-4000:])
        temporary_output = output.with_suffix(output.suffix + ".part")
        shutil.copy2(built, temporary_output)
        temporary_output.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_pdf(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
