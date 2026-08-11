from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_cover_letter_pdf import markdown_to_latex_source


def test_markdown_cover_letter_keeps_emphasis_links_and_signature_breaks() -> None:
    latex = markdown_to_latex_source(
        "Dear Editor,\n\n"
        "We submit **A graph–Transformer study** to the *International Journal*.\n\n"
        "Code: https://github.com/example/repo\n\n"
        "Jian Wang  \nCorresponding author  \nEmail: wjfttt@mail.ustc.edu.cn\n"
    )
    assert r"\textbf{A graph--Transformer study}" in latex
    assert r"\textit{International Journal}" in latex
    assert r"\url{https://github.com/example/repo}" in latex
    assert r"Jian Wang\\Corresponding author\\Email: wjfttt@mail.ustc.edu.cn" in latex
    assert r"\pagestyle{empty}" in latex
    assert r"\pdfinfoomitdate=1" in latex
