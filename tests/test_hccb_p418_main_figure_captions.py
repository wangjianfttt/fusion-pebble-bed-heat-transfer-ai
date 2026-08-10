from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "manuscript/main.tex",
    ROOT / "manuscript/methods_condensed.tex",
    ROOT / "manuscript/results_condensed.tex",
)


def main_figure_captions() -> list[tuple[str, str]]:
    captions: list[tuple[str, str]] = []
    pattern = re.compile(
        r"\\caption\{(.*?)\}\s*\\label\{(fig:[^}]+)\}", re.DOTALL
    )
    for source in SOURCES:
        text = source.read_text(encoding="utf-8")
        for caption, label in pattern.findall(text):
            captions.append((label, re.sub(r"\s+", " ", caption).strip()))
    return captions


def test_seven_main_figure_captions_are_compact_and_final_in_tone() -> None:
    captions = main_figure_captions()
    assert len(captions) == 7
    labels = [label for label, _ in captions]
    assert len(labels) == len(set(labels))
    forbidden = ("draft", "pending", "will be inserted", "to be generated")
    for label, caption in captions:
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", caption)
        assert len(words) <= 65, (label, len(words))
        assert caption.endswith(".")
        assert not any(token in caption.lower() for token in forbidden)


def test_panelled_captions_name_each_plotted_panel_range() -> None:
    captions = dict(main_figure_captions())
    assert "(a)" in captions["fig:workflow"]
    assert "(d)" in captions["fig:workflow"]
    assert "(a)" in captions["fig:physical_response"]
    assert "(f)" in captions["fig:physical_response"]
    assert "(a)" in captions["fig:external_evidence"]
    assert "(d)" in captions["fig:external_evidence"]
    assert "(a)" in captions["fig:cross_packing_integral"]
    assert "(d)" in captions["fig:cross_packing_integral"]
    assert "(a)" in captions["fig:transient_model_comparison"]
    assert "(f)" in captions["fig:transient_model_comparison"]
