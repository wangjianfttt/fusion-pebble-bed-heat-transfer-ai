from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "研究主线_简明版_CN.md"


def test_plain_summary_covers_the_physical_research_chain() -> None:
    text = SUMMARY.read_text(encoding="utf-8")
    for term in (
        "三维OpenFOAM",
        "PINN",
        "图--Transformer",
        "DMDc和POD",
        "扩散模型",
        "60组",
        "12条",
        "质量和能量收支",
        "新颗粒排列",
        "物理参数从哪里来",
    ):
        assert term in text


def test_plain_summary_avoids_unnecessary_management_words() -> None:
    text = SUMMARY.read_text(encoding="utf-8")
    for term in ("审计", "门槛", "门控", "代理模型"):
        assert term not in text


def test_readme_links_the_plain_summary() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "研究主线_简明版_CN.md" in text
