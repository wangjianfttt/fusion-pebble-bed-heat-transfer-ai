from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
READER = ROOT / "manuscript/P418_论文中文便读版.md"


def test_chinese_preview_is_concise_and_matches_the_current_scope() -> None:
    text = READER.read_text(encoding="utf-8")
    assert len(text) < 12_000
    assert all(f"## {index}." in text for index in range(1, 8))
    for statement in (
        "60 个三维共轭换热工况",
        "12 条固定流场、0--300 s 的热阶跃曲线",
        "6 条不参与训练的高速度固定流场曲线",
        "seed202 的 9 个匹配稳态工况",
        "不代表阀门或泵速变化后的最初流动启动",
        "没有启动稳态求解",
        "不能声称已经验证了完整的初始动量瞬态",
        "目前已经完成`20/30`项",
        "当前英文PDF为`23`页",
        "主文不设附录",
    ):
        assert statement in text
    assert "200秒" not in text
    assert "全文域稳态求解已完成" not in text
    assert "全耦合时间步收敛已验证" not in text
    assert "40/75" not in text
    assert "19/30" not in text
    assert "next_epoch=34/500" not in text
    assert "当前英文PDF为`22`页" not in text
    assert "`7062`个英文词" not in text
    assert "约`7018`个英文词" not in text
    assert "程序正在为训练、验证和独立测试曲线生成" not in text
    assert not any(term in text for term in ("审计", "门槛", "门控", "代理"))
