from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from plot_hccb_p418_learning_curve import METRICS, STYLE, plot  # noqa: E402


def test_learning_curve_uses_equal_slightly_wide_panels(tmp_path: Path) -> None:
    source = tmp_path / "learning_curve.csv"
    fieldnames = [
        "architecture",
        "train_case_count",
        "openfoam_training_core_hours_32ranks",
        *(metric for metric, _ in METRICS),
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for architecture_index, architecture in enumerate(STYLE):
            for count, cost in zip((9, 18, 27, 36), (100.0, 250.0, 420.0, 530.0)):
                value = 0.1 + 0.01 * architecture_index + 1.0 / count
                writer.writerow(
                    {
                        "architecture": architecture,
                        "train_case_count": count,
                        "openfoam_training_core_hours_32ranks": cost,
                        **{metric: value for metric, _ in METRICS},
                    }
                )

    summary = plot(source, tmp_path / "figure")

    assert summary["panel_width_spread"] < 1.0e-12
    assert summary["panel_height_spread"] < 1.0e-12
    assert 1.15 <= summary["panel_width_to_height_ratio"] <= 1.35
    assert (tmp_path / "figure" / "learning_curve_efficiency.pdf").is_file()


def test_learning_curve_panel_labels_use_parentheses() -> None:
    source = (ROOT / "code" / "plot_hccb_p418_learning_curve.py").read_text(
        encoding="utf-8"
    )
    assert "f\"({chr(ord('a') + panel)})\"" in source
    assert ".set_title(" not in source
