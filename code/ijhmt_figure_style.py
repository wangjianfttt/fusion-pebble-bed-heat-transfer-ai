"""Shared matplotlib settings for the IJHMT packed-bed manuscript figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_ijhmt_style(
    *,
    font_size: float = 8.0,
    label_size: float = 8.3,
    tick_size: float = 7.3,
    legend_size: float = 7.0,
    axis_width: float = 0.75,
) -> None:
    """Use the restrained sans-serif style found in recent IJHMT papers."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": font_size,
            "axes.labelsize": label_size,
            "axes.titlesize": label_size,
            "axes.titleweight": "normal",
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "axes.linewidth": axis_width,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": axis_width,
            "ytick.major.width": axis_width,
            "xtick.minor.width": axis_width,
            "ytick.minor.width": axis_width,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )
