#!/usr/bin/env python3
"""Build the source-backed high-velocity OpenFOAM response table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CASE_ORDER = (
    "high_re_temperature_up_u0p25_q6p85",
    "high_re_temperature_down_u0p25_q6p85",
    "high_re_velocity_up_T300_q6p85",
    "high_re_velocity_down_T300_q6p85",
    "high_re_source_up_u0p25_T300",
    "high_re_source_down_u0p25_T300",
)

LABELS = {
    "high_re_temperature_up_u0p25_q6p85": ("Inlet temperature", r"$300\rightarrow900$ K"),
    "high_re_temperature_down_u0p25_q6p85": ("Inlet temperature", r"$900\rightarrow300$ K"),
    "high_re_velocity_up_T300_q6p85": ("Inlet velocity", r"$0.05\rightarrow0.25$ m s$^{-1}$"),
    "high_re_velocity_down_T300_q6p85": ("Inlet velocity", r"$0.25\rightarrow0.05$ m s$^{-1}$"),
    "high_re_source_up_u0p25_T300": ("Solid heat source", r"$4.85\rightarrow8.85$ MW m$^{-3}$"),
    "high_re_source_down_u0p25_T300": ("Solid heat source", r"$8.85\rightarrow4.85$ MW m$^{-3}$"),
}


def load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed_p418_high_re_step_response_analysis":
        raise ValueError("high-velocity response analysis is not complete")
    if payload.get("curve_count") != 6 or payload.get("points_per_curve") != 16401:
        raise ValueError("expected six complete 16401-point response curves")
    if payload.get("time_range_s") != [0.0, 300.0]:
        raise ValueError("expected the formal 0--300 s observation interval")
    return payload


def build_table(payload: dict) -> str:
    cases = {case["sequence_id"]: case for case in payload["cases"]}
    if set(cases) != set(CASE_ORDER):
        raise ValueError("unexpected or missing high-velocity sequence")

    rows = []
    for sequence_id in CASE_ORDER:
        case = cases[sequence_id]
        if case.get("time_point_count") != 16401:
            raise ValueError(f"{sequence_id}: incomplete time history")
        response = case["responses"]["outlet_temperature_K"]
        mass_residual = float(case["maximum_absolute_mass_residual_kg_s"])
        if mass_residual > 1.0e-12:
            raise ValueError(f"{sequence_id}: mass residual exceeds table limit")
        quantity, change = LABELS[sequence_id]
        rows.append(
            f"{quantity} & {change} & "
            f"{float(response['change']):+.2f} & "
            f"{float(response['t50_s']):.2f} & "
            f"{float(response['t90_s']):.2f} \\\\"
        )

    return "\n".join(
        [
            r"\begin{table*}[htbp]",
            r"\centering",
            r"\caption{Reference response of the six independent high-velocity fixed-flow OpenFOAM trajectories. "
            r"$\Delta T_{\mathrm{out}}$ is the final minus initial outlet temperature. "
            r"$t_{50}$ and $t_{90}$ are the first times at which 50\% and 90\% of that signed change are reached. "
            r"All histories span 0--300 s and contain 16\,401 time points.}",
            r"\label{tab:high_re_reference_response}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Perturbed quantity & Source$\rightarrow$target & $\Delta T_{\mathrm{out}}$ (K) & $t_{50}$ (s) & $t_{90}$ (s) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = build_table(load_summary(args.summary.resolve()))
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
