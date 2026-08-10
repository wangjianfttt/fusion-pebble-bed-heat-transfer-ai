#!/usr/bin/env python3
"""Build the manuscript table for the P418 thermal time-step comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "outlet_temperature_K": r"Outlet temperature",
    "cooling_wall_power_W": r"Cooling-wall power",
    "maximum_solid_temperature_K": r"Maximum solid temperature",
    "volume_average_fluid_temperature_K": r"Volume-averaged fluid temperature",
    "volume_average_solid_temperature_K": r"Volume-averaged solid temperature",
}


def gci_text(row: dict) -> str:
    status = row["convergence_status"]
    value = row.get("fine_gci_fraction")
    if value is not None:
        return f"{100.0 * float(value):.3g}\\%"
    return {
        "oscillatory_no_gci_reported": "oscillatory",
        "not_monotonically_reducing_no_gci_reported": "non-reducing",
    }.get(status, "not reported")


def build_table(payload: dict) -> str:
    if payload.get("status") != "completed_p418_thermal_timestep_sensitivity":
        raise ValueError("thermal time-step comparison is incomplete")
    selected = float(payload["selected_delta_t_s"])
    declared = [float(value) for value in payload["delta_t_s"]]
    if selected != min(declared):
        raise ValueError("formal thermal histories do not use the finest time step")
    schedule = payload.get("selected_time_step_schedule")
    if not schedule:
        raise ValueError("thermal time-step comparison does not contain the staged schedule")
    schedule_text = ", ".join(
        rf"\num{{{float(row['delta_t_s']):.6g}}}\,s over \num{{{float(row['start_s']):g}}}--\num{{{float(row['end_s']):g}}}\,s"
        for row in schedule
    )
    ordered = [name for name in LABELS if any(row["signal"] == name for row in payload["comparisons"])]
    if not ordered:
        raise ValueError("time-step comparison contains no manuscript quantities")
    finest_pair = {}
    for row in payload["comparisons"]:
        if float(row["fine_delta_t_s"]) == selected:
            finest_pair[row["signal"]] = row
    gci = {(row["signal"], row["quantity"]): row for row in payload["gci_results"]}

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\caption{{Temporal-resolution comparison for the representative heat-source step. The computed trajectories use the finest staged schedule: {schedule_text}. The second column reports the largest next-coarser--finest curve difference normalized by the finest-curve response span. GCI values are reported only for monotonically converging three-resolution triplets.}}",
        r"\label{tab:thermal_timestep}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        (
            r"Quantity & \shortstack{Finest-pair curve\\difference} "
            r"& \shortstack{Endpoint\\GCI} "
            r"& \shortstack{Curve-maximum\\GCI} \\"
        ),
        r"\midrule",
    ]
    for signal in ordered:
        if signal not in finest_pair:
            raise ValueError(f"missing finest-pair comparison for {signal}")
        difference = 100.0 * float(finest_pair[signal]["maximum_difference_over_response_span"])
        endpoint = gci_text(gci[(signal, "endpoint")])
        maximum = gci_text(gci[(signal, "curve_maximum")])
        lines.append(f"{LABELS[signal]} & {difference:.3g}\\% & {endpoint} & {maximum} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.resolve().read_text(encoding="utf-8"))
    text = build_table(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
