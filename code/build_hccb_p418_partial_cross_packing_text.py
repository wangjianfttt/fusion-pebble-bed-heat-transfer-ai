#!/usr/bin/env python3
"""Write the partial seed101--seed202 integral-response comparison for the SI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_metric(summary: dict[str, object], metric: str, field: str) -> float:
    value = float(summary["metric_summary"][metric][field])  # type: ignore[index]
    if not math.isfinite(value):
        raise ValueError(f"non-finite {metric}.{field}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    status = payload.get("status")
    allowed = {
        "partial_seed101_seed202_integral_response_comparison",
        "completed_seed101_seed202_integral_response_comparison",
    }
    if status not in allowed:
        raise ValueError("unexpected cross-packing summary status")
    complete = payload.get("complete_nine_case_comparison") is True
    if complete:
        if int(payload.get("accepted_common_case_count", -1)) != 9:
            raise ValueError("the complete comparison requires nine common cases")
        if int(payload.get("failed_seed202_case_count", -1)) != 0:
            raise ValueError("the complete comparison cannot contain failed cases")
    else:
        if int(payload.get("accepted_common_case_count", -1)) != 7:
            raise ValueError("the partial comparison requires seven common cases")
        if int(payload.get("failed_seed202_case_count", -1)) != 2:
            raise ValueError("the partial comparison must identify two failed cases")

    failed_cases = payload.get("failed_seed202_cases")
    if complete:
        if failed_cases != []:
            raise ValueError(f"unexpected failed seed202 cases: {failed_cases}")
    else:
        expected_failed = ["u0p25_T900_q4p85", "u0p25_T900_q8p85"]
        if failed_cases != expected_failed:
            raise ValueError(f"unexpected failed seed202 cases: {failed_cases}")

    outlet_mean = finite_metric(
        payload, "outlet_temperature_K", "mean_absolute_relative_change_percent"
    )
    outlet_max = finite_metric(
        payload, "outlet_temperature_K", "maximum_absolute_relative_change_percent"
    )
    solid_mean = finite_metric(
        payload,
        "maximum_solid_temperature_K",
        "mean_absolute_relative_change_percent",
    )
    solid_max = finite_metric(
        payload,
        "maximum_solid_temperature_K",
        "maximum_absolute_relative_change_percent",
    )
    pressure_mean = finite_metric(
        payload, "pressure_drop_Pa", "mean_absolute_relative_change_percent"
    )
    pressure_min, pressure_max = [
        float(value)
        for value in payload["metric_summary"]["pressure_drop_Pa"][  # type: ignore[index]
            "relative_change_percent_range"
        ]
    ]
    if not all(math.isfinite(value) for value in (pressure_min, pressure_max)):
        raise ValueError("non-finite pressure-change range")

    if complete:
        opening = (
            "All nine tested operating conditions have valid finite-volume "
            "solutions on both the seed101 and seed202 pebble arrangements. "
        )
        closing = (
            "The comparison therefore covers the eight corners and the interior "
            "point of the independent-packing test matrix. It quantifies "
            "packing sensitivity of integral thermal and hydraulic responses; it "
            "does not replace the separate frozen-model field prediction test.\n"
        )
    else:
        opening = (
            "Seven operating conditions have valid finite-volume solutions on both "
            "the seed101 and seed202 pebble arrangements. "
        )
        closing = (
            "This comparison is not a complete nine-condition packing study: the "
            "two seed202 cases at $u=\\SI{0.25}{m.s^{-1}}$ and "
            "$T_{\\mathrm{in}}=\\SI{900}{K}$ are excluded because their nonlinear "
            "pressure iterates exceed the declared helium viscosity-table range, "
            "and no extrapolated material property is introduced.\n"
        )
    text = (
        opening
        + "Across these paired cases, "
        "the mean absolute relative changes in outlet temperature and maximum "
        f"solid temperature are {outlet_mean:.3f}\\% and {solid_mean:.3f}\\%, "
        f"with maxima of {outlet_max:.3f}\\% and {solid_max:.3f}\\%, respectively. "
        "The seed202 pressure drop is higher in every paired condition, by "
        f"{pressure_min:.2f}--{pressure_max:.2f}\\% "
        f"(mean absolute change {pressure_mean:.2f}\\%). "
        f"Thus, for the two arrangements and "
        f"{int(payload['accepted_common_case_count'])} common conditions tested here, "
        "the integral thermal response is substantially less sensitive to the "
        "packing realization than the hydraulic resistance. "
        + closing
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
