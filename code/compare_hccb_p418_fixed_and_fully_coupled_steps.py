#!/usr/bin/env python3
"""Compare fixed-hydrodynamics and fully coupled P418 step histories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_SIGNALS = (
    "pressure_drop_Pa",
    "outlet_temperature_K",
    "maximum_solid_temperature_K",
    "volume_average_fluid_temperature_K",
    "volume_average_solid_temperature_K",
    "cooling_wall_power_W",
    "signed_mass_residual_kg_s",
    "net_outward_enthalpy_flow_W",
)

SIGNAL_UNITS = {
    "pressure_drop_Pa": "Pa",
    "outlet_temperature_K": "K",
    "maximum_solid_temperature_K": "K",
    "volume_average_fluid_temperature_K": "K",
    "volume_average_solid_temperature_K": "K",
    "cooling_wall_power_W": "W",
    "signed_mass_residual_kg_s": "kg s^-1",
    "net_outward_enthalpy_flow_W": "W",
}


def load_curves(path: Path) -> dict[str, dict[str, object]]:
    data = np.load(path, allow_pickle=True)
    names = [str(value) for value in data["signal_names"]]
    result: dict[str, dict[str, object]] = {}
    for index, raw_id in enumerate(data["case_id"]):
        mask = data["time_mask"][index].astype(bool)
        result[str(raw_id)] = {
            "complete": bool(data["complete"][index]),
            "time_s": data["time_s"][index, mask].astype(np.float64),
            "values": data["values"][index, mask].astype(np.float64),
            "signal_names": names,
        }
    return result


def compare(
    fixed_path: Path,
    coupled_path: Path,
    requested_signals: tuple[str, ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    fixed = load_curves(fixed_path)
    coupled = load_curves(coupled_path)
    if set(fixed) != set(coupled):
        raise ValueError(
            "fixed and fully coupled observables do not contain the same sequence ids"
        )
    rows: list[dict[str, object]] = []
    for sequence_id in sorted(fixed):
        fixed_curve = fixed[sequence_id]
        coupled_curve = coupled[sequence_id]
        if not fixed_curve["complete"] or not coupled_curve["complete"]:
            raise ValueError(f"incomplete response history for {sequence_id}")
        fixed_names = fixed_curve["signal_names"]
        coupled_names = coupled_curve["signal_names"]
        missing = [
            signal
            for signal in requested_signals
            if signal not in fixed_names or signal not in coupled_names
        ]
        if missing:
            raise ValueError(f"{sequence_id} lacks comparison signals {missing}")
        fixed_time = fixed_curve["time_s"]
        coupled_time = coupled_curve["time_s"]
        start = max(float(fixed_time[0]), float(coupled_time[0]))
        end = min(float(fixed_time[-1]), float(coupled_time[-1]))
        common_time = fixed_time[(fixed_time >= start) & (fixed_time <= end)]
        if common_time.size < 2:
            raise ValueError(f"{sequence_id} has fewer than two common time points")
        for signal in requested_signals:
            fixed_values = fixed_curve["values"][:, fixed_names.index(signal)]
            coupled_values = coupled_curve["values"][:, coupled_names.index(signal)]
            if not np.isfinite(fixed_values).all() or not np.isfinite(coupled_values).all():
                raise ValueError(f"{sequence_id} contains non-finite {signal}")
            fixed_common = np.interp(common_time, fixed_time, fixed_values)
            coupled_common = np.interp(common_time, coupled_time, coupled_values)
            difference = fixed_common - coupled_common
            response_span = float(np.max(coupled_common) - np.min(coupled_common))
            response_scale = max(
                response_span,
                abs(float(coupled_common[-1] - coupled_common[0])),
                1.0e-30,
            )
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "signal": signal,
                    "common_time_point_count": int(common_time.size),
                    "comparison_start_s": start,
                    "comparison_end_s": end,
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "maximum_absolute_difference": float(
                        np.max(np.abs(difference))
                    ),
                    "endpoint_absolute_difference": float(abs(difference[-1])),
                    "maximum_difference_over_fully_coupled_response_span": float(
                        np.max(np.abs(difference)) / response_scale
                    ),
                }
            )
    aggregate = {}
    for signal in requested_signals:
        signal_rows = [row for row in rows if row["signal"] == signal]
        worst_normalized_row = max(
            signal_rows,
            key=lambda row: row[
                "maximum_difference_over_fully_coupled_response_span"
            ],
        )
        worst_absolute_row = max(
            signal_rows,
            key=lambda row: row["maximum_absolute_difference"],
        )
        aggregate[signal] = {
            "unit": SIGNAL_UNITS.get(signal, ""),
            "trajectory_count": len(signal_rows),
            "median_rmse": float(
                np.median([float(row["rmse"]) for row in signal_rows])
            ),
            "largest_absolute_difference": worst_absolute_row[
                "maximum_absolute_difference"
            ],
            "largest_absolute_difference_sequence_id": worst_absolute_row[
                "sequence_id"
            ],
            "largest_difference_over_fully_coupled_response_span": (
                worst_normalized_row[
                    "maximum_difference_over_fully_coupled_response_span"
                ]
            ),
            "largest_normalized_difference_sequence_id": worst_normalized_row[
                "sequence_id"
            ],
            "maximum_absolute_difference_at_largest_normalized_difference": (
                worst_normalized_row["maximum_absolute_difference"]
            ),
            "median_difference_over_fully_coupled_response_span": float(
                np.median(
                    [
                        float(
                            row[
                                "maximum_difference_over_fully_coupled_response_span"
                            ]
                        )
                        for row in signal_rows
                    ]
                )
            ),
            "maximum_difference_over_fully_coupled_response_span": (
                worst_normalized_row[
                "maximum_difference_over_fully_coupled_response_span"
                ]
            ),
        }
    summary = {
        "status": "completed_p418_fixed_vs_fully_coupled_step_comparison",
        "fixed_observables": str(fixed_path),
        "fully_coupled_observables": str(coupled_path),
        "sequence_count": len(fixed),
        "signals": list(requested_signals),
        "comparison_rule": (
            "Interpolate the fully coupled history to fixed-hydrodynamics output "
            "times inside their common interval and report dimensional and "
            "response-normalized differences. No fitted acceptance percentage is used."
        ),
        "aggregate_by_signal": aggregate,
        "worst_case_by_signal": {
            signal: {
                "sequence_id": values[
                    "largest_normalized_difference_sequence_id"
                ],
                "maximum_absolute_difference": values[
                    "maximum_absolute_difference_at_largest_normalized_difference"
                ],
                "maximum_difference_over_fully_coupled_response_span": values[
                    "largest_difference_over_fully_coupled_response_span"
                ],
            }
            for signal, values in aggregate.items()
        },
        "new_physical_parameters": [],
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-observables", type=Path, required=True)
    parser.add_argument("--fully-coupled-observables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal", action="append", default=[])
    args = parser.parse_args()
    signals = tuple(args.signal) if args.signal else DEFAULT_SIGNALS
    rows, summary = compare(
        args.fixed_observables.resolve(),
        args.fully_coupled_observables.resolve(),
        signals,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "fixed_vs_fully_coupled_steps.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["artifact"] = str(csv_path)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
