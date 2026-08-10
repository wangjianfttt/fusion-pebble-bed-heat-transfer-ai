#!/usr/bin/env python3
"""Compare matched-initial fixed-flow and fully coupled 0--0.01 s responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SIGNALS = (
    "inlet_temperature_K",
    "outlet_temperature_K",
    "inlet_pressure_Pa",
    "outlet_pressure_Pa",
    "inlet_mass_flow_kg_s",
    "outlet_mass_flow_kg_s",
    "inlet_enthalpy_flow_W",
    "outlet_enthalpy_flow_W",
    "cooling_wall_power_W",
    "maximum_solid_temperature_K",
    "volume_average_fluid_temperature_K",
    "volume_average_solid_temperature_K",
    "pressure_drop_Pa",
    "signed_mass_residual_kg_s",
    "net_outward_enthalpy_flow_W",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"{path} contains fewer than two rows")
    missing = [name for name in ("time_s", *SIGNALS) if name not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    time = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in SIGNALS
    }
    if not np.isfinite(time).all() or any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError(f"{path} contains non-finite values")
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{path} time is not strictly increasing")
    return time, values


def compare(fixed_csv: Path, coupled_csv: Path, output: Path) -> dict[str, object]:
    fixed_time, fixed = load_csv(fixed_csv)
    coupled_time, coupled = load_csv(coupled_csv)
    tolerance = 1.0e-12
    if fixed_time[0] > tolerance or coupled_time[0] > tolerance:
        raise ValueError("both responses must include time zero")
    if fixed_time[-1] < 0.01 - tolerance or coupled_time[-1] < 0.01 - tolerance:
        raise ValueError("both responses must reach 0.01 s")

    mask = coupled_time <= 0.01 + tolerance
    comparison_time = coupled_time[mask]
    metrics: dict[str, dict[str, float | None]] = {}
    for name in SIGNALS:
        direct = coupled[name][mask]
        reference = np.interp(comparison_time, fixed_time, fixed[name])
        difference = direct - reference
        fixed_change = float(reference[-1] - reference[0])
        direct_change = float(direct[-1] - direct[0])
        characteristic = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
        change_relative = (
            float((direct_change - fixed_change) / abs(fixed_change))
            if abs(fixed_change) > 1.0e-14 * characteristic
            else None
        )
        metrics[name] = {
            "fixed_initial": float(reference[0]),
            "coupled_initial": float(direct[0]),
            "initial_difference": float(difference[0]),
            "fixed_final": float(reference[-1]),
            "coupled_final": float(direct[-1]),
            "final_difference": float(difference[-1]),
            "fixed_change": fixed_change,
            "coupled_change": direct_change,
            "change_difference": float(direct_change - fixed_change),
            "change_relative_difference": change_relative,
            "time_series_rmse": float(np.sqrt(np.mean(difference**2))),
            "maximum_absolute_time_series_difference": float(np.max(np.abs(difference))),
        }

    payload: dict[str, object] = {
        "status": "p418_matched_initial_fixed_flow_fully_coupled_short_comparison_complete",
        "sequence_id": "source_up_u0p15_T700",
        "comparison_start_time_s": float(comparison_time[0]),
        "comparison_end_time_s": float(comparison_time[-1]),
        "fully_coupled_time_point_count": int(comparison_time.size),
        "fixed_flow_time_point_count": int(fixed_time.size),
        "fixed_flow_interpolated_at_fully_coupled_times": True,
        "signal_count": len(SIGNALS),
        "signals": metrics,
        "inputs": {
            "fixed_flow_csv": str(fixed_csv),
            "fixed_flow_csv_sha256": sha256(fixed_csv),
            "fully_coupled_csv": str(coupled_csv),
            "fully_coupled_csv_sha256": sha256(coupled_csv),
        },
        "openfoam_solver_started_by_this_comparison": False,
        "new_physical_parameters": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-flow-csv", type=Path, required=True)
    parser.add_argument("--fully-coupled-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(
        args.fixed_flow_csv.resolve(),
        args.fully_coupled_csv.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
