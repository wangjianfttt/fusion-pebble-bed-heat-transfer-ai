#!/usr/bin/env python3
"""Extract the fixed-flow 0--0.01 s reference for the matched-initial smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "hccb_p418_fixed_timestep_source_up_u0p15_T700_v7"
    / "dt_1em05"
    / "hccb_p418_transient_observables.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "hccb_p418_matched_initial_fixed_flow_reference_0p01s_20260809"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def extract(input_path: Path, end_time_s: float) -> tuple[list[dict[str, float]], dict]:
    data = np.load(input_path, allow_pickle=True)
    case_ids = [str(value) for value in data["case_id"]]
    if case_ids != ["source_up_u0p15_T700"]:
        raise ValueError(f"unexpected case ids: {case_ids}")
    if data["complete"].tolist() != [True]:
        raise ValueError("fixed-flow reference is incomplete")
    source_mask = data["time_mask"][0].astype(bool)
    source_time = data["time_s"][0, source_mask].astype(np.float64)
    source_values = data["values"][0, source_mask].astype(np.float64)
    signal_names = [str(value) for value in data["signal_names"]]
    if source_values.shape != (source_time.size, len(signal_names)):
        raise ValueError("time/value shape mismatch")
    if not np.isfinite(source_time).all() or not np.isfinite(source_values).all():
        raise ValueError("fixed-flow reference contains non-finite values")
    if not np.all(np.diff(source_time) > 0.0):
        raise ValueError("fixed-flow times are not strictly increasing")
    keep = source_time <= end_time_s + 1.0e-14
    time_s = source_time[keep]
    values = source_values[keep]
    if time_s.size < 2 or abs(float(time_s[0])) > 1.0e-15:
        raise ValueError("short reference does not start at zero")
    if abs(float(time_s[-1]) - end_time_s) > 1.0e-12:
        raise ValueError("short reference does not end at the requested time")

    rows = []
    for index, time_value in enumerate(time_s):
        row = {"time_s": float(time_value)}
        row.update(
            {name: float(values[index, column]) for column, name in enumerate(signal_names)}
        )
        rows.append(row)
    stats = {}
    for column, name in enumerate(signal_names):
        column_values = values[:, column]
        stats[name] = {
            "initial": float(column_values[0]),
            "final": float(column_values[-1]),
            "change": float(column_values[-1] - column_values[0]),
            "minimum": float(np.min(column_values)),
            "maximum": float(np.max(column_values)),
        }
    summary = {
        "status": "p418_fixed_flow_matched_initial_0p01s_reference_ready",
        "sequence_id": "source_up_u0p15_T700",
        "history_mode": "fixed_hydrodynamics_thermal",
        "source_observables": portable_path(input_path),
        "source_observables_sha256": sha256(input_path),
        "start_time_s": float(time_s[0]),
        "end_time_s": float(time_s[-1]),
        "time_point_count": int(time_s.size),
        "minimum_time_spacing_s": float(np.min(np.diff(time_s))),
        "maximum_time_spacing_s": float(np.max(np.diff(time_s))),
        "signal_count": len(signal_names),
        "signal_names": signal_names,
        "signal_statistics": stats,
        "interpolation_used": False,
        "openfoam_solver_started_by_this_extraction": False,
        "new_physical_parameters": [],
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--end-time-s", type=float, default=0.01)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = extract(input_path, args.end_time_s)
    csv_path = output / "fixed_flow_reference_0p01s.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["csv"] = portable_path(csv_path)
    summary["csv_sha256"] = sha256(csv_path)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
