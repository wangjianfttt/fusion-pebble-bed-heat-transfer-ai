#!/usr/bin/env python3
"""Export one completed matched-initial fully coupled smoke to comparison data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from export_hccb_p418_transient_observables import (
    DERIVED_SIGNALS,
    REQUIRED_SIGNALS,
    SIGNALS,
    signal_history,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(case: Path, completion_path: Path, output: Path) -> dict[str, object]:
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "matched_initial_direct_transport_representative_smoke_complete":
        raise ValueError("representative smoke completion record is missing")
    if completion.get("sequence_id") != "source_up_u0p15_T700":
        raise ValueError("unexpected sequence id")
    if int(completion.get("mpi_ranks", -1)) != 32:
        raise ValueError("representative smoke did not use 32 MPI ranks")
    if any(bool(value) for value in completion.get("error_scan", {}).values()):
        raise ValueError("representative smoke completion contains a solver error")
    final_time = float(completion["final_common_complete_time_s"])
    if final_time < 0.01:
        raise ValueError("representative smoke did not reach 0.01 s")

    histories = {
        name: signal_history(case, relative_dir, filename)
        for name, (relative_dir, filename) in SIGNALS.items()
    }
    common_times = sorted(
        set.intersection(*(set(histories[name]) for name in REQUIRED_SIGNALS))
    )
    common_times = [time for time in common_times if 0.0 <= time <= final_time + 1.0e-12]
    if len(common_times) < 2:
        raise ValueError("fewer than two common OpenFOAM observable times")
    if common_times[-1] < 0.01 - 1.0e-12:
        raise ValueError("OpenFOAM observable histories do not reach 0.01 s")

    signal_names = list(SIGNALS) + DERIVED_SIGNALS
    values = np.full((1, len(common_times), len(signal_names)), np.nan, dtype=np.float64)
    long_rows: list[dict[str, object]] = []
    for row_index, time_s in enumerate(common_times):
        row = {name: histories[name].get(time_s, float("nan")) for name in SIGNALS}
        row["pressure_drop_Pa"] = row["inlet_pressure_Pa"] - row["outlet_pressure_Pa"]
        row["signed_mass_residual_kg_s"] = (
            row["inlet_mass_flow_kg_s"] + row["outlet_mass_flow_kg_s"]
        )
        row["net_outward_enthalpy_flow_W"] = (
            row["inlet_enthalpy_flow_W"] + row["outlet_enthalpy_flow_W"]
        )
        values[0, row_index] = [row[name] for name in signal_names]
        long_rows.append(
            {
                "condition_id": "source_up_u0p15_T700",
                "complete": 1,
                "time_s": time_s,
                **row,
            }
        )
    if not np.isfinite(values).all():
        raise ValueError("fully coupled observable export contains a non-finite value")

    output.mkdir(parents=True, exist_ok=True)
    time_array = np.asarray(common_times, dtype=np.float64)[None, :]
    npz_path = output / "hccb_p418_transient_observables.npz"
    np.savez_compressed(
        npz_path,
        case_id=np.asarray(["source_up_u0p15_T700"], dtype=object),
        complete=np.asarray([True]),
        conditions=np.empty((1, 0), dtype=np.float64),
        condition_names=np.asarray([], dtype=object),
        time_s=time_array,
        time_mask=np.ones(time_array.shape, dtype=bool),
        values=values,
        signal_names=np.asarray(signal_names, dtype=object),
    )
    csv_path = output / "hccb_p418_transient_observables_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    summary = {
        "status": "completed_p418_matched_initial_direct_transport_observable_export",
        "sequence_id": "source_up_u0p15_T700",
        "history_kind": "fully_coupled_flow_heat_response",
        "case": str(case),
        "completion_record": str(completion_path),
        "completion_record_sha256": sha256(completion_path),
        "time_point_count": len(common_times),
        "start_time_s": float(common_times[0]),
        "end_time_s": float(common_times[-1]),
        "signal_names": signal_names,
        "all_values_finite": True,
        "time_points_are_direct_openfoam_function_object_outputs": True,
        "new_physical_parameters": [],
        "openfoam_solver_started_by_this_export": False,
        "artifacts": {
            "npz": str(npz_path),
            "npz_sha256": sha256(npz_path),
            "long_csv": str(csv_path),
            "long_csv_sha256": sha256(csv_path),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = export(args.case.resolve(), args.completion.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
