#!/usr/bin/env python3
"""Extract exact steady solid hot cells and their movement across P418 cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_sample(case: Path, marker: dict[str, object]) -> Path:
    declared = Path(str(marker["training_sample"]))
    candidates = [declared, case / declared.name]
    if declared.parent.name:
        candidates.append(case / declared.parent.name / declared.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"training sample is missing for {case.name}: {declared}")


def hotspot_records(matrix_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for marker_path in sorted(matrix_root.glob("*/formal_sample_complete.json")):
        case = marker_path.parent
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        time_name = str(marker["time"])
        summary_path = case / f"cht_result_summary_{time_name}.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sample_path = resolve_sample(case, marker)
        with np.load(sample_path, allow_pickle=False) as loaded:
            temperature = loaded["solid_temperature_K"].astype(np.float64)
            centroid = loaded["solid_cell_centroid_m"].astype(np.float64)
        if temperature.ndim != 1 or centroid.shape != (len(temperature), 3):
            raise ValueError(f"invalid solid fields in {sample_path}")
        if np.any(~np.isfinite(temperature)) or np.any(~np.isfinite(centroid)):
            raise ValueError(f"non-finite solid fields in {sample_path}")
        hot_index = int(np.argmax(temperature))
        hot_temperature = float(temperature[hot_index])
        reported = float(summary["temperature"]["solid_maximum_K"])
        if not np.isclose(hot_temperature, reported, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"solid maximum differs between field and summary in {case.name}: "
                f"{hot_temperature} != {reported}"
            )
        condition = summary["physical_conditions"]
        records.append(
            {
                "condition_id": case.name,
                "completion_time_s": time_name,
                "inlet_velocity_m_s": float(condition["inlet_velocity_m_s"]),
                "inlet_temperature_K": float(condition["inlet_temperature_K"]),
                "solid_heat_source_MW_m3": float(
                    condition["solid_heat_source_W_m3"]
                )
                / 1.0e6,
                "solid_hot_cell_index": hot_index,
                "solid_maximum_temperature_K": hot_temperature,
                "hotspot_x_m": float(centroid[hot_index, 0]),
                "hotspot_y_m": float(centroid[hot_index, 1]),
                "hotspot_z_m": float(centroid[hot_index, 2]),
            }
        )
    if not records:
        raise ValueError(f"no completed steady hotspot fields in {matrix_root}")
    return records


def adjacent_movements(records: list[dict[str, object]]) -> list[dict[str, object]]:
    factors = (
        (
            "inlet_temperature_K",
            ("inlet_velocity_m_s", "solid_heat_source_MW_m3"),
        ),
        (
            "solid_heat_source_MW_m3",
            ("inlet_velocity_m_s", "inlet_temperature_K"),
        ),
        (
            "inlet_velocity_m_s",
            ("inlet_temperature_K", "solid_heat_source_MW_m3"),
        ),
    )
    output: list[dict[str, object]] = []
    for varied, fixed in factors:
        groups: dict[tuple[float, float], list[dict[str, object]]] = {}
        for record in records:
            key = (float(record[fixed[0]]), float(record[fixed[1]]))
            groups.setdefault(key, []).append(record)
        for fixed_values, members in sorted(groups.items()):
            ordered = sorted(members, key=lambda item: float(item[varied]))
            for low, high in zip(ordered[:-1], ordered[1:]):
                low_xyz = np.asarray(
                    [low["hotspot_x_m"], low["hotspot_y_m"], low["hotspot_z_m"]],
                    dtype=np.float64,
                )
                high_xyz = np.asarray(
                    [high["hotspot_x_m"], high["hotspot_y_m"], high["hotspot_z_m"]],
                    dtype=np.float64,
                )
                output.append(
                    {
                        "varied_factor": varied,
                        "fixed_factor_1": fixed[0],
                        "fixed_factor_1_value": fixed_values[0],
                        "fixed_factor_2": fixed[1],
                        "fixed_factor_2_value": fixed_values[1],
                        "low_condition_id": str(low["condition_id"]),
                        "high_condition_id": str(high["condition_id"]),
                        "low_factor_value": float(low[varied]),
                        "high_factor_value": float(high[varied]),
                        "hotspot_distance_m": float(np.linalg.norm(high_xyz - low_xyz)),
                        "solid_maximum_temperature_change_K": float(
                            high["solid_maximum_temperature_K"]
                        )
                        - float(low["solid_maximum_temperature_K"]),
                        "same_hot_cell": bool(
                            int(high["solid_hot_cell_index"])
                            == int(low["solid_hot_cell_index"])
                        ),
                    }
                )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int)
    args = parser.parse_args()
    records = hotspot_records(args.matrix_root.resolve())
    if args.expected_case_count is not None and len(records) != args.expected_case_count:
        raise ValueError(
            f"expected {args.expected_case_count} completed hotspot fields, found {len(records)}"
        )
    movements = adjacent_movements(records)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "steady_hotspots.csv", records)
    write_csv(output / "steady_hotspot_movements.csv", movements)
    factor_summary = []
    for factor in (
        "inlet_temperature_K",
        "solid_heat_source_MW_m3",
        "inlet_velocity_m_s",
    ):
        members = [item for item in movements if item["varied_factor"] == factor]
        if not members:
            continue
        largest = max(members, key=lambda item: float(item["hotspot_distance_m"]))
        factor_summary.append(
            {
                "varied_factor": factor,
                "adjacent_pair_count": len(members),
                "same_hot_cell_pair_count": sum(bool(item["same_hot_cell"]) for item in members),
                "maximum_adjacent_hotspot_distance_m": float(
                    largest["hotspot_distance_m"]
                ),
                "maximum_distance_condition_pair": [
                    str(largest["low_condition_id"]),
                    str(largest["high_condition_id"]),
                ],
            }
        )
    summary = {
        "status": "p418_steady_hotspots_ready",
        "completed_case_count": len(records),
        "adjacent_movement_count": len(movements),
        "factor_summary": factor_summary,
        "definition": (
            "The hotspot is the centroid of the native solid cell with the largest "
            "OpenFOAM temperature. Exact cell identity can switch between nearly equal "
            "neighbouring maxima, so distance is reported together with the maximum "
            "temperature change and is not treated as an independent material property."
        ),
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
