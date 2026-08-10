#!/usr/bin/env python3
"""Check that P418 uses absolute pressure consistently in helium density."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from hccb_source_backed_thermophysical import helium_density


NUMERICAL_DENSITY_TOLERANCE = 1.0e-4


def check_condition(
    *, condition: dict[str, object], dataset_root: Path
) -> dict[str, object]:
    field_path = dataset_root / str(condition["field_file"])
    with np.load(field_path, allow_pickle=False) as loaded:
        pressure = loaded["fluid_pressure_Pa"].astype(np.float64)
        temperature = loaded["fluid_temperature_K"].astype(np.float64)
        density = loaded["fluid_density_kg_m3"].astype(np.float64)
    if not all(np.all(np.isfinite(values)) for values in (pressure, temperature, density)):
        raise ValueError(f"{condition['condition_id']} contains non-finite p, T or rho")
    outlet_pressure = float(condition["outlet_pressure_Pa"])
    if outlet_pressure <= 0.0:
        raise ValueError("declared outlet absolute pressure must be positive")
    if float(np.median(pressure)) <= 0.5 * outlet_pressure:
        raise ValueError(
            f"{condition['condition_id']} pressure resembles gauge rather than absolute pressure"
        )
    calculated = helium_density(
        torch.as_tensor(pressure), torch.as_tensor(temperature)
    ).detach().cpu().numpy()
    relative_error = np.abs(calculated - density) / np.maximum(
        np.abs(density), np.finfo(np.float64).tiny
    )
    maximum_relative_error = float(relative_error.max())
    return {
        "condition_id": str(condition["condition_id"]),
        "pressure_field": "OpenFOAM fluid/p",
        "pressure_interpretation": "absolute thermodynamic pressure in Pa",
        "outlet_pressure_Pa": outlet_pressure,
        "minimum_pressure_Pa": float(pressure.min()),
        "mean_pressure_Pa": float(pressure.mean()),
        "maximum_pressure_Pa": float(pressure.max()),
        "minimum_temperature_K": float(temperature.min()),
        "maximum_temperature_K": float(temperature.max()),
        "minimum_openfoam_density_kg_m3": float(density.min()),
        "maximum_openfoam_density_kg_m3": float(density.max()),
        "mean_density_relative_difference": float(relative_error.mean()),
        "maximum_density_relative_difference": maximum_relative_error,
        "numerically_consistent": maximum_relative_error
        <= NUMERICAL_DENSITY_TOLERANCE,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    index_path = args.dataset_index.resolve()
    dataset_root = index_path.parent
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = [
        check_condition(condition=condition, dataset_root=dataset_root)
        for condition in index["conditions"]
    ]
    if not rows:
        raise ValueError("dataset contains no P418 conditions")
    failed = [row["condition_id"] for row in rows if not row["numerically_consistent"]]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "pressure_density_consistency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": (
            "p418_pressure_density_consistency_ready" if not failed else "failed"
        ),
        "case_count": len(rows),
        "pressure_field": "OpenFOAM fluid/p",
        "pressure_interpretation": "absolute thermodynamic pressure in Pa",
        "density_relation": (
            "the same source-backed helium p-T relation used by the transient energy equation"
        ),
        "numerical_density_relative_tolerance": NUMERICAL_DENSITY_TOLERANCE,
        "overall_maximum_density_relative_difference": max(
            float(row["maximum_density_relative_difference"]) for row in rows
        ),
        "failed_conditions": failed,
        "table": csv_path.name,
        "new_physical_parameters": [],
        "interpretation": (
            "This checks field meaning and numerical consistency. The tolerance is not "
            "a pebble-bed material or operating parameter."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
