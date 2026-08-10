#!/usr/bin/env python3
"""Compare one P418 CHT condition on coarse, medium and fine meshes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = {
    "pressure_drop_Pa": ("flow", "pressure_drop_Pa"),
    "outlet_temperature_K": ("temperature", "outlet_average_K"),
    "solid_maximum_temperature_K": ("temperature", "solid_maximum_K"),
    "cooling_wall_heat_flow_W": ("heat_balance", "cooling_wall_heat_flow_W"),
    "generated_power_W": ("heat_balance", "solid_generated_power_W"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(payload: dict, keys: tuple[str, str]) -> float:
    return float(payload[keys[0]][keys[1]])


def relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-300)


def equivalent_cell_size(mesh: dict) -> float:
    total_cells = int(mesh["fluid"]["cells"]) + int(mesh["solid"]["cells"])
    total_volume = float(mesh["fluid"]["volume_m3"]) + float(
        mesh["solid"]["volume_m3"]
    )
    if total_cells <= 0 or total_volume <= 0.0:
        raise ValueError("mesh cell count and volume must be positive")
    return (total_volume / total_cells) ** (1.0 / 3.0)


def generalized_gci_triplet(
    coarse: float,
    medium: float,
    fine: float,
    h_coarse: float,
    h_medium: float,
    h_fine: float,
    safety_factor: float = 1.25,
) -> dict[str, float | str | None]:
    """Three-grid GCI for unequal unstructured-mesh refinement ratios."""
    if not h_coarse > h_medium > h_fine > 0.0 or safety_factor <= 0.0:
        raise ValueError("equivalent mesh sizes must decrease from coarse to fine")
    r21 = h_medium / h_fine
    r32 = h_coarse / h_medium
    epsilon_32 = coarse - medium
    epsilon_21 = medium - fine
    scale = max(abs(coarse), abs(medium), abs(fine), 1.0)
    tiny = math.ulp(1.0) * scale
    result: dict[str, float | str | None] = {
        "coarse_value": float(coarse),
        "medium_value": float(medium),
        "fine_value": float(fine),
        "coarse_to_medium_refinement_ratio": float(r32),
        "medium_to_fine_refinement_ratio": float(r21),
        "observed_order": None,
        "richardson_extrapolated_value": None,
        "fine_gci_fraction": None,
        "fine_gci_absolute": None,
    }
    if min(coarse, medium, fine) <= 0.0 <= max(coarse, medium, fine):
        result["convergence_status"] = "zero_crossing_no_gci_reported"
        return result
    if abs(epsilon_32) <= tiny and abs(epsilon_21) <= tiny:
        result.update(
            {
                "convergence_status": "identical_within_float64_resolution",
                "richardson_extrapolated_value": float(fine),
                "fine_gci_fraction": 0.0,
                "fine_gci_absolute": 0.0,
            }
        )
        return result
    if abs(epsilon_21) <= tiny:
        result.update(
            {
                "convergence_status": "fine_pair_identical",
                "richardson_extrapolated_value": float(fine),
                "fine_gci_fraction": 0.0,
                "fine_gci_absolute": 0.0,
            }
        )
        return result
    if epsilon_32 * epsilon_21 <= 0.0:
        result["convergence_status"] = "oscillatory_no_gci_reported"
        return result

    difference_ratio = abs(epsilon_32 / epsilon_21)
    order = max(abs(math.log(difference_ratio) / math.log(r21)), 1.0e-8)
    converged = False
    for _ in range(1000):
        numerator = r21**order - 1.0
        denominator = r32**order - 1.0
        if numerator <= 0.0 or denominator <= 0.0:
            break
        correction = math.log(numerator / denominator)
        updated = abs(math.log(difference_ratio) + correction) / math.log(r21)
        updated = 0.5 * order + 0.5 * updated
        if abs(updated - order) <= 1.0e-10 * max(1.0, updated):
            order = updated
            converged = True
            break
        order = updated
    if not converged or not math.isfinite(order) or order <= 0.0:
        result.update(
            {
                "convergence_status": "apparent_order_not_resolved_no_gci_reported",
                "observed_order": float(order) if math.isfinite(order) else None,
            }
        )
        return result

    denominator = r21**order - 1.0
    if denominator <= 0.0:
        result["convergence_status"] = "apparent_order_not_resolved_no_gci_reported"
        return result
    extrapolated = fine + (fine - medium) / denominator
    absolute_gci = safety_factor * abs(fine - medium) / denominator
    result.update(
        {
            "convergence_status": "monotonic_gci_reported",
            "observed_order": float(order),
            "richardson_extrapolated_value": float(extrapolated),
            "fine_gci_fraction": (
                float(absolute_gci / abs(fine)) if abs(fine) > tiny else None
            ),
            "fine_gci_absolute": float(absolute_gci),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for level in ("coarse", "medium", "fine"):
        parser.add_argument(f"--{level}-mesh", type=Path, required=True)
        parser.add_argument(f"--{level}-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    meshes = {
        level: read_json(getattr(args, f"{level}_mesh"))
        for level in ("coarse", "medium", "fine")
    }
    results = {
        level: read_json(getattr(args, f"{level}_result"))
        for level in ("coarse", "medium", "fine")
    }
    conditions = [result["physical_conditions"] for result in results.values()]
    if any(condition != conditions[0] for condition in conditions[1:]):
        raise ValueError("the three mesh calculations do not use identical physical conditions")

    rows: list[dict[str, object]] = []
    for level in ("coarse", "medium", "fine"):
        mesh = meshes[level]
        result = results[level]
        inlet_temperature = float(result["physical_conditions"]["inlet_temperature_K"])
        generated_power = float(result["heat_balance"]["solid_generated_power_W"])
        row: dict[str, object] = {
            "mesh_level": level,
            "fluid_cells": int(mesh["fluid"]["cells"]),
            "solid_cells": int(mesh["solid"]["cells"]),
            "total_cells": int(mesh["fluid"]["cells"]) + int(mesh["solid"]["cells"]),
            "equivalent_cell_size_m": equivalent_cell_size(mesh),
            "cell_volume_porosity": float(mesh["cell_volume_porosity"]),
            "fluid_basic_check_passes": bool(mesh["checks"]["fluid_mesh_passes"]),
            "solid_basic_check_passes": bool(mesh["checks"]["solid_mesh_passes"]),
            "relative_mass_difference": float(result["flow"]["relative_mass_difference"]),
            "relative_energy_difference": float(
                result["heat_balance"]["relative_energy_difference"]
            ),
            "outlet_temperature_change_K": (
                float(result["temperature"]["outlet_average_K"]) - inlet_temperature
            ),
            "solid_maximum_temperature_change_K": (
                float(result["temperature"]["solid_maximum_K"]) - inlet_temperature
            ),
            "cooling_wall_heat_fraction": (
                float(result["heat_balance"]["cooling_wall_heat_flow_W"])
                / generated_power
            ),
        }
        for name, keys in METRICS.items():
            row[name] = nested(result, keys)
        rows.append(row)

    fine = rows[-1]
    compared_metrics = [
        "pressure_drop_Pa",
        "outlet_temperature_change_K",
        "solid_maximum_temperature_change_K",
        "cooling_wall_heat_fraction",
    ]
    for row in rows:
        for metric in compared_metrics:
            row[f"{metric}_relative_difference_from_fine"] = relative_difference(
                float(row[metric]), float(fine[metric])
            )

    trends: dict[str, dict[str, object]] = {}
    for metric in compared_metrics:
        values = [float(row[metric]) for row in rows]
        coarse_medium = abs(values[1] - values[0])
        medium_fine = abs(values[2] - values[1])
        trends[metric] = {
            "coarse_to_medium_absolute_change": coarse_medium,
            "medium_to_fine_absolute_change": medium_fine,
            "change_reduces_on_refinement": medium_fine < coarse_medium,
            "monotonic_values": (
                values[0] <= values[1] <= values[2]
                or values[0] >= values[1] >= values[2]
            ),
        }

    h_coarse, h_medium, h_fine = [
        float(row["equivalent_cell_size_m"]) for row in rows
    ]
    grid_convergence = []
    for metric in compared_metrics:
        grid_convergence.append(
            {
                "metric": metric,
                **generalized_gci_triplet(
                    float(rows[0][metric]),
                    float(rows[1][metric]),
                    float(rows[2][metric]),
                    h_coarse,
                    h_medium,
                    h_fine,
                ),
            }
        )

    payload = {
        "status": "completed_three_mesh_p418_cht_comparison",
        "physical_conditions": conditions[0],
        "mesh_levels": rows,
        "refinement_trends": trends,
        "grid_convergence": grid_convergence,
        "grid_convergence_method": (
            "Celik et al. three-grid GCI with actual unequal equivalent-cell-size "
            "refinement ratios and safety factor 1.25"
        ),
        "grid_convergence_safety_factor": 1.25,
        "interpretation": (
            "This is a three-mesh sensitivity comparison for engineering observables. "
            "It does not by itself prove complete mesh independence."
        ),
        "new_physical_parameters": [],
    }
    if not all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in (
            "pressure_drop_Pa",
            "outlet_temperature_K",
            "solid_maximum_temperature_K",
            "cooling_wall_heat_flow_W",
        )
    ):
        raise ValueError("mesh comparison contains a non-finite engineering observable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "engineering_observables.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "mesh_gci.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_convergence[0]))
        writer.writeheader()
        writer.writerows(grid_convergence)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
