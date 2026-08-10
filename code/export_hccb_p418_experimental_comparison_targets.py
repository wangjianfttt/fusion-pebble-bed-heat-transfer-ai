#!/usr/bin/env python3
"""Export P418 OpenFOAM results in the form used by pebble-bed experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OBSERVABLES = (
    ("入口质量流量或表观速度", "inlet_velocity_m_s", "m/s", "direct"),
    ("入口与出口气体温度", "inlet_temperature_K", "K", "direct_inlet"),
    ("入口与出口气体温度", "outlet_temperature_K", "K", "direct_outlet"),
    ("轴向压降", "pressure_drop_Pa", "Pa", "direct"),
    ("冷却壁或冷却板温度", "cooling_wall_temperature_K", "K", "direct"),
    ("颗粒最高温度与热点位置", "solid_maximum_temperature_K", "K", "sparse_measurement_plus_model"),
    ("输入加热功率", "generated_power_W", "W", "direct"),
    ("入口与出口气体温度", "net_outward_enthalpy_flow_W", "W", "mass_flow_and_temperature"),
    ("冷却侧带走的热功率", "cooling_wall_heat_into_fluid_W", "W", "coolant_balance"),
    ("气固界面净换热", "interphase_heat_from_solid_balance_W", "W", "steady_solid_energy_balance"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def keyed(rows: list[dict[str, str]], source: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        condition = row["condition_id"].strip()
        if not condition or condition in result:
            raise ValueError(f"invalid or duplicate condition_id in {source}: {condition}")
        result[condition] = row
    return result


def build_targets(
    physics_rows: list[dict[str, str]],
    dimensionless_rows: list[dict[str, str]],
    observable_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    physics = keyed(physics_rows, Path("completed physics"))
    dimensionless = keyed(dimensionless_rows, Path("dimensionless heat transfer"))
    if set(physics) != set(dimensionless):
        raise ValueError(
            "completed-physics and dimensionless-heat case sets differ: "
            f"physics_only={sorted(set(physics) - set(dimensionless))}, "
            f"heat_only={sorted(set(dimensionless) - set(physics))}"
        )
    observable_map = {row["观测量"].strip(): row for row in observable_rows}
    missing = sorted({item[0] for item in OBSERVABLES} - set(observable_map))
    if missing:
        raise ValueError(f"observable matrix is missing: {missing}")

    long_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    for condition in sorted(physics):
        row = dict(physics[condition])
        heat = dimensionless[condition]
        generated = float(row["generated_power_W"])
        solid_wall = float(heat["openfoam_solid_wall_heat_into_solid_W"])
        inferred_interphase = generated + solid_wall
        resolved_interphase = float(heat["openfoam_interphase_heat_into_fluid_W"])
        row["interphase_heat_from_solid_balance_W"] = inferred_interphase
        difference = inferred_interphase - resolved_interphase
        balance_rows.append(
            {
                "condition_id": condition,
                "generated_power_W": generated,
                "solid_wall_heat_into_solid_W": solid_wall,
                "interphase_heat_from_solid_balance_W": inferred_interphase,
                "openfoam_resolved_interphase_heat_into_fluid_W": resolved_interphase,
                "balance_minus_resolved_W": difference,
                "absolute_difference_over_generated": abs(difference) / abs(generated),
                "heat_direction_agrees_with_phase_temperature_difference": heat[
                    "openfoam_interface_flux_and_phase_temperature_sign_agree"
                ],
            }
        )
        for observable_name, field, unit, value_role in OBSERVABLES:
            definition = observable_map[observable_name]
            long_rows.append(
                {
                    "condition_id": condition,
                    "observable": observable_name,
                    "model_field": field,
                    "model_value": float(row[field]),
                    "unit": unit,
                    "value_role": value_role,
                    "experimental_method": definition["测量方法"],
                    "experimental_obtaining_method": definition["获得方式"],
                    "physical_comparison": definition["主要检验内容"],
                    "literature_parameter_ids": definition["文献参数编号"],
                    "limitation": definition["限制"],
                }
            )
    return long_rows, balance_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed-physics-csv", type=Path, required=True)
    parser.add_argument("--dimensionless-heat-csv", type=Path, required=True)
    parser.add_argument("--observable-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    long_rows, balance_rows = build_targets(
        read_csv(args.completed_physics_csv),
        read_csv(args.dimensionless_heat_csv),
        read_csv(args.observable_matrix),
    )
    output = args.output_dir.resolve()
    write_csv(output / "experimental_comparison_targets.csv", long_rows)
    write_csv(output / "interphase_heat_balance_check.csv", balance_rows)
    maximum_difference = max(
        float(row["absolute_difference_over_generated"]) for row in balance_rows
    )
    summary = {
        "status": "p418_experimental_comparison_targets_ready",
        "counts": {
            "cases": len(balance_rows),
            "experimental_comparison_rows": len(long_rows),
            "observables_per_case": len(OBSERVABLES),
        },
        "maximum_interphase_balance_difference_over_generated": maximum_difference,
        "interphase_heat_definition": (
            "At steady state, interphase heat into the fluid is inferred from solid "
            "generation plus wall heat into the solid. It is not a direct sensor value."
        ),
        "observable_matrix": str(args.observable_matrix.resolve()),
        "new_physical_parameters": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
