#!/usr/bin/env python3
"""Export time-resolved P418 solver-relaxation observables from 3D CHT cases.

The exporter reads the function-object histories written by the formal
OpenFOAM runs.  It does not synthesize time points or physical parameters.
Completed and still-running cases may both be exported; a time mask keeps
their different sequence lengths explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SIGNALS = {
    "inlet_temperature_K": ("fluid/inletTemperature", "surfaceFieldValue.dat"),
    "outlet_temperature_K": ("fluid/outletTemperature", "surfaceFieldValue.dat"),
    "inlet_pressure_Pa": ("fluid/inletPressure", "surfaceFieldValue.dat"),
    "outlet_pressure_Pa": ("fluid/outletPressure", "surfaceFieldValue.dat"),
    "inlet_mass_flow_kg_s": ("fluid/inletMassFlow", "surfaceFieldValue.dat"),
    "outlet_mass_flow_kg_s": ("fluid/outletMassFlow", "surfaceFieldValue.dat"),
    "inlet_enthalpy_flow_W": ("fluid/inletEnthalpyFlow", "surfaceFieldValue.dat"),
    "outlet_enthalpy_flow_W": ("fluid/outletEnthalpyFlow", "surfaceFieldValue.dat"),
    "cooling_wall_power_W": ("fluid/coolingWallPower", "surfaceFieldValue.dat"),
    "maximum_solid_temperature_K": ("solid/solidTemperatureMaximum", "volFieldValue.dat"),
    "volume_average_fluid_temperature_K": (
        "fluid/fluidTemperatureVolumeAverage",
        "volFieldValue.dat",
    ),
    "volume_average_solid_temperature_K": (
        "solid/solidTemperatureVolumeAverage",
        "volFieldValue.dat",
    ),
}

# Pressure histories were added after one early pilot run.  Thermal and flow
# histories are complete for that case, so pressure is retained as an optional
# channel instead of discarding an otherwise valid thermal sequence.
OPTIONAL_SIGNALS = {
    "inlet_pressure_Pa",
    "outlet_pressure_Pa",
    "volume_average_fluid_temperature_K",
    "volume_average_solid_temperature_K",
}
REQUIRED_SIGNALS = [name for name in SIGNALS if name not in OPTIONAL_SIGNALS]

DERIVED_SIGNALS = [
    "pressure_drop_Pa",
    "signed_mass_residual_kg_s",
    "net_outward_enthalpy_flow_W",
]


def numeric_rows(path: Path) -> dict[float, float]:
    """Return the last scalar value at each reported time."""
    rows: dict[float, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        try:
            values = [float(item) for item in parts]
        except ValueError:
            continue
        if len(values) >= 2:
            rows[values[0]] = values[-1]
    return rows


def signal_history(case: Path, relative_dir: str, filename: str) -> dict[float, float]:
    root = case / "postProcessing" / relative_dir
    values: dict[float, float] = {}
    if not root.is_dir():
        return values
    starts = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: float(path.name),
    )
    for start in starts:
        data_path = start / filename
        if data_path.is_file():
            values.update(numeric_rows(data_path))
    return values


def case_metadata(case: Path) -> dict[str, object]:
    path = case / "cht_smoke_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing case metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def export_matrix(
    matrix_root: Path,
    output_dir: Path,
    minimum_points: int = 2,
    history_kind: str = "solver_relaxation",
) -> dict[str, object]:
    manifest_path = matrix_root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    condition_rows = manifest["published_conditions"]
    signal_names = list(SIGNALS) + DERIVED_SIGNALS
    step_history_kinds = {
        "physical_step_response",
        "fully_coupled_flow_heat_response",
    }
    if history_kind in step_history_kinds:
        condition_names = [
            "source_inlet_velocity_m_s",
            "source_inlet_temperature_K",
            "source_solid_heat_source_MW_m3",
            "target_inlet_velocity_m_s",
            "target_inlet_temperature_K",
            "target_solid_heat_source_MW_m3",
            "outlet_pressure_Pa",
            "cooling_wall_temperature_K",
        ]
    else:
        condition_names = [
            "inlet_velocity_m_s",
            "inlet_temperature_K",
            "solid_heat_source_MW_m3",
            "outlet_pressure_Pa",
            "cooling_wall_temperature_K",
        ]
    cases: list[dict[str, object]] = []

    for condition in condition_rows:
        condition_id = str(condition["condition_id"])
        case = matrix_root / condition_id
        if not case.is_dir():
            continue
        histories = {
            name: signal_history(case, relative_dir, filename)
            for name, (relative_dir, filename) in SIGNALS.items()
        }
        common_times = sorted(set.intersection(*(set(histories[name]) for name in REQUIRED_SIGNALS)))
        if len(common_times) < minimum_points:
            continue
        metadata = case_metadata(case)
        if history_kind in step_history_kinds:
            step_metadata = (
                "fully_coupled_step_metadata.json"
                if history_kind == "fully_coupled_flow_heat_response"
                else "step_case_metadata.json"
            )
            step = json.loads((case / step_metadata).read_text(encoding="utf-8"))
            source = step["source_parameters"]
            target = step["target_parameters"]
            condition_values = {
                "source_inlet_velocity_m_s": float(source["inlet_velocity_m_s"]),
                "source_inlet_temperature_K": float(source["inlet_temperature_K"]),
                "source_solid_heat_source_MW_m3": float(source["solid_heat_source_MW_m3"]),
                "target_inlet_velocity_m_s": float(target["inlet_velocity_m_s"]),
                "target_inlet_temperature_K": float(target["inlet_temperature_K"]),
                "target_solid_heat_source_MW_m3": float(target["solid_heat_source_MW_m3"]),
                "outlet_pressure_Pa": float(metadata["outlet_pressure_Pa"]),
                "cooling_wall_temperature_K": float(metadata["cooling_wall_temperature_K"]),
            }
        else:
            condition_values = {
                "inlet_velocity_m_s": float(metadata["inlet_velocity_m_s"]),
                "inlet_temperature_K": float(metadata["inlet_temperature_K"]),
                "solid_heat_source_MW_m3": float(metadata["solid_heat_source_W_m3"]) / 1.0e6,
                "outlet_pressure_Pa": float(metadata["outlet_pressure_Pa"]),
                "cooling_wall_temperature_K": float(metadata["cooling_wall_temperature_K"]),
            }
        rows = []
        for time_s in common_times:
            row = {name: histories[name].get(time_s, float("nan")) for name in SIGNALS}
            row["pressure_drop_Pa"] = row["inlet_pressure_Pa"] - row["outlet_pressure_Pa"]
            row["signed_mass_residual_kg_s"] = (
                row["inlet_mass_flow_kg_s"] + row["outlet_mass_flow_kg_s"]
            )
            row["net_outward_enthalpy_flow_W"] = (
                row["inlet_enthalpy_flow_W"] + row["outlet_enthalpy_flow_W"]
            )
            rows.append({"time_s": time_s, **row})
        cases.append(
            {
                "condition_id": condition_id,
                "complete": (
                    (
                        case
                        / (
                            "fully_coupled_step_response_complete.json"
                            if history_kind == "fully_coupled_flow_heat_response"
                            else "step_response_complete.json"
                        )
                    ).is_file()
                    if history_kind in step_history_kinds
                    else (case / "formal_sample_complete.json").is_file()
                ),
                "condition_values": condition_values,
                "rows": rows,
            }
        )

    if not cases:
        raise ValueError(f"no cases with at least {minimum_points} common time points in {matrix_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = max(len(case["rows"]) for case in cases)
    values = np.full((len(cases), max_steps, len(signal_names)), np.nan, dtype="float64")
    time_s = np.full((len(cases), max_steps), np.nan, dtype="float64")
    time_mask = np.zeros((len(cases), max_steps), dtype=bool)
    conditions = np.zeros((len(cases), len(condition_names)), dtype="float64")
    case_ids = np.empty(len(cases), dtype=object)
    complete = np.zeros(len(cases), dtype=bool)
    long_rows: list[dict[str, object]] = []
    for i, case in enumerate(cases):
        case_ids[i] = case["condition_id"]
        complete[i] = bool(case["complete"])
        conditions[i] = [case["condition_values"][name] for name in condition_names]
        for j, row in enumerate(case["rows"]):
            time_s[i, j] = row["time_s"]
            time_mask[i, j] = True
            values[i, j] = [row[name] for name in signal_names]
            long_rows.append(
                {
                    "condition_id": case["condition_id"],
                    "complete": int(case["complete"]),
                    **case["condition_values"],
                    **row,
                }
            )

    npz_path = output_dir / "hccb_p418_transient_observables.npz"
    np.savez_compressed(
        npz_path,
        case_id=case_ids,
        complete=complete,
        conditions=conditions,
        condition_names=np.asarray(condition_names, dtype=object),
        time_s=time_s,
        time_mask=time_mask,
        values=values,
        signal_names=np.asarray(signal_names, dtype=object),
    )
    csv_path = output_dir / "hccb_p418_transient_observables_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)

    if history_kind == "physical_step_response":
        status = "completed_p418_3d_openfoam_physical_step_observable_export"
        scope = (
            "Computed three-dimensional coupled fluid-solid thermal responses after one inlet temperature, inlet "
            "velocity or solid heat-source input is stepped between exact published P418 endpoints on one fixed "
            "pebble packing. The converged target hydrodynamic field is held fixed, so these histories resolve the "
            "thermal response after flow adjustment rather than the first sub-second momentum transient."
        )
    elif history_kind == "fully_coupled_flow_heat_response":
        status = "completed_p418_3d_openfoam_fully_coupled_flow_heat_observable_export"
        scope = (
            "Computed three-dimensional coupled flow and heat-transfer responses after one inlet temperature, "
            "inlet velocity or solid heat-source input is stepped between exact published P418 endpoints on one "
            "fixed pebble packing. Velocity, pressure, face mass flux and fluid-solid temperature evolve together."
        )
    else:
        status = "completed_p418_3d_openfoam_relaxation_observable_export"
        scope = (
            "Pseudo-transient solver-relaxation histories under fixed published P418 operating conditions. "
            "They support steady-solver acceleration studies, but are not physical start-up, step-change or accident sequences."
        )
    summary = {
        "status": status,
        "history_kind": history_kind,
        "source_matrix": str(matrix_root),
        "source_title": manifest["source_title"],
        "source_doi": manifest["source_doi"],
        "case_count_with_time_histories": len(cases),
        "completed_case_count": int(complete.sum()),
        "maximum_time_points": max_steps,
        "time_points_are_direct_openfoam_function_object_outputs": True,
        "physical_conditions_are_exact_published_p418_matrix_points": True,
        "new_physical_parameters": [],
        "signal_names": signal_names,
        "condition_names": condition_names,
        "required_time_resolved_signals": REQUIRED_SIGNALS,
        "optional_time_resolved_signals": sorted(OPTIONAL_SIGNALS | {"pressure_drop_Pa"}),
        "derived_signal_definitions": {
            "pressure_drop_Pa": "inlet_pressure_Pa - outlet_pressure_Pa",
            "signed_mass_residual_kg_s": "inlet_mass_flow_kg_s + outlet_mass_flow_kg_s",
            "net_outward_enthalpy_flow_W": "inlet_enthalpy_flow_W + outlet_enthalpy_flow_W",
        },
        "artifacts": {"npz": str(npz_path), "long_csv": str(csv_path)},
        "scientific_scope": scope,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-points", type=int, default=2)
    parser.add_argument(
        "--history-kind",
        choices=(
            "solver_relaxation",
            "physical_step_response",
            "fully_coupled_flow_heat_response",
        ),
        default="solver_relaxation",
    )
    args = parser.parse_args()
    summary = export_matrix(
        args.matrix_root.resolve(),
        args.output_dir.resolve(),
        args.minimum_points,
        args.history_kind,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
