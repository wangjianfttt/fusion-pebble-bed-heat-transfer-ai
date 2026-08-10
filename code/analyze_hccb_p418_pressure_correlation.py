#!/usr/bin/env python3
"""Compare resolved crop pressure drops with the published P420/P421 model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_PARAMETERS = ("P048", "P420", "P421", "P422", "P426")


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["parameter_id"]: row for row in csv.DictReader(stream)}
    missing = [key for key in REQUIRED_PARAMETERS if key not in rows]
    if missing:
        raise ValueError(f"missing literature parameters: {missing}")
    if any(rows[key]["status"] != "extracted" for key in REQUIRED_PARAMETERS):
        raise ValueError("pressure-correlation inputs must be literature extracted")
    if not all(
        token in rows["P420"]["value"]
        for token in ("-9.181", "-12.238", "-5.320", "-8.062")
    ):
        raise ValueError("P420 coefficients differ from the registered equation")
    if "T=300 K" not in rows["P420"].get("notes", ""):
        raise ValueError("P420 reference temperature is not recorded in the source notes")
    if not all(
        token in rows["P421"]["value"]
        for token in ("180*xi_f", "mu_IN/d_p^2*u_IN")
    ):
        raise ValueError("P421 differs from the registered pressure equation")
    return rows


def helium_viscosity(temperature_k: float | np.ndarray) -> np.ndarray:
    return 0.4646 * np.asarray(temperature_k, dtype=np.float64) ** 0.66 * 1.0e-6


def area_mean(values: np.ndarray, area: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sum(values[mask] * area[mask]) / np.sum(area[mask]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--parameter-manifest", required=True, type=Path)
    parser.add_argument("--physical-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-case-count", type=int)
    args = parser.parse_args()

    matrix_root = args.matrix_root.resolve()
    parameters = read_manifest(args.parameter_manifest.resolve())
    particle_diameter_m = float(parameters["P048"]["value"]) * 1.0e-3
    source_maximum_difference_percent = float(parameters["P422"]["value"])
    reference_pressure_mpa = float(parameters["P426"]["value"])
    # P420 states this reference condition explicitly; it is not selected from
    # the computed fields or fitted to the pressure result.
    reference_temperature_k = 300.0
    reference_density = 480.19 * reference_pressure_mpa / reference_temperature_k
    reference_viscosity = float(helium_viscosity(reference_temperature_k))

    with args.physical_csv.resolve().open(newline="", encoding="utf-8") as stream:
        physical = {row["condition_id"]: row for row in csv.DictReader(stream)}
    marker_paths = sorted(matrix_root.glob("*/formal_sample_complete.json"))
    if args.expected_case_count is not None and len(marker_paths) != args.expected_case_count:
        raise ValueError(
            f"completed case count {len(marker_paths)} != {args.expected_case_count}"
        )
    if not marker_paths:
        raise ValueError("no completed P418 cases were found")

    rows: list[dict[str, object]] = []
    for marker_path in marker_paths:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        condition_id = str(marker["condition_id"])
        if condition_id not in physical:
            raise ValueError(f"physical summary lacks {condition_id}")
        sample_path = Path(str(marker["training_sample"]))
        if not sample_path.is_absolute():
            sample_path = marker_path.parent / sample_path
        sample_metadata = json.loads(
            (sample_path.parent / "metadata.json").read_text(encoding="utf-8")
        )
        fluid_names = list(sample_metadata["fluid_patch_names"])
        solid_names = list(sample_metadata["solid_patch_names"])
        inlet_fluid = fluid_names.index("inlet")
        outlet_fluid = fluid_names.index("outlet")
        inlet_solid = solid_names.index("inlet")

        with np.load(sample_path, allow_pickle=False) as sample:
            fluid_patch = sample["fluid_boundary_face_patch"].astype(np.int64)
            solid_patch = sample["solid_boundary_face_patch"].astype(np.int64)
            fluid_area = sample["fluid_boundary_face_area_m2"].astype(np.float64)
            solid_area = sample["solid_boundary_face_area_m2"].astype(np.float64)
            inlet = fluid_patch == inlet_fluid
            outlet = fluid_patch == outlet_fluid
            solid_inlet = solid_patch == inlet_solid
            if not inlet.any() or not outlet.any() or not solid_inlet.any():
                raise ValueError(f"{condition_id}: incomplete inlet/outlet boundary")
            temperature = sample["fluid_boundary_temperature_K"].astype(np.float64)
            pressure = sample["fluid_boundary_pressure_Pa"].astype(np.float64)
            density = sample["fluid_boundary_density_kg_m3"].astype(np.float64)
            mass_flow = sample["fluid_boundary_face_mass_flow_kg_s"].astype(np.float64)
            centroid = sample["fluid_boundary_face_centroid_m"].astype(np.float64)
            fluid_volume = float(np.sum(sample["fluid_cell_volume_m3"]))
            solid_volume = float(np.sum(sample["solid_cell_volume_m3"]))

            inlet_temperature = area_mean(temperature, fluid_area, inlet)
            outlet_temperature = area_mean(temperature, fluid_area, outlet)
            inlet_density = area_mean(density, fluid_area, inlet)
            outlet_density = area_mean(density, fluid_area, outlet)
            inlet_pressure = area_mean(pressure, fluid_area, inlet)
            outlet_pressure = area_mean(pressure, fluid_area, outlet)
            inlet_fluid_area = float(np.sum(fluid_area[inlet]))
            total_inlet_area = inlet_fluid_area + float(np.sum(solid_area[solid_inlet]))
            volume_flow = abs(float(np.sum(mass_flow[inlet] / density[inlet])))
            pore_opening_velocity = volume_flow / inlet_fluid_area
            superficial_velocity = volume_flow / total_inlet_area
            packed_length = abs(
                area_mean(centroid[:, 2], fluid_area, outlet)
                - area_mean(centroid[:, 2], fluid_area, inlet)
            )
            porosity = fluid_volume / (fluid_volume + solid_volume)

        inlet_viscosity = float(helium_viscosity(inlet_temperature))
        outlet_viscosity = float(helium_viscosity(outlet_temperature))
        xi_f = (
            (outlet_density / inlet_density) ** -9.181
            * (outlet_viscosity / inlet_viscosity) ** -12.238
            * (inlet_density / reference_density) ** -5.320
            * (inlet_viscosity / reference_viscosity) ** -8.062
        )
        predicted_pressure_drop = (
            180.0
            * xi_f
            * (1.0 - porosity) ** 2
            / porosity**3
            * inlet_viscosity
            / particle_diameter_m**2
            * superficial_velocity
            * packed_length
        )
        resolved_pressure_drop = float(physical[condition_id]["pressure_drop_Pa"])
        boundary_pressure_drop = inlet_pressure - outlet_pressure
        signed_difference_percent = (
            100.0 * (predicted_pressure_drop - resolved_pressure_drop) / resolved_pressure_drop
        )
        physical_conditions = sample_metadata["physical_conditions"]
        source_channel_velocity = float(physical_conditions["inlet_velocity_m_s"])
        prescribed_pore_velocity = float(
            physical_conditions.get(
                "pore_opening_boundary_velocity_m_s", source_channel_velocity
            )
        )
        rows.append(
            {
                "condition_id": condition_id,
                "inlet_temperature_K": inlet_temperature,
                "outlet_temperature_K": outlet_temperature,
                "source_inlet_channel_velocity_m_s": source_channel_velocity,
                "prescribed_pore_opening_velocity_m_s": prescribed_pore_velocity,
                "reconstructed_pore_opening_velocity_m_s": pore_opening_velocity,
                "superficial_velocity_m_s": superficial_velocity,
                "inlet_open_area_fraction": inlet_fluid_area / total_inlet_area,
                "volume_porosity": porosity,
                "packed_length_m": packed_length,
                "xi_f": xi_f,
                "resolved_pressure_drop_Pa": resolved_pressure_drop,
                "boundary_pressure_drop_Pa": boundary_pressure_drop,
                "P420_P421_pressure_drop_Pa": predicted_pressure_drop,
                "signed_difference_percent": signed_difference_percent,
                "absolute_difference_percent": abs(signed_difference_percent),
                "inside_source_P422_4p6_percent": abs(signed_difference_percent)
                <= source_maximum_difference_percent,
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "pressure_correlation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    absolute = np.asarray([float(row["absolute_difference_percent"]) for row in rows])
    boundary_difference = np.asarray(
        [
            abs(
                float(row["boundary_pressure_drop_Pa"])
                - float(row["resolved_pressure_drop_Pa"])
            )
            / float(row["resolved_pressure_drop_Pa"])
            for row in rows
        ]
    )
    velocity_difference = np.asarray(
        [
            abs(
                float(row["reconstructed_pore_opening_velocity_m_s"])
                - float(row["prescribed_pore_opening_velocity_m_s"])
            )
            / float(row["prescribed_pore_opening_velocity_m_s"])
            for row in rows
        ]
    )
    summary = {
        "status": "p418_local_crop_pressure_correlation_complete",
        "case_count": len(rows),
        "parameter_ids": list(REQUIRED_PARAMETERS),
        "median_absolute_difference_percent": float(np.median(absolute)),
        "maximum_absolute_difference_percent": float(np.max(absolute)),
        "inside_source_P422_case_count": sum(
            bool(row["inside_source_P422_4p6_percent"]) for row in rows
        ),
        "source_P422_full_domain_maximum_difference_percent": source_maximum_difference_percent,
        "maximum_boundary_vs_reported_pressure_difference_fraction": float(
            np.max(boundary_difference)
        ),
        "maximum_reconstructed_vs_prescribed_pore_velocity_difference_fraction": float(
            np.max(velocity_difference)
        ),
        "superficial_to_pore_velocity_ratio_range": [
            min(
                float(row["superficial_velocity_m_s"])
                / float(row["reconstructed_pore_opening_velocity_m_s"])
                for row in rows
            ),
            max(
                float(row["superficial_velocity_m_s"])
                / float(row["reconstructed_pore_opening_velocity_m_s"])
                for row in rows
            ),
        ],
        "maximum_superficial_vs_source_channel_velocity_difference_fraction": float(
            np.max(
                [
                    abs(
                        float(row["superficial_velocity_m_s"])
                        - float(row["source_inlet_channel_velocity_m_s"])
                    )
                    / float(row["source_inlet_channel_velocity_m_s"])
                    for row in rows
                ]
            )
        ),
        "table": str(csv_path),
        "interpretation": (
            "P420/P421 is evaluated with the resolved crop porosity, crop length and "
            "superficial velocity through the complete inlet cross-section. For the formal "
            "source-flow matrix, the pore-opening boundary velocity is area-corrected so that "
            "this superficial velocity equals the published inlet-channel velocity. The pore "
            "velocity itself is not inserted directly into the packed-bed relation. P422 describes the source "
            "paper's full domain and is retained as a reference, not a fitted tolerance "
            "for the smaller wall-adjacent crop."
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
