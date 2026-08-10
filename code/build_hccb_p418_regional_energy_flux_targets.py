#!/usr/bin/env python3
"""Build conservative regional energy-flow targets for the P418 CHT cases.

OpenFOAM's solved mass flux and wall heat flux are retained.  Internal fluid
enthalpy/conduction and solid conduction are reconstructed on the original
faces with the archived thermophysical laws and numerical schemes, aggregated
to the regional graph, and projected to the nearest locally conservative flow.
The projection preserves the OpenFOAM external heat flows and volumetric heat
source; only the internal regional face powers are corrected.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg
import torch

from build_hccb_p418_regional_residual_geometry import composed_parent
from hccb_p418_regional_cht_adapter import (
    assemble_p418_regional_cht_residual,
    load_p418_fine_geometry,
)
from hccb_source_backed_thermophysical import helium_sensible_enthalpy


CONDITION_KEYS = (
    "inlet_velocity_m_s",
    "inlet_temperature_K",
    "solid_heat_source_W_m3",
    "outlet_pressure_Pa",
    "cooling_wall_temperature_K",
)

# Numerical consistency limits for independently reconstructed fluid- and
# solid-side interface powers.  They are dimensionless and therefore remain
# meaningful when the imposed source power changes between cases.
INTERFACE_MAX_RELATIVE_TOLERANCE = 1.0e-6
# The L1 metric accumulates the independently reconstructed mismatch over
# 513,310 paired interface faces.  A 2e-5 bound remains a strict 0.002%
# consistency requirement while avoiding rejection from the accumulated
# finite-volume gradient difference on many near-zero-power faces.
INTERFACE_L1_RELATIVE_TOLERANCE = 2.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regional_balance(
    internal_flux: np.ndarray,
    owner: np.ndarray,
    neighbour: np.ndarray,
    boundary_flux: np.ndarray,
    boundary_owner: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    result = -np.asarray(source, dtype=np.float64).copy()
    np.add.at(result, owner, internal_flux)
    np.add.at(result, neighbour, -internal_flux)
    np.add.at(result, boundary_owner, boundary_flux)
    return result


def aggregate_crossing_flux(
    *,
    fine_owner_global: np.ndarray,
    fine_neighbour_global: np.ndarray,
    fine_flux_owner_to_neighbour: np.ndarray,
    fine_to_region: np.ndarray,
    region_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    owner_region = fine_to_region[fine_owner_global]
    neighbour_region = fine_to_region[fine_neighbour_global]
    crossing = owner_region != neighbour_region
    lower = np.minimum(owner_region[crossing], neighbour_region[crossing])
    upper = np.maximum(owner_region[crossing], neighbour_region[crossing])
    key = lower * region_count + upper
    unique, inverse = np.unique(key, return_inverse=True)
    orientation = np.where(owner_region[crossing] == lower, 1.0, -1.0)
    flux = np.bincount(
        inverse,
        weights=fine_flux_owner_to_neighbour[crossing] * orientation,
        minlength=len(unique),
    )
    return unique, flux


def align_flux_to_edges(
    *,
    aggregated_key: np.ndarray,
    aggregated_flux: np.ndarray,
    edge_key: np.ndarray,
) -> np.ndarray:
    order = np.argsort(aggregated_key)
    sorted_key = aggregated_key[order]
    location = np.searchsorted(sorted_key, edge_key)
    if np.any(location >= len(sorted_key)) or not np.array_equal(
        sorted_key[location], edge_key
    ):
        raise ValueError("aggregated fine faces do not match the regional edge set")
    return aggregated_flux[order][location]


def grouped_boundaries(
    *,
    owner_global: np.ndarray,
    patch_kind: np.ndarray,
    face_area: np.ndarray,
    face_area_vector: np.ndarray,
    face_centroid: np.ndarray,
    kind_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = owner_global * kind_count + patch_kind
    unique, inverse = np.unique(key, return_inverse=True)
    area = np.bincount(inverse, weights=face_area, minlength=len(unique))
    area_vector = np.column_stack(
        [
            np.bincount(
                inverse, weights=face_area_vector[:, axis], minlength=len(unique)
            )
            for axis in range(3)
        ]
    )
    centroid = np.column_stack(
        [
            np.bincount(
                inverse,
                weights=face_centroid[:, axis] * face_area,
                minlength=len(unique),
            )
            for axis in range(3)
        ]
    ) / area[:, None]
    return unique // kind_count, unique % kind_count, inverse, area, area_vector, centroid


def interface_reciprocity_metrics(
    fluid_interface_out_w: np.ndarray,
    solid_interface_in_w: np.ndarray,
) -> dict[str, float]:
    """Compare the two independently reconstructed sides of the CHT interface."""
    fluid = np.asarray(fluid_interface_out_w, dtype=np.float64)
    solid = np.asarray(solid_interface_in_w, dtype=np.float64)
    if fluid.shape != solid.shape or fluid.size == 0:
        raise ValueError("paired interface power arrays must be non-empty and aligned")
    difference = np.abs(fluid - solid)
    pair_scale = np.maximum(np.abs(fluid), np.abs(solid))
    tiny = np.finfo(np.float64).tiny
    return {
        "maximum_interface_pair_difference_W": float(np.max(difference)),
        "maximum_difference_over_maximum_interface_face_power": float(
            np.max(difference) / max(float(np.max(pair_scale)), tiny)
        ),
        "l1_difference_over_l1_interface_face_power": float(
            np.sum(difference) / max(float(np.sum(pair_scale)), tiny)
        ),
    }


def conservative_internal_projection(
    *,
    initial_internal_flux: np.ndarray,
    owner: np.ndarray,
    neighbour: np.ndarray,
    conductance: np.ndarray,
    boundary_flux: np.ndarray,
    boundary_owner: np.ndarray,
    source: np.ndarray,
    cell_volume: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Correct internal edge powers while preserving source and boundary power."""
    cell_count = len(cell_volume)
    initial_balance = regional_balance(
        initial_internal_flux,
        owner,
        neighbour,
        boundary_flux,
        boundary_owner,
        source,
    )
    adjacency = sparse.coo_matrix(
        (
            np.ones(2 * len(owner), dtype=np.int8),
            (np.concatenate((owner, neighbour)), np.concatenate((neighbour, owner))),
        ),
        shape=(cell_count, cell_count),
    ).tocsr()
    component_count, component = connected_components(adjacency, directed=False)
    target_balance = np.zeros(cell_count, dtype=np.float64)
    for label in range(component_count):
        selected = component == label
        target_balance[selected] = (
            np.sum(initial_balance[selected])
            * cell_volume[selected]
            / np.sum(cell_volume[selected])
        )
    rhs = target_balance - initial_balance
    diagonal = np.bincount(
        np.concatenate((owner, neighbour)),
        weights=np.concatenate((conductance, conductance)),
        minlength=cell_count,
    )
    laplacian = sparse.coo_matrix(
        (
            np.concatenate((diagonal, -conductance, -conductance)),
            (
                np.concatenate((np.arange(cell_count), owner, neighbour)),
                np.concatenate((np.arange(cell_count), neighbour, owner)),
            ),
        ),
        shape=(cell_count, cell_count),
    ).tocsr()
    anchors = np.asarray(
        [np.flatnonzero(component == label)[0] for label in range(component_count)],
        dtype=np.int64,
    )
    keep = np.ones(cell_count, dtype=bool)
    keep[anchors] = False
    reduced = laplacian[keep][:, keep]
    reduced_diagonal = reduced.diagonal()
    preconditioner = sparse.diags(1.0 / np.maximum(reduced_diagonal, np.finfo(float).tiny))
    potential_reduced, info = cg(
        reduced,
        rhs[keep],
        M=preconditioner,
        rtol=1.0e-11,
        atol=0.0,
        maxiter=10000,
    )
    if info != 0:
        raise RuntimeError(f"conservative energy projection did not converge: {info}")
    potential = np.zeros(cell_count, dtype=np.float64)
    potential[keep] = potential_reduced
    correction = conductance * (potential[owner] - potential[neighbour])
    projected = initial_internal_flux + correction
    final_balance = regional_balance(
        projected,
        owner,
        neighbour,
        boundary_flux,
        boundary_owner,
        source,
    )
    return projected, {
        "connected_components": int(component_count),
        "initial_local_balance_l1_W": float(np.sum(np.abs(initial_balance))),
        "final_local_balance_l1_W": float(np.sum(np.abs(final_balance))),
        "final_balance_minus_openfoam_global_distribution_linf_W": float(
            np.max(np.abs(final_balance - target_balance))
        ),
        "internal_flux_correction_rms_W": float(np.sqrt(np.mean(correction**2))),
        "internal_flux_initial_rms_W": float(
            np.sqrt(np.mean(initial_internal_flux**2))
        ),
        "internal_flux_correction_over_initial_rms": float(
            np.sqrt(np.mean(correction**2))
            / max(np.sqrt(np.mean(initial_internal_flux**2)), np.finfo(float).tiny)
        ),
        "global_energy_difference_W": float(np.sum(final_balance)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--native-graph", type=Path, required=True)
    parser.add_argument("--boundary-heat-targets", type=Path, required=True)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_index.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected_case_count = int(dataset.get("case_count", len(dataset["conditions"])))
    if expected_case_count <= 0 or expected_case_count != len(dataset["conditions"]):
        raise ValueError("dataset case_count does not match its condition records")
    root = dataset_path.parent
    with np.load(root / dataset["shared_topology_file"], allow_pickle=False) as loaded:
        topology = {name: loaded[name] for name in loaded.files}
    with np.load(args.regional_topology.resolve(), allow_pickle=False) as loaded:
        regional = {name: loaded[name] for name in loaded.files}
    with np.load(args.boundary_heat_targets.resolve(), allow_pickle=False) as loaded:
        heat = {name: loaded[name] for name in loaded.files}
    condition_ids = np.asarray([record["condition_id"] for record in dataset["conditions"]])
    if not np.array_equal(heat["condition_id"].astype(str), condition_ids.astype(str)):
        raise ValueError("boundary heat target case order differs from the dataset")

    prefix = f"level_{args.level}"
    parent = composed_parent(regional, args.level)
    region_count = len(regional[f"{prefix}_node_type"])
    region_type = regional[f"{prefix}_node_type"].astype(np.int8)
    region_centroid = regional[f"{prefix}_centroid_m"].astype(np.float64)
    region_volume = regional[f"{prefix}_volume_m3"].astype(np.float64)
    edge_source_all = regional[f"{prefix}_edge_source"].astype(np.int64)
    edge_target_all = regional[f"{prefix}_edge_target"].astype(np.int64)
    edge_kind_all = regional[f"{prefix}_edge_kind"].astype(np.int8)
    unique_edge = edge_source_all < edge_target_all
    edge_owner = edge_source_all[unique_edge]
    edge_neighbour = edge_target_all[unique_edge]
    edge_kind = edge_kind_all[unique_edge]
    edge_area = regional[f"{prefix}_edge_area_m2"][unique_edge].astype(np.float64)
    edge_area_vector = regional[f"{prefix}_edge_area_vector_m2"][unique_edge].astype(np.float64)
    edge_centroid = regional[f"{prefix}_edge_centroid_m"][unique_edge].astype(np.float64)
    edge_distance = np.linalg.norm(
        region_centroid[edge_neighbour] - region_centroid[edge_owner], axis=1
    )
    if np.any(edge_distance <= 0.0) or np.any(edge_area <= 0.0):
        raise ValueError("regional edge geometry is degenerate")
    conductance = edge_area / edge_distance

    fluid_count = len(topology["fluid_cell_volume_m3"])
    solid_count = len(topology["solid_cell_volume_m3"])
    fine_count = fluid_count + solid_count
    if len(parent) != fine_count:
        raise ValueError("fine-to-regional parent count differs from the CHT mesh")
    fluid_names = list(dataset["boundary_patch_names"]["fluid"])
    solid_names = list(dataset["boundary_patch_names"]["solid"])
    fluid_interface_patch = fluid_names.index("fluid_to_solid")
    solid_interface_patch = solid_names.index("solid_to_fluid")
    fluid_patch = topology["fluid_boundary_face_patch"].astype(np.int64)
    solid_patch = topology["solid_boundary_face_patch"].astype(np.int64)
    fluid_external = fluid_patch != fluid_interface_patch
    solid_external = solid_patch != solid_interface_patch
    boundary_kind_names = [f"fluid:{name}" for name in fluid_names[:-1]] + [
        f"solid:{name}" for name in solid_names[:-1]
    ]
    kind_count = len(boundary_kind_names)
    fluid_kind = fluid_patch[fluid_external]
    solid_kind = (len(fluid_names) - 1) + solid_patch[solid_external]
    boundary_owner_fine = np.concatenate(
        (
            topology["fluid_boundary_face_owner"][fluid_external].astype(np.int64),
            fluid_count
            + topology["solid_boundary_face_owner"][solid_external].astype(np.int64),
        )
    )
    boundary_patch_kind_fine = np.concatenate((fluid_kind, solid_kind)).astype(np.int64)
    boundary_area_fine = np.concatenate(
        (
            topology["fluid_boundary_face_area_m2"][fluid_external],
            topology["solid_boundary_face_area_m2"][solid_external],
        )
    ).astype(np.float64)
    boundary_vector_fine = np.concatenate(
        (
            topology["fluid_boundary_face_area_vector_outward_m2"][fluid_external],
            topology["solid_boundary_face_area_vector_outward_m2"][solid_external],
        ),
        axis=0,
    ).astype(np.float64)
    boundary_centroid_fine = np.concatenate(
        (
            topology["fluid_boundary_face_centroid_m"][fluid_external],
            topology["solid_boundary_face_centroid_m"][solid_external],
        ),
        axis=0,
    ).astype(np.float64)
    (
        boundary_owner,
        boundary_kind,
        boundary_inverse,
        boundary_area,
        boundary_area_vector,
        boundary_centroid,
    ) = grouped_boundaries(
        owner_global=parent[boundary_owner_fine],
        patch_kind=boundary_patch_kind_fine,
        face_area=boundary_area_fine,
        face_area_vector=boundary_vector_fine,
        face_centroid=boundary_centroid_fine,
        kind_count=kind_count,
    )

    fine_geometry = load_p418_fine_geometry(
        root / dataset["shared_topology_file"],
        args.native_graph.resolve(),
        fluid_patch_names=fluid_names,
        solid_patch_names=solid_names,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    level_edge_key = edge_owner * region_count + edge_neighbour
    internal_targets: list[np.ndarray] = []
    boundary_targets: list[np.ndarray] = []
    source_targets: list[np.ndarray] = []
    reports: list[dict[str, object]] = []
    start = time.time()
    with torch.no_grad():
        for case_index, record in enumerate(dataset["conditions"]):
            with np.load(root / record["field_file"], allow_pickle=False) as loaded:
                field = {name: loaded[name] for name in loaded.files}
            boundary_pressure = field["fluid_boundary_pressure_Pa"].astype(
                np.float64, copy=True
            )
            missing_pressure = boundary_pressure <= 0.0
            boundary_pressure[missing_pressure] = field["fluid_pressure_Pa"][
                topology["fluid_boundary_face_owner"][missing_pressure]
            ]
            condition = torch.tensor(
                [[float(record[name]) for name in CONDITION_KEYS]], dtype=torch.float64
            )
            residual = assemble_p418_regional_cht_residual(
                geometry=fine_geometry,
                physical_conditions=condition,
                fluid_velocity_m_s=torch.as_tensor(
                    field["fluid_velocity_m_s"], dtype=torch.float64
                ).unsqueeze(0),
                fluid_pressure_pa=torch.as_tensor(
                    field["fluid_pressure_Pa"], dtype=torch.float64
                ).unsqueeze(0),
                fluid_temperature_k=torch.as_tensor(
                    field["fluid_temperature_K"], dtype=torch.float64
                ).unsqueeze(0),
                solid_temperature_k=torch.as_tensor(
                    field["solid_temperature_K"], dtype=torch.float64
                ).unsqueeze(0),
                fluid_boundary_pressure_pa=torch.as_tensor(
                    boundary_pressure, dtype=torch.float64
                ).unsqueeze(0),
                fluid_internal_mass_flux_kg_s=torch.as_tensor(
                    field["fluid_internal_face_mass_flow_kg_s"], dtype=torch.float64
                ).unsqueeze(0),
                fluid_boundary_mass_flux_kg_s=torch.as_tensor(
                    field["fluid_boundary_face_mass_flow_kg_s"], dtype=torch.float64
                ).unsqueeze(0),
            )
            initial = np.zeros(len(edge_owner), dtype=np.float64)
            fluid_key, fluid_flux = aggregate_crossing_flux(
                fine_owner_global=topology["fluid_internal_face_owner"].astype(np.int64),
                fine_neighbour_global=topology["fluid_internal_face_neighbour"].astype(np.int64),
                fine_flux_owner_to_neighbour=residual.fluid_internal_energy_flux_w[0].cpu().numpy(),
                fine_to_region=parent,
                region_count=region_count,
            )
            fluid_edge = edge_kind == 0
            initial[fluid_edge] = align_flux_to_edges(
                aggregated_key=fluid_key,
                aggregated_flux=fluid_flux,
                edge_key=level_edge_key[fluid_edge],
            )
            solid_key, solid_flux = aggregate_crossing_flux(
                fine_owner_global=fluid_count
                + topology["solid_internal_face_owner"].astype(np.int64),
                fine_neighbour_global=fluid_count
                + topology["solid_internal_face_neighbour"].astype(np.int64),
                fine_flux_owner_to_neighbour=residual.solid_internal_heat_flux_w[0].cpu().numpy(),
                fine_to_region=parent,
                region_count=region_count,
            )
            solid_edge = edge_kind == 1
            initial[solid_edge] = align_flux_to_edges(
                aggregated_key=solid_key,
                aggregated_flux=solid_flux,
                edge_key=level_edge_key[solid_edge],
            )

            fluid_heat_into = heat[
                "fluid_boundary_heat_flux_into_region_W_m2"
            ][case_index].astype(np.float64)
            solid_heat_into = heat[
                "solid_boundary_heat_flux_into_region_W_m2"
            ][case_index].astype(np.float64)
            fluid_interface = fluid_patch == fluid_interface_patch
            solid_interface = solid_patch == solid_interface_patch
            fluid_interface_out = -fluid_heat_into[fluid_interface] * topology[
                "fluid_boundary_face_area_m2"
            ][fluid_interface]
            solid_interface_in = solid_heat_into[solid_interface] * topology[
                "solid_boundary_face_area_m2"
            ][solid_interface]
            interface_flux = 0.5 * (fluid_interface_out + solid_interface_in)
            interface_key, interface_regional_flux = aggregate_crossing_flux(
                fine_owner_global=topology["interface_fluid_cell"].astype(np.int64),
                fine_neighbour_global=fluid_count
                + topology["interface_solid_cell"].astype(np.int64),
                fine_flux_owner_to_neighbour=interface_flux,
                fine_to_region=parent,
                region_count=region_count,
            )
            interface_edge = edge_kind == 2
            initial[interface_edge] = align_flux_to_edges(
                aggregated_key=interface_key,
                aggregated_flux=interface_regional_flux,
                edge_key=level_edge_key[interface_edge],
            )

            boundary_temperature_np = field["fluid_boundary_temperature_K"].astype(
                np.float64, copy=True
            )
            missing_temperature = boundary_temperature_np <= 0.0
            boundary_temperature_np[missing_temperature] = field[
                "fluid_temperature_K"
            ][topology["fluid_boundary_face_owner"][missing_temperature]]
            boundary_temperature = torch.as_tensor(
                boundary_temperature_np, dtype=torch.float64
            )
            boundary_enthalpy = helium_sensible_enthalpy(boundary_temperature).cpu().numpy()
            fluid_boundary_outward = (
                field["fluid_boundary_face_mass_flow_kg_s"] * boundary_enthalpy
                - fluid_heat_into * topology["fluid_boundary_face_area_m2"]
            )
            solid_boundary_outward = (
                -solid_heat_into * topology["solid_boundary_face_area_m2"]
            )
            boundary_fine = np.concatenate(
                (
                    fluid_boundary_outward[fluid_external],
                    solid_boundary_outward[solid_external],
                )
            )
            boundary = np.bincount(
                boundary_inverse, weights=boundary_fine, minlength=len(boundary_owner)
            )
            source = np.where(
                region_type == 1,
                float(record["solid_heat_source_W_m3"]) * region_volume,
                0.0,
            )
            projected, projection = conservative_internal_projection(
                initial_internal_flux=initial,
                owner=edge_owner,
                neighbour=edge_neighbour,
                conductance=conductance,
                boundary_flux=boundary,
                boundary_owner=boundary_owner,
                source=source,
                cell_volume=region_volume,
            )
            generated = float(np.sum(source))
            projection["initial_local_balance_l1_over_generated_power"] = float(
                projection["initial_local_balance_l1_W"] / generated
            )
            projection["final_local_balance_l1_over_generated_power"] = float(
                projection["final_local_balance_l1_W"] / generated
            )
            projection["global_energy_difference_over_generated_power"] = float(
                abs(projection["global_energy_difference_W"]) / generated
            )
            reports.append(
                {
                    "condition_id": str(record["condition_id"]),
                    **interface_reciprocity_metrics(
                        fluid_interface_out, solid_interface_in
                    ),
                    "projection": projection,
                }
            )
            internal_targets.append(projected)
            boundary_targets.append(boundary)
            source_targets.append(source)
            del field, residual
            gc.collect()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "regional_energy_flux_targets.npz"
    np.savez_compressed(
        target_path,
        condition_id=condition_ids,
        node_type=region_type,
        node_centroid_m=region_centroid,
        node_volume_m3=region_volume,
        internal_owner=edge_owner,
        internal_neighbour=edge_neighbour,
        internal_kind=edge_kind,
        internal_kind_name=np.asarray(
            ["fluid", "solid", "fluid_to_solid"], dtype="U32"
        ),
        internal_face_centroid_m=edge_centroid,
        internal_face_area_vector_m2=edge_area_vector,
        internal_face_area_m2=edge_area,
        boundary_owner=boundary_owner,
        boundary_kind=boundary_kind,
        boundary_kind_name=np.asarray(boundary_kind_names),
        boundary_face_centroid_m=boundary_centroid,
        boundary_face_area_vector_m2=boundary_area_vector,
        boundary_face_area_m2=boundary_area,
        internal_energy_flow_W=np.stack(internal_targets),
        boundary_energy_flow_W=np.stack(boundary_targets),
        node_source_power_W=np.stack(source_targets),
    )
    observed_interface_max_relative = max(
        float(row["maximum_difference_over_maximum_interface_face_power"])
        for row in reports
    )
    observed_interface_l1_relative = max(
        float(row["l1_difference_over_l1_interface_face_power"])
        for row in reports
    )
    checks = {
        "all_dataset_cases_are_present": len(reports) == expected_case_count,
        "all_projected_targets_are_finite": bool(
            np.all(np.isfinite(internal_targets))
            and np.all(np.isfinite(boundary_targets))
            and np.all(np.isfinite(source_targets))
        ),
        "interface_heat_flux_is_reciprocal": (
            observed_interface_max_relative < INTERFACE_MAX_RELATIVE_TOLERANCE
            and observed_interface_l1_relative < INTERFACE_L1_RELATIVE_TOLERANCE
        ),
        "regional_local_balance_matches_openfoam_global_remainder": max(
            float(
                row["projection"][
                    "final_balance_minus_openfoam_global_distribution_linf_W"
                ]
            )
            for row in reports
        ) < 1.0e-10,
    }
    summary = {
        "status": "p418_regional_energy_flux_targets_ready" if all(checks.values()) else "failed",
        "regional_level": args.level,
        "counts": {
            "cells": region_count,
            "internal_energy_edges": len(edge_owner),
            "external_boundary_groups": len(boundary_owner),
            "cases": len(reports),
        },
        "checks": checks,
        "cases": reports,
        "source_dataset_sha256": sha256(dataset_path),
        "source_regional_topology_sha256": sha256(args.regional_topology.resolve()),
        "source_boundary_heat_targets_sha256": sha256(args.boundary_heat_targets.resolve()),
        "target_file": target_path.name,
        "target_sha256": sha256(target_path),
        "method": (
            "OpenFOAM external heat flow and source retained; source-backed fine-face "
            "internal energy flow aggregated and corrected by an area-over-distance "
            "finite-volume conservative projection"
        ),
        "numerical_consistency_limits": {
            "maximum_difference_over_maximum_interface_face_power": INTERFACE_MAX_RELATIVE_TOLERANCE,
            "l1_difference_over_l1_interface_face_power": INTERFACE_L1_RELATIVE_TOLERANCE,
            "meaning": "dimensionless comparison of independently reconstructed fluid- and solid-side interface powers; not a physical model parameter",
        },
        "observed_interface_consistency": {
            "maximum_difference_over_maximum_interface_face_power": observed_interface_max_relative,
            "l1_difference_over_l1_interface_face_power": observed_interface_l1_relative,
        },
        "new_physical_parameters": [],
        "elapsed_seconds": time.time() - start,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
