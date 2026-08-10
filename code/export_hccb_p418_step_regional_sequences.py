#!/usr/bin/env python3
"""Export P418 thermal-step fields as compact regional time sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from export_hccb_cht_training_sample import boundary_field, field, region_description


STATE_NAMES = ("Ux_m_s", "Uy_m_s", "Uz_m_s", "pressure_Pa", "temperature_K")
CONDITION_NAMES = (
    "source_inlet_velocity_m_s",
    "source_inlet_temperature_K",
    "source_solid_heat_source_MW_m3",
    "target_inlet_velocity_m_s",
    "target_inlet_temperature_K",
    "target_solid_heat_source_MW_m3",
    "target_outlet_pressure_Pa",
    "target_cooling_wall_temperature_K",
)
HISTORY_MODES = ("fixed_hydrodynamics_thermal", "fully_coupled_flow_heat")


def volume_mean(
    values: np.ndarray,
    volume: np.ndarray,
    regional_global: np.ndarray,
    selected_global: np.ndarray,
) -> np.ndarray:
    """Volume-average fine cells into the selected regional nodes."""
    output = np.zeros((len(selected_global),) + values.shape[1:], dtype=np.float64)
    global_to_local = np.full(int(selected_global.max()) + 1, -1, dtype=np.int64)
    global_to_local[selected_global] = np.arange(len(selected_global), dtype=np.int64)
    local = global_to_local[regional_global]
    if np.any(local < 0):
        raise ValueError("fine cells map outside the selected material regions")
    denominator = np.bincount(local, weights=volume, minlength=len(selected_global))
    if values.ndim == 1:
        numerator = np.bincount(
            local, weights=values * volume, minlength=len(selected_global)
        )
        return numerator / denominator
    for channel in range(values.shape[1]):
        output[:, channel] = np.bincount(
            local,
            weights=values[:, channel] * volume,
            minlength=len(selected_global),
        ) / denominator
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_time_directory(case: Path, requested: float) -> str:
    matches = []
    for path in case.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if abs(value - requested) <= 1.0e-9 * max(1.0, abs(requested)):
            matches.append(path.name)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one directory for t={requested} in {case}, found {matches}")
    return matches[0]


def aggregate_state(
    *,
    fluid_velocity: np.ndarray,
    fluid_pressure: np.ndarray,
    fluid_temperature: np.ndarray,
    solid_temperature: np.ndarray,
    fluid_volume: np.ndarray,
    solid_volume: np.ndarray,
    fluid_parent: np.ndarray,
    solid_parent: np.ndarray,
    fluid_global: np.ndarray,
    solid_global: np.ndarray,
) -> np.ndarray:
    state = np.zeros((len(fluid_global) + len(solid_global), 5), dtype=np.float64)
    state[fluid_global, :3] = volume_mean(
        fluid_velocity, fluid_volume, fluid_parent, fluid_global
    )
    state[fluid_global, 3] = volume_mean(
        fluid_pressure, fluid_volume, fluid_parent, fluid_global
    )
    state[fluid_global, 4] = volume_mean(
        fluid_temperature, fluid_volume, fluid_parent, fluid_global
    )
    state[solid_global, 4] = volume_mean(
        solid_temperature, solid_volume, solid_parent, solid_global
    )
    if not np.all(np.isfinite(state)):
        raise ValueError("regional state contains non-finite values")
    return state


def aggregate_regional_mass_flux(
    *,
    internal_fine: np.ndarray,
    boundary_fine: np.ndarray,
    internal_inverse: np.ndarray,
    internal_crossing: np.ndarray,
    internal_orientation: np.ndarray,
    internal_count: int,
    boundary_inverse: np.ndarray,
    boundary_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum solved OpenFOAM ``phi`` on each oriented regional face."""
    internal = np.bincount(
        internal_inverse,
        weights=internal_fine[internal_crossing] * internal_orientation,
        minlength=internal_count,
    )
    boundary = np.bincount(
        boundary_inverse,
        weights=boundary_fine,
        minlength=boundary_count,
    )
    if not np.all(np.isfinite(internal)) or not np.all(np.isfinite(boundary)):
        raise ValueError("regional mass flux contains non-finite values")
    return internal, boundary


def preserve_openfoam_subface_mass_flux(
    *,
    internal_fine: np.ndarray,
    boundary_fine: np.ndarray,
    internal_openfoam_face: np.ndarray,
    internal_orientation: np.ndarray,
    boundary_openfoam_face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map solved OpenFOAM ``phi`` directly to retained residual faces."""
    if len(internal_openfoam_face) != len(internal_orientation):
        raise ValueError("internal OpenFOAM face map and orientation differ")
    if np.any(internal_openfoam_face < 0) or np.any(
        internal_openfoam_face >= len(internal_fine)
    ):
        raise ValueError("internal OpenFOAM face map is out of range")
    if np.any(np.abs(internal_orientation) != 1.0):
        raise ValueError("internal phi orientation must be +1 or -1")
    external = boundary_openfoam_face >= 0
    if np.any(boundary_openfoam_face[external] >= len(boundary_fine)):
        raise ValueError("boundary OpenFOAM face map is out of range")
    internal = internal_fine[internal_openfoam_face] * internal_orientation
    boundary = np.zeros(len(boundary_openfoam_face), dtype=np.float64)
    boundary[external] = boundary_fine[boundary_openfoam_face[external]]
    if not np.all(np.isfinite(internal)) or not np.all(np.isfinite(boundary)):
        raise ValueError("preserved OpenFOAM mass flux contains non-finite values")
    return internal, boundary


def condition_vector(
    metadata: dict[str, object], boundary_metadata: dict[str, object]
) -> np.ndarray:
    source = metadata["source_parameters"]
    target = metadata["target_parameters"]
    return np.asarray(
        [
            source["inlet_velocity_m_s"],
            source["inlet_temperature_K"],
            source["solid_heat_source_MW_m3"],
            target["inlet_velocity_m_s"],
            target["inlet_temperature_K"],
            target["solid_heat_source_MW_m3"],
            boundary_metadata["outlet_pressure_Pa"],
            boundary_metadata["cooling_wall_temperature_K"],
        ],
        dtype=np.float64,
    )


def validate_sequence_arrays(
    *,
    times: np.ndarray,
    state: np.ndarray,
    internal_mass_flux: np.ndarray,
    boundary_mass_flux: np.ndarray,
    history_mode: str,
) -> dict[str, object]:
    """Check one complete curve before it is written to the training dataset."""
    if history_mode not in HISTORY_MODES:
        raise ValueError(f"unknown history mode: {history_mode}")
    if times.ndim != 1 or len(times) < 2 or np.any(~np.isfinite(times)):
        raise ValueError("sequence time must be a finite one-dimensional array")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("sequence times must increase strictly")
    if (
        state.ndim != 3
        or state.shape[0] != len(times)
        or state.shape[2] != len(STATE_NAMES)
    ):
        raise ValueError("state must have shape [Nt,Nnode,5]")
    if np.any(~np.isfinite(state)):
        raise ValueError("state contains non-finite values")
    if history_mode == "fully_coupled_flow_heat":
        if internal_mass_flux.ndim != 2 or boundary_mass_flux.ndim != 2:
            raise ValueError("fully coupled mass flux must retain shape [Nt,Nface]")
        if (
            internal_mass_flux.shape[0] != len(times)
            or boundary_mass_flux.shape[0] != len(times)
        ):
            raise ValueError("fully coupled mass flux time axis differs from the state")
    elif internal_mass_flux.ndim != 1 or boundary_mass_flux.ndim != 1:
        raise ValueError("fixed-hydrodynamics mass flux must have shape [Nface]")
    if np.any(~np.isfinite(internal_mass_flux)) or np.any(
        ~np.isfinite(boundary_mass_flux)
    ):
        raise ValueError("mass flux contains non-finite values")
    return {
        "history_mode": history_mode,
        "time_point_count": len(times),
        "regional_node_count": state.shape[1],
        "state_channel_count": state.shape[2],
        "mass_flux_time_dependent": history_mode == "fully_coupled_flow_heat",
    }


def matching_regional_graph(
    regional_topology_path: Path, node_type: np.ndarray
) -> tuple[int, dict[str, np.ndarray]]:
    """Return the unique graph level that matches the exported regional nodes."""
    with np.load(regional_topology_path, allow_pickle=False) as loaded:
        matching_levels = []
        level = 0
        while f"level_{level}_node_type" in loaded.files:
            if len(loaded[f"level_{level}_node_type"]) == len(node_type):
                matching_levels.append(level)
            level += 1
        if len(matching_levels) != 1:
            raise ValueError(
                f"expected one regional level with {len(node_type)} nodes, found {matching_levels}"
            )
        graph_level = matching_levels[0]
        if not np.array_equal(loaded[f"level_{graph_level}_node_type"], node_type):
            raise ValueError("regional graph node types differ from residual geometry")
        graph_arrays = {
            "edge_source": loaded[f"level_{graph_level}_edge_source"],
            "edge_target": loaded[f"level_{graph_level}_edge_target"],
            "edge_kind": loaded[f"level_{graph_level}_edge_kind"],
            "edge_area_m2": loaded[f"level_{graph_level}_edge_area_m2"],
            "edge_area_vector_m2": loaded[f"level_{graph_level}_edge_area_vector_m2"],
            "edge_centroid_m": loaded[f"level_{graph_level}_edge_centroid_m"],
        }
    return graph_level, graph_arrays


def regional_boundary_features(
    model_geometry_path: Path,
    graph_level: int,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load geometry-derived boundary fractions for the selected graph level."""
    key = f"level_{graph_level}_boundary_volume_fraction"
    with np.load(model_geometry_path, allow_pickle=False) as loaded:
        if key not in loaded.files or "boundary_role_names" not in loaded.files:
            raise ValueError(
                f"model geometry does not contain {key} and boundary_role_names"
            )
        boundary = loaded[key].astype(np.float32)
        names = loaded["boundary_role_names"].astype("U")
    if boundary.ndim != 2 or boundary.shape[0] != node_count:
        raise ValueError("regional boundary features differ from the selected graph")
    if boundary.shape[1] != len(names) or len(set(names.tolist())) != len(names):
        raise ValueError("boundary feature columns and role names are inconsistent")
    if np.any(~np.isfinite(boundary)) or np.any(boundary < 0.0) or np.any(boundary > 1.0):
        raise ValueError("regional boundary fractions must be finite and between zero and one")
    return boundary, names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-root", type=Path, required=True)
    parser.add_argument("--shared-topology", type=Path, required=True)
    parser.add_argument("--steady-dataset-index", type=Path, required=True)
    parser.add_argument("--subface-geometry", type=Path, required=True)
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--model-geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--history-mode",
        choices=HISTORY_MODES,
        default="fixed_hydrodynamics_thermal",
    )
    args = parser.parse_args()

    fully_coupled = args.history_mode == "fully_coupled_flow_heat"
    metadata_filename = (
        "fully_coupled_step_metadata.json"
        if fully_coupled
        else "step_case_metadata.json"
    )
    completion_filename = (
        "fully_coupled_step_response_complete.json"
        if fully_coupled
        else "step_response_complete.json"
    )

    step_root = args.step_root.resolve()
    topology_path = args.shared_topology.resolve()
    steady_dataset_path = args.steady_dataset_index.resolve()
    geometry_path = args.subface_geometry.resolve()
    regional_topology_path = args.regional_topology.resolve()
    model_geometry_path = args.model_geometry.resolve()
    output = args.output_dir.resolve()
    sequence_dir = output / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)

    with np.load(topology_path, allow_pickle=False) as loaded:
        topology = {name: loaded[name] for name in loaded.files}
        fluid_volume = topology["fluid_cell_volume_m3"].astype(np.float64)
        solid_volume = topology["solid_cell_volume_m3"].astype(np.float64)
    steady_dataset = json.loads(steady_dataset_path.read_text(encoding="utf-8"))
    boundary_patch_names = steady_dataset["boundary_patch_names"]
    with np.load(geometry_path, allow_pickle=False) as loaded:
        parent = loaded["fine_to_regional_global"].astype(np.int64)
        fluid_global = loaded["fluid_global_region"].astype(np.int64)
        solid_global = loaded["solid_global_region"].astype(np.int64)
        node_volume = np.zeros(len(fluid_global) + len(solid_global), dtype=np.float64)
        node_volume[fluid_global] = loaded["fluid_cell_volume_m3"]
        node_volume[solid_global] = loaded["solid_cell_volume_m3"]
        node_centroid = np.zeros((len(node_volume), 3), dtype=np.float64)
        node_centroid[fluid_global] = loaded["fluid_cell_centroid_m"]
        node_centroid[solid_global] = loaded["solid_cell_centroid_m"]
        internal_openfoam_face = loaded[
            "fluid_internal_subface_openfoam_face"
        ].astype(np.int64)
        internal_phi_orientation = loaded[
            "fluid_internal_subface_phi_orientation"
        ].astype(np.float64)
        boundary_openfoam_face = loaded["fluid_boundary_openfoam_face"].astype(
            np.int64
        )
    fluid_count = len(fluid_volume)
    if len(parent) != fluid_count + len(solid_volume):
        raise ValueError("fine-to-regional map does not cover the shared topology")
    fluid_parent = parent[:fluid_count]
    solid_parent = parent[fluid_count:]
    node_type = np.ones(len(node_volume), dtype=np.int8)
    node_type[fluid_global] = 0

    graph_level, graph_arrays = matching_regional_graph(
        regional_topology_path, node_type
    )
    node_boundary_fraction, boundary_role_names = regional_boundary_features(
        model_geometry_path,
        graph_level,
        len(node_volume),
    )

    geometry_output = output / "regional_sequence_geometry.npz"
    np.savez_compressed(
        geometry_output,
        node_centroid_m=node_centroid,
        node_volume_m3=node_volume,
        node_type=node_type,
        fluid_global_region=fluid_global,
        solid_global_region=solid_global,
        regional_graph_level=np.asarray(graph_level, dtype=np.int64),
        node_boundary_fraction=node_boundary_fraction,
        boundary_role_names=boundary_role_names,
        **graph_arrays,
    )

    cases = sorted(path for path in step_root.iterdir() if path.is_dir())
    records: list[dict[str, object]] = []
    waiting: list[str] = []
    for case in cases:
        metadata_path = case / metadata_filename
        if not metadata_path.is_file():
            continue
        complete = (case / completion_filename).is_file()
        if not complete:
            waiting.append(case.name)
            if args.require_complete:
                continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        boundary_metadata = json.loads(
            (case / "cht_smoke_metadata.json").read_text(encoding="utf-8")
        )
        requested_times = [float(value) for value in metadata["snapshot_times_s"]]
        available = []
        for value in requested_times:
            try:
                available.append((value, numeric_time_directory(case, value)))
            except FileNotFoundError:
                if args.require_complete:
                    raise
        if len(available) < 2:
            continue

        fluid_description = region_description(case, "fluid")
        solid_description = region_description(case, "solid")
        if fluid_description["n_cells"] != len(fluid_volume):
            raise ValueError(f"fluid cell count differs in {case.name}")
        if solid_description["n_cells"] != len(solid_volume):
            raise ValueError(f"solid cell count differs in {case.name}")

        states = []
        times = []
        regional_phi = []
        for physical_time, time_name in available:
            states.append(
                aggregate_state(
                    fluid_velocity=field(case, time_name, "fluid", "U", fluid_description),
                    fluid_pressure=field(case, time_name, "fluid", "p", fluid_description)[:, 0],
                    fluid_temperature=field(case, time_name, "fluid", "T", fluid_description)[:, 0],
                    solid_temperature=field(case, time_name, "solid", "T", solid_description)[:, 0],
                    fluid_volume=fluid_volume,
                    solid_volume=solid_volume,
                    fluid_parent=fluid_parent,
                    solid_parent=solid_parent,
                    fluid_global=fluid_global,
                    solid_global=solid_global,
                )
            )
            internal_phi = field(
                case, time_name, "fluid", "phi", fluid_description
            )[:, 0]
            boundary_phi, boundary_phi_mask = boundary_field(
                case,
                time_name,
                "fluid",
                "phi",
                fluid_description,
                require_all=True,
            )
            if not np.all(boundary_phi_mask):
                raise ValueError(f"fluid/phi boundary values are incomplete in {case.name}")
            interface_patch = boundary_patch_names["fluid"].index("fluid_to_solid")
            interface_fine = (
                topology["fluid_boundary_face_patch"] == interface_patch
            )
            if np.max(np.abs(boundary_phi[interface_fine, 0]), initial=0.0) > 1.0e-14:
                raise ValueError(
                    f"nonzero mass flux crosses fluid-solid interface in {case.name}"
                )
            regional_phi.append(
                preserve_openfoam_subface_mass_flux(
                    internal_fine=internal_phi,
                    boundary_fine=boundary_phi[:, 0],
                    internal_openfoam_face=internal_openfoam_face,
                    internal_orientation=internal_phi_orientation,
                    boundary_openfoam_face=boundary_openfoam_face,
                )
            )
            times.append(physical_time)
        state = np.stack(states).astype(np.float32)
        internal_mass_flux = np.stack([value[0] for value in regional_phi])
        boundary_mass_flux = np.stack([value[1] for value in regional_phi])
        internal_flux_output = (
            internal_mass_flux if fully_coupled else internal_mass_flux[0]
        ).astype(np.float64)
        boundary_flux_output = (
            boundary_mass_flux if fully_coupled else boundary_mass_flux[0]
        ).astype(np.float64)
        if not fully_coupled and (
            not np.allclose(
                internal_mass_flux,
                internal_mass_flux[0:1],
                rtol=5.0e-10,
                atol=5.0e-14,
            )
            or not np.allclose(
                boundary_mass_flux,
                boundary_mass_flux[0:1],
                rtol=5.0e-10,
                atol=5.0e-14,
            )
        ):
            raise ValueError(f"frozen OpenFOAM phi changed during {case.name}")
        fluid_nodes = fluid_global
        hydrodynamic_change = float(
            np.max(np.abs(state[:, fluid_nodes, :4] - state[0:1, fluid_nodes, :4]))
        )
        array_check = validate_sequence_arrays(
            times=np.asarray(times, dtype=np.float64),
            state=state,
            internal_mass_flux=internal_flux_output,
            boundary_mass_flux=boundary_flux_output,
            history_mode=args.history_mode,
        )
        sequence_path = sequence_dir / f"{case.name}.npz"
        np.savez_compressed(
            sequence_path,
            sequence_id=np.asarray(case.name),
            time_s=np.asarray(times, dtype=np.float64),
            condition_physical=condition_vector(metadata, boundary_metadata),
            state_physical=state,
            fluid_internal_mass_flux_kg_s=internal_flux_output,
            fluid_boundary_mass_flux_kg_s=boundary_flux_output,
        )
        records.append(
            {
                "sequence_id": case.name,
                "family": metadata["family"],
                "source_condition_id": metadata["source_condition_id"],
                "target_condition_id": metadata["target_condition_id"],
                "time_points": len(times),
                "complete": complete,
                "hydrodynamic_maximum_absolute_change": hydrodynamic_change,
                "mass_flux_time_dependent": array_check["mass_flux_time_dependent"],
                "frozen_phi_maximum_absolute_change_kg_s": float(
                    max(
                        np.max(np.abs(internal_mass_flux - internal_mass_flux[0:1])),
                        np.max(np.abs(boundary_mass_flux - boundary_mass_flux[0:1])),
                    )
                ),
                "sequence_file": str(sequence_path.relative_to(output)),
                "sequence_sha256": sha256(sequence_path),
            }
        )

    if args.require_complete and waiting:
        raise RuntimeError(f"{len(waiting)} sequences are not complete: {waiting}")
    if not records:
        raise ValueError("no regional step sequences were exported")
    summary = {
        "status": (
            "p418_regional_fully_coupled_flow_heat_sequences_ready"
            if fully_coupled
            else "p418_regional_thermal_step_sequences_ready"
        ),
        "history_mode": args.history_mode,
        "transient_model": (
            "fully_coupled_flow_momentum_fluid_energy_solid_energy"
            if fully_coupled
            else "thermal_step_with_quasi_steady_target_hydrodynamics"
        ),
        "sequence_count": len(records),
        "waiting_sequence_count": len(waiting),
        "regional_node_count": len(node_volume),
        "state_names": list(STATE_NAMES),
        "condition_names": list(CONDITION_NAMES),
        "mass_flux_definition": (
            "time-dependent solved OpenFOAM phi retained at every saved time on each "
            "oriented residual subface; the impermeable fluid-solid interface mass flux is zero"
            if fully_coupled
            else "frozen solved OpenFOAM phi retained on every residual subface with "
            "explicit owner-neighbour orientation; the impermeable fluid-solid interface mass flux is zero"
        ),
        "boundary_patch_names": boundary_patch_names,
        "steady_dataset_index": str(steady_dataset_path),
        "steady_dataset_index_sha256": sha256(steady_dataset_path),
        "shared_topology_sha256": sha256(topology_path),
        "subface_geometry_sha256": sha256(geometry_path),
        "regional_topology_sha256": sha256(regional_topology_path),
        "model_geometry_sha256": sha256(model_geometry_path),
        "regional_graph_level": graph_level,
        "boundary_role_names": boundary_role_names.tolist(),
        "boundary_feature_definition": (
            "volume-weighted fractions of regional cells touching the inlet, outlet, "
            "cooling wall, symmetry boundaries and fluid-solid interface"
        ),
        "regional_geometry_file": geometry_output.name,
        "regional_geometry_sha256": sha256(geometry_output),
        "sequences": records,
        "new_physical_parameters": [],
        "scientific_scope": (
            "Volume-weighted regional fields from reconstructed fully coupled OpenFOAM snapshots; "
            "velocity, pressure, face mass flux and fluid/solid temperature evolve in time."
            if fully_coupled
            else "Volume-weighted regional fields from reconstructed OpenFOAM thermal-step snapshots. "
            "Target hydrodynamics are fixed; fluid and solid temperatures evolve in time."
        ),
    }
    (output / "dataset_index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
