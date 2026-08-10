#!/usr/bin/env python3
"""P418 adapter from regional neural states to conservative CHT residuals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hccb_multiregion_steady_cht_residual import (
    CoupledInterfaceMap,
    RegionMesh,
    SteadyChtResiduals,
    ThermalBoundaryConditions,
    VelocityBoundaryConditions,
    assemble_steady_hccb_cht_residual,
)


@dataclass(frozen=True)
class P418SubfaceGeometry:
    fluid_mesh: RegionMesh
    solid_mesh: RegionMesh
    interface: CoupledInterfaceMap
    fluid_boundary_patch: torch.Tensor
    solid_boundary_patch: torch.Tensor
    fluid_patch_names: tuple[str, ...]
    solid_patch_names: tuple[str, ...]
    fine_to_regional_global: np.ndarray
    fluid_global_region: np.ndarray
    solid_global_region: np.ndarray


def _tensor(values: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if np.issubdtype(values.dtype, np.integer):
        return torch.as_tensor(values, device=device, dtype=torch.long)
    return torch.as_tensor(values, device=device, dtype=dtype)


def load_p418_subface_geometry(
    path: Path,
    *,
    fluid_patch_names: list[str] | tuple[str, ...],
    solid_patch_names: list[str] | tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> P418SubfaceGeometry:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}

    def mesh(region: str) -> RegionMesh:
        return RegionMesh(
            cell_centroid=_tensor(arrays[f"{region}_cell_centroid_m"], device=device, dtype=dtype),
            cell_volume=_tensor(arrays[f"{region}_cell_volume_m3"], device=device, dtype=dtype),
            internal_face_centroid=_tensor(arrays[f"{region}_internal_subface_centroid_m"], device=device, dtype=dtype),
            internal_area_vector=_tensor(arrays[f"{region}_internal_subface_area_vector_m2"], device=device, dtype=dtype),
            internal_owner=_tensor(arrays[f"{region}_internal_subface_owner"], device=device, dtype=dtype),
            internal_neighbour=_tensor(arrays[f"{region}_internal_subface_neighbour"], device=device, dtype=dtype),
            boundary_face_centroid=_tensor(arrays[f"{region}_boundary_face_centroid_m"], device=device, dtype=dtype),
            boundary_area_vector=_tensor(arrays[f"{region}_boundary_face_area_vector_m2"], device=device, dtype=dtype),
            boundary_owner=_tensor(arrays[f"{region}_boundary_face_owner"], device=device, dtype=dtype),
        )

    return P418SubfaceGeometry(
        fluid_mesh=mesh("fluid"),
        solid_mesh=mesh("solid"),
        interface=CoupledInterfaceMap(
            fluid_boundary_face=_tensor(arrays["interface_fluid_boundary_face"], device=device, dtype=dtype),
            solid_boundary_face=_tensor(arrays["interface_solid_boundary_face"], device=device, dtype=dtype),
        ),
        fluid_boundary_patch=_tensor(arrays["fluid_boundary_face_patch"], device=device, dtype=dtype),
        solid_boundary_patch=_tensor(arrays["solid_boundary_face_patch"], device=device, dtype=dtype),
        fluid_patch_names=tuple(fluid_patch_names),
        solid_patch_names=tuple(solid_patch_names),
        fine_to_regional_global=arrays["fine_to_regional_global"].astype(np.int64),
        fluid_global_region=arrays["fluid_global_region"].astype(np.int64),
        solid_global_region=arrays["solid_global_region"].astype(np.int64),
    )


def load_p418_fine_geometry(
    topology_path: Path,
    native_graph_path: Path,
    *,
    fluid_patch_names: list[str] | tuple[str, ...],
    solid_patch_names: list[str] | tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> P418SubfaceGeometry:
    """Load the original OpenFOAM cells/faces in owner-to-neighbour order."""
    with np.load(topology_path, allow_pickle=False) as loaded:
        topology = {name: loaded[name] for name in loaded.files}
    with np.load(native_graph_path, allow_pickle=False) as loaded:
        native = {name: loaded[name] for name in loaded.files}
    fluid_count = len(topology["fluid_cell_volume_m3"])
    solid_count = len(topology["solid_cell_volume_m3"])

    def original_internal(region: str, kind: int, offset: int) -> dict[str, np.ndarray]:
        owner = topology[f"{region}_internal_face_owner"].astype(np.int64)
        neighbour = topology[f"{region}_internal_face_neighbour"].astype(np.int64)
        edge = native["edge_kind"] == kind
        local_face = native["edge_local_face"][edge].astype(np.int64)
        source = native["edge_source_global"][edge].astype(np.int64) - offset
        target = native["edge_target_global"][edge].astype(np.int64) - offset
        original = (source == owner[local_face]) & (target == neighbour[local_face])
        selected_face = local_face[original]
        order = np.argsort(selected_face)
        if not np.array_equal(selected_face[order], np.arange(len(owner))):
            raise ValueError(f"native graph does not recover every {region} owner face")
        native_indices = np.flatnonzero(edge)[original][order]
        return {
            "owner": owner,
            "neighbour": neighbour,
            "centroid": native["edge_face_centroid_m"][native_indices].astype(np.float64),
            "area_vector": native["edge_area_vector_m2"][native_indices].astype(np.float64),
        }

    fluid_internal = original_internal("fluid", 0, 0)
    solid_internal = original_internal("solid", 1, fluid_count)

    def fine_mesh(region: str, internal: dict[str, np.ndarray]) -> RegionMesh:
        return RegionMesh(
            cell_centroid=_tensor(topology[f"{region}_cell_centroid_m"], device=device, dtype=dtype),
            cell_volume=_tensor(topology[f"{region}_cell_volume_m3"], device=device, dtype=dtype),
            internal_face_centroid=_tensor(internal["centroid"], device=device, dtype=dtype),
            internal_area_vector=_tensor(internal["area_vector"], device=device, dtype=dtype),
            internal_owner=_tensor(internal["owner"], device=device, dtype=dtype),
            internal_neighbour=_tensor(internal["neighbour"], device=device, dtype=dtype),
            boundary_face_centroid=_tensor(topology[f"{region}_boundary_face_centroid_m"], device=device, dtype=dtype),
            boundary_area_vector=_tensor(topology[f"{region}_boundary_face_area_vector_outward_m2"], device=device, dtype=dtype),
            boundary_owner=_tensor(topology[f"{region}_boundary_face_owner"], device=device, dtype=dtype),
        )

    fluid_patch = topology["fluid_boundary_face_patch"].astype(np.int64)
    solid_patch = topology["solid_boundary_face_patch"].astype(np.int64)
    fluid_interface = np.flatnonzero(
        fluid_patch == tuple(fluid_patch_names).index("fluid_to_solid")
    )
    solid_interface = np.flatnonzero(
        solid_patch == tuple(solid_patch_names).index("solid_to_fluid")
    )
    if not np.array_equal(
        topology["fluid_boundary_face_owner"][fluid_interface],
        topology["interface_fluid_cell"],
    ) or not np.array_equal(
        topology["solid_boundary_face_owner"][solid_interface],
        topology["interface_solid_cell"],
    ):
        raise ValueError("shared interface order differs from the boundary-face order")
    return P418SubfaceGeometry(
        fluid_mesh=fine_mesh("fluid", fluid_internal),
        solid_mesh=fine_mesh("solid", solid_internal),
        interface=CoupledInterfaceMap(
            fluid_boundary_face=torch.as_tensor(fluid_interface, device=device, dtype=torch.long),
            solid_boundary_face=torch.as_tensor(solid_interface, device=device, dtype=torch.long),
        ),
        fluid_boundary_patch=torch.as_tensor(fluid_patch, device=device, dtype=torch.long),
        solid_boundary_patch=torch.as_tensor(solid_patch, device=device, dtype=torch.long),
        fluid_patch_names=tuple(fluid_patch_names),
        solid_patch_names=tuple(solid_patch_names),
        fine_to_regional_global=np.arange(fluid_count + solid_count, dtype=np.int64),
        fluid_global_region=np.arange(fluid_count, dtype=np.int64),
        solid_global_region=np.arange(fluid_count, fluid_count + solid_count, dtype=np.int64),
    )


def _patch_mask(patch: torch.Tensor, names: tuple[str, ...], name: str) -> torch.Tensor:
    if name not in names:
        raise ValueError(f"missing boundary patch {name!r}")
    return patch == names.index(name)


def _batch_face_values(
    conditions: torch.Tensor,
    patch: torch.Tensor,
    names: tuple[str, ...],
    *,
    vector: bool,
) -> torch.Tensor:
    batch = conditions.shape[0]
    if vector:
        values = conditions.new_zeros((batch, len(patch), 3))
        inlet = _patch_mask(patch, names, "inlet")
        values[:, inlet, 2] = conditions[:, 0:1]
        return values
    values = conditions.new_zeros((batch, len(patch)))
    inlet = _patch_mask(patch, names, "inlet")
    cooling = _patch_mask(patch, names, "coolingWall")
    values[:, inlet] = conditions[:, 1:2]
    values[:, cooling] = conditions[:, 4:5]
    return values


def p418_boundary_conditions(
    geometry: P418SubfaceGeometry,
    physical_conditions: torch.Tensor,
) -> tuple[VelocityBoundaryConditions, ThermalBoundaryConditions, ThermalBoundaryConditions]:
    """Build the exact boundary-condition families used by the P418 cases."""
    if physical_conditions.ndim != 2 or physical_conditions.shape[1] != 5:
        raise ValueError("physical conditions must be [Uin,Tin,q,pout,Twall]")
    fp = geometry.fluid_boundary_patch
    sp = geometry.solid_boundary_patch
    fn = geometry.fluid_patch_names
    sn = geometry.solid_patch_names
    fluid_interface = _patch_mask(fp, fn, "fluid_to_solid")
    solid_interface = _patch_mask(sp, sn, "solid_to_fluid")
    fluid_symmetry = _patch_mask(fp, fn, "symmetryWalls")
    solid_symmetry = _patch_mask(sp, sn, "symmetryWalls")
    fluid_velocity = VelocityBoundaryConditions(
        fixed_value_mask=(
            _patch_mask(fp, fn, "inlet")
            | _patch_mask(fp, fn, "coolingWall")
            | fluid_interface
        ),
        fixed_reference_value=_batch_face_values(
            physical_conditions, fp, fn, vector=True
        ),
        pressure_inlet_outlet_mask=_patch_mask(fp, fn, "outlet"),
        symmetry_or_empty_mask=fluid_symmetry,
    )
    fluid_reference_temperature = _batch_face_values(
        physical_conditions, fp, fn, vector=False
    )
    fluid_temperature = ThermalBoundaryConditions(
        fixed_value_mask=(
            _patch_mask(fp, fn, "inlet") | _patch_mask(fp, fn, "coolingWall")
        ),
        fixed_reference_value=fluid_reference_temperature,
        zero_gradient_or_symmetry_mask=fluid_symmetry,
        inlet_outlet_mask=_patch_mask(fp, fn, "outlet"),
        inlet_reference_value=physical_conditions[:, 1:2].expand(-1, len(fp)),
        coupled_temperature_mask=fluid_interface,
    )
    solid_reference_temperature = _batch_face_values(
        physical_conditions, sp, sn, vector=False
    )
    solid_temperature = ThermalBoundaryConditions(
        fixed_value_mask=_patch_mask(sp, sn, "coolingWall"),
        fixed_reference_value=solid_reference_temperature,
        zero_gradient_or_symmetry_mask=(
            _patch_mask(sp, sn, "inlet")
            | _patch_mask(sp, sn, "outlet")
            | solid_symmetry
        ),
        inlet_outlet_mask=torch.zeros_like(sp, dtype=torch.bool),
        inlet_reference_value=physical_conditions[:, 1:2].expand(-1, len(sp)),
        coupled_temperature_mask=solid_interface,
    )
    return fluid_velocity, fluid_temperature, solid_temperature


def assemble_p418_regional_cht_residual(
    *,
    geometry: P418SubfaceGeometry,
    physical_conditions: torch.Tensor,
    fluid_velocity_m_s: torch.Tensor,
    fluid_pressure_pa: torch.Tensor,
    fluid_temperature_k: torch.Tensor,
    solid_temperature_k: torch.Tensor,
    fluid_boundary_pressure_pa: torch.Tensor | None = None,
    fluid_internal_mass_flux_kg_s: torch.Tensor | None = None,
    fluid_boundary_mass_flux_kg_s: torch.Tensor | None = None,
) -> SteadyChtResiduals:
    velocity_bc, fluid_temperature_bc, solid_temperature_bc = p418_boundary_conditions(
        geometry, physical_conditions
    )
    owner_pressure = fluid_pressure_pa[:, geometry.fluid_mesh.boundary_owner]
    outlet = _patch_mask(
        geometry.fluid_boundary_patch, geometry.fluid_patch_names, "outlet"
    )
    boundary_pressure = fluid_boundary_pressure_pa
    if boundary_pressure is None:
        boundary_pressure = torch.where(
            outlet.view(1, -1),
            physical_conditions[:, 3:4],
            owner_pressure,
        )
    solid_source = physical_conditions[:, 2:3].expand_as(solid_temperature_k)
    return assemble_steady_hccb_cht_residual(
        fluid_mesh=geometry.fluid_mesh,
        solid_mesh=geometry.solid_mesh,
        interface=geometry.interface,
        fluid_velocity_bc=velocity_bc,
        fluid_temperature_bc=fluid_temperature_bc,
        solid_temperature_bc=solid_temperature_bc,
        fluid_pressure_pa=fluid_pressure_pa,
        fluid_velocity_m_s=fluid_velocity_m_s,
        fluid_temperature_k=fluid_temperature_k,
        solid_temperature_k=solid_temperature_k,
        fluid_boundary_pressure_pa=boundary_pressure,
        solid_volumetric_heat_source_w_m3=solid_source,
        fluid_internal_mass_flux_override_kg_s=fluid_internal_mass_flux_kg_s,
        fluid_boundary_mass_flux_override_kg_s=fluid_boundary_mass_flux_kg_s,
    )


def _volume_mean(
    values: np.ndarray,
    volume: np.ndarray,
    regional_global: np.ndarray,
    selected_global: np.ndarray,
) -> np.ndarray:
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


def volume_average_reference_fields(
    *,
    geometry: P418SubfaceGeometry,
    topology: dict[str, np.ndarray],
    field: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    fluid_count = len(topology["fluid_cell_volume_m3"])
    fluid_parent = geometry.fine_to_regional_global[:fluid_count]
    solid_parent = geometry.fine_to_regional_global[fluid_count:]
    fluid_volume = topology["fluid_cell_volume_m3"].astype(np.float64)
    solid_volume = topology["solid_cell_volume_m3"].astype(np.float64)
    return {
        "fluid_velocity_m_s": _volume_mean(field["fluid_velocity_m_s"], fluid_volume, fluid_parent, geometry.fluid_global_region),
        "fluid_pressure_pa": _volume_mean(field["fluid_pressure_Pa"], fluid_volume, fluid_parent, geometry.fluid_global_region),
        "fluid_temperature_k": _volume_mean(field["fluid_temperature_K"], fluid_volume, fluid_parent, geometry.fluid_global_region),
        "solid_temperature_k": _volume_mean(field["solid_temperature_K"], solid_volume, solid_parent, geometry.solid_global_region),
    }


def conservation_metrics(
    *,
    residual: SteadyChtResiduals,
    geometry: P418SubfaceGeometry,
    inlet_mass_flow_kg_s: float,
    generated_heat_w: float,
) -> dict[str, float]:
    fluid_mass = residual.fluid_mass_kg_m3_s * geometry.fluid_mesh.cell_volume
    fluid_energy = residual.fluid_energy_w_m3 * geometry.fluid_mesh.cell_volume
    solid_energy = residual.solid_energy_w_m3 * geometry.solid_mesh.cell_volume
    mass_scale = max(abs(inlet_mass_flow_kg_s), np.finfo(np.float64).tiny)
    heat_scale = max(abs(generated_heat_w), np.finfo(np.float64).tiny)
    return {
        "global_mass_imbalance_over_inlet": float(torch.abs(fluid_mass.sum()).detach() / mass_scale),
        "local_mass_l1_over_two_inlet": float(torch.abs(fluid_mass).sum().detach() / (2.0 * mass_scale)),
        "global_fluid_plus_solid_energy_imbalance_over_generated_heat": float(torch.abs(fluid_energy.sum() + solid_energy.sum()).detach() / heat_scale),
        "local_fluid_plus_solid_energy_l1_over_generated_heat": float((torch.abs(fluid_energy).sum() + torch.abs(solid_energy).sum()).detach() / heat_scale),
        "maximum_interface_flux_sum_over_generated_heat": float(torch.max(torch.abs(residual.interface_flux_reciprocity_w)).detach() / heat_scale),
        "maximum_interface_temperature_jump_k": float(torch.max(torch.abs(residual.interface_temperature_jump_k)).detach()),
    }
