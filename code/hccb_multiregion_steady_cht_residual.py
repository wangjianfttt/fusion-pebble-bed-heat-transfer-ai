#!/usr/bin/env python3
"""Steady HCCB fluid/solid mass and energy residuals on OpenFOAM faces.

This module combines source-backed thermophysical functions with the archived
OpenFOAM-13 interpolation, upwind convection, corrected normal gradient and
coupled-temperature interface formulas.  It returns dimensional residuals and
does not choose loss weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hccb_source_backed_thermophysical import (
    helium_density,
    helium_sensible_enthalpy,
    helium_thermal_conductivity,
    load_hccb_thermophysical_parameters,
    steady_li4sio4_conductivity_like,
)
from multiregion_finite_volume_balance import (
    finite_volume_cell_balance,
    interface_flux_reciprocity,
    interface_temperature_jump,
)
from openfoam13_face_flux_reconstruction import (
    boundary_conductive_heat_flux,
    boundary_mass_flux,
    boundary_scalar_from_conditions,
    boundary_upwind_enthalpy_flux,
    boundary_velocity_from_conditions,
    corrected_internal_sn_grad_scalar,
    coupled_temperature_interface,
    gauss_cell_gradient_scalar,
    internal_conductive_heat_flux,
    internal_mass_flux,
    internal_upwind_enthalpy_flux,
    linear_internal_face_interpolate,
    openfoam_boundary_delta_coeff,
    openfoam_linear_weights,
    openfoam_nonorthogonal_geometry,
)


@dataclass(frozen=True)
class RegionMesh:
    cell_centroid: torch.Tensor
    cell_volume: torch.Tensor
    internal_face_centroid: torch.Tensor
    internal_area_vector: torch.Tensor
    internal_owner: torch.Tensor
    internal_neighbour: torch.Tensor
    boundary_face_centroid: torch.Tensor
    boundary_area_vector: torch.Tensor
    boundary_owner: torch.Tensor


@dataclass(frozen=True)
class VelocityBoundaryConditions:
    fixed_value_mask: torch.Tensor
    fixed_reference_value: torch.Tensor
    pressure_inlet_outlet_mask: torch.Tensor
    symmetry_or_empty_mask: torch.Tensor


@dataclass(frozen=True)
class ThermalBoundaryConditions:
    fixed_value_mask: torch.Tensor
    fixed_reference_value: torch.Tensor
    zero_gradient_or_symmetry_mask: torch.Tensor
    inlet_outlet_mask: torch.Tensor
    inlet_reference_value: torch.Tensor
    coupled_temperature_mask: torch.Tensor


@dataclass(frozen=True)
class CoupledInterfaceMap:
    fluid_boundary_face: torch.Tensor
    solid_boundary_face: torch.Tensor


@dataclass(frozen=True)
class SteadyChtResiduals:
    fluid_mass_kg_m3_s: torch.Tensor
    fluid_energy_w_m3: torch.Tensor
    solid_energy_w_m3: torch.Tensor
    interface_flux_reciprocity_w: torch.Tensor
    interface_temperature_jump_k: torch.Tensor
    outlet_backflow_mask: torch.Tensor
    fluid_internal_mass_flux_kg_s: torch.Tensor
    fluid_boundary_mass_flux_kg_s: torch.Tensor
    fluid_internal_energy_flux_w: torch.Tensor
    fluid_boundary_energy_flux_w: torch.Tensor
    solid_internal_heat_flux_w: torch.Tensor
    solid_boundary_heat_flux_w: torch.Tensor
    interface_temperature_k: torch.Tensor
    fluid_boundary_temperature_k: torch.Tensor
    solid_boundary_temperature_k: torch.Tensor


def _thermal_boundary_temperature(
    *,
    owner_temperature: torch.Tensor,
    switching_flux: torch.Tensor,
    conditions: ThermalBoundaryConditions,
) -> torch.Tensor:
    if conditions.coupled_temperature_mask.dtype != torch.bool:
        raise ValueError("coupled_temperature_mask must be boolean")
    if conditions.coupled_temperature_mask.shape != (owner_temperature.shape[1],):
        raise ValueError("coupled_temperature_mask must contain one value per boundary face")
    overlap = conditions.coupled_temperature_mask & (
        conditions.fixed_value_mask
        | conditions.zero_gradient_or_symmetry_mask
        | conditions.inlet_outlet_mask
    )
    if torch.any(overlap):
        raise ValueError("coupled-temperature faces cannot select another thermal branch")
    return boundary_scalar_from_conditions(
        owner_scalar=owner_temperature,
        boundary_flux=switching_flux,
        fixed_value_mask=conditions.fixed_value_mask,
        fixed_reference_value=conditions.fixed_reference_value,
        zero_gradient_or_symmetry_mask=(
            conditions.zero_gradient_or_symmetry_mask
            | conditions.coupled_temperature_mask
        ),
        inlet_outlet_mask=conditions.inlet_outlet_mask,
        inlet_reference_value=conditions.inlet_reference_value,
    )


def assemble_steady_hccb_cht_residual(
    *,
    fluid_mesh: RegionMesh,
    solid_mesh: RegionMesh,
    interface: CoupledInterfaceMap,
    fluid_velocity_bc: VelocityBoundaryConditions,
    fluid_temperature_bc: ThermalBoundaryConditions,
    solid_temperature_bc: ThermalBoundaryConditions,
    fluid_pressure_pa: torch.Tensor,
    fluid_velocity_m_s: torch.Tensor,
    fluid_temperature_k: torch.Tensor,
    solid_temperature_k: torch.Tensor,
    fluid_boundary_pressure_pa: torch.Tensor,
    solid_volumetric_heat_source_w_m3: torch.Tensor,
    fluid_volumetric_heat_source_w_m3: torch.Tensor | None = None,
    fluid_internal_mass_flux_override_kg_s: torch.Tensor | None = None,
    fluid_boundary_mass_flux_override_kg_s: torch.Tensor | None = None,
) -> SteadyChtResiduals:
    """Assemble dimensional steady mass and energy residuals.

    ``fluid_boundary_pressure_pa`` is the already evaluated absolute pressure
    on each boundary face.  Conversion from ``p_rgh`` is intentionally left to
    the case adapter because it depends on gravity and the pressure reference.
    """
    batch, fluid_cells = fluid_temperature_k.shape
    if fluid_pressure_pa.shape != (batch, fluid_cells):
        raise ValueError("fluid pressure and temperature shapes differ")
    if fluid_velocity_m_s.shape != (batch, fluid_cells, 3):
        raise ValueError("fluid velocity must have [batch,fluid_cell,3] shape")
    if solid_temperature_k.ndim != 2 or solid_temperature_k.shape[0] != batch:
        raise ValueError("solid temperature must have [batch,solid_cell] shape")
    if solid_volumetric_heat_source_w_m3.shape != solid_temperature_k.shape:
        raise ValueError("solid heat source must match solid temperature shape")
    fluid_boundary_count = len(fluid_mesh.boundary_owner)
    solid_boundary_count = len(solid_mesh.boundary_owner)
    if fluid_boundary_pressure_pa.shape != (batch, fluid_boundary_count):
        raise ValueError("fluid boundary pressure must have [batch,boundary_face] shape")
    if fluid_boundary_mass_flux_override_kg_s is not None:
        expected_shape = (batch, fluid_boundary_count)
        if fluid_boundary_mass_flux_override_kg_s.shape != expected_shape:
            raise ValueError(
                "fluid boundary mass-flux override must have "
                f"shape {expected_shape}"
            )
        if not torch.all(torch.isfinite(fluid_boundary_mass_flux_override_kg_s)):
            raise ValueError("fluid boundary mass-flux override contains non-finite values")
    if interface.fluid_boundary_face.dtype != torch.long or interface.solid_boundary_face.dtype != torch.long:
        raise ValueError("interface face maps must use torch.long indices")
    if interface.fluid_boundary_face.shape != interface.solid_boundary_face.shape:
        raise ValueError("fluid and solid interface maps must have equal length")
    if interface.fluid_boundary_face.numel() and (
        int(interface.fluid_boundary_face.min()) < 0
        or int(interface.fluid_boundary_face.max()) >= fluid_boundary_count
        or int(interface.solid_boundary_face.min()) < 0
        or int(interface.solid_boundary_face.max()) >= solid_boundary_count
    ):
        raise ValueError("interface map contains an out-of-range boundary face")
    if not torch.all(fluid_temperature_bc.coupled_temperature_mask[interface.fluid_boundary_face]):
        raise ValueError("all mapped fluid interface faces must use coupled temperature")
    if not torch.all(solid_temperature_bc.coupled_temperature_mask[interface.solid_boundary_face]):
        raise ValueError("all mapped solid interface faces must use coupled temperature")

    params = load_hccb_thermophysical_parameters()
    fluid_density = helium_density(fluid_pressure_pa, fluid_temperature_k)
    fluid_conductivity = helium_thermal_conductivity(
        fluid_pressure_pa, fluid_temperature_k
    )
    fluid_enthalpy = helium_sensible_enthalpy(
        fluid_temperature_k, parameters=params
    )
    solid_conductivity = steady_li4sio4_conductivity_like(
        solid_temperature_k, parameters=params
    )

    fluid_weight = openfoam_linear_weights(
        cell_centroid=fluid_mesh.cell_centroid,
        face_centroid=fluid_mesh.internal_face_centroid,
        area_vector=fluid_mesh.internal_area_vector,
        owner=fluid_mesh.internal_owner,
        neighbour=fluid_mesh.internal_neighbour,
    )
    solid_weight = openfoam_linear_weights(
        cell_centroid=solid_mesh.cell_centroid,
        face_centroid=solid_mesh.internal_face_centroid,
        area_vector=solid_mesh.internal_area_vector,
        owner=solid_mesh.internal_owner,
        neighbour=solid_mesh.internal_neighbour,
    )

    fluid_boundary_owner_velocity = fluid_velocity_m_s[:, fluid_mesh.boundary_owner]
    fluid_boundary_velocity, backflow = boundary_velocity_from_conditions(
        owner_velocity=fluid_boundary_owner_velocity,
        outward_area_vector=fluid_mesh.boundary_area_vector,
        fixed_value_mask=fluid_velocity_bc.fixed_value_mask,
        fixed_reference_value=fluid_velocity_bc.fixed_reference_value,
        pressure_inlet_outlet_mask=fluid_velocity_bc.pressure_inlet_outlet_mask,
        symmetry_or_empty_mask=fluid_velocity_bc.symmetry_or_empty_mask,
    )
    volumetric_switching_flux = torch.sum(
        fluid_boundary_velocity * fluid_mesh.boundary_area_vector[None, :, :], dim=2
    )
    thermal_switching_flux = (
        volumetric_switching_flux
        if fluid_boundary_mass_flux_override_kg_s is None
        else fluid_boundary_mass_flux_override_kg_s
    )
    fluid_boundary_owner_temperature = fluid_temperature_k[:, fluid_mesh.boundary_owner]
    fluid_boundary_temperature = _thermal_boundary_temperature(
        owner_temperature=fluid_boundary_owner_temperature,
        switching_flux=thermal_switching_flux,
        conditions=fluid_temperature_bc,
    )
    solid_boundary_owner_temperature = solid_temperature_k[:, solid_mesh.boundary_owner]
    solid_zero_flux = solid_temperature_k.new_zeros((batch, solid_boundary_count))
    solid_boundary_temperature = _thermal_boundary_temperature(
        owner_temperature=solid_boundary_owner_temperature,
        switching_flux=solid_zero_flux,
        conditions=solid_temperature_bc,
    )

    fluid_interface_face = interface.fluid_boundary_face
    solid_interface_face = interface.solid_boundary_face
    fluid_interface_owner = fluid_mesh.boundary_owner[fluid_interface_face]
    solid_interface_owner = solid_mesh.boundary_owner[solid_interface_face]
    interface_temperature, fluid_interface_heat, solid_interface_heat = (
        coupled_temperature_interface(
            fluid_cell_temperature=fluid_temperature_k[:, fluid_interface_owner],
            solid_cell_temperature=solid_temperature_k[:, solid_interface_owner],
            fluid_conductivity=fluid_conductivity[:, fluid_interface_owner],
            solid_conductivity=solid_conductivity[:, solid_interface_owner],
            fluid_cell_centroid=fluid_mesh.cell_centroid[fluid_interface_owner],
            solid_cell_centroid=solid_mesh.cell_centroid[solid_interface_owner],
            face_centroid=fluid_mesh.boundary_face_centroid[fluid_interface_face],
            fluid_outward_area_vector=fluid_mesh.boundary_area_vector[fluid_interface_face],
        )
    )
    fluid_boundary_temperature = fluid_boundary_temperature.clone()
    solid_boundary_temperature = solid_boundary_temperature.clone()
    fluid_boundary_temperature[:, fluid_interface_face] = interface_temperature
    solid_boundary_temperature[:, solid_interface_face] = interface_temperature

    fluid_boundary_density = helium_density(
        fluid_boundary_pressure_pa, fluid_boundary_temperature
    )
    reconstructed_fluid_internal_mass = internal_mass_flux(
        density=fluid_density,
        velocity=fluid_velocity_m_s,
        area_vector=fluid_mesh.internal_area_vector,
        owner=fluid_mesh.internal_owner,
        neighbour=fluid_mesh.internal_neighbour,
        owner_weight=fluid_weight,
    )
    if fluid_internal_mass_flux_override_kg_s is None:
        fluid_internal_mass = reconstructed_fluid_internal_mass
    else:
        expected_shape = (batch, len(fluid_mesh.internal_owner))
        if fluid_internal_mass_flux_override_kg_s.shape != expected_shape:
            raise ValueError(
                "fluid internal mass-flux override must have "
                f"shape {expected_shape}"
            )
        if not torch.all(torch.isfinite(fluid_internal_mass_flux_override_kg_s)):
            raise ValueError("fluid internal mass-flux override contains non-finite values")
        fluid_internal_mass = fluid_internal_mass_flux_override_kg_s
    reconstructed_fluid_boundary_mass = boundary_mass_flux(
        density_face=fluid_boundary_density,
        velocity_face=fluid_boundary_velocity,
        outward_area_vector=fluid_mesh.boundary_area_vector,
    )
    fluid_boundary_mass = (
        reconstructed_fluid_boundary_mass
        if fluid_boundary_mass_flux_override_kg_s is None
        else fluid_boundary_mass_flux_override_kg_s
    )
    backflow = fluid_velocity_bc.pressure_inlet_outlet_mask[None, :] & (
        fluid_boundary_mass < 0
    )

    fluid_internal_temperature = linear_internal_face_interpolate(
        fluid_temperature_k,
        fluid_mesh.internal_owner,
        fluid_mesh.internal_neighbour,
        fluid_weight,
    )
    fluid_gradient = gauss_cell_gradient_scalar(
        cell_scalar=fluid_temperature_k,
        internal_face_scalar=fluid_internal_temperature,
        boundary_face_scalar=fluid_boundary_temperature,
        internal_area_vector=fluid_mesh.internal_area_vector,
        boundary_area_vector=fluid_mesh.boundary_area_vector,
        internal_owner=fluid_mesh.internal_owner,
        internal_neighbour=fluid_mesh.internal_neighbour,
        boundary_owner=fluid_mesh.boundary_owner,
        cell_volume=fluid_mesh.cell_volume,
    )
    fluid_delta, fluid_correction = openfoam_nonorthogonal_geometry(
        cell_centroid=fluid_mesh.cell_centroid,
        area_vector=fluid_mesh.internal_area_vector,
        owner=fluid_mesh.internal_owner,
        neighbour=fluid_mesh.internal_neighbour,
    )
    fluid_sn_grad = corrected_internal_sn_grad_scalar(
        cell_scalar=fluid_temperature_k,
        cell_gradient=fluid_gradient,
        owner=fluid_mesh.internal_owner,
        neighbour=fluid_mesh.internal_neighbour,
        owner_weight=fluid_weight,
        nonorthogonal_delta_coeff=fluid_delta,
        correction_vector=fluid_correction,
    )
    fluid_internal_conduction = internal_conductive_heat_flux(
        conductivity=fluid_conductivity,
        corrected_sn_grad=fluid_sn_grad,
        face_area=torch.linalg.vector_norm(fluid_mesh.internal_area_vector, dim=1),
        owner=fluid_mesh.internal_owner,
        neighbour=fluid_mesh.internal_neighbour,
        owner_weight=fluid_weight,
    )
    fluid_internal_advection = internal_upwind_enthalpy_flux(
        mass_flux=fluid_internal_mass,
        enthalpy=fluid_enthalpy,
        owner=fluid_mesh.internal_owner,
        neighbour=fluid_mesh.internal_neighbour,
    )
    fluid_internal_energy = fluid_internal_advection + fluid_internal_conduction

    fluid_boundary_enthalpy_owner = fluid_enthalpy[:, fluid_mesh.boundary_owner]
    fluid_boundary_inlet_enthalpy = helium_sensible_enthalpy(
        fluid_temperature_bc.inlet_reference_value, parameters=params
    )
    fluid_boundary_advection = boundary_upwind_enthalpy_flux(
        mass_flux=fluid_boundary_mass,
        owner_enthalpy=fluid_boundary_enthalpy_owner,
        inlet_enthalpy=fluid_boundary_inlet_enthalpy,
    )
    fluid_boundary_conductivity = helium_thermal_conductivity(
        fluid_boundary_pressure_pa, fluid_boundary_temperature
    )
    fluid_boundary_fixed_or_inflow = (
        fluid_temperature_bc.fixed_value_mask[None, :]
        | (
            fluid_temperature_bc.inlet_outlet_mask[None, :]
            & (fluid_boundary_mass < 0)
        )
        | fluid_temperature_bc.coupled_temperature_mask[None, :]
    )
    fluid_boundary_conduction = boundary_conductive_heat_flux(
        owner_temperature=fluid_boundary_owner_temperature,
        face_temperature=fluid_boundary_temperature,
        conductivity_face=fluid_boundary_conductivity,
        face_area=torch.linalg.vector_norm(fluid_mesh.boundary_area_vector, dim=1),
        boundary_delta_coeff=openfoam_boundary_delta_coeff(
            cell_centroid=fluid_mesh.cell_centroid,
            face_centroid=fluid_mesh.boundary_face_centroid,
            boundary_owner=fluid_mesh.boundary_owner,
        ),
        fixed_or_inflow_mask=fluid_boundary_fixed_or_inflow,
    )
    fluid_boundary_energy = fluid_boundary_advection + fluid_boundary_conduction
    fluid_boundary_energy = fluid_boundary_energy.clone()
    fluid_boundary_energy[:, fluid_interface_face] = (
        fluid_boundary_advection[:, fluid_interface_face] + fluid_interface_heat
    )

    solid_internal_temperature = linear_internal_face_interpolate(
        solid_temperature_k,
        solid_mesh.internal_owner,
        solid_mesh.internal_neighbour,
        solid_weight,
    )
    solid_gradient = gauss_cell_gradient_scalar(
        cell_scalar=solid_temperature_k,
        internal_face_scalar=solid_internal_temperature,
        boundary_face_scalar=solid_boundary_temperature,
        internal_area_vector=solid_mesh.internal_area_vector,
        boundary_area_vector=solid_mesh.boundary_area_vector,
        internal_owner=solid_mesh.internal_owner,
        internal_neighbour=solid_mesh.internal_neighbour,
        boundary_owner=solid_mesh.boundary_owner,
        cell_volume=solid_mesh.cell_volume,
    )
    solid_delta, solid_correction = openfoam_nonorthogonal_geometry(
        cell_centroid=solid_mesh.cell_centroid,
        area_vector=solid_mesh.internal_area_vector,
        owner=solid_mesh.internal_owner,
        neighbour=solid_mesh.internal_neighbour,
    )
    solid_sn_grad = corrected_internal_sn_grad_scalar(
        cell_scalar=solid_temperature_k,
        cell_gradient=solid_gradient,
        owner=solid_mesh.internal_owner,
        neighbour=solid_mesh.internal_neighbour,
        owner_weight=solid_weight,
        nonorthogonal_delta_coeff=solid_delta,
        correction_vector=solid_correction,
    )
    solid_internal_heat = internal_conductive_heat_flux(
        conductivity=solid_conductivity,
        corrected_sn_grad=solid_sn_grad,
        face_area=torch.linalg.vector_norm(solid_mesh.internal_area_vector, dim=1),
        owner=solid_mesh.internal_owner,
        neighbour=solid_mesh.internal_neighbour,
        owner_weight=solid_weight,
    )
    solid_boundary_fixed = (
        solid_temperature_bc.fixed_value_mask[None, :]
        | solid_temperature_bc.coupled_temperature_mask[None, :]
    ).expand(solid_temperature_k.shape[0], -1)
    solid_boundary_conductivity = steady_li4sio4_conductivity_like(
        solid_boundary_temperature, parameters=params
    )
    solid_boundary_heat = boundary_conductive_heat_flux(
        owner_temperature=solid_boundary_owner_temperature,
        face_temperature=solid_boundary_temperature,
        conductivity_face=solid_boundary_conductivity,
        face_area=torch.linalg.vector_norm(solid_mesh.boundary_area_vector, dim=1),
        boundary_delta_coeff=openfoam_boundary_delta_coeff(
            cell_centroid=solid_mesh.cell_centroid,
            face_centroid=solid_mesh.boundary_face_centroid,
            boundary_owner=solid_mesh.boundary_owner,
        ),
        fixed_or_inflow_mask=solid_boundary_fixed,
    )
    solid_boundary_heat = solid_boundary_heat.clone()
    solid_boundary_heat[:, solid_interface_face] = solid_interface_heat

    fluid_source = fluid_volumetric_heat_source_w_m3
    if fluid_source is None:
        fluid_source = fluid_temperature_k.new_zeros(fluid_temperature_k.shape)
    if fluid_source.shape != fluid_temperature_k.shape:
        raise ValueError("fluid heat source must match fluid temperature shape")

    fluid_mass_residual = finite_volume_cell_balance(
        internal_face_flux=fluid_internal_mass,
        boundary_face_flux=fluid_boundary_mass,
        internal_owner=fluid_mesh.internal_owner,
        internal_neighbour=fluid_mesh.internal_neighbour,
        boundary_owner=fluid_mesh.boundary_owner,
        cell_volume=fluid_mesh.cell_volume,
    )
    fluid_energy_residual = finite_volume_cell_balance(
        internal_face_flux=fluid_internal_energy,
        boundary_face_flux=fluid_boundary_energy,
        internal_owner=fluid_mesh.internal_owner,
        internal_neighbour=fluid_mesh.internal_neighbour,
        boundary_owner=fluid_mesh.boundary_owner,
        cell_volume=fluid_mesh.cell_volume,
        volumetric_source=fluid_source,
    )
    solid_energy_residual = finite_volume_cell_balance(
        internal_face_flux=solid_internal_heat,
        boundary_face_flux=solid_boundary_heat,
        internal_owner=solid_mesh.internal_owner,
        internal_neighbour=solid_mesh.internal_neighbour,
        boundary_owner=solid_mesh.boundary_owner,
        cell_volume=solid_mesh.cell_volume,
        volumetric_source=solid_volumetric_heat_source_w_m3,
    )

    return SteadyChtResiduals(
        fluid_mass_kg_m3_s=fluid_mass_residual,
        fluid_energy_w_m3=fluid_energy_residual,
        solid_energy_w_m3=solid_energy_residual,
        interface_flux_reciprocity_w=interface_flux_reciprocity(
            fluid_interface_heat, solid_interface_heat
        ),
        interface_temperature_jump_k=interface_temperature_jump(
            interface_temperature, interface_temperature
        ),
        outlet_backflow_mask=backflow,
        fluid_internal_mass_flux_kg_s=fluid_internal_mass,
        fluid_boundary_mass_flux_kg_s=fluid_boundary_mass,
        fluid_internal_energy_flux_w=fluid_internal_energy,
        fluid_boundary_energy_flux_w=fluid_boundary_energy,
        solid_internal_heat_flux_w=solid_internal_heat,
        solid_boundary_heat_flux_w=solid_boundary_heat,
        interface_temperature_k=interface_temperature,
        fluid_boundary_temperature_k=fluid_boundary_temperature,
        solid_boundary_temperature_k=solid_boundary_temperature,
    )
