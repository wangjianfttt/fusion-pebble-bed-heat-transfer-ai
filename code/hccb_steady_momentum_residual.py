#!/usr/bin/env python3
"""Steady OpenFOAM-13-style momentum residual for pore-resolved HCCB flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hccb_multiregion_steady_cht_residual import RegionMesh, VelocityBoundaryConditions
from hccb_source_backed_thermophysical import (
    helium_density,
    helium_dynamic_viscosity,
)
from multiregion_finite_volume_balance import finite_volume_cell_vector_balance
from openfoam13_face_flux_reconstruction import (
    boundary_mass_flux,
    boundary_velocity_from_conditions,
    corrected_internal_sn_grad_vector,
    gauss_cell_gradient_vector,
    internal_mass_flux,
    linear_internal_face_interpolate,
    openfoam_boundary_delta_coeff,
    openfoam_linear_weights,
    openfoam_nonorthogonal_geometry,
    upwind_internal_face_interpolate,
)


@dataclass(frozen=True)
class PressureBoundaryConditions:
    fixed_value_mask: torch.Tensor
    fixed_reference_value_pa: torch.Tensor
    fixed_flux_or_zero_gradient_mask: torch.Tensor


@dataclass(frozen=True)
class SteadyMomentumResiduals:
    momentum_n_m3: torch.Tensor
    convection_n_m3: torch.Tensor
    pressure_n_m3: torch.Tensor
    viscous_n_m3: torch.Tensor
    internal_momentum_flux_n: torch.Tensor
    boundary_momentum_flux_n: torch.Tensor
    internal_mass_flux_kg_s: torch.Tensor
    boundary_mass_flux_kg_s: torch.Tensor
    absolute_pressure_pa: torch.Tensor
    boundary_absolute_pressure_pa: torch.Tensor
    density_kg_m3: torch.Tensor
    dynamic_viscosity_pa_s: torch.Tensor
    outlet_backflow_mask: torch.Tensor


def helium_absolute_pressure_from_prgh(
    *,
    pressure_rgh_pa: torch.Tensor,
    temperature_k: torch.Tensor,
    position_m: torch.Tensor,
    gravity_m_s2: torch.Tensor,
    reference_height_m: float = 0.0,
    reference_pressure_pa: float = 0.0,
) -> torch.Tensor:
    """Convert ``p_rgh`` to absolute pressure using P389 helium density.

    OpenFOAM 13 defines ``p_rgh = p - rho*gh - pRef`` and
    ``gh = g dot x + |g|*hRef``.  Because P389 is linear in absolute pressure,
    the relation can be solved exactly rather than iterated.
    """
    if pressure_rgh_pa.shape != temperature_k.shape:
        raise ValueError("p_rgh and temperature must have identical shapes")
    if pressure_rgh_pa.ndim != 2:
        raise ValueError("p_rgh and temperature must have [batch,location] shape")
    if position_m.shape != (pressure_rgh_pa.shape[1], 3):
        raise ValueError("position_m must have [location,3] shape")
    if gravity_m_s2.shape != (3,):
        raise ValueError("gravity_m_s2 must contain three components")
    if torch.any(~torch.isfinite(pressure_rgh_pa)) or torch.any(~torch.isfinite(temperature_k)):
        raise ValueError("p_rgh and temperature must be finite")
    if torch.any(temperature_k <= 0):
        raise ValueError("temperature must be positive")
    gh = (
        torch.sum(position_m * gravity_m_s2[None, :], dim=1)
        + torch.linalg.vector_norm(gravity_m_s2) * reference_height_m
    )
    density_per_pressure = 480.19 / (1.0e6 * temperature_k)
    denominator = 1.0 - density_per_pressure * gh[None, :]
    if torch.any(denominator <= 0):
        raise ValueError("p_rgh conversion produced a non-positive denominator")
    pressure = (pressure_rgh_pa + reference_pressure_pa) / denominator
    if torch.any(pressure <= 0) or torch.any(~torch.isfinite(pressure)):
        raise ValueError("converted absolute pressure must be finite and positive")
    return pressure


def pressure_boundary_from_conditions(
    *,
    owner_pressure_rgh_pa: torch.Tensor,
    conditions: PressureBoundaryConditions,
) -> torch.Tensor:
    """Evaluate fixed-value and zero-normal-gradient pressure boundaries."""
    if owner_pressure_rgh_pa.ndim != 2:
        raise ValueError("owner boundary pressure must have [batch,face] shape")
    n_faces = owner_pressure_rgh_pa.shape[1]
    masks = (conditions.fixed_value_mask, conditions.fixed_flux_or_zero_gradient_mask)
    if any(mask.shape != (n_faces,) or mask.dtype != torch.bool for mask in masks):
        raise ValueError("pressure boundary masks must be boolean [face] tensors")
    if torch.any(
        conditions.fixed_value_mask.to(torch.int8)
        + conditions.fixed_flux_or_zero_gradient_mask.to(torch.int8)
        != 1
    ):
        raise ValueError("each pressure boundary face must select exactly one branch")
    if conditions.fixed_reference_value_pa.shape != owner_pressure_rgh_pa.shape:
        raise ValueError("fixed pressure reference must have [batch,face] shape")
    return torch.where(
        conditions.fixed_value_mask[None, :],
        conditions.fixed_reference_value_pa,
        owner_pressure_rgh_pa,
    )


def _dev2_transpose_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Return OpenFOAM ``dev2(T(grad(U)))`` for ``[d/dx,U]`` gradients."""
    transpose = gradient.transpose(-1, -2)
    trace = torch.diagonal(transpose, dim1=-2, dim2=-1).sum(dim=-1)
    identity = torch.eye(3, dtype=gradient.dtype, device=gradient.device)
    return transpose - (2.0 / 3.0) * trace[..., None, None] * identity


def assemble_steady_momentum_from_properties(
    *,
    mesh: RegionMesh,
    velocity_m_s: torch.Tensor,
    boundary_velocity_m_s: torch.Tensor,
    pressure_pa: torch.Tensor,
    boundary_pressure_pa: torch.Tensor,
    density_kg_m3: torch.Tensor,
    boundary_density_kg_m3: torch.Tensor,
    dynamic_viscosity_pa_s: torch.Tensor,
    boundary_dynamic_viscosity_pa_s: torch.Tensor,
    internal_mass_flux_override_kg_s: torch.Tensor | None = None,
    boundary_mass_flux_override_kg_s: torch.Tensor | None = None,
    volumetric_momentum_source_n_m3: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Assemble convection, pressure and laminar viscous momentum terms."""
    if velocity_m_s.ndim != 3 or velocity_m_s.shape[2] != 3:
        raise ValueError("velocity must have [batch,cell,3] shape")
    batch, cells, _ = velocity_m_s.shape
    if boundary_velocity_m_s.shape != (batch, len(mesh.boundary_owner), 3):
        raise ValueError("boundary velocity has the wrong shape")
    for name, field in {
        "pressure": pressure_pa,
        "density": density_kg_m3,
        "dynamic viscosity": dynamic_viscosity_pa_s,
    }.items():
        if field.shape != (batch, cells):
            raise ValueError(f"{name} must have [batch,cell] shape")
    for name, field in {
        "boundary pressure": boundary_pressure_pa,
        "boundary density": boundary_density_kg_m3,
        "boundary dynamic viscosity": boundary_dynamic_viscosity_pa_s,
    }.items():
        if field.shape != (batch, len(mesh.boundary_owner)):
            raise ValueError(f"{name} must have [batch,boundary_face] shape")
    if torch.any(density_kg_m3 <= 0) or torch.any(boundary_density_kg_m3 <= 0):
        raise ValueError("density must be positive")
    if torch.any(dynamic_viscosity_pa_s <= 0) or torch.any(boundary_dynamic_viscosity_pa_s <= 0):
        raise ValueError("dynamic viscosity must be positive")

    weight = openfoam_linear_weights(
        cell_centroid=mesh.cell_centroid,
        face_centroid=mesh.internal_face_centroid,
        area_vector=mesh.internal_area_vector,
        owner=mesh.internal_owner,
        neighbour=mesh.internal_neighbour,
    )
    reconstructed_mass_internal = internal_mass_flux(
        density=density_kg_m3,
        velocity=velocity_m_s,
        area_vector=mesh.internal_area_vector,
        owner=mesh.internal_owner,
        neighbour=mesh.internal_neighbour,
        owner_weight=weight,
    )
    if internal_mass_flux_override_kg_s is None:
        mass_internal = reconstructed_mass_internal
    else:
        expected_shape = (batch, len(mesh.internal_owner))
        if internal_mass_flux_override_kg_s.shape != expected_shape:
            raise ValueError(
                "internal mass-flux override must have [batch,internal_face] shape"
            )
        if torch.any(~torch.isfinite(internal_mass_flux_override_kg_s)):
            raise ValueError("internal mass-flux override must be finite")
        mass_internal = internal_mass_flux_override_kg_s
    reconstructed_mass_boundary = boundary_mass_flux(
        density_face=boundary_density_kg_m3,
        velocity_face=boundary_velocity_m_s,
        outward_area_vector=mesh.boundary_area_vector,
    )
    if boundary_mass_flux_override_kg_s is None:
        mass_boundary = reconstructed_mass_boundary
    else:
        expected_shape = (batch, len(mesh.boundary_owner))
        if boundary_mass_flux_override_kg_s.shape != expected_shape:
            raise ValueError(
                "boundary mass-flux override must have [batch,boundary_face] shape"
            )
        if torch.any(~torch.isfinite(boundary_mass_flux_override_kg_s)):
            raise ValueError("boundary mass-flux override must be finite")
        mass_boundary = boundary_mass_flux_override_kg_s
    upwind_velocity = upwind_internal_face_interpolate(
        velocity_m_s, mass_internal, mesh.internal_owner, mesh.internal_neighbour
    )
    convection_internal = mass_internal[:, :, None] * upwind_velocity
    convection_boundary = mass_boundary[:, :, None] * boundary_velocity_m_s

    pressure_internal = linear_internal_face_interpolate(
        pressure_pa, mesh.internal_owner, mesh.internal_neighbour, weight
    )
    pressure_force_internal = pressure_internal[:, :, None] * mesh.internal_area_vector[None, :, :]
    pressure_force_boundary = boundary_pressure_pa[:, :, None] * mesh.boundary_area_vector[None, :, :]

    velocity_internal = linear_internal_face_interpolate(
        velocity_m_s, mesh.internal_owner, mesh.internal_neighbour, weight
    )
    velocity_gradient = gauss_cell_gradient_vector(
        cell_vector=velocity_m_s,
        internal_face_vector=velocity_internal,
        boundary_face_vector=boundary_velocity_m_s,
        internal_area_vector=mesh.internal_area_vector,
        boundary_area_vector=mesh.boundary_area_vector,
        internal_owner=mesh.internal_owner,
        internal_neighbour=mesh.internal_neighbour,
        boundary_owner=mesh.boundary_owner,
        cell_volume=mesh.cell_volume,
    )
    delta_coeff, correction_vector = openfoam_nonorthogonal_geometry(
        cell_centroid=mesh.cell_centroid,
        area_vector=mesh.internal_area_vector,
        owner=mesh.internal_owner,
        neighbour=mesh.internal_neighbour,
    )
    internal_sn_grad = corrected_internal_sn_grad_vector(
        cell_vector=velocity_m_s,
        cell_gradient=velocity_gradient,
        owner=mesh.internal_owner,
        neighbour=mesh.internal_neighbour,
        owner_weight=weight,
        nonorthogonal_delta_coeff=delta_coeff,
        correction_vector=correction_vector,
    )
    face_gradient = linear_internal_face_interpolate(
        velocity_gradient, mesh.internal_owner, mesh.internal_neighbour, weight
    )
    internal_area = torch.linalg.vector_norm(mesh.internal_area_vector, dim=1)
    internal_normal = mesh.internal_area_vector / internal_area[:, None]
    internal_dev2 = _dev2_transpose_gradient(face_gradient)
    internal_transpose_part = torch.einsum(
        "fi,bfij->bfj", internal_normal, internal_dev2
    )
    viscosity_internal = linear_internal_face_interpolate(
        dynamic_viscosity_pa_s,
        mesh.internal_owner,
        mesh.internal_neighbour,
        weight,
    )
    viscous_force_internal = (
        -viscosity_internal[:, :, None]
        * (internal_transpose_part + internal_sn_grad)
        * internal_area[None, :, None]
    )

    boundary_area = torch.linalg.vector_norm(mesh.boundary_area_vector, dim=1)
    boundary_normal = mesh.boundary_area_vector / boundary_area[:, None]
    boundary_delta = openfoam_boundary_delta_coeff(
        cell_centroid=mesh.cell_centroid,
        face_centroid=mesh.boundary_face_centroid,
        boundary_owner=mesh.boundary_owner,
    )
    boundary_sn_grad = boundary_delta[None, :, None] * (
        boundary_velocity_m_s - velocity_m_s[:, mesh.boundary_owner]
    )
    boundary_dev2 = _dev2_transpose_gradient(
        velocity_gradient[:, mesh.boundary_owner]
    )
    boundary_transpose_part = torch.einsum(
        "fi,bfij->bfj", boundary_normal, boundary_dev2
    )
    viscous_force_boundary = (
        -boundary_dynamic_viscosity_pa_s[:, :, None]
        * (boundary_transpose_part + boundary_sn_grad)
        * boundary_area[None, :, None]
    )

    convection_cell = finite_volume_cell_vector_balance(
        internal_face_flux=convection_internal,
        boundary_face_flux=convection_boundary,
        internal_owner=mesh.internal_owner,
        internal_neighbour=mesh.internal_neighbour,
        boundary_owner=mesh.boundary_owner,
        cell_volume=mesh.cell_volume,
    )
    pressure_cell = finite_volume_cell_vector_balance(
        internal_face_flux=pressure_force_internal,
        boundary_face_flux=pressure_force_boundary,
        internal_owner=mesh.internal_owner,
        internal_neighbour=mesh.internal_neighbour,
        boundary_owner=mesh.boundary_owner,
        cell_volume=mesh.cell_volume,
    )
    viscous_cell = finite_volume_cell_vector_balance(
        internal_face_flux=viscous_force_internal,
        boundary_face_flux=viscous_force_boundary,
        internal_owner=mesh.internal_owner,
        internal_neighbour=mesh.internal_neighbour,
        boundary_owner=mesh.boundary_owner,
        cell_volume=mesh.cell_volume,
    )
    source = (
        torch.zeros_like(convection_cell)
        if volumetric_momentum_source_n_m3 is None
        else volumetric_momentum_source_n_m3
    )
    if source.shape != convection_cell.shape:
        raise ValueError("momentum source must have [batch,cell,3] shape")
    total = convection_cell + pressure_cell + viscous_cell - source
    return {
        "momentum_n_m3": total,
        "convection_n_m3": convection_cell,
        "pressure_n_m3": pressure_cell,
        "viscous_n_m3": viscous_cell,
        "internal_momentum_flux_n": convection_internal + pressure_force_internal + viscous_force_internal,
        "boundary_momentum_flux_n": convection_boundary + pressure_force_boundary + viscous_force_boundary,
        "internal_mass_flux_kg_s": mass_internal,
        "reconstructed_internal_mass_flux_kg_s": reconstructed_mass_internal,
        "boundary_mass_flux_kg_s": mass_boundary,
        "reconstructed_boundary_mass_flux_kg_s": reconstructed_mass_boundary,
        "velocity_gradient_s_inv": velocity_gradient,
        "internal_viscous_force_n": viscous_force_internal,
        "boundary_viscous_force_n": viscous_force_boundary,
    }


def assemble_steady_hccb_momentum_residual(
    *,
    mesh: RegionMesh,
    velocity_bc: VelocityBoundaryConditions,
    pressure_bc: PressureBoundaryConditions,
    pressure_rgh_pa: torch.Tensor,
    velocity_m_s: torch.Tensor,
    temperature_k: torch.Tensor,
    boundary_temperature_k: torch.Tensor,
    gravity_m_s2: torch.Tensor,
    reference_height_m: float = 0.0,
    reference_pressure_pa: float = 0.0,
    fluid_internal_mass_flux_override_kg_s: torch.Tensor | None = None,
    fluid_boundary_mass_flux_override_kg_s: torch.Tensor | None = None,
    volumetric_momentum_source_n_m3: torch.Tensor | None = None,
) -> SteadyMomentumResiduals:
    """Assemble the dimensional steady HCCB momentum residual."""
    if pressure_rgh_pa.shape != temperature_k.shape:
        raise ValueError("p_rgh and temperature must have the same cell shape")
    if velocity_m_s.shape != (*pressure_rgh_pa.shape, 3):
        raise ValueError("velocity must have [batch,cell,3] shape")
    if boundary_temperature_k.shape != (pressure_rgh_pa.shape[0], len(mesh.boundary_owner)):
        raise ValueError("boundary temperature has the wrong shape")
    boundary_velocity, backflow = boundary_velocity_from_conditions(
        owner_velocity=velocity_m_s[:, mesh.boundary_owner],
        outward_area_vector=mesh.boundary_area_vector,
        fixed_value_mask=velocity_bc.fixed_value_mask,
        fixed_reference_value=velocity_bc.fixed_reference_value,
        pressure_inlet_outlet_mask=velocity_bc.pressure_inlet_outlet_mask,
        symmetry_or_empty_mask=velocity_bc.symmetry_or_empty_mask,
    )
    boundary_prgh = pressure_boundary_from_conditions(
        owner_pressure_rgh_pa=pressure_rgh_pa[:, mesh.boundary_owner],
        conditions=pressure_bc,
    )
    pressure = helium_absolute_pressure_from_prgh(
        pressure_rgh_pa=pressure_rgh_pa,
        temperature_k=temperature_k,
        position_m=mesh.cell_centroid,
        gravity_m_s2=gravity_m_s2,
        reference_height_m=reference_height_m,
        reference_pressure_pa=reference_pressure_pa,
    )
    boundary_pressure = helium_absolute_pressure_from_prgh(
        pressure_rgh_pa=boundary_prgh,
        temperature_k=boundary_temperature_k,
        position_m=mesh.boundary_face_centroid,
        gravity_m_s2=gravity_m_s2,
        reference_height_m=reference_height_m,
        reference_pressure_pa=reference_pressure_pa,
    )
    density = helium_density(pressure, temperature_k)
    boundary_density = helium_density(boundary_pressure, boundary_temperature_k)
    viscosity = helium_dynamic_viscosity(pressure, temperature_k)
    boundary_viscosity = helium_dynamic_viscosity(
        boundary_pressure, boundary_temperature_k
    )
    terms = assemble_steady_momentum_from_properties(
        mesh=mesh,
        velocity_m_s=velocity_m_s,
        boundary_velocity_m_s=boundary_velocity,
        pressure_pa=pressure,
        boundary_pressure_pa=boundary_pressure,
        density_kg_m3=density,
        boundary_density_kg_m3=boundary_density,
        dynamic_viscosity_pa_s=viscosity,
        boundary_dynamic_viscosity_pa_s=boundary_viscosity,
        internal_mass_flux_override_kg_s=fluid_internal_mass_flux_override_kg_s,
        boundary_mass_flux_override_kg_s=fluid_boundary_mass_flux_override_kg_s,
        volumetric_momentum_source_n_m3=volumetric_momentum_source_n_m3,
    )
    backflow = velocity_bc.pressure_inlet_outlet_mask[None, :] & (
        terms["boundary_mass_flux_kg_s"] < 0
    )
    return SteadyMomentumResiduals(
        momentum_n_m3=terms["momentum_n_m3"],
        convection_n_m3=terms["convection_n_m3"],
        pressure_n_m3=terms["pressure_n_m3"],
        viscous_n_m3=terms["viscous_n_m3"],
        internal_momentum_flux_n=terms["internal_momentum_flux_n"],
        boundary_momentum_flux_n=terms["boundary_momentum_flux_n"],
        internal_mass_flux_kg_s=terms["internal_mass_flux_kg_s"],
        boundary_mass_flux_kg_s=terms["boundary_mass_flux_kg_s"],
        absolute_pressure_pa=pressure,
        boundary_absolute_pressure_pa=boundary_pressure,
        density_kg_m3=density,
        dynamic_viscosity_pa_s=viscosity,
        outlet_backflow_mask=backflow,
    )
