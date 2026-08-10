#!/usr/bin/env python3
"""Transient regional energy residual for the P418 thermal-step model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hccb_p418_regional_cht_adapter import (
    P418SubfaceGeometry,
    assemble_p418_regional_cht_residual,
    p418_boundary_conditions,
)
from hccb_source_backed_thermophysical import (
    helium_density,
    helium_sensible_enthalpy,
    li4sio4_sensible_internal_energy,
    load_hccb_thermophysical_parameters,
)
from multiregion_finite_volume_balance import finite_volume_cell_balance
from openfoam13_face_flux_reconstruction import (
    boundary_upwind_enthalpy_flux,
    boundary_velocity_from_conditions,
    internal_upwind_enthalpy_flux,
)


TRANSIENT_STORAGE_PARAMETER_IDS = ("P388", "P389", "P403", "P428", "P429", "P430", "P431")


@dataclass(frozen=True)
class P418TransientRegionalResidual:
    fluid_energy_w_m3: torch.Tensor
    solid_energy_w_m3: torch.Tensor
    fluid_storage_w_m3: torch.Tensor
    solid_storage_w_m3: torch.Tensor
    fluid_enthalpy_storage_w_m3: torch.Tensor
    fluid_kinetic_storage_w_m3: torch.Tensor
    fluid_pressure_work_w_m3: torch.Tensor
    fluid_kinetic_advection_w_m3: torch.Tensor
    fluid_steady_energy_w_m3: torch.Tensor
    solid_steady_energy_w_m3: torch.Tensor
    fluid_mass_kg_m3_s: torch.Tensor
    interface_flux_reciprocity_w: torch.Tensor
    fluid_internal_energy_flux_w: torch.Tensor
    solid_internal_heat_flux_w: torch.Tensor


@dataclass(frozen=True)
class P418FluidStorageTerms:
    """Conservative OpenFOAM-13 fluid storage terms in W/m3."""

    total_w_m3: torch.Tensor
    enthalpy_w_m3: torch.Tensor
    kinetic_w_m3: torch.Tensor
    pressure_work_w_m3: torch.Tensor


def _expanded_fixed_face_flux(
    values: torch.Tensor | None,
    *,
    batch: int,
    time_count: int,
    face_count: int,
    name: str,
) -> torch.Tensor | None:
    if values is None:
        return None
    if values.shape == (batch, face_count):
        values = values[:, None, :].expand(-1, time_count, -1)
    if values.shape != (batch, time_count, face_count):
        raise ValueError(
            f"{name} must have [batch,face] or [batch,time,face] shape"
        )
    if torch.any(~torch.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values.reshape(batch * time_count, face_count)


def time_derivative(values: torch.Tensor, time_s: torch.Tensor) -> torch.Tensor:
    """Differentiate reported histories on a strictly increasing time grid.

    Three-point, second-order formulas are used for irregularly spaced times.
    With only two reported times, the only available first-order slope is used.
    """
    if values.ndim < 2:
        raise ValueError("time-dependent values must have shape [batch,time,...]")
    batch, count = values.shape[:2]
    if count < 2:
        raise ValueError("at least two reported times are required")
    if time_s.ndim == 1:
        time_s = time_s.unsqueeze(0).expand(batch, -1)
    if time_s.shape != (batch, count):
        raise ValueError("time must have shape [time] or [batch,time]")
    interval = time_s[:, 1:] - time_s[:, :-1]
    if torch.any(interval <= 0):
        raise ValueError("reported times must be strictly increasing")
    coefficient_shape = (batch,) + (1,) * (values.ndim - 2)
    derivative = torch.empty_like(values)
    if count == 2:
        slope = (values[:, 1] - values[:, 0]) / interval[:, 0].reshape(
            coefficient_shape
        )
        derivative[:, 0] = slope
        derivative[:, 1] = slope
        return derivative

    interval_shape = (batch, count - 1) + (1,) * (values.ndim - 2)
    slope = (values[:, 1:] - values[:, :-1]) / interval.reshape(interval_shape)
    left = interval[:, :-1]
    right = interval[:, 1:]
    interior_shape = (batch, count - 2) + (1,) * (values.ndim - 2)
    derivative[:, 1:-1] = (
        (right / (left + right)).reshape(interior_shape) * slope[:, :-1]
        + (left / (left + right)).reshape(interior_shape) * slope[:, 1:]
    )

    first_left = interval[:, 0]
    first_right = interval[:, 1]
    derivative[:, 0] = (
        ((2.0 * first_left + first_right) / (first_left + first_right)).reshape(
            coefficient_shape
        )
        * slope[:, 0]
        - (first_left / (first_left + first_right)).reshape(coefficient_shape)
        * slope[:, 1]
    )

    last_left = interval[:, -2]
    last_right = interval[:, -1]
    derivative[:, -1] = (
        ((last_left + 2.0 * last_right) / (last_left + last_right)).reshape(
            coefficient_shape
        )
        * slope[:, -1]
        - (last_right / (last_left + last_right)).reshape(coefficient_shape)
        * slope[:, -2]
    )
    return derivative


def conservative_fluid_storage_terms(
    fluid_state: torch.Tensor,
    time_s: torch.Tensor,
) -> P418FluidStorageTerms:
    """Evaluate ``d(rho*h)/dt + d(rho*K)/dt - dp/dt``.

    ``fluid_state`` follows the project state order ``[Ux, Uy, Uz, p, T]`` and
    has shape ``[batch,time,node,5]``.  The expression is the storage and
    pressure-work part of the OpenFOAM-13 sensible-enthalpy equation used by
    the formal thermal-step cases.
    """
    if fluid_state.ndim != 4 or fluid_state.shape[-1] != 5:
        raise ValueError("fluid state must have shape [batch,time,node,5]")
    pressure = fluid_state[..., 3]
    temperature = fluid_state[..., 4]
    density = helium_density(pressure, temperature)
    enthalpy = helium_sensible_enthalpy(temperature)
    kinetic_energy = 0.5 * fluid_state[..., :3].square().sum(dim=-1)
    enthalpy_storage = time_derivative(density * enthalpy, time_s)
    kinetic_storage = time_derivative(density * kinetic_energy, time_s)
    pressure_work = -time_derivative(pressure, time_s)
    return P418FluidStorageTerms(
        total_w_m3=enthalpy_storage + kinetic_storage + pressure_work,
        enthalpy_w_m3=enthalpy_storage,
        kinetic_w_m3=kinetic_storage,
        pressure_work_w_m3=pressure_work,
    )


def solid_internal_energy_storage(
    solid_temperature_k: torch.Tensor,
    time_s: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``rho_s*d(e_s)/dt`` from the P428--P431 Li4SiO4 relation."""
    if solid_temperature_k.ndim < 3:
        raise ValueError("solid temperature must have shape [batch,time,node,...]")
    params = load_hccb_thermophysical_parameters()
    specific_internal_energy = li4sio4_sensible_internal_energy(
        solid_temperature_k, parameters=params
    )
    return params.solid_density_kg_m3 * time_derivative(
        specific_internal_energy, time_s
    )


def fluid_kinetic_advection_from_mass_flux(
    *,
    geometry: P418SubfaceGeometry,
    physical_conditions: torch.Tensor,
    fluid_state: torch.Tensor,
    internal_mass_flux_kg_s: torch.Tensor,
    boundary_mass_flux_kg_s: torch.Tensor,
) -> torch.Tensor:
    """Return ``div(phi*K)`` using the same face flux as mass and enthalpy.

    Inputs are flattened over any leading sample/time dimension.  The fluid
    state order is ``[Ux, Uy, Uz, p, T]`` and both mass-flux arrays retain the
    OpenFOAM owner-to-neighbour/outward sign convention.
    """
    if fluid_state.ndim != 3 or fluid_state.shape[-1] != 5:
        raise ValueError("fluid state must have shape [sample,cell,5]")
    sample_count, fluid_count, _ = fluid_state.shape
    if fluid_count != len(geometry.fluid_mesh.cell_volume):
        raise ValueError("fluid state and geometry have different cell counts")
    if physical_conditions.shape != (sample_count, 5):
        raise ValueError("physical conditions must have shape [sample,5]")
    if internal_mass_flux_kg_s.shape != (
        sample_count,
        len(geometry.fluid_mesh.internal_owner),
    ):
        raise ValueError("internal mass flux has the wrong shape")
    if boundary_mass_flux_kg_s.shape != (
        sample_count,
        len(geometry.fluid_mesh.boundary_owner),
    ):
        raise ValueError("boundary mass flux has the wrong shape")
    if any(
        torch.any(~torch.isfinite(values))
        for values in (
            fluid_state,
            physical_conditions,
            internal_mass_flux_kg_s,
            boundary_mass_flux_kg_s,
        )
    ):
        raise ValueError("kinetic-advection inputs must be finite")

    kinetic_energy = 0.5 * fluid_state[..., :3].square().sum(dim=-1)
    internal_flux = internal_upwind_enthalpy_flux(
        mass_flux=internal_mass_flux_kg_s,
        enthalpy=kinetic_energy,
        owner=geometry.fluid_mesh.internal_owner,
        neighbour=geometry.fluid_mesh.internal_neighbour,
    )
    velocity_bc, _, _ = p418_boundary_conditions(geometry, physical_conditions)
    boundary_owner_velocity = fluid_state[:, geometry.fluid_mesh.boundary_owner, :3]
    boundary_velocity, _ = boundary_velocity_from_conditions(
        owner_velocity=boundary_owner_velocity,
        outward_area_vector=geometry.fluid_mesh.boundary_area_vector,
        fixed_value_mask=velocity_bc.fixed_value_mask,
        fixed_reference_value=velocity_bc.fixed_reference_value,
        pressure_inlet_outlet_mask=velocity_bc.pressure_inlet_outlet_mask,
        symmetry_or_empty_mask=velocity_bc.symmetry_or_empty_mask,
    )
    boundary_kinetic = 0.5 * boundary_velocity.square().sum(dim=-1)
    boundary_flux = boundary_upwind_enthalpy_flux(
        mass_flux=boundary_mass_flux_kg_s,
        owner_enthalpy=kinetic_energy[:, geometry.fluid_mesh.boundary_owner],
        inlet_enthalpy=boundary_kinetic,
    )
    return finite_volume_cell_balance(
        internal_face_flux=internal_flux,
        boundary_face_flux=boundary_flux,
        internal_owner=geometry.fluid_mesh.internal_owner,
        internal_neighbour=geometry.fluid_mesh.internal_neighbour,
        boundary_owner=geometry.fluid_mesh.boundary_owner,
        cell_volume=geometry.fluid_mesh.cell_volume,
    )


def target_physical_conditions(step_condition: torch.Tensor) -> torch.Tensor:
    """Convert the eight step inputs to [Uin,Tin,q(W/m3),pout,Twall]."""
    if step_condition.ndim != 2 or step_condition.shape[1] != 8:
        raise ValueError("step condition must have eight source/target inputs")
    return torch.stack(
        (
            step_condition[:, 3],
            step_condition[:, 4],
            step_condition[:, 5] * 1.0e6,
            step_condition[:, 6],
            step_condition[:, 7],
        ),
        dim=1,
    )


def assemble_p418_transient_regional_residual(
    *,
    geometry: P418SubfaceGeometry,
    step_condition: torch.Tensor,
    state_physical: torch.Tensor,
    time_s: torch.Tensor,
    fluid_internal_mass_flux_kg_s: torch.Tensor | None = None,
    fluid_boundary_mass_flux_kg_s: torch.Tensor | None = None,
) -> P418TransientRegionalResidual:
    """Assemble the OpenFOAM-13 conservative transient CHT equations.

    Formal thermal-step runs pass the frozen OpenFOAM ``phi`` field through the
    two mass-flux arguments.  The optional reconstruction is retained only for
    small code checks that predate the formal step-response data.
    """
    if state_physical.ndim != 4 or state_physical.shape[-1] != 5:
        raise ValueError("state must have shape [batch,time,node,5]")
    batch, time_count, node_count, _ = state_physical.shape
    if node_count != len(geometry.fluid_global_region) + len(geometry.solid_global_region):
        raise ValueError("state node count differs from the regional CHT geometry")
    fluid_index = torch.as_tensor(
        geometry.fluid_global_region, dtype=torch.long, device=state_physical.device
    )
    solid_index = torch.as_tensor(
        geometry.solid_global_region, dtype=torch.long, device=state_physical.device
    )
    fluid = state_physical[:, :, fluid_index]
    solid = state_physical[:, :, solid_index]
    target = target_physical_conditions(step_condition)
    repeated_target = target[:, None, :].expand(-1, time_count, -1).reshape(-1, 5)
    internal_mass_flux = _expanded_fixed_face_flux(
        fluid_internal_mass_flux_kg_s,
        batch=batch,
        time_count=time_count,
        face_count=len(geometry.fluid_mesh.internal_owner),
        name="fluid internal mass flux",
    )
    boundary_mass_flux = _expanded_fixed_face_flux(
        fluid_boundary_mass_flux_kg_s,
        batch=batch,
        time_count=time_count,
        face_count=len(geometry.fluid_mesh.boundary_owner),
        name="fluid boundary mass flux",
    )
    steady = assemble_p418_regional_cht_residual(
        geometry=geometry,
        physical_conditions=repeated_target,
        fluid_velocity_m_s=fluid[..., :3].reshape(-1, len(fluid_index), 3),
        fluid_pressure_pa=fluid[..., 3].reshape(-1, len(fluid_index)),
        fluid_temperature_k=fluid[..., 4].reshape(-1, len(fluid_index)),
        solid_temperature_k=solid[..., 4].reshape(-1, len(solid_index)),
        fluid_internal_mass_flux_kg_s=internal_mass_flux,
        fluid_boundary_mass_flux_kg_s=boundary_mass_flux,
    )

    fluid_temperature = fluid[..., 4]
    solid_temperature = solid[..., 4]
    fluid_storage_terms = conservative_fluid_storage_terms(fluid, time_s)
    fluid_storage = fluid_storage_terms.total_w_m3
    solid_storage = solid_internal_energy_storage(solid_temperature, time_s)
    kinetic_advection_flat = fluid_kinetic_advection_from_mass_flux(
        geometry=geometry,
        physical_conditions=repeated_target,
        fluid_state=fluid.reshape(batch * time_count, len(fluid_index), 5),
        internal_mass_flux_kg_s=steady.fluid_internal_mass_flux_kg_s,
        boundary_mass_flux_kg_s=steady.fluid_boundary_mass_flux_kg_s,
    )
    kinetic_advection = kinetic_advection_flat.reshape(batch, time_count, -1)

    fluid_steady = (
        steady.fluid_energy_w_m3.reshape(batch, time_count, -1)
        + kinetic_advection
    )
    solid_steady = steady.solid_energy_w_m3.reshape(batch, time_count, -1)
    return P418TransientRegionalResidual(
        fluid_energy_w_m3=fluid_storage + fluid_steady,
        solid_energy_w_m3=solid_storage + solid_steady,
        fluid_storage_w_m3=fluid_storage,
        solid_storage_w_m3=solid_storage,
        fluid_enthalpy_storage_w_m3=fluid_storage_terms.enthalpy_w_m3,
        fluid_kinetic_storage_w_m3=fluid_storage_terms.kinetic_w_m3,
        fluid_pressure_work_w_m3=fluid_storage_terms.pressure_work_w_m3,
        fluid_kinetic_advection_w_m3=kinetic_advection,
        fluid_steady_energy_w_m3=fluid_steady,
        solid_steady_energy_w_m3=solid_steady,
        fluid_mass_kg_m3_s=steady.fluid_mass_kg_m3_s.reshape(batch, time_count, -1),
        interface_flux_reciprocity_w=steady.interface_flux_reciprocity_w.reshape(
            batch, time_count, -1
        ),
        fluid_internal_energy_flux_w=(
            steady.fluid_internal_energy_flux_w
            + internal_upwind_enthalpy_flux(
                mass_flux=steady.fluid_internal_mass_flux_kg_s,
                enthalpy=(
                    0.5
                    * fluid[..., :3]
                    .square()
                    .sum(dim=-1)
                    .reshape(batch * time_count, len(fluid_index))
                ),
                owner=geometry.fluid_mesh.internal_owner,
                neighbour=geometry.fluid_mesh.internal_neighbour,
            )
        ).reshape(batch, time_count, -1),
        solid_internal_heat_flux_w=steady.solid_internal_heat_flux_w.reshape(
            batch, time_count, -1
        ),
    )


def volume_weighted_mean_square(
    values: torch.Tensor, cell_volume_m3: torch.Tensor
) -> torch.Tensor:
    """Mean squared field value with the finite-volume cell measure."""
    if values.ndim < 3 or values.shape[-1] != len(cell_volume_m3):
        raise ValueError("field and cell-volume shapes differ")
    if torch.any(~torch.isfinite(cell_volume_m3)) or torch.any(cell_volume_m3 <= 0.0):
        raise ValueError("cell volumes must be finite and positive")
    weight = cell_volume_m3 / cell_volume_m3.sum()
    return (values.square() * weight).sum(dim=-1).mean()


def dimensionless_transient_energy_loss(
    residual: P418TransientRegionalResidual,
    step_condition: torch.Tensor,
    fluid_volume_m3: torch.Tensor,
    solid_volume_m3: torch.Tensor,
) -> torch.Tensor:
    """Equal fluid/solid volume-weighted residual scaled by heat generation."""
    source_scale = (step_condition[:, 5] * 1.0e6).clamp_min(
        torch.finfo(step_condition.dtype).tiny
    )
    fluid = residual.fluid_energy_w_m3 / source_scale[:, None, None]
    solid = residual.solid_energy_w_m3 / source_scale[:, None, None]
    return 0.5 * (
        volume_weighted_mean_square(fluid, fluid_volume_m3)
        + volume_weighted_mean_square(solid, solid_volume_m3)
    )
