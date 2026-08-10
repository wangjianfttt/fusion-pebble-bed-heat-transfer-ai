#!/usr/bin/env python3
"""Fully coupled transient FV relations for P418 regional predictions.

The module combines the existing source-backed helium/Li4SiO4 properties,
P418 boundary conditions and steady finite-volume operators with conservative
time-storage terms.  It adds no material coefficient or fitted physical
parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from hccb_conservative_physics_residual import (
    ConservativeHCCBPhysicsResiduals,
    assemble_conservative_hccb_physics_residuals,
)
from hccb_p418_regional_cht_adapter import (
    P418SubfaceGeometry,
    p418_boundary_conditions,
)
from hccb_p418_transient_regional_physics import (
    conservative_fluid_storage_terms,
    fluid_kinetic_advection_from_mass_flux,
    solid_internal_energy_storage,
    target_physical_conditions,
    time_derivative,
    volume_weighted_mean_square,
)
from hccb_source_backed_thermophysical import helium_density


@dataclass(frozen=True)
class P418FullyCoupledTransientResidual:
    """Dimensional residuals and their transient/steady components."""

    continuity_kg_m3_s: torch.Tensor
    momentum_n_m3: torch.Tensor
    fluid_energy_w_m3: torch.Tensor
    solid_energy_w_m3: torch.Tensor
    density_storage_kg_m3_s: torch.Tensor
    momentum_storage_n_m3: torch.Tensor
    fluid_storage_w_m3: torch.Tensor
    solid_storage_w_m3: torch.Tensor
    steady_mass_kg_m3_s: torch.Tensor
    steady_momentum_n_m3: torch.Tensor
    steady_fluid_energy_w_m3: torch.Tensor
    steady_solid_energy_w_m3: torch.Tensor
    fluid_kinetic_advection_w_m3: torch.Tensor
    interface_flux_reciprocity_w: torch.Tensor
    interface_temperature_jump_k: torch.Tensor
    interface_heat_flux_w: torch.Tensor
    internal_mass_flux_kg_s: torch.Tensor
    boundary_mass_flux_kg_s: torch.Tensor
    internal_mass_flux_consistency_kg_s: torch.Tensor
    boundary_mass_flux_consistency_kg_s: torch.Tensor


@dataclass(frozen=True)
class P418FullyCoupledEquationScales:
    """Positive scales computed from training histories, never test curves."""

    continuity_kg_m3_s: torch.Tensor
    momentum_n_m3: torch.Tensor
    fluid_energy_w_m3: torch.Tensor
    solid_energy_w_m3: torch.Tensor
    interface_flux_w: torch.Tensor
    interface_temperature_k: torch.Tensor
    internal_mass_flux_kg_s: torch.Tensor
    boundary_mass_flux_kg_s: torch.Tensor


def density_and_momentum_storage(
    fluid_state: torch.Tensor,
    time_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``d(rho)/dt`` and ``d(rho*U)/dt`` on reported times."""
    if fluid_state.ndim != 4 or fluid_state.shape[-1] != 5:
        raise ValueError("fluid state must have shape [batch,time,cell,5]")
    if torch.any(~torch.isfinite(fluid_state)):
        raise ValueError("fluid state contains non-finite values")
    pressure = fluid_state[..., 3]
    temperature = fluid_state[..., 4]
    if torch.any(pressure <= 0.0) or torch.any(temperature <= 0.0):
        raise ValueError("fluid pressure and temperature must be positive")
    density = helium_density(pressure, temperature)
    return (
        time_derivative(density, time_s),
        time_derivative(density[..., None] * fluid_state[..., :3], time_s),
    )


def _time_dependent_flux(
    values: torch.Tensor,
    *,
    batch: int,
    time_count: int,
    face_count: int,
    name: str,
) -> torch.Tensor:
    expected = (batch, time_count, face_count)
    if values.shape != expected:
        raise ValueError(f"{name} must have shape {expected}; fixed flux is not fully coupled")
    if torch.any(~torch.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values.reshape(batch * time_count, face_count)


def _p418_boundary_pressure(
    *,
    geometry: P418SubfaceGeometry,
    fluid_pressure_pa: torch.Tensor,
    physical_conditions: torch.Tensor,
) -> torch.Tensor:
    owner_pressure = fluid_pressure_pa[:, geometry.fluid_mesh.boundary_owner]
    if "outlet" not in geometry.fluid_patch_names:
        raise ValueError("P418 fluid patches do not include outlet")
    outlet = geometry.fluid_boundary_patch == geometry.fluid_patch_names.index("outlet")
    return torch.where(
        outlet[None, :],
        physical_conditions[:, 3:4],
        owner_pressure,
    )


def assemble_p418_fully_coupled_transient_residual(
    *,
    geometry: P418SubfaceGeometry,
    step_condition: torch.Tensor,
    state_physical: torch.Tensor,
    time_s: torch.Tensor,
    fluid_internal_mass_flux_kg_s: torch.Tensor,
    fluid_boundary_mass_flux_kg_s: torch.Tensor,
) -> P418FullyCoupledTransientResidual:
    """Assemble transient mass, momentum and dual-region energy relations.

    ``state_physical`` follows ``[Ux,Uy,Uz,p,T]``.  Solid graph nodes use only
    temperature; their velocity and pressure channels are ignored here, as in
    the full-state model mask.  Predicted face flux is required at every time
    and is used unchanged in mass, momentum and both fluid energy advections.
    """
    if state_physical.ndim != 4 or state_physical.shape[-1] != 5:
        raise ValueError("state must have shape [batch,time,node,5]")
    if torch.any(~torch.isfinite(state_physical)):
        raise ValueError("state contains non-finite values")
    batch, time_count, node_count, _ = state_physical.shape
    fluid_count = len(geometry.fluid_global_region)
    solid_count = len(geometry.solid_global_region)
    if node_count != fluid_count + solid_count:
        raise ValueError("state node count differs from regional geometry")
    if step_condition.shape != (batch, 8):
        raise ValueError("step condition must have shape [batch,8]")

    fluid_index = torch.as_tensor(
        geometry.fluid_global_region,
        dtype=torch.long,
        device=state_physical.device,
    )
    solid_index = torch.as_tensor(
        geometry.solid_global_region,
        dtype=torch.long,
        device=state_physical.device,
    )
    fluid = state_physical[:, :, fluid_index]
    solid = state_physical[:, :, solid_index]
    target = target_physical_conditions(step_condition)
    repeated_target = target[:, None, :].expand(-1, time_count, -1).reshape(-1, 5)
    internal_flux = _time_dependent_flux(
        fluid_internal_mass_flux_kg_s,
        batch=batch,
        time_count=time_count,
        face_count=len(geometry.fluid_mesh.internal_owner),
        name="fluid internal mass flux",
    )
    boundary_flux = _time_dependent_flux(
        fluid_boundary_mass_flux_kg_s,
        batch=batch,
        time_count=time_count,
        face_count=len(geometry.fluid_mesh.boundary_owner),
        name="fluid boundary mass flux",
    )

    fluid_flat = fluid.reshape(batch * time_count, fluid_count, 5)
    solid_flat = solid.reshape(batch * time_count, solid_count, 5)
    ordered_state = torch.cat((fluid_flat, solid_flat), dim=1)
    velocity_bc, fluid_temperature_bc, solid_temperature_bc = p418_boundary_conditions(
        geometry, repeated_target
    )
    boundary_pressure = _p418_boundary_pressure(
        geometry=geometry,
        fluid_pressure_pa=fluid_flat[..., 3],
        physical_conditions=repeated_target,
    )
    steady: ConservativeHCCBPhysicsResiduals = (
        assemble_conservative_hccb_physics_residuals(
            fluid_mesh=geometry.fluid_mesh,
            solid_mesh=geometry.solid_mesh,
            interface=geometry.interface,
            fluid_velocity_bc=velocity_bc,
            fluid_temperature_bc=fluid_temperature_bc,
            solid_temperature_bc=solid_temperature_bc,
            cell_state_physical=ordered_state,
            fluid_internal_mass_flux_kg_s=internal_flux,
            fluid_boundary_mass_flux_kg_s=boundary_flux,
            fluid_boundary_pressure_pa=boundary_pressure,
            solid_volumetric_heat_source_w_m3=repeated_target[:, 2:3].expand(
                -1, solid_count
            ),
        )
    )

    density_storage, momentum_storage = density_and_momentum_storage(fluid, time_s)
    fluid_storage = conservative_fluid_storage_terms(fluid, time_s).total_w_m3
    solid_storage = solid_internal_energy_storage(solid[..., 4], time_s)
    kinetic_advection = fluid_kinetic_advection_from_mass_flux(
        geometry=geometry,
        physical_conditions=repeated_target,
        fluid_state=fluid_flat,
        internal_mass_flux_kg_s=internal_flux,
        boundary_mass_flux_kg_s=boundary_flux,
    ).reshape(batch, time_count, fluid_count)

    steady_mass = steady.cht.fluid_mass_kg_m3_s.reshape(
        batch, time_count, fluid_count
    )
    steady_momentum = steady.momentum.momentum_n_m3.reshape(
        batch, time_count, fluid_count, 3
    )
    steady_fluid_energy = (
        steady.cht.fluid_energy_w_m3.reshape(batch, time_count, fluid_count)
        + kinetic_advection
    )
    steady_solid_energy = steady.cht.solid_energy_w_m3.reshape(
        batch, time_count, solid_count
    )
    return P418FullyCoupledTransientResidual(
        continuity_kg_m3_s=density_storage + steady_mass,
        momentum_n_m3=momentum_storage + steady_momentum,
        fluid_energy_w_m3=fluid_storage + steady_fluid_energy,
        solid_energy_w_m3=solid_storage + steady_solid_energy,
        density_storage_kg_m3_s=density_storage,
        momentum_storage_n_m3=momentum_storage,
        fluid_storage_w_m3=fluid_storage,
        solid_storage_w_m3=solid_storage,
        steady_mass_kg_m3_s=steady_mass,
        steady_momentum_n_m3=steady_momentum,
        steady_fluid_energy_w_m3=steady_fluid_energy,
        steady_solid_energy_w_m3=steady_solid_energy,
        fluid_kinetic_advection_w_m3=kinetic_advection,
        interface_flux_reciprocity_w=steady.cht.interface_flux_reciprocity_w.reshape(
            batch, time_count, -1
        ),
        interface_temperature_jump_k=steady.cht.interface_temperature_jump_k.reshape(
            batch, time_count, -1
        ),
        interface_heat_flux_w=steady.cht.fluid_boundary_energy_flux_w[
            :, geometry.interface.fluid_boundary_face
        ].reshape(batch, time_count, -1),
        internal_mass_flux_kg_s=internal_flux.reshape(batch, time_count, -1),
        boundary_mass_flux_kg_s=boundary_flux.reshape(batch, time_count, -1),
        internal_mass_flux_consistency_kg_s=(
            steady.internal_mass_flux_consistency_kg_s.reshape(
                batch, time_count, -1
            )
        ),
        boundary_mass_flux_consistency_kg_s=(
            steady.boundary_mass_flux_consistency_kg_s.reshape(
                batch, time_count, -1
            )
        ),
    )


def dimensional_fully_coupled_fields(
    residual: P418FullyCoupledTransientResidual,
) -> Mapping[str, torch.Tensor]:
    """Expose every equation separately before training-only normalization."""
    return {
        "continuity_kg_m3_s": residual.continuity_kg_m3_s,
        "momentum_n_m3": residual.momentum_n_m3,
        "fluid_energy_w_m3": residual.fluid_energy_w_m3,
        "solid_energy_w_m3": residual.solid_energy_w_m3,
        "interface_flux_reciprocity_w": residual.interface_flux_reciprocity_w,
        "interface_temperature_jump_k": residual.interface_temperature_jump_k,
        "internal_mass_flux_consistency_kg_s": (
            residual.internal_mass_flux_consistency_kg_s
        ),
        "boundary_mass_flux_consistency_kg_s": (
            residual.boundary_mass_flux_consistency_kg_s
        ),
    }


def dimensionless_fully_coupled_equation_terms(
    *,
    residual: P418FullyCoupledTransientResidual,
    scales: P418FullyCoupledEquationScales,
    fluid_volume_m3: torch.Tensor,
    solid_volume_m3: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Return equation-wise mean squares using training-derived scales.

    The terms are deliberately not summed here.  Relative equation weights
    belong to the declared training procedure and must not be hidden inside
    the finite-volume relation.
    """
    scale_values = {
        "continuity": scales.continuity_kg_m3_s,
        "momentum": scales.momentum_n_m3,
        "fluid_energy": scales.fluid_energy_w_m3,
        "solid_energy": scales.solid_energy_w_m3,
        "interface_flux": scales.interface_flux_w,
        "interface_temperature": scales.interface_temperature_k,
        "internal_mass_flux": scales.internal_mass_flux_kg_s,
        "boundary_mass_flux": scales.boundary_mass_flux_kg_s,
    }
    for name, scale in scale_values.items():
        if torch.any(~torch.isfinite(scale)) or torch.any(scale <= 0.0):
            raise ValueError(f"{name} scale must be finite and positive")
    momentum_square = (
        residual.momentum_n_m3 / scales.momentum_n_m3
    ).square().sum(dim=-1)
    fluid_weight = fluid_volume_m3 / fluid_volume_m3.sum()
    return {
        "continuity": volume_weighted_mean_square(
            residual.continuity_kg_m3_s / scales.continuity_kg_m3_s,
            fluid_volume_m3,
        ),
        "momentum": (momentum_square * fluid_weight).sum(dim=-1).mean(),
        "fluid_energy": volume_weighted_mean_square(
            residual.fluid_energy_w_m3 / scales.fluid_energy_w_m3,
            fluid_volume_m3,
        ),
        "solid_energy": volume_weighted_mean_square(
            residual.solid_energy_w_m3 / scales.solid_energy_w_m3,
            solid_volume_m3,
        ),
        "interface_flux": (
            residual.interface_flux_reciprocity_w / scales.interface_flux_w
        ).square().mean(),
        "interface_temperature": (
            residual.interface_temperature_jump_k
            / scales.interface_temperature_k
        ).square().mean(),
        "internal_mass_flux": (
            residual.internal_mass_flux_consistency_kg_s
            / scales.internal_mass_flux_kg_s
        ).square().mean(),
        "boundary_mass_flux": (
            residual.boundary_mass_flux_consistency_kg_s
            / scales.boundary_mass_flux_kg_s
        ).square().mean(),
    }
