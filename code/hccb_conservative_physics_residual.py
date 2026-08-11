#!/usr/bin/env python3
"""Couple conservative HCCB cell/face predictions to all steady FV equations.

The adapter accepts dimensional cell states ``[Ux, Uy, Uz, p, T]``, one
owner-to-neighbour mass flux per fluid internal face and, optionally, one
outward mass flux per fluid boundary face.  The same mass-flux tensors are
used by fluid mass, momentum convection and enthalpy convection.
No residual normalization or loss weight is selected in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from hccb_multiregion_steady_cht_residual import (
    CoupledInterfaceMap,
    RegionMesh,
    SteadyChtResiduals,
    ThermalBoundaryConditions,
    VelocityBoundaryConditions,
    assemble_steady_hccb_cht_residual,
)
from hccb_source_backed_thermophysical import (
    helium_density,
    helium_dynamic_viscosity,
)
from hccb_steady_momentum_residual import (
    SteadyMomentumResiduals,
    assemble_steady_momentum_from_properties,
)
from openfoam13_face_flux_reconstruction import boundary_velocity_from_conditions


@dataclass(frozen=True)
class ConservativeHCCBPhysicsResiduals:
    cht: SteadyChtResiduals
    momentum: SteadyMomentumResiduals
    internal_mass_flux_consistency_kg_s: torch.Tensor
    boundary_mass_flux_consistency_kg_s: torch.Tensor


def _split_dimensional_cell_state(
    cell_state: torch.Tensor,
    *,
    fluid_cells: int,
    solid_cells: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell_state.ndim != 3 or cell_state.shape[2] != 5:
        raise ValueError("cell state must have [batch,node,5] shape")
    if cell_state.shape[1] != fluid_cells + solid_cells:
        raise ValueError("cell state does not match fluid plus solid cell counts")
    if torch.any(~torch.isfinite(cell_state)):
        raise ValueError("cell state must be finite")
    fluid = cell_state[:, :fluid_cells]
    solid = cell_state[:, fluid_cells:]
    velocity = fluid[:, :, :3]
    pressure = fluid[:, :, 3]
    fluid_temperature = fluid[:, :, 4]
    solid_temperature = solid[:, :, 4]
    if torch.any(pressure <= 0):
        raise ValueError("fluid absolute pressure must be positive")
    if torch.any(fluid_temperature <= 0) or torch.any(solid_temperature <= 0):
        raise ValueError("fluid and solid temperatures must be positive")
    return velocity, pressure, fluid_temperature, solid_temperature


def assemble_conservative_hccb_physics_residuals(
    *,
    fluid_mesh: RegionMesh,
    solid_mesh: RegionMesh,
    interface: CoupledInterfaceMap,
    fluid_velocity_bc: VelocityBoundaryConditions,
    fluid_temperature_bc: ThermalBoundaryConditions,
    solid_temperature_bc: ThermalBoundaryConditions,
    cell_state_physical: torch.Tensor,
    fluid_internal_mass_flux_kg_s: torch.Tensor,
    fluid_boundary_pressure_pa: torch.Tensor,
    solid_volumetric_heat_source_w_m3: torch.Tensor,
    fluid_boundary_mass_flux_kg_s: torch.Tensor | None = None,
    fluid_volumetric_heat_source_w_m3: torch.Tensor | None = None,
    volumetric_momentum_source_n_m3: torch.Tensor | None = None,
) -> ConservativeHCCBPhysicsResiduals:
    """Return dimensional steady CHT and momentum residuals from one prediction."""
    fluid_cells = len(fluid_mesh.cell_volume)
    solid_cells = len(solid_mesh.cell_volume)
    velocity, pressure, fluid_temperature, solid_temperature = (
        _split_dimensional_cell_state(
            cell_state_physical,
            fluid_cells=fluid_cells,
            solid_cells=solid_cells,
        )
    )
    expected_flux_shape = (
        cell_state_physical.shape[0],
        len(fluid_mesh.internal_owner),
    )
    if fluid_internal_mass_flux_kg_s.shape != expected_flux_shape:
        raise ValueError(
            "fluid internal mass flux must have "
            f"shape {expected_flux_shape}"
        )
    if torch.any(~torch.isfinite(fluid_internal_mass_flux_kg_s)):
        raise ValueError("fluid internal mass flux must be finite")
    if fluid_boundary_mass_flux_kg_s is not None:
        expected_boundary_shape = (
            cell_state_physical.shape[0],
            len(fluid_mesh.boundary_owner),
        )
        if fluid_boundary_mass_flux_kg_s.shape != expected_boundary_shape:
            raise ValueError(
                "fluid boundary mass flux must have "
                f"shape {expected_boundary_shape}"
            )
        if torch.any(~torch.isfinite(fluid_boundary_mass_flux_kg_s)):
            raise ValueError("fluid boundary mass flux must be finite")

    cht = assemble_steady_hccb_cht_residual(
        fluid_mesh=fluid_mesh,
        solid_mesh=solid_mesh,
        interface=interface,
        fluid_velocity_bc=fluid_velocity_bc,
        fluid_temperature_bc=fluid_temperature_bc,
        solid_temperature_bc=solid_temperature_bc,
        fluid_pressure_pa=pressure,
        fluid_velocity_m_s=velocity,
        fluid_temperature_k=fluid_temperature,
        solid_temperature_k=solid_temperature,
        fluid_boundary_pressure_pa=fluid_boundary_pressure_pa,
        solid_volumetric_heat_source_w_m3=solid_volumetric_heat_source_w_m3,
        fluid_volumetric_heat_source_w_m3=fluid_volumetric_heat_source_w_m3,
        fluid_internal_mass_flux_override_kg_s=fluid_internal_mass_flux_kg_s,
        fluid_boundary_mass_flux_override_kg_s=fluid_boundary_mass_flux_kg_s,
    )

    boundary_velocity, backflow = boundary_velocity_from_conditions(
        owner_velocity=velocity[:, fluid_mesh.boundary_owner],
        outward_area_vector=fluid_mesh.boundary_area_vector,
        fixed_value_mask=fluid_velocity_bc.fixed_value_mask,
        fixed_reference_value=fluid_velocity_bc.fixed_reference_value,
        pressure_inlet_outlet_mask=fluid_velocity_bc.pressure_inlet_outlet_mask,
        symmetry_or_empty_mask=fluid_velocity_bc.symmetry_or_empty_mask,
    )
    density = helium_density(pressure, fluid_temperature)
    boundary_density = helium_density(
        fluid_boundary_pressure_pa, cht.fluid_boundary_temperature_k
    )
    viscosity = helium_dynamic_viscosity(pressure, fluid_temperature)
    boundary_viscosity = helium_dynamic_viscosity(
        fluid_boundary_pressure_pa, cht.fluid_boundary_temperature_k
    )
    momentum_terms = assemble_steady_momentum_from_properties(
        mesh=fluid_mesh,
        velocity_m_s=velocity,
        boundary_velocity_m_s=boundary_velocity,
        pressure_pa=pressure,
        boundary_pressure_pa=fluid_boundary_pressure_pa,
        density_kg_m3=density,
        boundary_density_kg_m3=boundary_density,
        dynamic_viscosity_pa_s=viscosity,
        boundary_dynamic_viscosity_pa_s=boundary_viscosity,
        internal_mass_flux_override_kg_s=fluid_internal_mass_flux_kg_s,
        boundary_mass_flux_override_kg_s=fluid_boundary_mass_flux_kg_s,
        volumetric_momentum_source_n_m3=volumetric_momentum_source_n_m3,
    )
    backflow = fluid_velocity_bc.pressure_inlet_outlet_mask[None, :] & (
        momentum_terms["boundary_mass_flux_kg_s"] < 0
    )
    momentum = SteadyMomentumResiduals(
        momentum_n_m3=momentum_terms["momentum_n_m3"],
        convection_n_m3=momentum_terms["convection_n_m3"],
        pressure_n_m3=momentum_terms["pressure_n_m3"],
        viscous_n_m3=momentum_terms["viscous_n_m3"],
        internal_momentum_flux_n=momentum_terms["internal_momentum_flux_n"],
        boundary_momentum_flux_n=momentum_terms["boundary_momentum_flux_n"],
        internal_mass_flux_kg_s=momentum_terms["internal_mass_flux_kg_s"],
        boundary_mass_flux_kg_s=momentum_terms["boundary_mass_flux_kg_s"],
        absolute_pressure_pa=pressure,
        boundary_absolute_pressure_pa=fluid_boundary_pressure_pa,
        density_kg_m3=density,
        dynamic_viscosity_pa_s=viscosity,
        outlet_backflow_mask=backflow,
    )
    return ConservativeHCCBPhysicsResiduals(
        cht=cht,
        momentum=momentum,
        internal_mass_flux_consistency_kg_s=(
            momentum_terms["internal_mass_flux_kg_s"]
            - momentum_terms["reconstructed_internal_mass_flux_kg_s"]
        ),
        boundary_mass_flux_consistency_kg_s=(
            momentum_terms["boundary_mass_flux_kg_s"]
            - momentum_terms["reconstructed_boundary_mass_flux_kg_s"]
        ),
    )


def dimensional_residual_fields(
    residuals: ConservativeHCCBPhysicsResiduals,
) -> Mapping[str, torch.Tensor]:
    """Expose each dimensional equation separately for later normalization."""
    return {
        "fluid_mass_kg_m3_s": residuals.cht.fluid_mass_kg_m3_s,
        "fluid_momentum_n_m3": residuals.momentum.momentum_n_m3,
        "fluid_energy_w_m3": residuals.cht.fluid_energy_w_m3,
        "solid_energy_w_m3": residuals.cht.solid_energy_w_m3,
        "interface_flux_reciprocity_w": residuals.cht.interface_flux_reciprocity_w,
        "interface_temperature_jump_k": residuals.cht.interface_temperature_jump_k,
        "internal_mass_flux_consistency_kg_s": (
            residuals.internal_mass_flux_consistency_kg_s
        ),
        "boundary_mass_flux_consistency_kg_s": (
            residuals.boundary_mass_flux_consistency_kg_s
        ),
    }
