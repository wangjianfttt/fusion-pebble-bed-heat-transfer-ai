#!/usr/bin/env python3
"""Differentiable finite-volume balance operators on native OpenFOAM faces.

The functions consume explicit oriented face fluxes. They do not reconstruct
mass or heat flux from cell-centred states, because that reconstruction must
match the source OpenFOAM interpolation, convection and non-orthogonal
correction schemes of the target case.
"""

from __future__ import annotations

import torch


def _index(name: str, values: torch.Tensor, size: int) -> torch.Tensor:
    if values.ndim != 1 or values.dtype != torch.long:
        raise ValueError(f"{name} must be a one-dimensional torch.long tensor")
    if values.numel() and (int(values.min()) < 0 or int(values.max()) >= size):
        raise ValueError(f"{name} contains a cell index outside [0,{size})")
    return values


def finite_volume_cell_balance(
    *,
    internal_face_flux: torch.Tensor,
    boundary_face_flux: torch.Tensor,
    internal_owner: torch.Tensor,
    internal_neighbour: torch.Tensor,
    boundary_owner: torch.Tensor,
    cell_volume: torch.Tensor,
    volumetric_source: torch.Tensor | None = None,
    divide_by_volume: bool = True,
) -> torch.Tensor:
    """Return steady per-cell balance using OpenFOAM face orientation.

    Positive internal flux points from owner to neighbour. Positive boundary
    flux points outward from its owner cell. For an energy equation with a
    positive generated source, the residual is ``net outward flux - source*V``.
    """
    if internal_face_flux.ndim != 2 or boundary_face_flux.ndim != 2:
        raise ValueError("face fluxes must have [batch,face] shape")
    if internal_face_flux.shape[0] != boundary_face_flux.shape[0]:
        raise ValueError("internal and boundary fluxes must share the batch size")
    if cell_volume.ndim != 1 or torch.any(cell_volume <= 0):
        raise ValueError("cell_volume must be a positive one-dimensional tensor")
    n_cells = len(cell_volume)
    _index("internal_owner", internal_owner, n_cells)
    _index("internal_neighbour", internal_neighbour, n_cells)
    _index("boundary_owner", boundary_owner, n_cells)
    if len(internal_owner) != internal_face_flux.shape[1] or len(internal_neighbour) != len(
        internal_owner
    ):
        raise ValueError("internal face topology does not match internal fluxes")
    if len(boundary_owner) != boundary_face_flux.shape[1]:
        raise ValueError("boundary face topology does not match boundary fluxes")

    residual = internal_face_flux.new_zeros((internal_face_flux.shape[0], n_cells))
    residual.index_add_(1, internal_owner, internal_face_flux)
    residual.index_add_(1, internal_neighbour, -internal_face_flux)
    residual.index_add_(1, boundary_owner, boundary_face_flux)
    if volumetric_source is not None:
        if volumetric_source.shape != residual.shape:
            raise ValueError("volumetric_source must have [batch,cell] shape")
        residual = residual - volumetric_source * cell_volume.view(1, -1)
    if divide_by_volume:
        residual = residual / cell_volume.view(1, -1)
    return residual


def finite_volume_cell_vector_balance(
    *,
    internal_face_flux: torch.Tensor,
    boundary_face_flux: torch.Tensor,
    internal_owner: torch.Tensor,
    internal_neighbour: torch.Tensor,
    boundary_owner: torch.Tensor,
    cell_volume: torch.Tensor,
    volumetric_source: torch.Tensor | None = None,
    divide_by_volume: bool = True,
) -> torch.Tensor:
    """Return a steady three-component finite-volume balance per cell."""
    if internal_face_flux.ndim != 3 or internal_face_flux.shape[2] != 3:
        raise ValueError("internal vector flux must have [batch,face,3] shape")
    if boundary_face_flux.ndim != 3 or boundary_face_flux.shape[2] != 3:
        raise ValueError("boundary vector flux must have [batch,face,3] shape")
    if internal_face_flux.shape[0] != boundary_face_flux.shape[0]:
        raise ValueError("internal and boundary fluxes must share the batch size")
    if cell_volume.ndim != 1 or torch.any(cell_volume <= 0):
        raise ValueError("cell_volume must be a positive one-dimensional tensor")
    n_cells = len(cell_volume)
    _index("internal_owner", internal_owner, n_cells)
    _index("internal_neighbour", internal_neighbour, n_cells)
    _index("boundary_owner", boundary_owner, n_cells)
    if len(internal_owner) != internal_face_flux.shape[1] or len(internal_neighbour) != len(internal_owner):
        raise ValueError("internal face topology does not match vector fluxes")
    if len(boundary_owner) != boundary_face_flux.shape[1]:
        raise ValueError("boundary face topology does not match vector fluxes")
    residual = internal_face_flux.new_zeros(
        (internal_face_flux.shape[0], n_cells, 3)
    )
    residual.index_add_(1, internal_owner, internal_face_flux)
    residual.index_add_(1, internal_neighbour, -internal_face_flux)
    residual.index_add_(1, boundary_owner, boundary_face_flux)
    if volumetric_source is not None:
        if volumetric_source.shape != residual.shape:
            raise ValueError("volumetric_source must have [batch,cell,3] shape")
        residual = residual - volumetric_source * cell_volume[None, :, None]
    if divide_by_volume:
        residual = residual / cell_volume[None, :, None]
    return residual


def interface_flux_reciprocity(
    fluid_outward_heat_flux: torch.Tensor,
    solid_outward_heat_flux: torch.Tensor,
) -> torch.Tensor:
    """Return q_fluid,out + q_solid,out for aligned reciprocal interface faces."""
    if fluid_outward_heat_flux.shape != solid_outward_heat_flux.shape:
        raise ValueError("fluid and solid interface flux arrays must have identical shape")
    if fluid_outward_heat_flux.ndim != 2:
        raise ValueError("interface flux arrays must have [batch,pair] shape")
    return fluid_outward_heat_flux + solid_outward_heat_flux


def interface_temperature_jump(
    fluid_face_temperature: torch.Tensor,
    solid_face_temperature: torch.Tensor,
) -> torch.Tensor:
    """Return aligned fluid minus solid interface temperature."""
    if fluid_face_temperature.shape != solid_face_temperature.shape:
        raise ValueError("fluid and solid interface temperature arrays must have identical shape")
    if fluid_face_temperature.ndim != 2:
        raise ValueError("interface temperatures must have [batch,pair] shape")
    return fluid_face_temperature - solid_face_temperature
