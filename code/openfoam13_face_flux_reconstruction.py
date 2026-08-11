#!/usr/bin/env python3
"""Differentiable OpenFOAM-13-style face interpolation and heat fluxes.

The internal-face formulas follow the archived OpenFOAM 13 implementations of
``surfaceInterpolation``, ``linear``, ``upwind``, ``gaussGrad`` and
``correctedSnGrad``.  Coupled-temperature faces use the same two-sided
conductance weighting as ``coupledTemperature`` for the no-contact-resistance,
no-surface-source case.
"""

from __future__ import annotations

import torch


def _cell_indices(name: str, values: torch.Tensor, n_cells: int) -> torch.Tensor:
    if values.ndim != 1 or values.dtype != torch.long:
        raise ValueError(f"{name} must be a one-dimensional torch.long tensor")
    if values.numel() and (int(values.min()) < 0 or int(values.max()) >= n_cells):
        raise ValueError(f"{name} contains an index outside [0,{n_cells})")
    return values


def _face_geometry(
    cell_centroid: torch.Tensor,
    face_centroid: torch.Tensor,
    area_vector: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
) -> None:
    if cell_centroid.ndim != 2 or cell_centroid.shape[1] != 3:
        raise ValueError("cell_centroid must have [cell,3] shape")
    if face_centroid.ndim != 2 or face_centroid.shape[1] != 3:
        raise ValueError("face_centroid must have [face,3] shape")
    if area_vector.shape != face_centroid.shape:
        raise ValueError("area_vector must have [face,3] shape")
    if len(owner) != len(face_centroid) or len(neighbour) != len(owner):
        raise ValueError("internal-face geometry and topology have inconsistent sizes")
    _cell_indices("owner", owner, len(cell_centroid))
    _cell_indices("neighbour", neighbour, len(cell_centroid))
    if torch.any(~torch.isfinite(cell_centroid)) or torch.any(~torch.isfinite(face_centroid)):
        raise ValueError("centroids must be finite")
    if torch.any(~torch.isfinite(area_vector)) or torch.any(torch.linalg.vector_norm(area_vector, dim=1) <= 0):
        raise ValueError("area vectors must be finite and nonzero")


def openfoam_linear_weights(
    *,
    cell_centroid: torch.Tensor,
    face_centroid: torch.Tensor,
    area_vector: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
) -> torch.Tensor:
    """Return OpenFOAM owner weights for linear internal-face interpolation."""
    _face_geometry(cell_centroid, face_centroid, area_vector, owner, neighbour)
    owner_delta = face_centroid - cell_centroid[owner]
    neighbour_delta = cell_centroid[neighbour] - face_centroid
    sf_d_owner = torch.abs(torch.sum(area_vector * owner_delta, dim=1))
    sf_d_neighbour = torch.abs(torch.sum(area_vector * neighbour_delta, dim=1))
    projected_sum = sf_d_owner + sf_d_neighbour
    distance_owner = torch.linalg.vector_norm(owner_delta, dim=1)
    distance_neighbour = torch.linalg.vector_norm(neighbour_delta, dim=1)
    distance_sum = distance_owner + distance_neighbour
    tiny = torch.finfo(cell_centroid.dtype).tiny
    if torch.any(distance_sum <= tiny):
        raise ValueError("a face centroid coincides with both adjacent cell centroids")
    projected = sf_d_neighbour / torch.clamp(projected_sum, min=tiny)
    distance = distance_neighbour / distance_sum
    # OpenFOAM switches to Euclidean distance only for the vGreat overflow case.
    # The finite tensor implementation uses projected distance whenever nonzero.
    return torch.where(projected_sum > tiny, projected, distance)


def linear_internal_face_interpolate(
    cell_field: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    owner_weight: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate ``[batch,cell,...]`` values to internal faces."""
    if cell_field.ndim < 2:
        raise ValueError("cell_field must have [batch,cell,...] shape")
    n_cells = cell_field.shape[1]
    _cell_indices("owner", owner, n_cells)
    _cell_indices("neighbour", neighbour, n_cells)
    if owner_weight.ndim != 1 or len(owner_weight) != len(owner):
        raise ValueError("owner_weight must have one value per face")
    shape = (1, len(owner)) + (1,) * (cell_field.ndim - 2)
    weight = owner_weight.view(shape)
    return weight * cell_field[:, owner] + (1.0 - weight) * cell_field[:, neighbour]


def upwind_internal_face_interpolate(
    cell_field: torch.Tensor,
    face_flux: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
) -> torch.Tensor:
    """Apply OpenFOAM upwind selection; nonnegative flux selects the owner."""
    if cell_field.ndim < 2 or face_flux.ndim != 2:
        raise ValueError("cell_field and face_flux must start with [batch,cell/face]")
    if cell_field.shape[0] != face_flux.shape[0] or face_flux.shape[1] != len(owner):
        raise ValueError("cell field and face flux have inconsistent batch/face sizes")
    _cell_indices("owner", owner, cell_field.shape[1])
    _cell_indices("neighbour", neighbour, cell_field.shape[1])
    condition = face_flux >= 0
    for _ in range(cell_field.ndim - 2):
        condition = condition.unsqueeze(-1)
    return torch.where(condition, cell_field[:, owner], cell_field[:, neighbour])


def limited_linear_internal_face_interpolate_scalar(
    *,
    cell_scalar: torch.Tensor,
    cell_gradient: torch.Tensor,
    face_flux: torch.Tensor,
    cell_centroid: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    owner_weight: torch.Tensor,
    coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply OpenFOAM's scalar ``limitedLinear`` internal-face interpolation.

    The returned tensors are the face scalar, limiter and final owner weight.
    ``cell_gradient`` is the ``Gauss linear`` cell gradient used by OpenFOAM's
    NVD/TVD limiter.  The switches and clipping match OpenFOAM 13.
    """
    if cell_scalar.ndim != 2 or cell_gradient.shape != (*cell_scalar.shape, 3):
        raise ValueError("cell_scalar and cell_gradient must have [batch,cell] and [batch,cell,3] shapes")
    if face_flux.shape != (cell_scalar.shape[0], len(owner)):
        raise ValueError("face_flux must have [batch,face] shape")
    if cell_centroid.shape != (cell_scalar.shape[1], 3):
        raise ValueError("cell_centroid must have [cell,3] shape")
    if owner_weight.shape != (len(owner),):
        raise ValueError("owner_weight must contain one value per face")
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("limitedLinear coefficient must lie in [0,1]")
    _cell_indices("owner", owner, cell_scalar.shape[1])
    _cell_indices("neighbour", neighbour, cell_scalar.shape[1])

    d = cell_centroid[neighbour] - cell_centroid[owner]
    gradf = cell_scalar[:, neighbour] - cell_scalar[:, owner]
    gradcf_owner = torch.sum(cell_gradient[:, owner] * d[None, :, :], dim=2)
    gradcf_neighbour = torch.sum(cell_gradient[:, neighbour] * d[None, :, :], dim=2)
    # NVDTVD uses the neighbour gradient when faceFlux is exactly zero.
    gradcf = torch.where(face_flux > 0.0, gradcf_owner, gradcf_neighbour)

    large_gradient_ratio = torch.abs(gradcf) >= 1000.0 * torch.abs(gradf)
    large_r = 2000.0 * torch.sign(gradcf) * torch.sign(gradf) - 1.0
    safe_gradf = torch.where(gradf != 0.0, gradf, torch.ones_like(gradf))
    regular_r = 2.0 * (gradcf / safe_gradf) - 1.0
    r = torch.where(large_gradient_ratio, large_r, regular_r)

    small = torch.finfo(cell_scalar.dtype).eps
    limiter = torch.clamp((2.0 / max(coefficient, small)) * r, min=0.0, max=1.0)
    upwind_owner_weight = (face_flux >= 0.0).to(cell_scalar.dtype)
    face_owner_weight = (
        limiter * owner_weight[None, :]
        + (1.0 - limiter) * upwind_owner_weight
    )
    face_scalar = (
        face_owner_weight * cell_scalar[:, owner]
        + (1.0 - face_owner_weight) * cell_scalar[:, neighbour]
    )
    return face_scalar, limiter, face_owner_weight


def openfoam_nonorthogonal_geometry(
    *,
    cell_centroid: torch.Tensor,
    area_vector: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-orthogonal delta coefficients and correction vectors."""
    if cell_centroid.ndim != 2 or cell_centroid.shape[1] != 3:
        raise ValueError("cell_centroid must have [cell,3] shape")
    if area_vector.ndim != 2 or area_vector.shape[1] != 3 or len(area_vector) != len(owner):
        raise ValueError("area_vector must have [face,3] shape")
    _cell_indices("owner", owner, len(cell_centroid))
    _cell_indices("neighbour", neighbour, len(cell_centroid))
    area = torch.linalg.vector_norm(area_vector, dim=1)
    if torch.any(area <= 0):
        raise ValueError("area vectors must be nonzero")
    unit_area = area_vector / area[:, None]
    delta = cell_centroid[neighbour] - cell_centroid[owner]
    delta_magnitude = torch.linalg.vector_norm(delta, dim=1)
    if torch.any(delta_magnitude <= 0):
        raise ValueError("owner and neighbour cell centroids must differ")
    projected = torch.sum(unit_area * delta, dim=1)
    denominator = torch.maximum(projected, 0.05 * delta_magnitude)
    coefficient = 1.0 / denominator
    correction_vector = unit_area - delta * coefficient[:, None]
    return coefficient, correction_vector


def gauss_cell_gradient_scalar(
    *,
    cell_scalar: torch.Tensor,
    internal_face_scalar: torch.Tensor,
    boundary_face_scalar: torch.Tensor,
    internal_area_vector: torch.Tensor,
    boundary_area_vector: torch.Tensor,
    internal_owner: torch.Tensor,
    internal_neighbour: torch.Tensor,
    boundary_owner: torch.Tensor,
    cell_volume: torch.Tensor,
) -> torch.Tensor:
    """Compute the internal Gauss gradient from explicit face values."""
    if cell_scalar.ndim != 2:
        raise ValueError("cell_scalar must have [batch,cell] shape")
    if internal_face_scalar.shape != (cell_scalar.shape[0], len(internal_owner)):
        raise ValueError("internal_face_scalar has the wrong shape")
    if boundary_face_scalar.shape != (cell_scalar.shape[0], len(boundary_owner)):
        raise ValueError("boundary_face_scalar has the wrong shape")
    if internal_area_vector.shape != (len(internal_owner), 3):
        raise ValueError("internal_area_vector has the wrong shape")
    if boundary_area_vector.shape != (len(boundary_owner), 3):
        raise ValueError("boundary_area_vector has the wrong shape")
    if cell_volume.ndim != 1 or len(cell_volume) != cell_scalar.shape[1] or torch.any(cell_volume <= 0):
        raise ValueError("cell_volume must contain one positive value per cell")
    _cell_indices("internal_owner", internal_owner, cell_scalar.shape[1])
    _cell_indices("internal_neighbour", internal_neighbour, cell_scalar.shape[1])
    _cell_indices("boundary_owner", boundary_owner, cell_scalar.shape[1])
    gradient = cell_scalar.new_zeros((cell_scalar.shape[0], cell_scalar.shape[1], 3))
    internal_contribution = internal_face_scalar[:, :, None] * internal_area_vector[None, :, :]
    boundary_contribution = boundary_face_scalar[:, :, None] * boundary_area_vector[None, :, :]
    gradient.index_add_(1, internal_owner, internal_contribution)
    gradient.index_add_(1, internal_neighbour, -internal_contribution)
    gradient.index_add_(1, boundary_owner, boundary_contribution)
    return gradient / cell_volume[None, :, None]


def gauss_cell_gradient_vector(
    *,
    cell_vector: torch.Tensor,
    internal_face_vector: torch.Tensor,
    boundary_face_vector: torch.Tensor,
    internal_area_vector: torch.Tensor,
    boundary_area_vector: torch.Tensor,
    internal_owner: torch.Tensor,
    internal_neighbour: torch.Tensor,
    boundary_owner: torch.Tensor,
    cell_volume: torch.Tensor,
) -> torch.Tensor:
    """Return the OpenFOAM-oriented gradient of a cell vector field.

    The final two axes are ``[derivative direction, vector component]``,
    matching the tensor produced by the Gauss gradient of ``U`` in OpenFOAM.
    """
    if cell_vector.ndim != 3 or cell_vector.shape[2] != 3:
        raise ValueError("cell_vector must have [batch,cell,3] shape")
    if internal_face_vector.shape != (cell_vector.shape[0], len(internal_owner), 3):
        raise ValueError("internal_face_vector has the wrong shape")
    if boundary_face_vector.shape != (cell_vector.shape[0], len(boundary_owner), 3):
        raise ValueError("boundary_face_vector has the wrong shape")
    if internal_area_vector.shape != (len(internal_owner), 3):
        raise ValueError("internal_area_vector has the wrong shape")
    if boundary_area_vector.shape != (len(boundary_owner), 3):
        raise ValueError("boundary_area_vector has the wrong shape")
    if cell_volume.ndim != 1 or len(cell_volume) != cell_vector.shape[1] or torch.any(cell_volume <= 0):
        raise ValueError("cell_volume must contain one positive value per cell")
    _cell_indices("internal_owner", internal_owner, cell_vector.shape[1])
    _cell_indices("internal_neighbour", internal_neighbour, cell_vector.shape[1])
    _cell_indices("boundary_owner", boundary_owner, cell_vector.shape[1])
    gradient = cell_vector.new_zeros((cell_vector.shape[0], cell_vector.shape[1], 3, 3))
    internal_contribution = (
        internal_area_vector[None, :, :, None]
        * internal_face_vector[:, :, None, :]
    )
    boundary_contribution = (
        boundary_area_vector[None, :, :, None]
        * boundary_face_vector[:, :, None, :]
    )
    gradient.index_add_(1, internal_owner, internal_contribution)
    gradient.index_add_(1, internal_neighbour, -internal_contribution)
    gradient.index_add_(1, boundary_owner, boundary_contribution)
    return gradient / cell_volume[None, :, None, None]


def corrected_internal_sn_grad_scalar(
    *,
    cell_scalar: torch.Tensor,
    cell_gradient: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    owner_weight: torch.Tensor,
    nonorthogonal_delta_coeff: torch.Tensor,
    correction_vector: torch.Tensor,
) -> torch.Tensor:
    """Return corrected OpenFOAM normal gradient on internal faces."""
    if cell_scalar.ndim != 2 or cell_gradient.shape != (*cell_scalar.shape, 3):
        raise ValueError("cell scalar/gradient shapes are inconsistent")
    if nonorthogonal_delta_coeff.shape != (len(owner),) or correction_vector.shape != (len(owner), 3):
        raise ValueError("non-orthogonal geometry has the wrong shape")
    face_gradient = linear_internal_face_interpolate(
        cell_gradient, owner, neighbour, owner_weight
    )
    orthogonal = nonorthogonal_delta_coeff[None, :] * (
        cell_scalar[:, neighbour] - cell_scalar[:, owner]
    )
    correction = torch.sum(face_gradient * correction_vector[None, :, :], dim=2)
    return orthogonal + correction


def corrected_internal_sn_grad_vector(
    *,
    cell_vector: torch.Tensor,
    cell_gradient: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    owner_weight: torch.Tensor,
    nonorthogonal_delta_coeff: torch.Tensor,
    correction_vector: torch.Tensor,
) -> torch.Tensor:
    """Return corrected normal gradients for a vector field."""
    if cell_vector.ndim != 3 or cell_vector.shape[2] != 3:
        raise ValueError("cell_vector must have [batch,cell,3] shape")
    if cell_gradient.shape != (*cell_vector.shape[:2], 3, 3):
        raise ValueError("cell_gradient must have [batch,cell,3,3] shape")
    if nonorthogonal_delta_coeff.shape != (len(owner),) or correction_vector.shape != (len(owner), 3):
        raise ValueError("non-orthogonal geometry has the wrong shape")
    face_gradient = linear_internal_face_interpolate(
        cell_gradient, owner, neighbour, owner_weight
    )
    orthogonal = nonorthogonal_delta_coeff[None, :, None] * (
        cell_vector[:, neighbour] - cell_vector[:, owner]
    )
    # correction_vector contracts the derivative-direction axis.
    correction = torch.einsum("bfdc,fd->bfc", face_gradient, correction_vector)
    return orthogonal + correction


def internal_mass_flux(
    *,
    density: torch.Tensor,
    velocity: torch.Tensor,
    area_vector: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    owner_weight: torch.Tensor,
) -> torch.Tensor:
    """Return OpenFOAM's ``linearInterpolate(rho*U) & Sf`` mass flux."""
    if density.ndim != 2 or velocity.shape != (*density.shape, 3):
        raise ValueError("density and velocity must have [batch,cell] and [batch,cell,3] shapes")
    momentum_face = linear_internal_face_interpolate(
        density[:, :, None] * velocity,
        owner,
        neighbour,
        owner_weight,
    )
    return torch.sum(momentum_face * area_vector[None, :, :], dim=2)


def internal_upwind_enthalpy_flux(
    *,
    mass_flux: torch.Tensor,
    enthalpy: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
) -> torch.Tensor:
    """Return ``phi * h_upwind`` for ``div(phi,h) Gauss upwind``."""
    enthalpy_face = upwind_internal_face_interpolate(enthalpy, mass_flux, owner, neighbour)
    return mass_flux * enthalpy_face


def openfoam_boundary_delta_coeff(
    *,
    cell_centroid: torch.Tensor,
    face_centroid: torch.Tensor,
    boundary_owner: torch.Tensor,
) -> torch.Tensor:
    """Return OpenFOAM boundary ``deltaCoeffs = 1/mag(patch.delta())``."""
    if cell_centroid.ndim != 2 or cell_centroid.shape[1] != 3:
        raise ValueError("cell_centroid must have [cell,3] shape")
    if face_centroid.ndim != 2 or face_centroid.shape[1] != 3:
        raise ValueError("face_centroid must have [face,3] shape")
    _cell_indices("boundary_owner", boundary_owner, len(cell_centroid))
    if len(face_centroid) != len(boundary_owner):
        raise ValueError("face_centroid and boundary_owner sizes differ")
    distance = torch.linalg.vector_norm(face_centroid - cell_centroid[boundary_owner], dim=1)
    if torch.any(distance <= 0):
        raise ValueError("boundary face and owner-cell centroids must differ")
    return 1.0 / distance


def boundary_velocity_from_conditions(
    *,
    owner_velocity: torch.Tensor,
    outward_area_vector: torch.Tensor,
    fixed_value_mask: torch.Tensor,
    fixed_reference_value: torch.Tensor,
    pressure_inlet_outlet_mask: torch.Tensor,
    symmetry_or_empty_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate supported fluid boundary velocity branches.

    ``pressureInletOutletVelocity`` is evaluated on its intended outflow branch
    with the owner value.  A separate returned mask flags predicted backflow;
    such faces require a pressure-coupled normal velocity and are not silently
    treated as valid outflow.
    """
    if owner_velocity.ndim != 3 or owner_velocity.shape[2] != 3:
        raise ValueError("owner_velocity must have [batch,face,3] shape")
    if outward_area_vector.shape != owner_velocity.shape[1:]:
        raise ValueError("outward_area_vector must have [face,3] shape")
    n_faces = owner_velocity.shape[1]
    masks = (fixed_value_mask, pressure_inlet_outlet_mask, symmetry_or_empty_mask)
    if any(mask.shape != (n_faces,) or mask.dtype != torch.bool for mask in masks):
        raise ValueError("boundary-condition masks must be one-dimensional boolean tensors")
    coverage = fixed_value_mask.to(torch.int8) + pressure_inlet_outlet_mask.to(torch.int8) + symmetry_or_empty_mask.to(torch.int8)
    if torch.any(coverage != 1):
        raise ValueError("each boundary face must select exactly one velocity branch")
    if fixed_reference_value.shape != owner_velocity.shape:
        raise ValueError("fixed_reference_value must have [batch,face,3] shape")
    area = torch.linalg.vector_norm(outward_area_vector, dim=1)
    if torch.any(area <= 0):
        raise ValueError("outward area vectors must be nonzero")
    normal = outward_area_vector / area[:, None]
    tangential_owner = owner_velocity - torch.sum(
        owner_velocity * normal[None, :, :], dim=2, keepdim=True
    ) * normal[None, :, :]
    face_velocity = torch.where(
        fixed_value_mask[None, :, None],
        fixed_reference_value,
        torch.where(
            symmetry_or_empty_mask[None, :, None], tangential_owner, owner_velocity
        ),
    )
    provisional_flux = torch.sum(owner_velocity * outward_area_vector[None, :, :], dim=2)
    backflow = pressure_inlet_outlet_mask[None, :] & (provisional_flux < 0)
    return face_velocity, backflow


def boundary_mass_flux(
    *,
    density_face: torch.Tensor,
    velocity_face: torch.Tensor,
    outward_area_vector: torch.Tensor,
) -> torch.Tensor:
    """Return outward boundary mass flux ``rho * U dot Sf``."""
    if density_face.ndim != 2 or velocity_face.shape != (*density_face.shape, 3):
        raise ValueError("density_face and velocity_face shapes are inconsistent")
    if outward_area_vector.shape != velocity_face.shape[1:]:
        raise ValueError("outward_area_vector must have [face,3] shape")
    return density_face * torch.sum(velocity_face * outward_area_vector[None, :, :], dim=2)


def boundary_scalar_from_conditions(
    *,
    owner_scalar: torch.Tensor,
    boundary_flux: torch.Tensor,
    fixed_value_mask: torch.Tensor,
    fixed_reference_value: torch.Tensor,
    zero_gradient_or_symmetry_mask: torch.Tensor,
    inlet_outlet_mask: torch.Tensor,
    inlet_reference_value: torch.Tensor,
) -> torch.Tensor:
    """Evaluate fixed, zero-gradient and flux-switched scalar boundaries."""
    if owner_scalar.ndim != 2 or boundary_flux.shape != owner_scalar.shape:
        raise ValueError("owner_scalar and boundary_flux must share [batch,face] shape")
    n_faces = owner_scalar.shape[1]
    masks = (fixed_value_mask, zero_gradient_or_symmetry_mask, inlet_outlet_mask)
    if any(mask.shape != (n_faces,) or mask.dtype != torch.bool for mask in masks):
        raise ValueError("scalar boundary masks must be one-dimensional boolean tensors")
    coverage = sum(mask.to(torch.int8) for mask in masks)
    if torch.any(coverage != 1):
        raise ValueError("each scalar boundary face must select exactly one branch")
    if fixed_reference_value.shape != owner_scalar.shape or inlet_reference_value.shape != owner_scalar.shape:
        raise ValueError("scalar boundary reference values must have [batch,face] shape")
    switched = torch.where(boundary_flux < 0, inlet_reference_value, owner_scalar)
    return torch.where(
        fixed_value_mask[None, :],
        fixed_reference_value,
        torch.where(inlet_outlet_mask[None, :], switched, owner_scalar),
    )


def boundary_upwind_enthalpy_flux(
    *,
    mass_flux: torch.Tensor,
    owner_enthalpy: torch.Tensor,
    inlet_enthalpy: torch.Tensor,
) -> torch.Tensor:
    """Return outward ``phi*h``; inward flux uses prescribed inlet enthalpy."""
    if mass_flux.shape != owner_enthalpy.shape or inlet_enthalpy.shape != mass_flux.shape:
        raise ValueError("boundary enthalpy inputs must share [batch,face] shape")
    face_enthalpy = torch.where(mass_flux < 0, inlet_enthalpy, owner_enthalpy)
    return mass_flux * face_enthalpy


def boundary_conductive_heat_flux(
    *,
    owner_temperature: torch.Tensor,
    face_temperature: torch.Tensor,
    conductivity_face: torch.Tensor,
    face_area: torch.Tensor,
    boundary_delta_coeff: torch.Tensor,
    fixed_or_inflow_mask: torch.Tensor,
) -> torch.Tensor:
    """Return outward non-coupled heat flux; zero-gradient branches return zero."""
    shapes = {owner_temperature.shape, face_temperature.shape, conductivity_face.shape}
    if len(shapes) != 1 or owner_temperature.ndim != 2:
        raise ValueError("boundary thermal fields must share [batch,face] shape")
    n_faces = owner_temperature.shape[1]
    if face_area.shape != (n_faces,) or boundary_delta_coeff.shape != (n_faces,):
        raise ValueError("boundary geometry must contain one scalar per face")
    if fixed_or_inflow_mask.shape != owner_temperature.shape or fixed_or_inflow_mask.dtype != torch.bool:
        raise ValueError("fixed_or_inflow_mask must have [batch,face] boolean shape")
    if torch.any(conductivity_face <= 0):
        raise ValueError("boundary conductivity must be positive")
    gradient = boundary_delta_coeff[None, :] * (face_temperature - owner_temperature)
    return torch.where(
        fixed_or_inflow_mask,
        -conductivity_face * face_area[None, :] * gradient,
        torch.zeros_like(gradient),
    )


def internal_conductive_heat_flux(
    *,
    conductivity: torch.Tensor,
    corrected_sn_grad: torch.Tensor,
    face_area: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    owner_weight: torch.Tensor,
) -> torch.Tensor:
    """Return outward-owner conductive heat flux ``-k_f |Sf| snGrad(T)``."""
    if conductivity.ndim != 2 or corrected_sn_grad.ndim != 2:
        raise ValueError("conductivity and corrected_sn_grad must have [batch,cell/face] shape")
    conductivity_face = linear_internal_face_interpolate(
        conductivity, owner, neighbour, owner_weight
    )
    return -conductivity_face * face_area[None, :] * corrected_sn_grad


def coupled_temperature_interface(
    *,
    fluid_cell_temperature: torch.Tensor,
    solid_cell_temperature: torch.Tensor,
    fluid_conductivity: torch.Tensor,
    solid_conductivity: torch.Tensor,
    fluid_cell_centroid: torch.Tensor,
    solid_cell_centroid: torch.Tensor,
    face_centroid: torch.Tensor,
    fluid_outward_area_vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return common interface T and reciprocal outward fluid/solid heat flux.

    Inputs are aligned per exact reciprocal interface pair.  Conductivities have
    ``[batch,pair]`` shape; temperatures have ``[batch,pair]`` shape.
    """
    shapes = {
        fluid_cell_temperature.shape,
        solid_cell_temperature.shape,
        fluid_conductivity.shape,
        solid_conductivity.shape,
    }
    if len(shapes) != 1 or fluid_cell_temperature.ndim != 2:
        raise ValueError("interface temperatures and conductivities must share [batch,pair] shape")
    pair_count = fluid_cell_temperature.shape[1]
    for name, values in {
        "fluid_cell_centroid": fluid_cell_centroid,
        "solid_cell_centroid": solid_cell_centroid,
        "face_centroid": face_centroid,
        "fluid_outward_area_vector": fluid_outward_area_vector,
    }.items():
        if values.shape != (pair_count, 3):
            raise ValueError(f"{name} must have [pair,3] shape")
    if torch.any(fluid_conductivity <= 0) or torch.any(solid_conductivity <= 0):
        raise ValueError("interface conductivities must be positive")
    area = torch.linalg.vector_norm(fluid_outward_area_vector, dim=1)
    normal = fluid_outward_area_vector / area[:, None]
    fluid_delta = face_centroid - fluid_cell_centroid
    solid_delta = solid_cell_centroid - face_centroid
    fluid_distance = torch.abs(torch.sum(normal * fluid_delta, dim=1))
    solid_distance = torch.abs(torch.sum(normal * solid_delta, dim=1))
    fluid_magnitude = torch.linalg.vector_norm(fluid_delta, dim=1)
    solid_magnitude = torch.linalg.vector_norm(solid_delta, dim=1)
    fluid_distance = torch.maximum(fluid_distance, 0.05 * fluid_magnitude)
    solid_distance = torch.maximum(solid_distance, 0.05 * solid_magnitude)
    if torch.any(fluid_distance <= 0) or torch.any(solid_distance <= 0):
        raise ValueError("interface cell centres must be separated from the face")
    fluid_conductance = fluid_conductivity / fluid_distance[None, :]
    solid_conductance = solid_conductivity / solid_distance[None, :]
    common_temperature = (
        fluid_conductance * fluid_cell_temperature
        + solid_conductance * solid_cell_temperature
    ) / (fluid_conductance + solid_conductance)
    fluid_outward_flux = -fluid_conductance * area[None, :] * (
        common_temperature - fluid_cell_temperature
    )
    solid_outward_flux = -solid_conductance * area[None, :] * (
        common_temperature - solid_cell_temperature
    )
    return common_temperature, fluid_outward_flux, solid_outward_flux
