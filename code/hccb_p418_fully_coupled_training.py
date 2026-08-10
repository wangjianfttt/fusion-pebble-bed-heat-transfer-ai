#!/usr/bin/env python3
"""Training-side losses for the fully coupled P418 graph--Transformer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

import torch

from hccb_p418_fully_coupled_transient_physics import (
    P418FullyCoupledEquationScales,
    P418FullyCoupledTransientResidual,
    volume_weighted_mean_square,
)


PHYSICS_TERM_NAMES = (
    "continuity",
    "momentum",
    "fluid_energy",
    "solid_energy",
    "interface_flux",
    "interface_temperature",
    "internal_mass_flux_consistency",
    "boundary_mass_flux_consistency",
)


def _positive_rms(name: str, *values: torch.Tensor) -> torch.Tensor:
    joined = torch.cat([value.reshape(-1) for value in values])
    scale = torch.sqrt(torch.mean(joined.square()))
    if not torch.isfinite(scale) or scale <= torch.finfo(scale.dtype).eps:
        raise ValueError(f"training reference has no positive {name} scale")
    return scale.detach()


def training_equation_scales(
    reference_residuals: Sequence[P418FullyCoupledTransientResidual],
    reference_states: Sequence[torch.Tensor],
) -> P418FullyCoupledEquationScales:
    """Calculate characteristic equation scales from training curves only.

    Storage and spatial terms are both included, so a well-balanced reference
    curve does not produce a near-zero scale merely because the two cancel.
    The caller is responsible for passing only complete training curves.
    """
    if not reference_residuals or len(reference_residuals) != len(reference_states):
        raise ValueError("training residual and state lists must be non-empty and aligned")
    for state in reference_states:
        if state.ndim != 4 or state.shape[-1] != 5:
            raise ValueError("reference state must have shape [batch,time,node,5]")
        if torch.any(~torch.isfinite(state)) or torch.any(state[..., 4] <= 0.0):
            raise ValueError("reference state must be finite with positive temperature")
    temperature = torch.cat([state[..., 4].reshape(-1) for state in reference_states])
    temperature_scale = temperature.std(unbiased=False)
    if not torch.isfinite(temperature_scale) or temperature_scale <= torch.finfo(
        temperature_scale.dtype
    ).eps:
        raise ValueError("training curves have no temperature variation")
    return P418FullyCoupledEquationScales(
        continuity_kg_m3_s=_positive_rms(
            "continuity",
            *[
                value
                for residual in reference_residuals
                for value in (
                    residual.density_storage_kg_m3_s,
                    residual.steady_mass_kg_m3_s,
                )
            ],
        ),
        momentum_n_m3=_positive_rms(
            "momentum",
            *[
                value
                for residual in reference_residuals
                for value in (
                    residual.momentum_storage_n_m3,
                    residual.steady_momentum_n_m3,
                )
            ],
        ),
        fluid_energy_w_m3=_positive_rms(
            "fluid energy",
            *[
                value
                for residual in reference_residuals
                for value in (
                    residual.fluid_storage_w_m3,
                    residual.steady_fluid_energy_w_m3,
                )
            ],
        ),
        solid_energy_w_m3=_positive_rms(
            "solid energy",
            *[
                value
                for residual in reference_residuals
                for value in (
                    residual.solid_storage_w_m3,
                    residual.steady_solid_energy_w_m3,
                )
            ],
        ),
        interface_flux_w=_positive_rms(
            "interface heat flux",
            *[residual.interface_heat_flux_w for residual in reference_residuals],
        ),
        interface_temperature_k=temperature_scale.detach(),
        internal_mass_flux_kg_s=_positive_rms(
            "internal mass flux",
            *[residual.internal_mass_flux_kg_s for residual in reference_residuals],
        ),
        boundary_mass_flux_kg_s=_positive_rms(
            "boundary mass flux",
            *[residual.boundary_mass_flux_kg_s for residual in reference_residuals],
        ),
    )


def projection_aware_physics_terms(
    *,
    prediction: P418FullyCoupledTransientResidual,
    reference: P418FullyCoupledTransientResidual,
    scales: P418FullyCoupledEquationScales,
    fluid_volume_m3: torch.Tensor,
    solid_volume_m3: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Compare prediction and projected OpenFOAM equations term by term."""
    fluid_weight = fluid_volume_m3 / fluid_volume_m3.sum()
    momentum_difference = (
        prediction.momentum_n_m3 - reference.momentum_n_m3
    ) / scales.momentum_n_m3
    return {
        "continuity": volume_weighted_mean_square(
            (prediction.continuity_kg_m3_s - reference.continuity_kg_m3_s)
            / scales.continuity_kg_m3_s,
            fluid_volume_m3,
        ),
        "momentum": (
            momentum_difference.square().sum(dim=-1) * fluid_weight
        ).sum(dim=-1).mean(),
        "fluid_energy": volume_weighted_mean_square(
            (prediction.fluid_energy_w_m3 - reference.fluid_energy_w_m3)
            / scales.fluid_energy_w_m3,
            fluid_volume_m3,
        ),
        "solid_energy": volume_weighted_mean_square(
            (prediction.solid_energy_w_m3 - reference.solid_energy_w_m3)
            / scales.solid_energy_w_m3,
            solid_volume_m3,
        ),
        "interface_flux": (
            (prediction.interface_flux_reciprocity_w - reference.interface_flux_reciprocity_w)
            / scales.interface_flux_w
        ).square().mean(),
        "interface_temperature": (
            (
                prediction.interface_temperature_jump_k
                - reference.interface_temperature_jump_k
            )
            / scales.interface_temperature_k
        ).square().mean(),
        "internal_mass_flux_consistency": (
            (
                prediction.internal_mass_flux_consistency_kg_s
                - reference.internal_mass_flux_consistency_kg_s
            )
            / scales.internal_mass_flux_kg_s
        ).square().mean(),
        "boundary_mass_flux_consistency": (
            (
                prediction.boundary_mass_flux_consistency_kg_s
                - reference.boundary_mass_flux_consistency_kg_s
            )
            / scales.boundary_mass_flux_kg_s
        ).square().mean(),
    }


def supervised_fully_coupled_terms(
    *,
    predicted_state: torch.Tensor,
    reference_state: torch.Tensor,
    predicted_internal_mass_flux: torch.Tensor,
    reference_internal_mass_flux: torch.Tensor,
    predicted_boundary_mass_flux: torch.Tensor,
    reference_boundary_mass_flux: torch.Tensor,
    node_type: torch.Tensor,
    state_scale_by_node: torch.Tensor,
    internal_mass_flux_scale_kg_s: torch.Tensor,
    boundary_mass_flux_scale_kg_s: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Return state and face-flow errors without mixing material channels."""
    if predicted_state.shape != reference_state.shape or predicted_state.ndim != 4:
        raise ValueError("predicted and reference state histories must have equal 4D shape")
    if state_scale_by_node.shape != predicted_state.shape[2:]:
        raise ValueError("state scale must have shape [node,5]")
    if node_type.shape != (predicted_state.shape[2],):
        raise ValueError("node type must have shape [node]")
    if torch.any(state_scale_by_node <= 0.0):
        raise ValueError("state scales must be positive")
    normalized = (predicted_state - reference_state) / state_scale_by_node[None, None]
    fluid = node_type == 0
    solid = node_type == 1
    if not torch.any(fluid) or not torch.any(solid):
        raise ValueError("both fluid and solid nodes are required")
    state_loss = 0.5 * (
        normalized[:, :, fluid, :].square().mean()
        + normalized[:, :, solid, 4].square().mean()
    )
    for name, predicted, reference, scale in (
        (
            "internal mass flux",
            predicted_internal_mass_flux,
            reference_internal_mass_flux,
            internal_mass_flux_scale_kg_s,
        ),
        (
            "boundary mass flux",
            predicted_boundary_mass_flux,
            reference_boundary_mass_flux,
            boundary_mass_flux_scale_kg_s,
        ),
    ):
        if predicted.shape != reference.shape or predicted.ndim != 3:
            raise ValueError(f"{name} histories must have equal [batch,time,face] shape")
        if not torch.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{name} scale must be finite and positive")
    return {
        "state": state_loss,
        "internal_mass_flux": (
            (predicted_internal_mass_flux - reference_internal_mass_flux)
            / internal_mass_flux_scale_kg_s
        ).square().mean(),
        "boundary_mass_flux": (
            (predicted_boundary_mass_flux - reference_boundary_mass_flux)
            / boundary_mass_flux_scale_kg_s
        ).square().mean(),
    }


def combine_fully_coupled_loss_groups(
    *,
    supervised_terms: Mapping[str, torch.Tensor],
    physics_terms: Mapping[str, torch.Tensor],
    state_weight: float,
    face_flux_weight: float,
    physics_weight: float,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Combine declared loss groups; all three weights are explicit inputs."""
    if set(supervised_terms) != {"state", "internal_mass_flux", "boundary_mass_flux"}:
        raise ValueError("supervised loss terms are incomplete")
    if set(physics_terms) != set(PHYSICS_TERM_NAMES):
        raise ValueError("fully coupled physical loss terms are incomplete")
    weights = (state_weight, face_flux_weight, physics_weight)
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0.0 for value in weights):
        raise ValueError("loss-group weights must be finite and positive")
    groups = {
        "state_data": supervised_terms["state"],
        "face_flux_data": torch.stack(
            (
                supervised_terms["internal_mass_flux"],
                supervised_terms["boundary_mass_flux"],
            )
        ).mean(),
        "physics": torch.stack([physics_terms[name] for name in PHYSICS_TERM_NAMES]).mean(),
    }
    total = (
        state_weight * groups["state_data"]
        + face_flux_weight * groups["face_flux_data"]
        + physics_weight * groups["physics"]
    )
    return total, groups
