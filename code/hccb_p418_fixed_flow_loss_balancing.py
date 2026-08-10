#!/usr/bin/env python3
"""Loss balancing for the fixed-hydrodynamics P418 thermal operator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from hccb_p418_loss_balancing import (
    FixedLossBalancer,
    ReLoBRaLoLossBalancer,
    build_loss_balancer,
    weighted_group_loss,
)


FIXED_FLOW_GROUP_TO_GENERIC = {
    "temperature_data": "state_data",
    "reference_edge_energy_flux": "face_flux_data",
    "projection_aware_transient_energy": "physics",
}


def fixed_flow_loss_groups(
    *,
    temperature_data: torch.Tensor,
    reference_edge_energy_flux: torch.Tensor,
    projection_aware_transient_energy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map the three fixed-flow objectives to the shared balancing interface."""
    return {
        FIXED_FLOW_GROUP_TO_GENERIC["temperature_data"]: temperature_data,
        FIXED_FLOW_GROUP_TO_GENERIC[
            "reference_edge_energy_flux"
        ]: reference_edge_energy_flux,
        FIXED_FLOW_GROUP_TO_GENERIC[
            "projection_aware_transient_energy"
        ]: projection_aware_transient_energy,
    }


def fixed_flow_named_weights(
    weights: Mapping[str, torch.Tensor | float],
) -> dict[str, torch.Tensor | float]:
    """Return weights under the physical names used by the fixed-flow model."""
    return {
        physical_name: weights[generic_name]
        for physical_name, generic_name in FIXED_FLOW_GROUP_TO_GENERIC.items()
    }


def load_fixed_flow_candidate(
    source_path: Path, candidate_id: str
) -> dict[str, object]:
    """Load one recorded candidate without inventing numerical settings."""
    source = json.loads(source_path.read_text(encoding="utf-8"))
    candidates = source.get("formal_candidates")
    if not isinstance(candidates, list):
        raise ValueError("fixed-flow loss-balancing source has no formal candidates")
    matches = [
        row for row in candidates if row.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one fixed-flow loss-balancing candidate {candidate_id!r}"
        )
    candidate = dict(matches[0])
    if candidate.get("method") not in {"fixed", "relobralo"}:
        raise ValueError("fixed-flow loss-balancing method is unsupported")
    return candidate


def build_fixed_flow_loss_balancer(
    *,
    source_path: Path,
    candidate_id: str,
    seed: int,
) -> FixedLossBalancer | ReLoBRaLoLossBalancer:
    """Build the recorded fixed or ReLoBRaLo candidate."""
    candidate = load_fixed_flow_candidate(source_path, candidate_id)
    return build_loss_balancer(
        method=str(candidate["method"]),
        state_weight=float(candidate["temperature_data_weight"]),
        face_flux_weight=float(candidate["reference_edge_energy_flux_weight"]),
        physics_weight=float(
            candidate["projection_aware_transient_energy_weight"]
        ),
        relobralo_temperature=(
            float(candidate["temperature"])
            if "temperature" in candidate
            else None
        ),
        relobralo_alpha=(
            float(candidate["alpha"]) if "alpha" in candidate else None
        ),
        relobralo_rho=(
            float(candidate["expected_rho"])
            if "expected_rho" in candidate
            else None
        ),
        seed=seed,
    )


def balanced_fixed_flow_loss(
    *,
    temperature_data: torch.Tensor,
    reference_edge_energy_flux: torch.Tensor,
    projection_aware_transient_energy: torch.Tensor,
    balancer: FixedLossBalancer | ReLoBRaLoLossBalancer,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Update weights and combine the three fixed-flow objectives."""
    groups = fixed_flow_loss_groups(
        temperature_data=temperature_data,
        reference_edge_energy_flux=reference_edge_energy_flux,
        projection_aware_transient_energy=projection_aware_transient_energy,
    )
    generic_weights = balancer.update(groups)
    total = weighted_group_loss(groups, generic_weights)
    return total, fixed_flow_named_weights(generic_weights)
