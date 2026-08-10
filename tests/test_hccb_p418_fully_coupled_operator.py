#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_fully_coupled_spatiotemporal_operator import (  # noqa: E402
    HCCBP418FullyCoupledRegionalOperator,
    P418FullyCoupledFluxGraph,
)
from hccb_p418_spatiotemporal_regional_operator import (  # noqa: E402
    P418ThermalStepRegionalGraph,
)


def small_graph() -> P418ThermalStepRegionalGraph:
    return P418ThermalStepRegionalGraph.from_tensors(
        centroid_m=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]),
        volume_m3=torch.ones(3),
        node_type=torch.tensor([0, 0, 1]),
        edge_source=torch.tensor([0, 1, 0, 2]),
        edge_target=torch.tensor([1, 0, 2, 0]),
        edge_kind=torch.tensor([0, 0, 2, 2]),
        edge_area_m2=torch.ones(4),
        edge_area_vector_m2=torch.ones((4, 3)),
        boundary_fraction=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]]
        ),
    )


def small_flux_graph() -> P418FullyCoupledFluxGraph:
    return P418FullyCoupledFluxGraph.from_tensors(
        internal_owner_global=torch.tensor([0]),
        internal_neighbour_global=torch.tensor([1]),
        internal_features=torch.zeros((1, 10)),
        boundary_owner_global=torch.tensor([0, 1]),
        boundary_features=torch.zeros((2, 12)),
        boundary_active=torch.tensor([True, False]),
        node_count=3,
    )


def test_fully_coupled_operator_preserves_source_state_at_zero_time() -> None:
    torch.manual_seed(7)
    graph = small_graph()
    flux_graph = small_flux_graph()
    model = HCCBP418FullyCoupledRegionalOperator(
        hidden_dim=8,
        local_pre_iterations=1,
        physics_attention_blocks=1,
        local_post_iterations=1,
        physics_attention_heads=1,
        physics_slices=2,
        temporal_layers=1,
        temporal_heads=1,
        temporal_node_chunk_size=None,
        boundary_role_count=graph.boundary_role_count,
    )
    initial_state = torch.randn((1, 3, 5), requires_grad=True)
    initial_internal = torch.tensor([[0.1]])
    initial_boundary = torch.tensor([[0.2, 0.0]])
    prediction = model(
        initial_state,
        initial_internal,
        initial_boundary,
        torch.zeros((1, 8)),
        torch.tensor([0.0, 0.5, 1.0]),
        graph,
        flux_graph,
    )
    assert prediction.state.shape == (1, 3, 3, 5)
    assert prediction.internal_mass_flux.shape == (1, 3, 1)
    assert prediction.boundary_mass_flux.shape == (1, 3, 2)
    torch.testing.assert_close(prediction.state[:, 0], initial_state)
    torch.testing.assert_close(prediction.internal_mass_flux[:, 0], initial_internal)
    torch.testing.assert_close(prediction.boundary_mass_flux[:, 0], initial_boundary)
    torch.testing.assert_close(
        prediction.state[:, :, 2, :4],
        initial_state[:, None, 2, :4].expand(-1, 3, -1),
    )
    assert torch.count_nonzero(prediction.boundary_mass_flux[..., 1]) == 0
    prediction.state.square().mean().backward()
    assert initial_state.grad is not None


def test_initial_face_flux_reaches_state_and_flux_branches() -> None:
    torch.manual_seed(13)
    graph = small_graph()
    flux_graph = small_flux_graph()
    model = HCCBP418FullyCoupledRegionalOperator(
        hidden_dim=8,
        local_pre_iterations=1,
        physics_attention_blocks=1,
        local_post_iterations=1,
        physics_attention_heads=1,
        physics_slices=2,
        temporal_layers=1,
        temporal_heads=1,
        temporal_node_chunk_size=None,
        boundary_role_count=graph.boundary_role_count,
    ).eval()
    initial_state = torch.randn((1, 3, 5))
    initial_internal = torch.tensor([[0.1]])
    initial_boundary = torch.tensor([[0.2, 0.0]])
    with torch.no_grad():
        reference = model(
            initial_state,
            initial_internal,
            initial_boundary,
            torch.zeros((1, 8)),
            torch.tensor([0.0, 0.5, 1.0]),
            graph,
            flux_graph,
        )
        changed = model(
            initial_state,
            1.2 * initial_internal,
            1.2 * initial_boundary,
            torch.zeros((1, 8)),
            torch.tensor([0.0, 0.5, 1.0]),
            graph,
            flux_graph,
        )
    torch.testing.assert_close(reference.state[:, 0], initial_state)
    torch.testing.assert_close(changed.state[:, 0], initial_state)
    assert torch.max(torch.abs(changed.state[:, -1] - reference.state[:, -1])) > 0
    assert (
        torch.max(
            torch.abs(
                changed.internal_mass_flux - reference.internal_mass_flux
            )
        )
        > 0
    )
