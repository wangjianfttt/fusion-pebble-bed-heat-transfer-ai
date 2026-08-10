#!/usr/bin/env python3
"""Graph--Transformer for fully coupled P418 flow and heat step histories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hccb_p418_parametric_regional_operator import RegionalPhysicsAttentionBlock
from hccb_p418_spatiotemporal_regional_operator import (
    LeakyMLP,
    LocalEdgeUpdatingBlock,
    P418ThermalStepRegionalGraph,
)
from hccb_p418_regional_cht_adapter import P418SubfaceGeometry


FULLY_COUPLED_ARCHITECTURE_REVISION = (
    "p418_fully_coupled_oriented_initial_face_flux_context_v2"
)
FULLY_COUPLED_FACE_FLUX_CONTEXT = {
    "initial_internal_face_flux_message": "owner:+m_dot;neighbour:-m_dot",
    "initial_active_boundary_face_flux_message": (
        "inlet/outlet to owner; other boundary faces excluded"
    ),
    "initial_face_flux_aggregation": "mean over incident active faces",
}


@dataclass(frozen=True)
class P418FullyCoupledFluxGraph:
    """Fluid-face indices and dimensionless geometry used by the flux decoders."""

    internal_owner_global: torch.Tensor
    internal_neighbour_global: torch.Tensor
    internal_features: torch.Tensor
    boundary_owner_global: torch.Tensor
    boundary_features: torch.Tensor
    boundary_active: torch.Tensor

    @classmethod
    def from_tensors(
        cls,
        *,
        internal_owner_global: torch.Tensor,
        internal_neighbour_global: torch.Tensor,
        internal_features: torch.Tensor,
        boundary_owner_global: torch.Tensor,
        boundary_features: torch.Tensor,
        boundary_active: torch.Tensor,
        node_count: int,
    ) -> "P418FullyCoupledFluxGraph":
        internal_count = len(internal_owner_global)
        boundary_count = len(boundary_owner_global)
        if internal_neighbour_global.shape != (internal_count,):
            raise ValueError("internal fluid-face owner and neighbour arrays differ")
        if internal_features.ndim != 2 or internal_features.shape[0] != internal_count:
            raise ValueError("internal fluid-face features have inconsistent dimensions")
        if boundary_features.ndim != 2 or boundary_features.shape[0] != boundary_count:
            raise ValueError("boundary fluid-face features have inconsistent dimensions")
        if boundary_active.shape != (boundary_count,):
            raise ValueError("boundary active mask has inconsistent dimensions")
        for name, indices in (
            ("internal owner", internal_owner_global),
            ("internal neighbour", internal_neighbour_global),
            ("boundary owner", boundary_owner_global),
        ):
            if torch.any(indices < 0) or torch.any(indices >= node_count):
                raise ValueError(f"{name} is outside the regional graph")
        if boundary_active.dtype != torch.bool:
            boundary_active = boundary_active.to(torch.bool)
        return cls(
            internal_owner_global=internal_owner_global.to(torch.long),
            internal_neighbour_global=internal_neighbour_global.to(torch.long),
            internal_features=internal_features,
            boundary_owner_global=boundary_owner_global.to(torch.long),
            boundary_features=boundary_features,
            boundary_active=boundary_active,
        )

    @property
    def internal_face_count(self) -> int:
        return len(self.internal_owner_global)

    @property
    def boundary_face_count(self) -> int:
        return len(self.boundary_owner_global)


@dataclass(frozen=True)
class P418FullyCoupledPrediction:
    state: torch.Tensor
    internal_mass_flux: torch.Tensor
    boundary_mass_flux: torch.Tensor


def build_p418_fully_coupled_flux_graph(
    *,
    geometry: P418SubfaceGeometry,
    graph: P418ThermalStepRegionalGraph,
) -> P418FullyCoupledFluxGraph:
    """Build face-decoder inputs from the same conservative regional geometry.

    Internal features contain centre-to-centre displacement, oriented area,
    log area and face offset from the owner-neighbour midpoint.  Boundary
    features contain owner-to-face displacement, outward area, log area and
    the five P418 boundary types.  Length and area scales come only from the
    fixed graph geometry; no material or operating parameter is introduced.
    """
    mesh = geometry.fluid_mesh
    device = graph.centroid_m.device
    dtype = graph.centroid_m.dtype
    fluid_global = torch.as_tensor(
        geometry.fluid_global_region, dtype=torch.long, device=device
    )
    if len(fluid_global) != len(mesh.cell_volume):
        raise ValueError("fluid regional indices and conservative cells differ")
    if torch.any(graph.node_type[fluid_global] != 0):
        raise ValueError("fluid conservative cells do not map to fluid graph nodes")
    if tuple(geometry.fluid_patch_names) != (
        "inlet",
        "outlet",
        "coolingWall",
        "symmetryWalls",
        "fluid_to_solid",
    ):
        required = {
            "inlet",
            "outlet",
            "coolingWall",
            "symmetryWalls",
            "fluid_to_solid",
        }
        if set(geometry.fluid_patch_names) != required:
            raise ValueError("fully coupled flux graph requires all five P418 fluid patches")

    coordinate_scale = graph.coordinate_scale_m.to(device=device, dtype=dtype)
    area_scale = graph.volume_scale_m3.to(device=device, dtype=dtype).pow(2.0 / 3.0)
    if torch.any(coordinate_scale <= 0.0) or area_scale <= 0.0:
        raise ValueError("graph geometry scales must be positive")

    owner_local = mesh.internal_owner.to(device=device)
    neighbour_local = mesh.internal_neighbour.to(device=device)
    owner_global = fluid_global[owner_local]
    neighbour_global = fluid_global[neighbour_local]
    owner_centroid = mesh.cell_centroid.to(device=device, dtype=dtype)[owner_local]
    neighbour_centroid = mesh.cell_centroid.to(device=device, dtype=dtype)[
        neighbour_local
    ]
    internal_centroid = mesh.internal_face_centroid.to(device=device, dtype=dtype)
    internal_area_vector = mesh.internal_area_vector.to(device=device, dtype=dtype)
    internal_area = torch.linalg.vector_norm(internal_area_vector, dim=1)
    if torch.any(internal_area <= 0.0):
        raise ValueError("internal fluid face area must be positive")
    internal_features = torch.cat(
        (
            (neighbour_centroid - owner_centroid) / coordinate_scale,
            internal_area_vector / area_scale,
            torch.log(internal_area / area_scale).unsqueeze(-1),
            (
                internal_centroid
                - 0.5 * (owner_centroid + neighbour_centroid)
            )
            / coordinate_scale,
        ),
        dim=1,
    )

    boundary_owner_local = mesh.boundary_owner.to(device=device)
    boundary_owner_global = fluid_global[boundary_owner_local]
    boundary_owner_centroid = mesh.cell_centroid.to(device=device, dtype=dtype)[
        boundary_owner_local
    ]
    boundary_centroid = mesh.boundary_face_centroid.to(device=device, dtype=dtype)
    boundary_area_vector = mesh.boundary_area_vector.to(device=device, dtype=dtype)
    boundary_area = torch.linalg.vector_norm(boundary_area_vector, dim=1)
    if torch.any(boundary_area <= 0.0):
        raise ValueError("boundary fluid face area must be positive")
    patch = geometry.fluid_boundary_patch.to(device=device)
    patch_columns = []
    for name in (
        "inlet",
        "outlet",
        "coolingWall",
        "symmetryWalls",
        "fluid_to_solid",
    ):
        patch_columns.append(
            (patch == geometry.fluid_patch_names.index(name)).to(dtype).unsqueeze(-1)
        )
    patch_features = torch.cat(patch_columns, dim=1)
    boundary_features = torch.cat(
        (
            (boundary_centroid - boundary_owner_centroid) / coordinate_scale,
            boundary_area_vector / area_scale,
            torch.log(boundary_area / area_scale).unsqueeze(-1),
            patch_features,
        ),
        dim=1,
    )
    inlet = patch == geometry.fluid_patch_names.index("inlet")
    outlet = patch == geometry.fluid_patch_names.index("outlet")
    return P418FullyCoupledFluxGraph.from_tensors(
        internal_owner_global=owner_global,
        internal_neighbour_global=neighbour_global,
        internal_features=internal_features,
        boundary_owner_global=boundary_owner_global,
        boundary_features=boundary_features,
        boundary_active=inlet | outlet,
        node_count=graph.node_count,
    )


class HCCBP418FullyCoupledRegionalOperator(nn.Module):
    """Predict time-dependent regional U, p, T and oriented fluid-face mass flux."""

    def __init__(
        self,
        *,
        condition_dim: int = 8,
        hidden_dim: int = 64,
        local_pre_iterations: int = 2,
        physics_attention_blocks: int = 2,
        local_post_iterations: int = 2,
        physics_attention_heads: int = 4,
        physics_slices: int = 128,
        temporal_layers: int = 3,
        temporal_heads: int = 1,
        spatial_time_chunk_size: int = 1,
        temporal_node_chunk_size: int | None = 4096,
        internal_face_feature_dim: int = 10,
        boundary_face_feature_dim: int = 12,
        boundary_role_count: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim % physics_attention_heads or hidden_dim % temporal_heads:
            raise ValueError("hidden dimension must be divisible by all attention heads")
        if min(
            hidden_dim,
            local_pre_iterations,
            physics_attention_blocks,
            local_post_iterations,
            temporal_layers,
            temporal_heads,
            spatial_time_chunk_size,
        ) <= 0:
            raise ValueError("all architecture dimensions must be positive")
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.spatial_time_chunk_size = spatial_time_chunk_size
        self.temporal_node_chunk_size = temporal_node_chunk_size
        self.internal_face_feature_dim = internal_face_feature_dim
        self.boundary_face_feature_dim = boundary_face_feature_dim
        self.state_encoder = LeakyMLP(5, hidden_dim, hidden_dim)
        self.condition_encoder = LeakyMLP(condition_dim, hidden_dim, hidden_dim)
        self.time_encoder = LeakyMLP(1, hidden_dim, hidden_dim)
        if boundary_role_count < 0:
            raise ValueError("boundary role count must be nonnegative")
        self.boundary_role_count = boundary_role_count
        self.structure_encoder = LeakyMLP(
            6 + boundary_role_count, hidden_dim, hidden_dim
        )
        self.edge_encoder = LeakyMLP(10, hidden_dim, hidden_dim)
        self.internal_face_encoder = LeakyMLP(
            internal_face_feature_dim, hidden_dim, hidden_dim
        )
        self.boundary_face_encoder = LeakyMLP(
            boundary_face_feature_dim, hidden_dim, hidden_dim
        )
        self.internal_initial_flux_encoder = LeakyMLP(
            1, hidden_dim, hidden_dim
        )
        self.boundary_initial_flux_encoder = LeakyMLP(
            1, hidden_dim, hidden_dim
        )
        self.local_pre = nn.ModuleList(
            LocalEdgeUpdatingBlock(hidden_dim) for _ in range(local_pre_iterations)
        )
        self.global_blocks = nn.ModuleList(
            RegionalPhysicsAttentionBlock(
                hidden_dim,
                heads=physics_attention_heads,
                slice_count=physics_slices,
                dropout=0.0,
            )
            for _ in range(physics_attention_blocks)
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=temporal_heads,
            dim_feedforward=hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers
        )
        self.local_post = nn.ModuleList(
            LocalEdgeUpdatingBlock(hidden_dim) for _ in range(local_post_iterations)
        )
        self.state_change = LeakyMLP(hidden_dim, 5, hidden_dim)
        self.internal_flux_change = LeakyMLP(3 * hidden_dim, 1, hidden_dim)
        self.boundary_flux_change = LeakyMLP(2 * hidden_dim, 1, hidden_dim)

    def _initial_flux_context(
        self,
        *,
        initial_internal_mass_flux: torch.Tensor,
        initial_boundary_mass_flux: torch.Tensor,
        graph: P418ThermalStepRegionalGraph,
        flux_graph: P418FullyCoupledFluxGraph,
    ) -> torch.Tensor:
        """Map oriented initial face fluxes to their adjacent regional nodes."""
        batch = initial_internal_mass_flux.shape[0]
        context = initial_internal_mass_flux.new_zeros(
            (batch, graph.node_count, self.hidden_dim)
        )
        count = initial_internal_mass_flux.new_zeros(
            (batch, graph.node_count, 1)
        )
        internal_owner = self.internal_initial_flux_encoder(
            initial_internal_mass_flux.unsqueeze(-1)
        )
        internal_neighbour = self.internal_initial_flux_encoder(
            -initial_internal_mass_flux.unsqueeze(-1)
        )
        context.index_add_(
            1, flux_graph.internal_owner_global, internal_owner
        )
        context.index_add_(
            1, flux_graph.internal_neighbour_global, internal_neighbour
        )
        one_internal = torch.ones_like(initial_internal_mass_flux).unsqueeze(-1)
        count.index_add_(1, flux_graph.internal_owner_global, one_internal)
        count.index_add_(1, flux_graph.internal_neighbour_global, one_internal)

        boundary_active = flux_graph.boundary_active.to(
            dtype=initial_boundary_mass_flux.dtype
        )
        active_boundary_flux = (
            initial_boundary_mass_flux * boundary_active[None]
        )
        boundary_message = self.boundary_initial_flux_encoder(
            active_boundary_flux.unsqueeze(-1)
        ) * boundary_active[None, :, None]
        context.index_add_(
            1, flux_graph.boundary_owner_global, boundary_message
        )
        count.index_add_(
            1,
            flux_graph.boundary_owner_global,
            boundary_active[None, :, None].expand(batch, -1, -1),
        )
        return context / count.clamp_min(1.0)

    def _spatial(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        graph: P418ThermalStepRegionalGraph,
        blocks: nn.ModuleList,
        include_global: bool,
    ) -> torch.Tensor:
        for block in blocks:
            nodes = block(nodes, edges, graph.edge_source, graph.edge_target)
        if include_global:
            for block in self.global_blocks:
                nodes = block(nodes)
        return nodes

    def _chunked_spatial(
        self,
        values: torch.Tensor,
        edges: torch.Tensor,
        graph: P418ThermalStepRegionalGraph,
        blocks: nn.ModuleList,
        include_global: bool,
    ) -> torch.Tensor:
        batch, time_count, node_count, hidden = values.shape
        output = []
        for start in range(0, time_count, self.spatial_time_chunk_size):
            stop = min(start + self.spatial_time_chunk_size, time_count)
            current = values[:, start:stop].reshape(
                batch * (stop - start), node_count, hidden
            )
            current = self._spatial(current, edges, graph, blocks, include_global)
            output.append(current.reshape(batch, stop - start, node_count, hidden))
        return torch.cat(output, dim=1)

    def _temporal_mix(self, values: torch.Tensor) -> torch.Tensor:
        batch, time_count, node_count, hidden = values.shape
        by_node = values.permute(0, 2, 1, 3).reshape(
            batch * node_count, time_count, hidden
        )
        chunk = self.temporal_node_chunk_size or len(by_node)
        output = []
        for start in range(0, len(by_node), chunk):
            current = by_node[start : start + chunk]
            if self.training and current.requires_grad:
                output.append(checkpoint(self.temporal, current, use_reentrant=False))
            else:
                output.append(self.temporal(current))
        joined = torch.cat(output).reshape(batch, node_count, time_count, hidden)
        return joined.permute(0, 2, 1, 3)

    def forward(
        self,
        initial_state: torch.Tensor,
        initial_internal_mass_flux: torch.Tensor,
        initial_boundary_mass_flux: torch.Tensor,
        normalized_condition: torch.Tensor,
        normalized_time: torch.Tensor,
        graph: P418ThermalStepRegionalGraph,
        flux_graph: P418FullyCoupledFluxGraph,
    ) -> P418FullyCoupledPrediction:
        if initial_state.ndim != 3 or initial_state.shape[1:] != (graph.node_count, 5):
            raise ValueError("initial state must have shape [batch,node,5]")
        batch = initial_state.shape[0]
        if normalized_condition.shape != (batch, self.condition_dim):
            raise ValueError(f"condition must have shape [batch,{self.condition_dim}]")
        if normalized_time.ndim == 1:
            normalized_time = normalized_time.unsqueeze(0).expand(batch, -1)
        if normalized_time.ndim != 2 or normalized_time.shape[0] != batch:
            raise ValueError("time must have shape [time] or [batch,time]")
        if torch.any(normalized_time < 0.0):
            raise ValueError("dimensionless time must be nonnegative")
        if initial_internal_mass_flux.shape != (batch, flux_graph.internal_face_count):
            raise ValueError("initial internal mass flux has inconsistent dimensions")
        if initial_boundary_mass_flux.shape != (batch, flux_graph.boundary_face_count):
            raise ValueError("initial boundary mass flux has inconsistent dimensions")
        if flux_graph.internal_features.shape[1] != self.internal_face_feature_dim:
            raise ValueError("internal face feature dimension differs from the model")
        if flux_graph.boundary_features.shape[1] != self.boundary_face_feature_dim:
            raise ValueError("boundary face feature dimension differs from the model")
        if graph.boundary_role_count != self.boundary_role_count:
            raise ValueError("graph boundary roles differ from the model")

        time_count = normalized_time.shape[1]
        initial_flux_context = self._initial_flux_context(
            initial_internal_mass_flux=initial_internal_mass_flux,
            initial_boundary_mass_flux=initial_boundary_mass_flux,
            graph=graph,
            flux_graph=flux_graph,
        )
        state_latent = (
            self.state_encoder(initial_state) + initial_flux_context
        )[:, None]
        condition_latent = self.condition_encoder(normalized_condition)[:, None, None]
        time_latent = self.time_encoder(normalized_time.unsqueeze(-1))[:, :, None]
        structure = self.structure_encoder(graph.structural_features())[None, None]
        latent = state_latent + condition_latent + time_latent + structure
        edges = self.edge_encoder(graph.edge_features())
        latent = self._chunked_spatial(
            latent, edges, graph, self.local_pre, include_global=True
        )
        latent = self._temporal_mix(latent)
        latent = self._chunked_spatial(
            latent, edges, graph, self.local_post, include_global=False
        )

        time_factor = normalized_time[:, :, None, None]
        channel_mask = F.one_hot(graph.node_type, num_classes=2).to(latent.dtype)
        fluid = channel_mask[:, 0:1]
        solid = channel_mask[:, 1:2]
        state_mask = torch.cat(
            (fluid.expand(-1, 4), torch.ones_like(fluid)), dim=-1
        )
        state_delta = self.state_change(latent) * state_mask[None, None]
        state = initial_state[:, None] + time_factor * state_delta

        internal_feature = self.internal_face_encoder(flux_graph.internal_features)
        internal_latent = torch.cat(
            (
                latent[:, :, flux_graph.internal_owner_global],
                latent[:, :, flux_graph.internal_neighbour_global],
                internal_feature[None, None].expand(batch, time_count, -1, -1),
            ),
            dim=-1,
        )
        internal_change = self.internal_flux_change(internal_latent).squeeze(-1)
        internal_flux = (
            initial_internal_mass_flux[:, None]
            + normalized_time[:, :, None] * internal_change
        )

        boundary_feature = self.boundary_face_encoder(flux_graph.boundary_features)
        boundary_latent = torch.cat(
            (
                latent[:, :, flux_graph.boundary_owner_global],
                boundary_feature[None, None].expand(batch, time_count, -1, -1),
            ),
            dim=-1,
        )
        boundary_change = self.boundary_flux_change(boundary_latent).squeeze(-1)
        boundary_flux = (
            initial_boundary_mass_flux[:, None]
            + normalized_time[:, :, None] * boundary_change
        )
        boundary_flux = torch.where(
            flux_graph.boundary_active[None, None],
            boundary_flux,
            torch.zeros_like(boundary_flux),
        )
        return P418FullyCoupledPrediction(
            state=state,
            internal_mass_flux=internal_flux,
            boundary_mass_flux=boundary_flux,
        )
