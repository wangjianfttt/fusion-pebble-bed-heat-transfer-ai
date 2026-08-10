#!/usr/bin/env python3
"""Regional graph--Transformer operator for P418 thermal step responses.

The OpenFOAM step calculations hold the converged target hydrodynamic field
fixed and advance only the coupled fluid/solid energy equations.  This model
uses the same definition: velocity and pressure are copied exactly from the
initial regional state, while the network predicts the subsequent temperature
change.  Multiplication by dimensionless time enforces the initial temperature
at t=0 exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hccb_p418_parametric_regional_operator import RegionalPhysicsAttentionBlock


FORMAL_ARCHITECTURE = {
    "hidden_dim": 64,
    "local_pre_iterations": 2,
    "physics_attention_blocks": 2,
    "local_post_iterations": 2,
    "physics_attention_heads": 4,
    "physics_slices": 128,
    "temporal_layers": 3,
    "temporal_heads": 1,
    "leaky_relu_negative_slope": 0.01,
}

SPATIAL_TEMPORAL_MODES = (
    "repeated_query_spatial",
    "factorized_static_spatial",
)

TEMPERATURE_OUTPUT_MODES = (
    "additive_normalized",
    "literature_bounded_logit",
    "literature_bounded_residual",
)

LITERATURE_BOUNDED_TEMPERATURE_OUTPUT_MODES = (
    "literature_bounded_logit",
    "literature_bounded_residual",
)


def temperature_output_bounds_by_node_type(
    record: object,
) -> np.ndarray:
    """Return fluid/solid temperature bounds from current or legacy summaries."""
    if isinstance(record, dict):
        if set(record) != {"fluid", "solid"}:
            raise ValueError(
                "temperature bounds must contain exactly fluid and solid"
            )
        record = [record["fluid"], record["solid"]]
    bounds = np.asarray(record, dtype=np.float32)
    if bounds.shape != (2, 2):
        raise ValueError("temperature bounds must have shape [2,2]")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("temperature bounds must be finite increasing pairs")
    return bounds


def segment_sum(values: torch.Tensor, target: torch.Tensor, count: int) -> torch.Tensor:
    """Sum edge values at their receiving regional node."""
    output = values.new_zeros((values.shape[0], count, values.shape[-1]))
    output.index_add_(1, target, values)
    return output


@dataclass(frozen=True)
class P418ThermalStepRegionalGraph:
    centroid_m: torch.Tensor
    volume_m3: torch.Tensor
    node_type: torch.Tensor
    edge_source: torch.Tensor
    edge_target: torch.Tensor
    edge_kind: torch.Tensor
    edge_area_m2: torch.Tensor
    edge_area_vector_m2: torch.Tensor
    boundary_fraction: torch.Tensor
    coordinate_center_m: torch.Tensor
    coordinate_scale_m: torch.Tensor
    volume_scale_m3: torch.Tensor

    @classmethod
    def from_npz(
        cls, path: Path, device: torch.device | str = "cpu"
    ) -> "P418ThermalStepRegionalGraph":
        with np.load(path, allow_pickle=False) as data:
            centroid = torch.as_tensor(
                data["node_centroid_m"], dtype=torch.float32, device=device
            )
            volume = torch.as_tensor(
                data["node_volume_m3"], dtype=torch.float32, device=device
            )
            node_type = torch.as_tensor(
                data["node_type"], dtype=torch.long, device=device
            )
            edge_source = torch.as_tensor(
                data["edge_source"], dtype=torch.long, device=device
            )
            edge_target = torch.as_tensor(
                data["edge_target"], dtype=torch.long, device=device
            )
            edge_kind = torch.as_tensor(
                data["edge_kind"], dtype=torch.long, device=device
            )
            edge_area = torch.as_tensor(
                data["edge_area_m2"], dtype=torch.float32, device=device
            )
            edge_area_vector = torch.as_tensor(
                data["edge_area_vector_m2"], dtype=torch.float32, device=device
            )
            boundary_fraction = torch.as_tensor(
                (
                    data["node_boundary_fraction"]
                    if "node_boundary_fraction" in data.files
                    else np.zeros((len(node_type), 0), dtype=np.float32)
                ),
                dtype=torch.float32,
                device=device,
            )
        return cls.from_tensors(
            centroid_m=centroid,
            volume_m3=volume,
            node_type=node_type,
            edge_source=edge_source,
            edge_target=edge_target,
            edge_kind=edge_kind,
            edge_area_m2=edge_area,
            edge_area_vector_m2=edge_area_vector,
            boundary_fraction=boundary_fraction,
        )

    @classmethod
    def from_tensors(
        cls,
        *,
        centroid_m: torch.Tensor,
        volume_m3: torch.Tensor,
        node_type: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        edge_kind: torch.Tensor,
        edge_area_m2: torch.Tensor,
        edge_area_vector_m2: torch.Tensor,
        boundary_fraction: torch.Tensor | None = None,
    ) -> "P418ThermalStepRegionalGraph":
        node_count = len(node_type)
        edge_count = len(edge_source)
        if centroid_m.shape != (node_count, 3) or volume_m3.shape != (node_count,):
            raise ValueError("regional node geometry has inconsistent dimensions")
        if edge_target.shape != (edge_count,) or edge_kind.shape != (edge_count,):
            raise ValueError("regional edge indices have inconsistent dimensions")
        if edge_area_m2.shape != (edge_count,) or edge_area_vector_m2.shape != (edge_count, 3):
            raise ValueError("regional edge geometry has inconsistent dimensions")
        if torch.any(volume_m3 <= 0) or torch.any(edge_area_m2 <= 0):
            raise ValueError("regional volumes and edge areas must be positive")
        if torch.any(node_type < 0) or torch.any(node_type > 1):
            raise ValueError("node type must be 0 for fluid or 1 for solid")
        if torch.any(edge_kind < 0) or torch.any(edge_kind > 2):
            raise ValueError("edge kind must be fluid, solid or interface")
        if torch.any(edge_source < 0) or torch.any(edge_source >= node_count):
            raise ValueError("edge source is outside the regional graph")
        if torch.any(edge_target < 0) or torch.any(edge_target >= node_count):
            raise ValueError("edge target is outside the regional graph")
        if boundary_fraction is None:
            boundary_fraction = centroid_m.new_zeros((node_count, 0))
        if boundary_fraction.ndim != 2 or boundary_fraction.shape[0] != node_count:
            raise ValueError("boundary fractions have inconsistent dimensions")
        if torch.any(~torch.isfinite(boundary_fraction)):
            raise ValueError("boundary fractions must be finite")
        if torch.any(boundary_fraction < 0.0) or torch.any(boundary_fraction > 1.0):
            raise ValueError("boundary fractions must lie between zero and one")
        coordinate_min = centroid_m.amin(dim=0)
        coordinate_max = centroid_m.amax(dim=0)
        coordinate_scale = (coordinate_max - coordinate_min).clamp_min(
            torch.finfo(centroid_m.dtype).eps
        )
        return cls(
            centroid_m=centroid_m,
            volume_m3=volume_m3,
            node_type=node_type,
            edge_source=edge_source,
            edge_target=edge_target,
            edge_kind=edge_kind,
            edge_area_m2=edge_area_m2,
            edge_area_vector_m2=edge_area_vector_m2,
            boundary_fraction=boundary_fraction,
            coordinate_center_m=0.5 * (coordinate_max + coordinate_min),
            coordinate_scale_m=coordinate_scale,
            volume_scale_m3=volume_m3.median(),
        )

    @property
    def node_count(self) -> int:
        return len(self.node_type)

    @property
    def boundary_role_count(self) -> int:
        return int(self.boundary_fraction.shape[1])

    @property
    def structural_feature_dim(self) -> int:
        return 6 + self.boundary_role_count

    def structural_features(self) -> torch.Tensor:
        coordinate = (self.centroid_m - self.coordinate_center_m) / self.coordinate_scale_m
        log_volume = torch.log(self.volume_m3 / self.volume_scale_m3).unsqueeze(-1)
        material = F.one_hot(self.node_type, num_classes=2).to(self.centroid_m.dtype)
        return torch.cat(
            (coordinate, log_volume, material, self.boundary_fraction), dim=-1
        )

    def edge_features(self) -> torch.Tensor:
        relative = (
            self.centroid_m[self.edge_target] - self.centroid_m[self.edge_source]
        ) / self.coordinate_scale_m
        area_scale = self.volume_scale_m3.pow(2.0 / 3.0)
        area_vector = self.edge_area_vector_m2 / area_scale
        log_area = torch.log(self.edge_area_m2 / area_scale).unsqueeze(-1)
        kind = F.one_hot(self.edge_kind, num_classes=3).to(relative.dtype)
        return torch.cat((relative, area_vector, log_area, kind), dim=-1)


class LeakyMLP(nn.Module):
    """64--32--64 path used by the source-backed formal architecture."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        bottleneck = max(hidden_dim // 2, 1)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, bottleneck),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(bottleneck, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class LocalEdgeUpdatingBlock(nn.Module):
    """Update local edge messages, sum them, and update receiving nodes."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_update = LeakyMLP(3 * hidden_dim, hidden_dim, hidden_dim)
        self.node_update = LeakyMLP(2 * hidden_dim, hidden_dim, hidden_dim)
        self.edge_normalization = nn.LayerNorm(hidden_dim)
        self.node_normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        base_edges: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        edges = base_edges.unsqueeze(0).expand(nodes.shape[0], -1, -1)
        edges = self.edge_normalization(
            edges
            + self.edge_update(
                torch.cat((nodes[:, source], nodes[:, target], edges), dim=-1)
            )
        )
        incoming = segment_sum(edges, target, nodes.shape[1])
        return self.node_normalization(
            nodes + self.node_update(torch.cat((nodes, incoming), dim=-1))
        )


class HCCBP418SpatiotemporalRegionalOperator(nn.Module):
    """Predict regional fluid/solid temperatures over a complete thermal step."""

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
        spatial_temporal_mode: str = "repeated_query_spatial",
        boundary_role_count: int = 0,
        temperature_output_mode: str = "additive_normalized",
        temperature_mean_k_by_node_type: torch.Tensor | None = None,
        temperature_std_k_by_node_type: torch.Tensor | None = None,
        temperature_bounds_k_by_node_type: torch.Tensor | None = None,
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
        ) <= 0:
            raise ValueError("all architecture dimensions must be positive")
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        if spatial_time_chunk_size <= 0:
            raise ValueError("spatial time chunk size must be positive")
        self.spatial_time_chunk_size = spatial_time_chunk_size
        self.temporal_node_chunk_size = temporal_node_chunk_size
        if spatial_temporal_mode not in SPATIAL_TEMPORAL_MODES:
            raise ValueError(
                f"spatial-temporal mode must be one of {SPATIAL_TEMPORAL_MODES}"
            )
        self.spatial_temporal_mode = spatial_temporal_mode
        if temperature_output_mode not in TEMPERATURE_OUTPUT_MODES:
            raise ValueError(
                f"temperature output mode must be one of {TEMPERATURE_OUTPUT_MODES}"
            )
        self.temperature_output_mode = temperature_output_mode
        if temperature_output_mode in LITERATURE_BOUNDED_TEMPERATURE_OUTPUT_MODES:
            if (
                temperature_mean_k_by_node_type is None
                or temperature_std_k_by_node_type is None
                or temperature_bounds_k_by_node_type is None
            ):
                raise ValueError(
                    "bounded temperature output requires mean, standard deviation "
                    "and literature bounds by node type"
                )
            if temperature_mean_k_by_node_type.shape != (2,):
                raise ValueError("temperature mean must contain fluid and solid values")
            if temperature_std_k_by_node_type.shape != (2,):
                raise ValueError(
                    "temperature standard deviation must contain fluid and solid values"
                )
            if temperature_bounds_k_by_node_type.shape != (2, 2):
                raise ValueError(
                    "temperature bounds must have [fluid/solid, lower/upper] shape"
                )
            if torch.any(temperature_std_k_by_node_type <= 0):
                raise ValueError("temperature standard deviations must be positive")
            if torch.any(
                temperature_bounds_k_by_node_type[:, 1]
                <= temperature_bounds_k_by_node_type[:, 0]
            ):
                raise ValueError("temperature upper bounds must exceed lower bounds")
        else:
            temperature_mean_k_by_node_type = torch.empty(0)
            temperature_std_k_by_node_type = torch.empty(0)
            temperature_bounds_k_by_node_type = torch.empty((0, 2))
        self.register_buffer(
            "temperature_mean_k_by_node_type",
            temperature_mean_k_by_node_type.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "temperature_std_k_by_node_type",
            temperature_std_k_by_node_type.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "temperature_bounds_k_by_node_type",
            temperature_bounds_k_by_node_type.detach().clone(),
            persistent=False,
        )
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
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        self.local_post = nn.ModuleList(
            LocalEdgeUpdatingBlock(hidden_dim) for _ in range(local_post_iterations)
        )
        self.temperature_change = LeakyMLP(hidden_dim, 1, hidden_dim)

    def _spatial_preprocess(
        self,
        nodes: torch.Tensor,
        base_edges: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.local_pre:
            nodes = block(nodes, base_edges, edge_source, edge_target)
        for block in self.global_blocks:
            nodes = block(nodes)
        return nodes

    def _spatial_refine(
        self,
        nodes: torch.Tensor,
        base_edges: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.local_post:
            nodes = block(nodes, base_edges, edge_source, edge_target)
        return nodes

    def _time_chunked_spatial(
        self,
        values: torch.Tensor,
        base_edges: torch.Tensor,
        graph: P418ThermalStepRegionalGraph,
        processor,
    ) -> torch.Tensor:
        batch, time_count, node_count, hidden = values.shape
        result = []
        for start in range(0, time_count, self.spatial_time_chunk_size):
            stop = min(start + self.spatial_time_chunk_size, time_count)
            current = values[:, start:stop].reshape(
                batch * (stop - start), node_count, hidden
            )
            if self.training and current.requires_grad:
                current = checkpoint(
                    processor,
                    current,
                    base_edges,
                    graph.edge_source,
                    graph.edge_target,
                    use_reentrant=False,
                )
            else:
                current = processor(
                    current, base_edges, graph.edge_source, graph.edge_target
                )
            result.append(current.reshape(batch, stop - start, node_count, hidden))
        return torch.cat(result, dim=1)

    def _single_spatial_pass(
        self,
        values: torch.Tensor,
        base_edges: torch.Tensor,
        graph: P418ThermalStepRegionalGraph,
        processor,
    ) -> torch.Tensor:
        if self.training and values.requires_grad:
            return checkpoint(
                processor,
                values,
                base_edges,
                graph.edge_source,
                graph.edge_target,
                use_reentrant=False,
            )
        return processor(values, base_edges, graph.edge_source, graph.edge_target)

    def _temporal_mix(self, values: torch.Tensor) -> torch.Tensor:
        # values: [batch,time,node,hidden]
        batch, time_count, node_count, hidden = values.shape
        by_node = values.permute(0, 2, 1, 3).reshape(batch * node_count, time_count, hidden)
        chunk = self.temporal_node_chunk_size or len(by_node)
        mixed = []
        for start in range(0, len(by_node), chunk):
            current = by_node[start : start + chunk]
            if self.training and current.requires_grad:
                mixed.append(
                    checkpoint(self.temporal, current, use_reentrant=False)
                )
            else:
                mixed.append(self.temporal(current))
        joined = torch.cat(mixed, dim=0).reshape(batch, node_count, time_count, hidden)
        return joined.permute(0, 2, 1, 3)

    def forward(
        self,
        initial_state: torch.Tensor,
        normalized_condition: torch.Tensor,
        normalized_time: torch.Tensor,
        graph: P418ThermalStepRegionalGraph,
    ) -> torch.Tensor:
        if initial_state.ndim != 3 or initial_state.shape[1:] != (graph.node_count, 5):
            raise ValueError("initial state must have shape [batch,node,5]")
        batch = initial_state.shape[0]
        if normalized_condition.shape != (batch, self.condition_dim):
            raise ValueError(
                f"condition must have shape [batch,{self.condition_dim}]"
            )
        if normalized_time.ndim == 1:
            normalized_time = normalized_time.unsqueeze(0).expand(batch, -1)
        if normalized_time.ndim != 2 or normalized_time.shape[0] != batch:
            raise ValueError("time must have shape [time] or [batch,time]")
        if torch.any(normalized_time < 0):
            raise ValueError("dimensionless time must be nonnegative")
        if graph.boundary_role_count != self.boundary_role_count:
            raise ValueError("graph boundary roles differ from the model")

        time_count = normalized_time.shape[1]
        state_latent = self.state_encoder(initial_state)[:, None]
        condition_latent = self.condition_encoder(normalized_condition)[:, None, None]
        time_latent = self.time_encoder(normalized_time.unsqueeze(-1))[:, :, None]
        structure_latent = self.structure_encoder(graph.structural_features())[None, None]
        fixed_latent = state_latent + condition_latent + structure_latent
        base_edges = self.edge_encoder(graph.edge_features())
        if self.spatial_temporal_mode == "repeated_query_spatial":
            latent = fixed_latent + time_latent
            latent = self._time_chunked_spatial(
                latent, base_edges, graph, self._spatial_preprocess
            )
        else:
            # Geometry, the converged hydrodynamic state, and the step
            # condition are fixed over one thermal-response curve.  Encode
            # that spatial context once, then add the time coordinate before
            # the temporal mixer.  The time-dependent local refinement and
            # full transient physics losses remain unchanged.
            fixed_latent = fixed_latent[:, 0]
            fixed_latent = self._single_spatial_pass(
                fixed_latent, base_edges, graph, self._spatial_preprocess
            )
            latent = fixed_latent[:, None] + time_latent
        latent = self._temporal_mix(latent)
        latent = self._time_chunked_spatial(
            latent, base_edges, graph, self._spatial_refine
        )
        latent = latent.reshape(batch * time_count, graph.node_count, self.hidden_dim)
        temperature_change = self.temperature_change(latent).reshape(
            batch, time_count, graph.node_count
        )

        output = initial_state[:, None].expand(-1, time_count, -1, -1).clone()
        if self.temperature_output_mode == "additive_normalized":
            output[..., 4] = (
                initial_state[:, None, :, 4]
                + normalized_time[:, :, None] * temperature_change
            )
        elif self.temperature_output_mode == "literature_bounded_logit":
            material = graph.node_type
            mean_k = self.temperature_mean_k_by_node_type[material]
            std_k = self.temperature_std_k_by_node_type[material]
            lower_k = self.temperature_bounds_k_by_node_type[material, 0]
            upper_k = self.temperature_bounds_k_by_node_type[material, 1]
            span_k = upper_k - lower_k
            initial_temperature_k = initial_state[..., 4] * std_k + mean_k
            if torch.any(initial_temperature_k < lower_k) or torch.any(
                initial_temperature_k > upper_k
            ):
                raise ValueError(
                    "initial temperature is outside the literature-backed output range"
                )
            fraction = (initial_temperature_k - lower_k) / span_k
            epsilon = torch.finfo(fraction.dtype).eps
            initial_logit = torch.logit(
                fraction.clamp(min=epsilon, max=1.0 - epsilon)
            )
            temperature_k = lower_k + span_k * torch.sigmoid(
                initial_logit[:, None]
                + normalized_time[:, :, None] * temperature_change
            )
            temperature_normalized = (temperature_k - mean_k) / std_k
            output[..., 4] = torch.where(
                normalized_time[:, :, None] == 0,
                initial_state[:, None, :, 4],
                temperature_normalized,
            )
        else:
            material = graph.node_type
            mean_k = self.temperature_mean_k_by_node_type[material]
            std_k = self.temperature_std_k_by_node_type[material]
            lower_k = self.temperature_bounds_k_by_node_type[material, 0]
            upper_k = self.temperature_bounds_k_by_node_type[material, 1]
            initial_temperature_k = initial_state[..., 4] * std_k + mean_k
            if torch.any(initial_temperature_k < lower_k) or torch.any(
                initial_temperature_k > upper_k
            ):
                raise ValueError(
                    "initial temperature is outside the literature-backed output range"
                )

            raw_delta_k = (
                normalized_time[:, :, None]
                * temperature_change
                * std_k[None, None]
            )
            positive_capacity_k = upper_k - initial_temperature_k
            negative_capacity_k = initial_temperature_k - lower_k
            capacity_k = torch.where(
                raw_delta_k >= 0,
                positive_capacity_k[:, None],
                negative_capacity_k[:, None],
            )
            scale_floor_k = torch.finfo(raw_delta_k.dtype).eps
            bounded_delta_k = capacity_k * torch.tanh(
                raw_delta_k / capacity_k.clamp_min(scale_floor_k)
            )
            temperature_k = initial_temperature_k[:, None] + bounded_delta_k
            temperature_normalized = (temperature_k - mean_k) / std_k
            output[..., 4] = torch.where(
                normalized_time[:, :, None] == 0,
                initial_state[:, None, :, 4],
                temperature_normalized,
            )
        return output
