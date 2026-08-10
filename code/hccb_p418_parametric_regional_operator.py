#!/usr/bin/env python3
"""Parametric multilevel regional operator for P418 CHT fields.

The learned weights are shared across compatible packed-bed meshes.  Each
packing supplies its own finite-volume coordinates, volumes, graph edges and
fine-to-regional maps at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def tensor(array: np.ndarray, *, integer: bool = False) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(array), dtype=torch.long if integer else torch.float32
    )


@dataclass
class P418RegionalLevel:
    centroid_m: torch.Tensor
    volume_m3: torch.Tensor
    node_type: torch.Tensor
    boundary_fraction: torch.Tensor
    parent_from_finer: torch.Tensor
    edge_source: torch.Tensor
    edge_target: torch.Tensor
    edge_kind: torch.Tensor
    edge_area_m2: torch.Tensor
    edge_area_vector_m2: torch.Tensor
    edge_face_centroid_m: torch.Tensor

    def to(self, device: torch.device | str) -> "P418RegionalLevel":
        return P418RegionalLevel(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()
            }
        )


@dataclass
class P418RegionalMesh:
    fine_centroid_m: torch.Tensor
    fine_volume_m3: torch.Tensor
    fine_node_type: torch.Tensor
    fine_boundary_role: torch.Tensor
    coordinate_center_m: torch.Tensor
    coordinate_scale_m: torch.Tensor
    volume_scale_m3: torch.Tensor
    levels: tuple[P418RegionalLevel, ...]

    @property
    def n_fine(self) -> int:
        return int(self.fine_node_type.numel())

    def to(self, device: torch.device | str) -> "P418RegionalMesh":
        return P418RegionalMesh(
            fine_centroid_m=self.fine_centroid_m.to(device),
            fine_volume_m3=self.fine_volume_m3.to(device),
            fine_node_type=self.fine_node_type.to(device),
            fine_boundary_role=self.fine_boundary_role.to(device),
            coordinate_center_m=self.coordinate_center_m.to(device),
            coordinate_scale_m=self.coordinate_scale_m.to(device),
            volume_scale_m3=self.volume_scale_m3.to(device),
            levels=tuple(level.to(device) for level in self.levels),
        )


def load_p418_regional_mesh(
    regional_topology_path: Path, model_geometry_path: Path
) -> P418RegionalMesh:
    regional = np.load(regional_topology_path, allow_pickle=False)
    geometry = np.load(model_geometry_path, allow_pickle=False)
    levels: list[P418RegionalLevel] = []
    level = 0
    while f"level_{level}_node_type" in regional.files:
        levels.append(
            P418RegionalLevel(
                centroid_m=tensor(regional[f"level_{level}_centroid_m"]),
                volume_m3=tensor(regional[f"level_{level}_volume_m3"]),
                node_type=tensor(regional[f"level_{level}_node_type"], integer=True),
                boundary_fraction=tensor(
                    geometry[f"level_{level}_boundary_volume_fraction"]
                ),
                parent_from_finer=tensor(
                    regional[f"level_{level}_parent_from_finer"], integer=True
                ),
                edge_source=tensor(
                    regional[f"level_{level}_edge_source"], integer=True
                ),
                edge_target=tensor(
                    regional[f"level_{level}_edge_target"], integer=True
                ),
                edge_kind=tensor(
                    regional[f"level_{level}_edge_kind"], integer=True
                ),
                edge_area_m2=tensor(regional[f"level_{level}_edge_area_m2"]),
                edge_area_vector_m2=tensor(
                    regional[f"level_{level}_edge_area_vector_m2"]
                ),
                edge_face_centroid_m=tensor(
                    regional[f"level_{level}_edge_centroid_m"]
                ),
            )
        )
        level += 1
    if not levels:
        raise ValueError("regional topology contains no active level")
    mesh = P418RegionalMesh(
        fine_centroid_m=tensor(regional["fine_node_centroid_m"]),
        fine_volume_m3=tensor(regional["fine_node_volume_m3"]),
        fine_node_type=tensor(regional["fine_node_type"], integer=True),
        fine_boundary_role=tensor(geometry["fine_boundary_role"]),
        coordinate_center_m=tensor(geometry["coordinate_center_m"]),
        coordinate_scale_m=tensor(geometry["coordinate_scale_m"]),
        volume_scale_m3=tensor(geometry["volume_scale_m3"]),
        levels=tuple(levels),
    )
    validate_mesh(mesh)
    return mesh


def validate_mesh(mesh: P418RegionalMesh) -> None:
    if mesh.fine_centroid_m.shape != (mesh.n_fine, 3):
        raise ValueError("fine centroids must have shape [cell,3]")
    if mesh.fine_boundary_role.shape[0] != mesh.n_fine:
        raise ValueError("fine boundary features do not match the cell count")
    previous_count = mesh.n_fine
    previous_type = mesh.fine_node_type
    for index, level in enumerate(mesh.levels):
        if level.parent_from_finer.shape != (previous_count,):
            raise ValueError(f"level {index} parent map has the wrong size")
        if torch.any(previous_type != level.node_type[level.parent_from_finer]):
            raise ValueError(f"level {index} mixes fluid and solid nodes")
        if level.edge_source.shape != level.edge_target.shape:
            raise ValueError(f"level {index} edge arrays differ in size")
        if level.edge_face_centroid_m.shape != (len(level.edge_source), 3):
            raise ValueError(f"level {index} edge face centroids have the wrong size")
        previous_count = len(level.node_type)
        previous_type = level.node_type


def collapse_mesh_to_level(
    mesh: P418RegionalMesh, level_index: int
) -> P418RegionalMesh:
    """Keep one regional graph and compose its map from all fine FV cells.

    This is the memory-bounded training representation: the native fine-cell
    geometry remains available to the decoder, while unused finer regional
    edge sets are not copied to the accelerator.
    """
    if not 0 <= level_index < len(mesh.levels):
        raise ValueError("regional level is outside the supplied hierarchy")
    parent = mesh.levels[0].parent_from_finer
    for index in range(1, level_index + 1):
        parent = mesh.levels[index].parent_from_finer[parent]
    source = mesh.levels[level_index]
    collapsed_level = P418RegionalLevel(
        centroid_m=source.centroid_m,
        volume_m3=source.volume_m3,
        node_type=source.node_type,
        boundary_fraction=source.boundary_fraction,
        parent_from_finer=parent,
        edge_source=source.edge_source,
        edge_target=source.edge_target,
        edge_kind=source.edge_kind,
        edge_area_m2=source.edge_area_m2,
        edge_area_vector_m2=source.edge_area_vector_m2,
        edge_face_centroid_m=source.edge_face_centroid_m,
    )
    collapsed = P418RegionalMesh(
        fine_centroid_m=mesh.fine_centroid_m,
        fine_volume_m3=mesh.fine_volume_m3,
        fine_node_type=mesh.fine_node_type,
        fine_boundary_role=mesh.fine_boundary_role,
        coordinate_center_m=mesh.coordinate_center_m,
        coordinate_scale_m=mesh.coordinate_scale_m,
        volume_scale_m3=mesh.volume_scale_m3,
        levels=(collapsed_level,),
    )
    validate_mesh(collapsed)
    return collapsed


def segment_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros((values.shape[0], size, values.shape[-1]))
    output.index_add_(1, index, values)
    count = values.new_zeros(size)
    count.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return output / torch.clamp_min(count, 1.0).view(1, -1, 1)


def segment_volume_mean(
    values: torch.Tensor,
    volume: torch.Tensor,
    parent: torch.Tensor,
    size: int,
) -> torch.Tensor:
    output = values.new_zeros((values.shape[0], size, values.shape[-1]))
    output.index_add_(1, parent, values * volume.view(1, -1, 1))
    denominator = values.new_zeros(size)
    denominator.index_add_(0, parent, volume)
    return output / denominator.view(1, -1, 1)


def allocate_processor_steps(total_steps: int, levels: int) -> tuple[int, ...]:
    """Distribute one total processor depth across the active regional levels."""
    if total_steps <= 0 or levels <= 0:
        raise ValueError("processor steps and levels must be positive")
    base, remainder = divmod(total_steps, levels)
    allocation = tuple(base + (index < remainder) for index in range(levels))
    if sum(allocation) != total_steps:
        raise RuntimeError("processor-step allocation changed the total depth")
    return allocation


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class RegionalMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = MLP(2 * hidden_dim + 10, hidden_dim, hidden_dim)
        self.update = MLP(2 * hidden_dim, hidden_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        edge = edge_features.unsqueeze(0).expand(nodes.shape[0], -1, -1)
        message = self.message(
            torch.cat((nodes[:, source], nodes[:, target], edge), dim=-1)
        )
        aggregate = segment_mean(message, target, nodes.shape[1])
        return self.normalization(
            nodes + self.update(torch.cat((nodes, aggregate), dim=-1))
        )


class RegionalEdgeAttentionBlock(nn.Module):
    """Sparse edge attention for coarser regional levels."""

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_bias = MLP(10, heads, hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.feed_forward = MLP(hidden_dim, hidden_dim, 2 * hidden_dim)
        self.normalization_1 = nn.LayerNorm(hidden_dim)
        self.normalization_2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        batch, node_count, hidden = nodes.shape
        query = self.query(nodes).view(batch, node_count, self.heads, self.head_dim)
        key = self.key(nodes).view(batch, node_count, self.heads, self.head_dim)
        value = self.value(nodes).view(batch, node_count, self.heads, self.head_dim)
        score = torch.sum(query[:, target] * key[:, source], dim=-1)
        score = score / self.head_dim ** 0.5
        score = score + self.edge_bias(edge_features).unsqueeze(0)
        target_index = target.view(1, -1, 1).expand(batch, -1, self.heads)
        maximum = score.new_full((batch, node_count, self.heads), -torch.inf)
        maximum.scatter_reduce_(1, target_index, score, reduce="amax", include_self=True)
        weight = torch.exp(score - maximum.gather(1, target_index))
        denominator = score.new_zeros((batch, node_count, self.heads))
        denominator.scatter_add_(1, target_index, weight)
        weight = weight / torch.clamp_min(
            denominator.gather(1, target_index), torch.finfo(weight.dtype).tiny
        )
        message = value[:, source] * weight.unsqueeze(-1)
        aggregate = nodes.new_zeros((batch, node_count, self.heads, self.head_dim))
        aggregate.index_add_(1, target, message)
        attended = self.projection(aggregate.reshape(batch, node_count, hidden))
        nodes = self.normalization_1(nodes + attended)
        return self.normalization_2(nodes + self.feed_forward(nodes))


class RegionalPhysicsAttentionBlock(nn.Module):
    """Transolver Physics-Attention applied to coarse regional nodes.

    The implementation follows the official irregular-mesh slice, attend and
    deslice sequence.  Native finite-volume aggregation is performed before
    this block, so attention is never formed over all original cells.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int = 8,
        slice_count: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        if slice_count <= 0:
            raise ValueError("physics slice count must be positive")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.slice_count = slice_count
        self.scale = self.head_dim**-0.5
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.input_x = nn.Linear(hidden_dim, hidden_dim)
        self.input_value = nn.Linear(hidden_dim, hidden_dim)
        self.slice_projection = nn.Linear(self.head_dim, slice_count)
        nn.init.orthogonal_(self.slice_projection.weight)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))
        self.normalization_1 = nn.LayerNorm(hidden_dim)
        self.normalization_2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_source: torch.Tensor | None = None,
        edge_target: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_source, edge_target, edge_features
        batch, node_count, hidden = nodes.shape
        normalized = self.normalization_1(nodes)
        x_mid = self.input_x(normalized).reshape(
            batch, node_count, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        value_mid = self.input_value(normalized).reshape(
            batch, node_count, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weight = torch.softmax(self.slice_projection(x_mid) / temperature, dim=-1)
        slice_norm = slice_weight.sum(dim=2).clamp_min(1.0e-5)
        slice_token = torch.einsum("bhnc,bhns->bhsc", value_mid, slice_weight)
        slice_token = slice_token / slice_norm.unsqueeze(-1)

        query = self.query(slice_token)
        key = self.key(slice_token)
        value = self.value(slice_token)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) * self.scale, dim=-1
        )
        attended_slice = torch.matmul(self.dropout(attention), value)
        attended_node = torch.einsum("bhsc,bhns->bhnc", attended_slice, slice_weight)
        attended_node = attended_node.permute(0, 2, 1, 3).reshape(
            batch, node_count, hidden
        )
        nodes = nodes + self.output(attended_node)
        return nodes + self.feed_forward(self.normalization_2(nodes))


class HCCBP418ParametricRegionalOperator(nn.Module):
    """Map five published operating inputs to pore-resolved steady CHT fields."""

    def __init__(
        self,
        *,
        boundary_role_count: int,
        hidden_dim: int = 128,
        processor_steps: int = 12,
        active_levels: int = 6,
        start_level: int = 0,
        condition_dim: int = 5,
        output_dim: int = 5,
        processor_kind: str = "message_passing",
        attention_heads: int = 8,
        attention_start_level: int = 3,
        physics_slices: int = 32,
    ) -> None:
        super().__init__()
        if min(hidden_dim, processor_steps, active_levels) <= 0 or start_level < 0:
            raise ValueError("model dimensions must be positive")
        self.boundary_role_count = boundary_role_count
        self.hidden_dim = hidden_dim
        self.active_levels = active_levels
        self.start_level = start_level
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        if processor_kind not in {
            "message_passing",
            "hybrid_attention",
            "hybrid_physics_attention",
        }:
            raise ValueError(
                "processor kind must be message_passing, hybrid_attention or "
                "hybrid_physics_attention"
            )
        if processor_kind != "message_passing" and not 0 <= attention_start_level < active_levels:
            raise ValueError("attention start level is outside the active hierarchy")
        self.processor_kind = processor_kind
        self.attention_start_level = attention_start_level
        self.processor_steps_by_level = allocate_processor_steps(
            processor_steps, active_levels
        )
        structural_dim = 3 + 1 + 2 + boundary_role_count
        self.condition_encoder = MLP(condition_dim, hidden_dim, hidden_dim)
        self.structure_encoder = MLP(structural_dim, hidden_dim, hidden_dim)
        self.processors = nn.ModuleList()
        for level, level_steps in enumerate(self.processor_steps_by_level):
            use_attention = (
                processor_kind != "message_passing" and level >= attention_start_level
            )
            if use_attention and processor_kind == "hybrid_physics_attention":
                block_factory = lambda: RegionalPhysicsAttentionBlock(
                    hidden_dim,
                    heads=attention_heads,
                    slice_count=physics_slices,
                    dropout=0.0,
                )
            elif use_attention:
                block_factory = lambda: RegionalEdgeAttentionBlock(
                    hidden_dim, attention_heads
                )
            else:
                block_factory = lambda: RegionalMessageBlock(hidden_dim)
            self.processors.append(
                nn.ModuleList(
                    block_factory()
                    for _ in range(level_steps)
                )
            )
        self.up_fuse = nn.ModuleList(
            MLP(2 * hidden_dim, hidden_dim, hidden_dim)
            for _ in range(active_levels - 1)
        )
        self.down_fuse = nn.ModuleList(
            MLP(2 * hidden_dim, hidden_dim, hidden_dim)
            for _ in range(active_levels - 1)
        )
        self.fine_decoder = MLP(
            2 * hidden_dim + structural_dim, hidden_dim, hidden_dim
        )
        self.output_head = MLP(hidden_dim, output_dim, hidden_dim)

    def validate_runtime_mesh(self, mesh: P418RegionalMesh) -> None:
        """Check that a packing-specific mesh matches the shared model inputs."""
        validate_mesh(mesh)
        if mesh.fine_boundary_role.ndim != 2:
            raise ValueError("fine boundary features must have shape [cell,role]")
        if mesh.fine_boundary_role.shape[1] != self.boundary_role_count:
            raise ValueError(
                "packing boundary-role count differs from the trained model"
            )
        for index, level in enumerate(mesh.levels):
            if level.boundary_fraction.ndim != 2:
                raise ValueError(
                    f"level {index} boundary features must have shape [node,role]"
                )
            if level.boundary_fraction.shape != (
                len(level.node_type),
                self.boundary_role_count,
            ):
                raise ValueError(
                    f"level {index} boundary features differ from the trained model"
                )
        if torch.any(mesh.coordinate_scale_m <= 0) or torch.any(
            mesh.volume_scale_m3 <= 0
        ):
            raise ValueError("packing geometry normalization scales must be positive")

    @staticmethod
    def _structural_features(
        centroid: torch.Tensor,
        volume: torch.Tensor,
        node_type: torch.Tensor,
        boundary: torch.Tensor,
        mesh: P418RegionalMesh,
    ) -> torch.Tensor:
        coordinate = (centroid - mesh.coordinate_center_m) / mesh.coordinate_scale_m
        log_volume = torch.log(volume / mesh.volume_scale_m3).unsqueeze(-1)
        type_feature = F.one_hot(node_type, num_classes=2).to(centroid.dtype)
        return torch.cat((coordinate, log_volume, type_feature, boundary), dim=-1)

    @staticmethod
    def _edge_features(
        level: P418RegionalLevel, mesh: P418RegionalMesh
    ) -> torch.Tensor:
        relative = (
            level.centroid_m[level.edge_target]
            - level.centroid_m[level.edge_source]
        ) / mesh.coordinate_scale_m
        area_scale = torch.pow(mesh.volume_scale_m3, 2.0 / 3.0)
        area_vector = level.edge_area_vector_m2 / area_scale
        log_area = torch.log(level.edge_area_m2 / area_scale).unsqueeze(-1)
        kind = F.one_hot(level.edge_kind, num_classes=3).to(relative.dtype)
        return torch.cat((relative, area_vector, log_area, kind), dim=-1)

    def encode_regions(
        self, normalized_condition: torch.Tensor, mesh: P418RegionalMesh
    ) -> torch.Tensor:
        self.validate_runtime_mesh(mesh)
        if normalized_condition.ndim != 2 or normalized_condition.shape[1] != self.condition_dim:
            raise ValueError(f"condition input must have shape [batch,{self.condition_dim}]")
        if self.start_level + self.active_levels > len(mesh.levels):
            raise ValueError("requested active levels exceed the supplied hierarchy")
        condition_latent = self.condition_encoder(normalized_condition)
        skip: list[torch.Tensor] = []
        current: torch.Tensor | None = None
        for local_index in range(self.active_levels):
            mesh_index = self.start_level + local_index
            level = mesh.levels[mesh_index]
            structure = self.structure_encoder(
                self._structural_features(
                    level.centroid_m,
                    level.volume_m3,
                    level.node_type,
                    level.boundary_fraction,
                    mesh,
                )
            ).unsqueeze(0)
            if local_index == 0:
                current = structure + condition_latent[:, None, :]
            else:
                assert current is not None
                current = self.up_fuse[local_index - 1](
                    torch.cat((current, structure.expand_as(current)), dim=-1)
                )
            for block in self.processors[local_index]:
                current = block(
                    current,
                    level.edge_source,
                    level.edge_target,
                    self._edge_features(level, mesh),
                )
            skip.append(current)
            if local_index + 1 < self.active_levels:
                next_level = mesh.levels[mesh_index + 1]
                current = segment_volume_mean(
                    current,
                    level.volume_m3,
                    next_level.parent_from_finer,
                    len(next_level.node_type),
                )

        assert current is not None
        for child in reversed(range(self.active_levels - 1)):
            parent = mesh.levels[self.start_level + child + 1].parent_from_finer
            current = self.down_fuse[child](
                torch.cat((skip[child], current[:, parent]), dim=-1)
            )
        return current

    def _fine_parent_for_start_level(
        self, mesh: P418RegionalMesh, start: int, stop: int
    ) -> torch.Tensor:
        """Map a fine-cell slice to the first regional level used by the model."""
        parent = mesh.levels[0].parent_from_finer[start:stop]
        for mesh_index in range(1, self.start_level + 1):
            parent = mesh.levels[mesh_index].parent_from_finer[parent]
        return parent

    def decode_fine_chunk(
        self,
        normalized_condition: torch.Tensor,
        level_zero_latent: torch.Tensor,
        mesh: P418RegionalMesh,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        if not 0 <= start < stop <= mesh.n_fine:
            raise ValueError("invalid fine-cell chunk")
        fine_slice = slice(start, stop)
        fine_structure = self._structural_features(
            mesh.fine_centroid_m[fine_slice],
            mesh.fine_volume_m3[fine_slice],
            mesh.fine_node_type[fine_slice],
            mesh.fine_boundary_role[fine_slice],
            mesh,
        )
        parent = self._fine_parent_for_start_level(mesh, start, stop)
        condition_latent = self.condition_encoder(normalized_condition)
        decoder_input = torch.cat(
            (
                level_zero_latent[:, parent],
                condition_latent[:, None, :].expand(-1, stop - start, -1),
                fine_structure.unsqueeze(0).expand(normalized_condition.shape[0], -1, -1),
            ),
            dim=-1,
        )
        return self.output_head(self.fine_decoder(decoder_input))

    def decode_active_regions(
        self,
        normalized_condition: torch.Tensor,
        regional_latent: torch.Tensor,
        mesh: P418RegionalMesh,
    ) -> torch.Tensor:
        """Decode physical channels on the finest active regional graph."""
        level = mesh.levels[self.start_level]
        if regional_latent.shape[:2] != (
            normalized_condition.shape[0],
            len(level.node_type),
        ):
            raise ValueError("regional latent and active regional mesh differ")
        structure = self._structural_features(
            level.centroid_m,
            level.volume_m3,
            level.node_type,
            level.boundary_fraction,
            mesh,
        )
        condition_latent = self.condition_encoder(normalized_condition)
        decoder_input = torch.cat(
            (
                regional_latent,
                condition_latent[:, None, :].expand(
                    -1, len(level.node_type), -1
                ),
                structure.unsqueeze(0).expand(
                    normalized_condition.shape[0], -1, -1
                ),
            ),
            dim=-1,
        )
        return self.output_head(self.fine_decoder(decoder_input))

    def iter_decode_chunks(
        self,
        normalized_condition: torch.Tensor,
        mesh: P418RegionalMesh,
        chunk_size: int,
    ) -> Iterator[tuple[int, int, torch.Tensor]]:
        if chunk_size <= 0:
            raise ValueError("chunk size must be positive")
        regional = self.encode_regions(normalized_condition, mesh)
        for start in range(0, mesh.n_fine, chunk_size):
            stop = min(start + chunk_size, mesh.n_fine)
            yield start, stop, self.decode_fine_chunk(
                normalized_condition, regional, mesh, start, stop
            )

    def forward(
        self,
        normalized_condition: torch.Tensor,
        mesh: P418RegionalMesh,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        return torch.cat(
            [values for _, _, values in self.iter_decode_chunks(normalized_condition, mesh, chunk_size)],
            dim=1,
        )


def output_mask(node_type: torch.Tensor) -> torch.Tensor:
    """Valid channels are [Ux,Uy,Uz,gauge pressure,T]."""
    mask = torch.zeros((len(node_type), 5), dtype=torch.bool, device=node_type.device)
    mask[node_type == 0, :] = True
    mask[node_type == 1, 4] = True
    return mask
