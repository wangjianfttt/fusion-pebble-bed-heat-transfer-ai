#!/usr/bin/env python3
"""Coordinate PINN baseline for P418 conjugate heat transfer meshes."""

from __future__ import annotations

from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from hccb_p418_parametric_regional_operator import P418RegionalMesh, validate_mesh


class HCCBP418CoordinatePINNOperator(nn.Module):
    """Pointwise tanh network constrained by the shared FV balance losses.

    The six hidden layers and width 50 follow the archived PINO-paper PINN
    baseline.  P418 operating conditions and packing-specific structural features
    replace the Navier--Stokes coordinates used by that reference code.
    """

    def __init__(
        self,
        *,
        boundary_role_count: int,
        hidden_dim: int = 50,
        hidden_layers: int = 6,
        condition_dim: int = 5,
        output_dim: int = 5,
        start_level: int = 0,
    ) -> None:
        super().__init__()
        if min(boundary_role_count, hidden_dim, hidden_layers, condition_dim, output_dim) <= 0:
            raise ValueError("PINN dimensions must be positive")
        if start_level < 0:
            raise ValueError("start level must be non-negative")
        self.boundary_role_count = boundary_role_count
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.start_level = start_level
        structural_dim = 3 + 1 + 2 + boundary_role_count
        input_dim = condition_dim + structural_dim
        layers: list[nn.Linear] = [nn.Linear(input_dim, hidden_dim)]
        layers.extend(nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1))
        self.hidden = nn.ModuleList(layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)
        # The common conservative face-output wrapper requires one condition
        # embedding.  It is separate from the six-layer state PINN.
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.Tanh(),
        )
        self.apply(self._glorot_normal)

    def validate_runtime_mesh(self, mesh: P418RegionalMesh) -> None:
        validate_mesh(mesh)
        if mesh.fine_boundary_role.ndim != 2 or int(
            mesh.fine_boundary_role.shape[1]
        ) != self.boundary_role_count:
            raise ValueError(
                "packing boundary-role count differs from the trained PINN"
            )
        for index, level in enumerate(mesh.levels):
            if level.boundary_fraction.shape != (
                len(level.node_type),
                self.boundary_role_count,
            ):
                raise ValueError(
                    f"level {index} boundary features differ from the trained PINN"
                )

    @staticmethod
    def _glorot_normal(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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

    def _encode_points(
        self,
        normalized_condition: torch.Tensor,
        structure: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_condition.ndim != 2 or normalized_condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"condition input must have shape [batch,{self.condition_dim}]"
            )
        batch = normalized_condition.shape[0]
        count = structure.shape[0]
        values = torch.cat(
            (
                normalized_condition[:, None, :].expand(-1, count, -1),
                structure.unsqueeze(0).expand(batch, -1, -1),
            ),
            dim=-1,
        )
        for layer in self.hidden:
            values = torch.tanh(layer(values))
        return values

    def encode_regions(
        self,
        normalized_condition: torch.Tensor,
        mesh: P418RegionalMesh,
    ) -> torch.Tensor:
        self.validate_runtime_mesh(mesh)
        if self.start_level >= len(mesh.levels):
            raise ValueError("requested PINN level exceeds the supplied hierarchy")
        level = mesh.levels[self.start_level]
        structure = self._structural_features(
            level.centroid_m,
            level.volume_m3,
            level.node_type,
            level.boundary_fraction,
            mesh,
        )
        return self._encode_points(normalized_condition, structure)

    def decode_active_regions(
        self,
        normalized_condition: torch.Tensor,
        regional_latent: torch.Tensor,
        mesh: P418RegionalMesh,
    ) -> torch.Tensor:
        level = mesh.levels[self.start_level]
        expected = (normalized_condition.shape[0], len(level.node_type), self.hidden_dim)
        if tuple(regional_latent.shape) != expected:
            raise ValueError("PINN latent field and active regional mesh differ")
        return self.output_head(regional_latent)

    def decode_fine_chunk(
        self,
        normalized_condition: torch.Tensor,
        regional_latent: torch.Tensor,
        mesh: P418RegionalMesh,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        del regional_latent
        if not 0 <= start < stop <= mesh.n_fine:
            raise ValueError("invalid fine-cell chunk")
        fine = slice(start, stop)
        structure = self._structural_features(
            mesh.fine_centroid_m[fine],
            mesh.fine_volume_m3[fine],
            mesh.fine_node_type[fine],
            mesh.fine_boundary_role[fine],
            mesh,
        )
        return self.output_head(self._encode_points(normalized_condition, structure))

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
            [
                values
                for _, _, values in self.iter_decode_chunks(
                    normalized_condition, mesh, chunk_size
                )
            ],
            dim=1,
        )
