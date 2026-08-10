#!/usr/bin/env python3
"""Regional state/face-flux operator for conservative P418 CHT modelling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hccb_p418_parametric_regional_operator import MLP, P418RegionalMesh


@dataclass(frozen=True)
class RegionalMassFluxGeometry:
    internal_owner: torch.Tensor
    internal_neighbour: torch.Tensor
    internal_face_centroid_m: torch.Tensor
    internal_face_area_vector_m2: torch.Tensor
    internal_face_area_m2: torch.Tensor
    boundary_owner: torch.Tensor
    boundary_patch: torch.Tensor
    boundary_face_centroid_m: torch.Tensor
    boundary_face_area_vector_m2: torch.Tensor
    boundary_face_area_m2: torch.Tensor
    patch_count: int


@dataclass(frozen=True)
class RegionalEnergyFluxGeometry:
    internal_owner: torch.Tensor
    internal_neighbour: torch.Tensor
    internal_kind: torch.Tensor
    internal_face_centroid_m: torch.Tensor
    internal_face_area_vector_m2: torch.Tensor
    internal_face_area_m2: torch.Tensor
    boundary_owner: torch.Tensor
    boundary_kind: torch.Tensor
    boundary_face_centroid_m: torch.Tensor
    boundary_face_area_vector_m2: torch.Tensor
    boundary_face_area_m2: torch.Tensor
    internal_kind_count: int
    boundary_kind_count: int


@dataclass(frozen=True)
class ConservativeRegionalOutput:
    regional_state: torch.Tensor
    internal_mass_flow_kg_s: torch.Tensor
    boundary_mass_flow_kg_s: torch.Tensor
    internal_energy_flow_W: torch.Tensor | None = None
    boundary_energy_flow_W: torch.Tensor | None = None


def load_regional_mass_flux_geometry(
    path: Path,
    *,
    patch_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> RegionalMassFluxGeometry:
    with np.load(path, allow_pickle=False) as loaded:
        fluid_global = loaded["fluid_global_region"].astype(np.int64)
        if not np.array_equal(fluid_global, np.arange(len(fluid_global))):
            raise ValueError("mass-flux operator requires leading contiguous fluid nodes")
        def floating(name: str) -> torch.Tensor:
            return torch.as_tensor(loaded[name], device=device, dtype=dtype)

        def index(name: str) -> torch.Tensor:
            return torch.as_tensor(loaded[name], device=device, dtype=torch.long)

        geometry = RegionalMassFluxGeometry(
            internal_owner=index("internal_owner"),
            internal_neighbour=index("internal_neighbour"),
            internal_face_centroid_m=floating("internal_face_centroid_m"),
            internal_face_area_vector_m2=floating("internal_face_area_vector_m2"),
            internal_face_area_m2=floating("internal_face_area_m2"),
            boundary_owner=index("boundary_owner"),
            boundary_patch=index("boundary_patch"),
            boundary_face_centroid_m=floating("boundary_face_centroid_m"),
            boundary_face_area_vector_m2=floating("boundary_face_area_vector_m2"),
            boundary_face_area_m2=floating("boundary_face_area_m2"),
            patch_count=patch_count,
        )
    if torch.any(geometry.boundary_patch < 0) or torch.any(
        geometry.boundary_patch >= patch_count
    ):
        raise ValueError("boundary patch index exceeds the declared patch count")
    return geometry


def load_regional_energy_flux_geometry(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> RegionalEnergyFluxGeometry:
    with np.load(path, allow_pickle=False) as loaded:
        def floating(name: str) -> torch.Tensor:
            return torch.as_tensor(loaded[name], device=device, dtype=dtype)

        def index(name: str) -> torch.Tensor:
            return torch.as_tensor(loaded[name], device=device, dtype=torch.long)

        internal_kind = index("internal_kind")
        boundary_kind = index("boundary_kind")
        geometry = RegionalEnergyFluxGeometry(
            internal_owner=index("internal_owner"),
            internal_neighbour=index("internal_neighbour"),
            internal_kind=internal_kind,
            internal_face_centroid_m=floating("internal_face_centroid_m"),
            internal_face_area_vector_m2=floating("internal_face_area_vector_m2"),
            internal_face_area_m2=floating("internal_face_area_m2"),
            boundary_owner=index("boundary_owner"),
            boundary_kind=boundary_kind,
            boundary_face_centroid_m=floating("boundary_face_centroid_m"),
            boundary_face_area_vector_m2=floating("boundary_face_area_vector_m2"),
            boundary_face_area_m2=floating("boundary_face_area_m2"),
            internal_kind_count=int(torch.max(internal_kind).item()) + 1,
            boundary_kind_count=int(torch.max(boundary_kind).item()) + 1,
        )
    return geometry


class HCCBP418ConservativeMixedOperator(nn.Module):
    """Predict regional fields and conservative mass-flux unknowns together."""

    def __init__(
        self,
        *,
        field_operator: nn.Module,
        patch_count: int,
        internal_mass_scale_kg_s: float = 1.0,
        boundary_mass_scale_kg_s: float = 1.0,
        internal_energy_scale_W: float | None = None,
        boundary_energy_scale_W: float | None = None,
        internal_energy_kind_count: int | None = None,
        boundary_energy_kind_count: int | None = None,
    ) -> None:
        super().__init__()
        if internal_mass_scale_kg_s <= 0.0 or boundary_mass_scale_kg_s <= 0.0:
            raise ValueError("mass-flow scales must be positive")
        self.field_operator = field_operator
        hidden = field_operator.hidden_dim
        self.internal_mass_head = MLP(2 * hidden + 7, 1, hidden)
        self.boundary_mass_head = MLP(2 * hidden + 7 + patch_count, 1, hidden)
        energy_values = (
            internal_energy_scale_W,
            boundary_energy_scale_W,
            internal_energy_kind_count,
            boundary_energy_kind_count,
        )
        if any(value is not None for value in energy_values) and not all(
            value is not None for value in energy_values
        ):
            raise ValueError("all energy-output settings must be supplied together")
        if internal_energy_scale_W is not None:
            if internal_energy_scale_W <= 0.0 or boundary_energy_scale_W <= 0.0:
                raise ValueError("energy-flow scales must be positive")
            if internal_energy_kind_count <= 0 or boundary_energy_kind_count <= 0:
                raise ValueError("energy kind counts must be positive")
            self.internal_energy_head = MLP(
                2 * hidden + 7 + internal_energy_kind_count, 1, hidden
            )
            self.boundary_energy_head = MLP(
                2 * hidden + 7 + boundary_energy_kind_count, 1, hidden
            )
            self.register_buffer(
                "internal_energy_scale_W", torch.tensor(float(internal_energy_scale_W))
            )
            self.register_buffer(
                "boundary_energy_scale_W", torch.tensor(float(boundary_energy_scale_W))
            )
        else:
            self.internal_energy_head = None
            self.boundary_energy_head = None
            self.register_buffer("internal_energy_scale_W", torch.tensor(1.0))
            self.register_buffer("boundary_energy_scale_W", torch.tensor(1.0))
        self.patch_count = patch_count
        self.register_buffer(
            "internal_mass_scale_kg_s", torch.tensor(float(internal_mass_scale_kg_s))
        )
        self.register_buffer(
            "boundary_mass_scale_kg_s", torch.tensor(float(boundary_mass_scale_kg_s))
        )

    @staticmethod
    def _face_geometry_features(
        *,
        owner_centroid: torch.Tensor,
        neighbour_or_face_centroid: torch.Tensor,
        area_vector: torch.Tensor,
        area: torch.Tensor,
        mesh: P418RegionalMesh,
    ) -> torch.Tensor:
        length = mesh.coordinate_scale_m
        area_scale = torch.pow(mesh.volume_scale_m3, 2.0 / 3.0)
        displacement = (neighbour_or_face_centroid - owner_centroid) / length
        scaled_vector = area_vector / area_scale
        log_area = torch.log(area / area_scale).unsqueeze(-1)
        return torch.cat((displacement, scaled_vector, log_area), dim=-1)

    def forward(
        self,
        normalized_condition: torch.Tensor,
        mesh: P418RegionalMesh,
        flux_geometry: RegionalMassFluxGeometry,
        energy_geometry: RegionalEnergyFluxGeometry | None = None,
    ) -> ConservativeRegionalOutput:
        latent = self.field_operator.encode_regions(normalized_condition, mesh)
        state = self.field_operator.decode_active_regions(
            normalized_condition, latent, mesh
        )
        level = mesh.levels[self.field_operator.start_level]
        owner = flux_geometry.internal_owner
        neighbour = flux_geometry.internal_neighbour
        internal_geometry = self._face_geometry_features(
            owner_centroid=level.centroid_m[owner],
            neighbour_or_face_centroid=level.centroid_m[neighbour],
            area_vector=flux_geometry.internal_face_area_vector_m2,
            area=flux_geometry.internal_face_area_m2,
            mesh=mesh,
        ).unsqueeze(0).expand(normalized_condition.shape[0], -1, -1)
        internal_mass = self.internal_mass_head(
            torch.cat((latent[:, owner], latent[:, neighbour], internal_geometry), dim=-1)
        ).squeeze(-1) * self.internal_mass_scale_kg_s

        boundary_owner = flux_geometry.boundary_owner
        boundary_geometry = self._face_geometry_features(
            owner_centroid=level.centroid_m[boundary_owner],
            neighbour_or_face_centroid=flux_geometry.boundary_face_centroid_m,
            area_vector=flux_geometry.boundary_face_area_vector_m2,
            area=flux_geometry.boundary_face_area_m2,
            mesh=mesh,
        )
        patch = F.one_hot(
            flux_geometry.boundary_patch, num_classes=self.patch_count
        ).to(boundary_geometry.dtype)
        condition_latent = self.field_operator.condition_encoder(normalized_condition)
        boundary_features = torch.cat((boundary_geometry, patch), dim=-1)
        boundary_features = boundary_features.unsqueeze(0).expand(
            normalized_condition.shape[0], -1, -1
        )
        boundary_mass = self.boundary_mass_head(
            torch.cat(
                (
                    latent[:, boundary_owner],
                    condition_latent[:, None, :].expand(
                        -1, len(boundary_owner), -1
                    ),
                    boundary_features,
                ),
                dim=-1,
            )
        ).squeeze(-1) * self.boundary_mass_scale_kg_s

        internal_energy = None
        boundary_energy = None
        if self.internal_energy_head is not None:
            if energy_geometry is None:
                raise ValueError("energy geometry is required when energy heads are enabled")
            energy_owner = energy_geometry.internal_owner
            energy_neighbour = energy_geometry.internal_neighbour
            energy_internal_geometry = self._face_geometry_features(
                owner_centroid=level.centroid_m[energy_owner],
                neighbour_or_face_centroid=level.centroid_m[energy_neighbour],
                area_vector=energy_geometry.internal_face_area_vector_m2,
                area=energy_geometry.internal_face_area_m2,
                mesh=mesh,
            )
            energy_internal_kind = F.one_hot(
                energy_geometry.internal_kind,
                num_classes=energy_geometry.internal_kind_count,
            ).to(energy_internal_geometry.dtype)
            energy_internal_features = torch.cat(
                (energy_internal_geometry, energy_internal_kind), dim=-1
            ).unsqueeze(0).expand(normalized_condition.shape[0], -1, -1)
            internal_energy = self.internal_energy_head(
                torch.cat(
                    (
                        latent[:, energy_owner],
                        latent[:, energy_neighbour],
                        energy_internal_features,
                    ),
                    dim=-1,
                )
            ).squeeze(-1) * self.internal_energy_scale_W

            energy_boundary_owner = energy_geometry.boundary_owner
            energy_boundary_geometry = self._face_geometry_features(
                owner_centroid=level.centroid_m[energy_boundary_owner],
                neighbour_or_face_centroid=energy_geometry.boundary_face_centroid_m,
                area_vector=energy_geometry.boundary_face_area_vector_m2,
                area=energy_geometry.boundary_face_area_m2,
                mesh=mesh,
            )
            energy_boundary_kind = F.one_hot(
                energy_geometry.boundary_kind,
                num_classes=energy_geometry.boundary_kind_count,
            ).to(energy_boundary_geometry.dtype)
            energy_boundary_features = torch.cat(
                (energy_boundary_geometry, energy_boundary_kind), dim=-1
            ).unsqueeze(0).expand(normalized_condition.shape[0], -1, -1)
            boundary_energy = self.boundary_energy_head(
                torch.cat(
                    (
                        latent[:, energy_boundary_owner],
                        condition_latent[:, None, :].expand(
                            -1, len(energy_boundary_owner), -1
                        ),
                        energy_boundary_features,
                    ),
                    dim=-1,
                )
            ).squeeze(-1) * self.boundary_energy_scale_W
        return ConservativeRegionalOutput(
            regional_state=state,
            internal_mass_flow_kg_s=internal_mass,
            boundary_mass_flow_kg_s=boundary_mass,
            internal_energy_flow_W=internal_energy,
            boundary_energy_flow_W=boundary_energy,
        )


def regional_mass_balance(
    output: ConservativeRegionalOutput,
    geometry: RegionalMassFluxGeometry,
    fluid_cell_count: int,
) -> torch.Tensor:
    balance = output.internal_mass_flow_kg_s.new_zeros(
        (output.internal_mass_flow_kg_s.shape[0], fluid_cell_count)
    )
    balance.index_add_(1, geometry.internal_owner, output.internal_mass_flow_kg_s)
    balance.index_add_(1, geometry.internal_neighbour, -output.internal_mass_flow_kg_s)
    balance.index_add_(1, geometry.boundary_owner, output.boundary_mass_flow_kg_s)
    return balance


def regional_energy_balance(
    output: ConservativeRegionalOutput,
    geometry: RegionalEnergyFluxGeometry,
    source_power_W: torch.Tensor,
) -> torch.Tensor:
    if output.internal_energy_flow_W is None or output.boundary_energy_flow_W is None:
        raise ValueError("energy-flow output is absent")
    if source_power_W.ndim != 2 or source_power_W.shape[0] != output.internal_energy_flow_W.shape[0]:
        raise ValueError("source power must have batch and regional-node dimensions")
    node_count = source_power_W.shape[1]
    if (
        int(torch.max(geometry.internal_owner).item()) >= node_count
        or int(torch.max(geometry.internal_neighbour).item()) >= node_count
        or int(torch.max(geometry.boundary_owner).item()) >= node_count
    ):
        raise ValueError("energy geometry contains a node outside the source array")
    balance = -source_power_W.clone()
    balance.index_add_(1, geometry.internal_owner, output.internal_energy_flow_W)
    balance.index_add_(1, geometry.internal_neighbour, -output.internal_energy_flow_W)
    balance.index_add_(1, geometry.boundary_owner, output.boundary_energy_flow_W)
    return balance
