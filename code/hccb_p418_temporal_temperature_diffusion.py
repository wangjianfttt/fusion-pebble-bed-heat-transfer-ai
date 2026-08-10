#!/usr/bin/env python3
"""Conditional diffusion correction for P418 regional temperature histories.

The deterministic graph--Transformer remains the forward model.  Diffusion is
restricted to its normalized temperature residual and may condition on an
externally supplied measurement mask.  The initial temperature and supplied
measurements are imposed exactly during sampling, so no unregistered guidance
weight is introduced.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from hccb_p418_parametric_regional_operator import MLP, RegionalPhysicsAttentionBlock
from hccb_p418_regional_diffusion_refiner import (
    pde_refiner_schedule,
    timestep_embedding,
)
from sparse_temperature_observation_operator import hard_condition_tensor


FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE = {
    "hidden_dim": 256,
    "spatial_layers": 5,
    "spatial_attention_heads": 8,
    "physics_slices": 32,
    "temporal_layers": 3,
    "temporal_heads": 1,
    "num_refinement_steps": 3,
    "minimum_noise_standard_deviation": 4.0e-7,
}


class P418TemporalTemperatureResidualRefiner(nn.Module):
    """Physics-Attention and temporal-Transformer residual velocity model."""

    def __init__(
        self,
        *,
        structural_dim: int,
        condition_dim: int = 8,
        hidden_dim: int = 256,
        spatial_layers: int = 5,
        spatial_attention_heads: int = 8,
        physics_slices: int = 32,
        temporal_layers: int = 3,
        temporal_heads: int = 1,
        num_refinement_steps: int = 3,
        spatial_time_chunk_size: int = 1,
        temporal_node_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if min(
            structural_dim,
            condition_dim,
            hidden_dim,
            spatial_layers,
            temporal_layers,
            num_refinement_steps,
            spatial_time_chunk_size,
            temporal_node_chunk_size,
        ) <= 0:
            raise ValueError("all temporal diffusion dimensions must be positive")
        if hidden_dim % spatial_attention_heads or hidden_dim % temporal_heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_refinement_steps = num_refinement_steps
        self.spatial_time_chunk_size = spatial_time_chunk_size
        self.temporal_node_chunk_size = temporal_node_chunk_size
        # baseline, noisy residual, observed residual, observation flag, time
        input_dim = 5 + condition_dim + structural_dim
        self.input_encoder = MLP(input_dim, hidden_dim, 2 * hidden_dim)
        self.refinement_encoder = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.spatial = nn.ModuleList(
            RegionalPhysicsAttentionBlock(
                hidden_dim,
                heads=spatial_attention_heads,
                slice_count=physics_slices,
                dropout=0.0,
            )
            for _ in range(spatial_layers)
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
        self.temporal = nn.TransformerEncoder(temporal_layer, temporal_layers)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def _spatial_mix(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.spatial:
            values = block(values)
        return values

    def _mix_spatial_time_chunks(self, values: torch.Tensor) -> torch.Tensor:
        batch, time_count, node_count, hidden = values.shape
        output = []
        for start in range(0, time_count, self.spatial_time_chunk_size):
            stop = min(start + self.spatial_time_chunk_size, time_count)
            current = values[:, start:stop].reshape(
                batch * (stop - start), node_count, hidden
            )
            if self.training and current.requires_grad:
                current = checkpoint(self._spatial_mix, current, use_reentrant=False)
            else:
                current = self._spatial_mix(current)
            output.append(current.reshape(batch, stop - start, node_count, hidden))
        return torch.cat(output, dim=1)

    def _mix_temporal_node_chunks(self, values: torch.Tensor) -> torch.Tensor:
        batch, time_count, node_count, hidden = values.shape
        by_node = values.permute(0, 2, 1, 3).reshape(batch * node_count, time_count, hidden)
        output = []
        for start in range(0, len(by_node), self.temporal_node_chunk_size):
            current = by_node[start : start + self.temporal_node_chunk_size]
            if self.training and current.requires_grad:
                current = checkpoint(self.temporal, current, use_reentrant=False)
            else:
                current = self.temporal(current)
            output.append(current)
        joined = torch.cat(output, dim=0).reshape(batch, node_count, time_count, hidden)
        return joined.permute(0, 2, 1, 3)

    def forward(
        self,
        baseline_temperature: torch.Tensor,
        noised_residual: torch.Tensor,
        normalized_condition: torch.Tensor,
        structural_features: torch.Tensor,
        normalized_time: torch.Tensor,
        observed_residual: torch.Tensor,
        observation_mask: torch.Tensor,
        refinement_step: torch.Tensor,
    ) -> torch.Tensor:
        if baseline_temperature.shape != noised_residual.shape or baseline_temperature.shape[-1] != 1:
            raise ValueError("baseline and residual must have shape [batch,time,node,1]")
        if observed_residual.shape != baseline_temperature.shape or observation_mask.shape != baseline_temperature.shape:
            raise ValueError("observations must match the baseline temperature shape")
        if observation_mask.dtype != torch.bool:
            raise ValueError("observation mask must be boolean")
        batch, time_count, node_count, _ = baseline_temperature.shape
        if normalized_condition.shape != (batch, self.condition_dim):
            raise ValueError("condition shape differs from the temporal diffusion model")
        if structural_features.ndim != 2 or structural_features.shape[0] != node_count:
            raise ValueError("structural features must contain one row per node")
        if normalized_time.ndim == 1:
            normalized_time = normalized_time.unsqueeze(0).expand(batch, -1)
        if normalized_time.shape != (batch, time_count):
            raise ValueError("time must have shape [time] or [batch,time]")
        if refinement_step.shape != (batch,):
            raise ValueError("refinement step must contain one index per curve")

        condition = normalized_condition[:, None, None].expand(-1, time_count, node_count, -1)
        structure = structural_features[None, None].expand(batch, time_count, -1, -1)
        time_feature = normalized_time[:, :, None, None].expand(-1, -1, node_count, -1)
        latent = self.input_encoder(
            torch.cat(
                (
                    baseline_temperature,
                    noised_residual,
                    observed_residual,
                    observation_mask.to(baseline_temperature.dtype),
                    time_feature,
                    condition,
                    structure,
                ),
                dim=-1,
            )
        )
        diffusion_time = refinement_step.float() * (1000.0 / self.num_refinement_steps)
        latent = latent + self.refinement_encoder(
            timestep_embedding(diffusion_time, self.hidden_dim)
        )[:, None, None]
        latent = self._mix_spatial_time_chunks(latent)
        latent = self._mix_temporal_node_chunks(latent)
        return self.output(latent)


@torch.no_grad()
def sample_temporal_temperature_residual(
    model: nn.Module,
    baseline_temperature: torch.Tensor,
    normalized_condition: torch.Tensor,
    structural_features: torch.Tensor,
    normalized_time: torch.Tensor,
    observed_residual: torch.Tensor,
    observation_mask: torch.Tensor,
    *,
    initial_noise: torch.Tensor | None = None,
    num_refinement_steps: int = 3,
    min_noise_std: float = 4.0e-7,
) -> torch.Tensor:
    """Sample a residual while imposing t=0 and supplied temperatures exactly."""
    if observed_residual.shape != baseline_temperature.shape:
        raise ValueError("observed residual and baseline temperatures must have identical shapes")
    if observation_mask.shape != baseline_temperature.shape or observation_mask.dtype != torch.bool:
        raise ValueError("observation mask must be boolean and match the temperature shape")
    hard_mask = observation_mask.clone()
    hard_mask[:, 0] = True
    observed_residual = hard_condition_tensor(
        torch.zeros_like(observed_residual), observed_residual, hard_mask
    )
    observed_residual[:, 0] = 0.0
    current = torch.randn_like(baseline_temperature) if initial_noise is None else initial_noise.clone()
    if current.shape != baseline_temperature.shape:
        raise ValueError("initial noise must match the baseline temperature shape")
    _, alpha = pde_refiner_schedule(
        num_refinement_steps=num_refinement_steps,
        min_noise_std=min_noise_std,
    )
    alpha = alpha.to(device=baseline_temperature.device, dtype=baseline_temperature.dtype)
    for step_index in reversed(range(num_refinement_steps + 1)):
        step = torch.full(
            (baseline_temperature.shape[0],),
            step_index,
            dtype=torch.long,
            device=baseline_temperature.device,
        )
        velocity = model(
            baseline_temperature,
            current,
            normalized_condition,
            structural_features,
            normalized_time,
            observed_residual,
            hard_mask,
            step,
        )
        signal = torch.sqrt(alpha[step_index])
        noise_scale = torch.sqrt(1.0 - alpha[step_index])
        clean = signal * current - noise_scale * velocity
        clean = hard_condition_tensor(clean, observed_residual, hard_mask)
        noise = noise_scale * current + signal * velocity
        if step_index == 0:
            current = clean
        else:
            previous_signal = torch.sqrt(alpha[step_index - 1])
            previous_noise = torch.sqrt(1.0 - alpha[step_index - 1])
            current = previous_signal * clean + previous_noise * noise
    current = hard_condition_tensor(current, observed_residual, hard_mask)
    return current
