#!/usr/bin/env python3
"""PDE-Refiner-style residual model for regional steady CHT fields.

The deterministic PINN/operator prediction remains the primary field.  This
module learns only the normalized residual on regional nodes and follows the
official PDE-Refiner noise schedule and velocity-prediction target.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from hccb_p418_parametric_regional_operator import (
    MLP,
    RegionalPhysicsAttentionBlock,
)


def timestep_embedding(
    timesteps: torch.Tensor, dimension: int, maximum_period: int = 10000
) -> torch.Tensor:
    """Sinusoidal embedding used by the archived Transolver implementation."""
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(maximum_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    argument = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat((torch.cos(argument), torch.sin(argument)), dim=-1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


def pde_refiner_schedule(
    *, num_refinement_steps: int = 3, min_noise_std: float = 4.0e-7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return official PDE-Refiner trained betas and cumulative signal factors."""
    if num_refinement_steps <= 0 or not 0.0 < min_noise_std < 1.0:
        raise ValueError("invalid PDE-Refiner schedule")
    betas = torch.tensor(
        [
            min_noise_std ** (k / num_refinement_steps)
            for k in reversed(range(num_refinement_steps + 1))
        ],
        dtype=torch.float32,
    )
    alpha_cumulative = torch.cumprod(1.0 - betas, dim=0)
    return betas, alpha_cumulative


def make_velocity_training_pair(
    residual: torch.Tensor,
    step: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    num_refinement_steps: int = 3,
    min_noise_std: float = 4.0e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Noise a normalized residual and form PDE-Refiner's v-prediction target."""
    if step.ndim != 1 or len(step) != residual.shape[0]:
        raise ValueError("refinement step must contain one index per batch item")
    if noise is None:
        noise = torch.randn_like(residual)
    if noise.shape != residual.shape:
        raise ValueError("noise and residual must have identical shapes")
    _, alpha = pde_refiner_schedule(
        num_refinement_steps=num_refinement_steps,
        min_noise_std=min_noise_std,
    )
    alpha = alpha.to(device=residual.device, dtype=residual.dtype)[step]
    alpha = alpha.view(-1, *([1] * (residual.ndim - 1)))
    signal = torch.sqrt(alpha)
    noise_scale = torch.sqrt(1.0 - alpha)
    noised_residual = signal * residual + noise_scale * noise
    velocity_target = signal * noise - noise_scale * residual
    return noised_residual, velocity_target


@torch.no_grad()
def sample_residual(
    model: nn.Module,
    baseline: torch.Tensor,
    normalized_condition: torch.Tensor,
    structural_features: torch.Tensor,
    *,
    initial_noise: torch.Tensor | None = None,
    num_refinement_steps: int = 3,
    min_noise_std: float = 4.0e-7,
) -> torch.Tensor:
    """Generate one residual sample with deterministic DDIM-style refinement."""
    if initial_noise is None:
        current = torch.randn_like(baseline)
    else:
        if initial_noise.shape != baseline.shape:
            raise ValueError("initial noise and baseline must have identical shapes")
        current = initial_noise.clone()
    _, alpha = pde_refiner_schedule(
        num_refinement_steps=num_refinement_steps,
        min_noise_std=min_noise_std,
    )
    alpha = alpha.to(device=baseline.device, dtype=baseline.dtype)
    for step_index in reversed(range(num_refinement_steps + 1)):
        step = torch.full(
            (baseline.shape[0],),
            step_index,
            dtype=torch.long,
            device=baseline.device,
        )
        velocity = model(
            baseline,
            current,
            normalized_condition,
            structural_features,
            step,
        )
        signal = torch.sqrt(alpha[step_index])
        noise_scale = torch.sqrt(1.0 - alpha[step_index])
        clean = signal * current - noise_scale * velocity
        noise = noise_scale * current + signal * velocity
        if step_index == 0:
            current = clean
        else:
            previous_signal = torch.sqrt(alpha[step_index - 1])
            previous_noise = torch.sqrt(1.0 - alpha[step_index - 1])
            current = previous_signal * clean + previous_noise * noise
    return current


class P418RegionalResidualRefiner(nn.Module):
    """Conditioned Physics-Attention model for steady regional-field residuals."""

    def __init__(
        self,
        *,
        structural_dim: int,
        field_dim: int = 5,
        condition_dim: int = 5,
        hidden_dim: int = 256,
        layers: int = 5,
        attention_heads: int = 8,
        physics_slices: int = 32,
        num_refinement_steps: int = 3,
    ) -> None:
        super().__init__()
        if min(structural_dim, field_dim, condition_dim, hidden_dim, layers) <= 0:
            raise ValueError("residual-refiner dimensions must be positive")
        self.field_dim = field_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_refinement_steps = num_refinement_steps
        input_dim = 2 * field_dim + condition_dim + structural_dim
        self.input_encoder = MLP(input_dim, hidden_dim, 2 * hidden_dim)
        self.time_encoder = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            RegionalPhysicsAttentionBlock(
                hidden_dim,
                heads=attention_heads,
                slice_count=physics_slices,
                dropout=0.0,
            )
            for _ in range(layers)
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, field_dim))

    def forward(
        self,
        baseline: torch.Tensor,
        noised_residual: torch.Tensor,
        normalized_condition: torch.Tensor,
        structural_features: torch.Tensor,
        refinement_step: torch.Tensor,
    ) -> torch.Tensor:
        if baseline.shape != noised_residual.shape or baseline.shape[-1] != self.field_dim:
            raise ValueError("baseline and residual field shapes are inconsistent")
        if normalized_condition.shape != (baseline.shape[0], self.condition_dim):
            raise ValueError("condition shape is inconsistent with the residual field")
        if structural_features.ndim != 2 or structural_features.shape[0] != baseline.shape[1]:
            raise ValueError("structural features must contain one row per regional node")
        if refinement_step.shape != (baseline.shape[0],):
            raise ValueError("refinement step must contain one index per batch item")
        condition = normalized_condition[:, None, :].expand(-1, baseline.shape[1], -1)
        structure = structural_features[None].expand(baseline.shape[0], -1, -1)
        latent = self.input_encoder(
            torch.cat((baseline, noised_residual, condition, structure), dim=-1)
        )
        time = refinement_step.float() * (1000.0 / self.num_refinement_steps)
        latent = latent + self.time_encoder(timestep_embedding(time, self.hidden_dim))[:, None]
        for block in self.blocks:
            latent = block(latent)
        return self.output(latent)
