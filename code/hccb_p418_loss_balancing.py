#!/usr/bin/env python3
"""Loss-group balancing for the fully coupled P418 PINN/operator model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


LOSS_GROUP_NAMES = ("state_data", "face_flux_data", "physics")


def _named_values(values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pair the three ordered loss names with values on Python 3.9+."""
    if len(values) != len(LOSS_GROUP_NAMES):
        raise ValueError("loss value count differs from declared loss groups")
    return dict(zip(LOSS_GROUP_NAMES, values))


def _ordered_losses(groups: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if set(groups) != set(LOSS_GROUP_NAMES):
        raise ValueError("loss groups must be state_data, face_flux_data and physics")
    losses = torch.stack([groups[name] for name in LOSS_GROUP_NAMES])
    if torch.any(~torch.isfinite(losses)) or torch.any(losses < 0.0):
        raise ValueError("loss groups must be finite and nonnegative")
    return losses


def common_validation_score(groups: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return the same dimensionless validation score for every weighting method."""
    return _ordered_losses(groups).mean()


def weighted_group_loss(
    groups: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor | float],
) -> torch.Tensor:
    """Combine loss groups with detached positive weights."""
    losses = _ordered_losses(groups)
    if set(weights) != set(LOSS_GROUP_NAMES):
        raise ValueError("loss weights must cover all three groups")
    values = torch.stack(
        [
            torch.as_tensor(
                weights[name],
                device=losses.device,
                dtype=losses.dtype,
            ).detach()
            for name in LOSS_GROUP_NAMES
        ]
    )
    if torch.any(~torch.isfinite(values)) or torch.any(values <= 0.0):
        raise ValueError("loss weights must be finite and positive")
    return torch.sum(values * losses)


class FixedLossBalancer:
    """Keep explicitly declared group weights constant."""

    method = "fixed"

    def __init__(self, weights: Mapping[str, float]) -> None:
        if set(weights) != set(LOSS_GROUP_NAMES):
            raise ValueError("fixed weights must cover all three groups")
        values = np.asarray([weights[name] for name in LOSS_GROUP_NAMES], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("fixed weights must be finite and positive")
        self._weights = values

    def update(
        self, groups: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        losses = _ordered_losses(groups)
        return self.weights(device=losses.device, dtype=losses.dtype)

    def weights(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> Mapping[str, torch.Tensor]:
        values = torch.as_tensor(self._weights, device=device, dtype=dtype)
        return _named_values(values)

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "weights": self._weights.tolist(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("method") != self.method:
            raise ValueError("loss-balancer method differs from checkpoint")
        values = np.asarray(state.get("weights"), dtype=float)
        if values.shape != (len(LOSS_GROUP_NAMES),):
            raise ValueError("fixed loss-balancer checkpoint has invalid weights")
        if not np.array_equal(values, self._weights):
            raise ValueError("fixed loss weights differ from checkpoint")


class ReLoBRaLoLossBalancer:
    """ReLoBRaLo following Bischof and Kraus Eq. (11) and official code.

    The first two updates use alpha=1 and alpha=0, respectively, matching the
    schedule in the authors' implementation. Initial reference losses are
    recorded after the second update. The Bernoulli lookback state is saved so
    an interrupted run resumes with the same subsequent weight sequence.
    """

    method = "relobralo"

    def __init__(
        self,
        *,
        temperature: float,
        alpha: float,
        expected_rho: float,
        seed: int,
        epsilon: float = 1.0e-12,
    ) -> None:
        values = np.asarray([temperature, alpha, expected_rho, epsilon], dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("ReLoBRaLo settings must be finite")
        if temperature <= 0.0 or epsilon <= 0.0:
            raise ValueError("ReLoBRaLo temperature and epsilon must be positive")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("ReLoBRaLo alpha must lie in [0, 1)")
        if not 0.0 <= expected_rho <= 1.0:
            raise ValueError("ReLoBRaLo expected rho must lie in [0, 1]")
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.expected_rho = float(expected_rho)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)
        self._step = 0
        self._weights = np.ones(len(LOSS_GROUP_NAMES), dtype=float)
        self._previous_losses = np.ones(len(LOSS_GROUP_NAMES), dtype=float)
        self._initial_losses = np.ones(len(LOSS_GROUP_NAMES), dtype=float)
        self._last_rho = 1.0

    @staticmethod
    def _balanced_weights(
        losses: torch.Tensor,
        reference: torch.Tensor,
        *,
        temperature: float,
        epsilon: float,
    ) -> torch.Tensor:
        logits = losses / (temperature * reference + epsilon)
        return len(LOSS_GROUP_NAMES) * torch.softmax(logits, dim=0)

    def update(
        self, groups: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        losses = _ordered_losses(groups).detach()
        previous = torch.as_tensor(
            self._previous_losses, device=losses.device, dtype=losses.dtype
        )
        initial = torch.as_tensor(
            self._initial_losses, device=losses.device, dtype=losses.dtype
        )
        previous_weights = torch.as_tensor(
            self._weights, device=losses.device, dtype=losses.dtype
        )
        relative_previous = self._balanced_weights(
            losses,
            previous,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )
        relative_initial = self._balanced_weights(
            losses,
            initial,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )
        if self._step == 0:
            alpha = 1.0
        elif self._step == 1:
            alpha = 0.0
        else:
            alpha = self.alpha
        rho = float(self._rng.uniform() < self.expected_rho)
        weights = (
            rho * alpha * previous_weights
            + (1.0 - rho) * alpha * relative_initial
            + (1.0 - alpha) * relative_previous
        ).detach()
        if torch.any(~torch.isfinite(weights)) or torch.any(weights <= 0.0):
            raise FloatingPointError("ReLoBRaLo produced invalid loss weights")

        current_losses = losses.cpu().double().numpy()
        self._weights = weights.cpu().double().numpy()
        self._previous_losses = current_losses
        if self._step == 1:
            self._initial_losses = current_losses.copy()
        self._last_rho = rho
        self._step += 1
        return _named_values(weights)

    def weights(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> Mapping[str, torch.Tensor]:
        values = torch.as_tensor(self._weights, device=device, dtype=dtype)
        return _named_values(values)

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "expected_rho": self.expected_rho,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "step": self._step,
            "weights": self._weights.tolist(),
            "previous_losses": self._previous_losses.tolist(),
            "initial_losses": self._initial_losses.tolist(),
            "last_rho": self._last_rho,
            "random_state": self._rng.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("method") != self.method:
            raise ValueError("loss-balancer method differs from checkpoint")
        settings = (
            ("temperature", self.temperature),
            ("alpha", self.alpha),
            ("expected_rho", self.expected_rho),
            ("epsilon", self.epsilon),
            ("seed", self.seed),
        )
        if any(state.get(name) != expected for name, expected in settings):
            raise ValueError("ReLoBRaLo settings differ from checkpoint")
        for name in ("weights", "previous_losses", "initial_losses"):
            values = np.asarray(state.get(name), dtype=float)
            if values.shape != (len(LOSS_GROUP_NAMES),):
                raise ValueError(f"ReLoBRaLo checkpoint has invalid {name}")
            setattr(self, f"_{name}", values)
        self._step = int(state["step"])
        self._last_rho = float(state["last_rho"])
        self._rng.set_state(state["random_state"])


def build_loss_balancer(
    *,
    method: str,
    state_weight: float,
    face_flux_weight: float,
    physics_weight: float,
    relobralo_temperature: float | None,
    relobralo_alpha: float | None,
    relobralo_rho: float | None,
    seed: int,
) -> FixedLossBalancer | ReLoBRaLoLossBalancer:
    fixed_weights = {
        "state_data": state_weight,
        "face_flux_data": face_flux_weight,
        "physics": physics_weight,
    }
    if method == "fixed":
        return FixedLossBalancer(fixed_weights)
    if method != "relobralo":
        raise ValueError(f"unknown loss balancing method: {method}")
    if fixed_weights != {name: 1.0 for name in LOSS_GROUP_NAMES}:
        raise ValueError("ReLoBRaLo must start from unit weights for all three groups")
    if None in (relobralo_temperature, relobralo_alpha, relobralo_rho):
        raise ValueError("ReLoBRaLo temperature, alpha and rho are required")
    return ReLoBRaLoLossBalancer(
        temperature=float(relobralo_temperature),
        alpha=float(relobralo_alpha),
        expected_rho=float(relobralo_rho),
        seed=seed,
    )
