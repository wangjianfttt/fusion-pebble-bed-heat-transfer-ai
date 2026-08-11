"""Shared sparse-temperature operations for PINNs, operators and diffusion models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # NumPy observation files remain usable without PyTorch.
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DenseTemperatureObservations:
    observed_temperature_K: np.ndarray
    observation_mask: np.ndarray
    source_kind: str
    hard_conditioning_allowed: bool
    uncertainty_interpretation: str

    def validate(self) -> None:
        if self.observed_temperature_K.shape != self.observation_mask.shape:
            raise ValueError("observed temperature and mask shapes differ")
        if self.observation_mask.dtype != np.bool_:
            raise ValueError("observation mask must be boolean")
        if not np.all(np.isfinite(self.observed_temperature_K[self.observation_mask])):
            raise ValueError("observed temperatures must be finite at measured positions")
        if not np.any(self.observation_mask):
            raise ValueError("observation package contains no measured positions")
        if not self.source_kind:
            raise ValueError("observation source kind must be declared")
        if not self.uncertainty_interpretation:
            raise ValueError("observation uncertainty interpretation must be declared")
        if self.source_kind == "external_experiment" and self.hard_conditioning_allowed:
            raise ValueError("external experimental temperatures cannot be exact hard conditions")

    @property
    def count(self) -> int:
        return int(self.observation_mask.sum())


def load_dense_temperature_observations(path: Path) -> DenseTemperatureObservations:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "observed_temperature_K",
            "observation_mask",
            "observation_source_kind",
            "hard_conditioning_allowed",
            "uncertainty_interpretation",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"observation file lacks {sorted(missing)}")
        source_kind = str(np.asarray(data["observation_source_kind"]).item())
        hard_conditioning_allowed = bool(
            np.asarray(data["hard_conditioning_allowed"]).item()
        )
        uncertainty_interpretation = str(
            np.asarray(data["uncertainty_interpretation"]).item()
        )
        observations = DenseTemperatureObservations(
            observed_temperature_K=data["observed_temperature_K"].astype(np.float64),
            observation_mask=data["observation_mask"].astype(bool),
            source_kind=source_kind,
            hard_conditioning_allowed=hard_conditioning_allowed,
            uncertainty_interpretation=uncertainty_interpretation,
        )
    observations.validate()
    return observations


def masked_rmse_K(prediction_K: np.ndarray, observations: DenseTemperatureObservations) -> float:
    if prediction_K.shape != observations.observed_temperature_K.shape:
        raise ValueError("prediction and observation shapes differ")
    error = prediction_K[observations.observation_mask] - observations.observed_temperature_K[
        observations.observation_mask
    ]
    return float(np.sqrt(np.mean(np.square(error))))


def masked_temperature_mse(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for tensor observation losses")
    if prediction.shape != observed.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, observation and mask shapes must match")
    if mask.dtype != torch.bool:
        raise ValueError("observation mask must be boolean")
    if not bool(mask.any()):
        raise ValueError("observation mask is empty")
    if not bool(torch.isfinite(observed[mask]).all()):
        raise ValueError("observations must be finite at measured positions")
    return (prediction[mask] - observed[mask]).square().mean()


def hard_condition_tensor(
    field: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Replace measured positions exactly, while ignoring NaNs outside the mask."""

    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for hard tensor conditioning")
    if field.shape != observed.shape or field.shape != mask.shape:
        raise ValueError("field, observation and mask shapes must match")
    if mask.dtype != torch.bool:
        raise ValueError("observation mask must be boolean")
    if not bool(torch.isfinite(observed[mask]).all()):
        raise ValueError("observations must be finite at measured positions")
    clean_observed = torch.where(mask, observed, torch.zeros_like(observed))
    return torch.where(mask, clean_observed, field)


def hard_condition_dense_observations(
    field: torch.Tensor,
    observed: torch.Tensor,
    observations: DenseTemperatureObservations,
) -> torch.Tensor:
    """Apply exact replacement only to sources that explicitly permit it."""

    observations.validate()
    if not observations.hard_conditioning_allowed:
        raise ValueError(
            f"hard conditioning is forbidden for observation source "
            f"{observations.source_kind!r}"
        )
    mask = torch.as_tensor(
        observations.observation_mask,
        dtype=torch.bool,
        device=field.device,
    )
    return hard_condition_tensor(field, observed, mask)


def normalized_observation_residual(
    observations_K: torch.Tensor,
    baseline_temperature_normalized: torch.Tensor,
    temperature_scale_K: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Convert measured kelvin temperatures to the normalized residual used by diffusion."""

    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for normalized observation residuals")
    if observations_K.shape != baseline_temperature_normalized.shape or mask.shape != observations_K.shape:
        raise ValueError("measured temperature, baseline and mask shapes must match")
    if temperature_scale_K.ndim != 1 or temperature_scale_K.shape[0] != observations_K.shape[-2]:
        raise ValueError("temperature scale must contain one value per spatial node")
    if not bool((temperature_scale_K > 0).all()):
        raise ValueError("temperature scales must be positive")
    clean_observations = torch.where(mask, observations_K, torch.zeros_like(observations_K))
    observed_normalized = clean_observations / temperature_scale_K[None, None, :, None]
    residual = observed_normalized - baseline_temperature_normalized
    return hard_condition_tensor(torch.zeros_like(residual), residual, mask)
