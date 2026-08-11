#!/usr/bin/env python3
"""Differentiable HCCB thermophysical functions from published correlations.

The helium functions implement rows P070, P071, P388 and P389 in the project
parameter manifest.  The Li4SiO4 solid properties implement P092, P403 and the
pure-compound calorimetry relation P428--P431.  The composition-mismatched P406
row is deliberately excluded from transient storage calculations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Material-table utilities can run without PyTorch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "parameters/literature_parameter_manifest.csv"

# OpenFOAM-13 etc/controlDict, DimensionedConstants/standard/Tstd.
# This is the software enthalpy reference, not a fitted HCCB property.
OPENFOAM_TSTD_K = 298.15


@dataclass(frozen=True)
class HccbThermophysicalParameters:
    helium_cp_j_kg_k: float
    solid_conductivity_w_m_k: float
    solid_density_kg_m3: float
    solid_molar_mass_kg_mol: float
    solid_cp_molar_coefficients: tuple[float, float, float]
    solid_cp_temperature_range_k: tuple[float, float]
    solid_transition_temperatures_k: tuple[float, float]
    source_ids: tuple[str, ...]
    transient_solid_cp_available: bool


EXPECTED_VALUES = {
    "P070": "mu=0.4646*T_K^0.66*1e-6",
    "P071": "lambda_f=0.1448*(T_K/273)^0.68*(1+2.5e-3*p_MPa^1.17*(T_K/273)^-1.85)",
    "P092": "1.42",
    "P388": "5200",
    "P389": "rho_f=480.19*p_MPa/T_K",
    "P403": "1526.4",
    "P428": "H_T_minus_H_298_J_mol=-17156+73.694*T_K+0.103210*T_K^2-4163115/T_K",
    "P429": "Cp_molar=73.694+0.206420*T_K+4163115/T_K^2",
    "P430": "Li=6.94;O=15.999;Si=28.085;M_Li4SiO4=119.841",
    "P431": "Tc1=938;Tc2=996",
}


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for differentiable thermophysical functions; "
            "the source-backed parameter loader remains available without it."
        )


def _normalise_formula(value: str) -> str:
    return "".join(value.split())


def _named_values(value: str) -> dict[str, float]:
    return {
        key: float(number)
        for key, number in (entry.split("=", 1) for entry in value.split(";"))
    }


def load_hccb_thermophysical_parameters(
    manifest_path: Path = MANIFEST,
) -> HccbThermophysicalParameters:
    """Load and verify the exact published values used by this module."""
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["parameter_id"]: row for row in csv.DictReader(handle)}
    missing = sorted(set(EXPECTED_VALUES) - set(rows))
    if missing:
        raise RuntimeError(f"missing HCCB property rows: {missing}")
    mismatched = {
        key: rows[key]["value"]
        for key, expected in EXPECTED_VALUES.items()
        if _normalise_formula(rows[key]["value"]) != _normalise_formula(expected)
    }
    if mismatched:
        raise RuntimeError(f"registered HCCB property equations changed: {mismatched}")
    if any(rows[key]["status"] != "extracted" for key in EXPECTED_VALUES):
        raise RuntimeError("all HCCB thermophysical rows must have extracted status")
    cp_terms = _normalise_formula(rows["P429"]["value"])
    prefix = "Cp_molar="
    if not cp_terms.startswith(prefix):
        raise RuntimeError("P429 does not contain the registered heat-capacity relation")
    # P429 is verified verbatim above.  Keep the coefficients explicit so that
    # changes in the source table cannot silently alter the differentiable model.
    cp_coefficients = (73.694, 0.206420, 4163115.0)
    mass_values = _named_values(rows["P430"]["value"])
    transitions = _named_values(rows["P431"]["value"])
    return HccbThermophysicalParameters(
        helium_cp_j_kg_k=float(rows["P388"]["value"]),
        solid_conductivity_w_m_k=float(rows["P092"]["value"]),
        solid_density_kg_m3=float(rows["P403"]["value"]),
        solid_molar_mass_kg_mol=mass_values["M_Li4SiO4"] / 1000.0,
        solid_cp_molar_coefficients=cp_coefficients,
        solid_cp_temperature_range_k=(298.0, 1300.0),
        solid_transition_temperatures_k=(transitions["Tc1"], transitions["Tc2"]),
        source_ids=tuple(EXPECTED_VALUES),
        transient_solid_cp_available=True,
    )


def _positive_pair(
    pressure_pa: torch.Tensor, temperature_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_torch()
    if pressure_pa.shape != temperature_k.shape:
        raise ValueError("pressure and temperature must have identical shapes")
    if not pressure_pa.is_floating_point() or not temperature_k.is_floating_point():
        raise ValueError("pressure and temperature must be floating-point tensors")
    if torch.any(~torch.isfinite(pressure_pa)) or torch.any(~torch.isfinite(temperature_k)):
        raise ValueError("pressure and temperature must be finite")
    if torch.any(pressure_pa <= 0) or torch.any(temperature_k <= 0):
        raise ValueError("absolute pressure and temperature must be positive")
    return pressure_pa, temperature_k


def helium_density(
    pressure_pa: torch.Tensor, temperature_k: torch.Tensor
) -> torch.Tensor:
    """P389: helium density in kg/m3; pressure input is absolute Pa."""
    pressure_pa, temperature_k = _positive_pair(pressure_pa, temperature_k)
    return 480.19 * (pressure_pa / 1.0e6) / temperature_k


def helium_dynamic_viscosity(
    pressure_pa: torch.Tensor, temperature_k: torch.Tensor
) -> torch.Tensor:
    """P070: helium dynamic viscosity in Pa s."""
    pressure_pa, temperature_k = _positive_pair(pressure_pa, temperature_k)
    del pressure_pa
    return 0.4646 * temperature_k.pow(0.66) * 1.0e-6


def helium_thermal_conductivity(
    pressure_pa: torch.Tensor, temperature_k: torch.Tensor
) -> torch.Tensor:
    """P071: helium thermal conductivity in W/(m K)."""
    pressure_pa, temperature_k = _positive_pair(pressure_pa, temperature_k)
    pressure_mpa = pressure_pa / 1.0e6
    theta = temperature_k / 273.0
    return 0.1448 * theta.pow(0.68) * (
        1.0 + 2.5e-3 * pressure_mpa.pow(1.17) * theta.pow(-1.85)
    )


def helium_sensible_enthalpy(
    temperature_k: torch.Tensor,
    *,
    reference_temperature_k: float = OPENFOAM_TSTD_K,
    parameters: HccbThermophysicalParameters | None = None,
) -> torch.Tensor:
    """P388/OpenFOAM hConst: ``Cp * (T - Tref)`` in J/kg."""
    _require_torch()
    if not temperature_k.is_floating_point() or torch.any(~torch.isfinite(temperature_k)):
        raise ValueError("temperature must be a finite floating-point tensor")
    if torch.any(temperature_k <= 0) or reference_temperature_k <= 0:
        raise ValueError("absolute and reference temperatures must be positive")
    params = parameters or load_hccb_thermophysical_parameters()
    return params.helium_cp_j_kg_k * (temperature_k - reference_temperature_k)


def steady_li4sio4_conductivity_like(
    temperature_k: torch.Tensor,
    *,
    parameters: HccbThermophysicalParameters | None = None,
) -> torch.Tensor:
    """Return the P092 constant conductivity with the input tensor shape."""
    _require_torch()
    if not temperature_k.is_floating_point() or torch.any(~torch.isfinite(temperature_k)):
        raise ValueError("temperature must be a finite floating-point tensor")
    if torch.any(temperature_k <= 0):
        raise ValueError("absolute temperature must be positive")
    params = parameters or load_hccb_thermophysical_parameters()
    return torch.full_like(temperature_k, params.solid_conductivity_w_m_k)


def li4sio4_specific_heat_capacity(
    temperature_k: torch.Tensor,
    *,
    parameters: HccbThermophysicalParameters | None = None,
) -> torch.Tensor:
    """P429 pure-Li4SiO4 heat capacity in J/(kg K).

    Kleykamp reports a smoothed molar relation over 298--1300 K.  P430 supplies
    the natural-composition molar mass used for the unit conversion.  The
    smoothed relation does not resolve the P431 transition anomalies.
    """
    _require_torch()
    if not temperature_k.is_floating_point() or torch.any(~torch.isfinite(temperature_k)):
        raise ValueError("temperature must be a finite floating-point tensor")
    params = parameters or load_hccb_thermophysical_parameters()
    low, high = params.solid_cp_temperature_range_k
    if torch.any(temperature_k < low) or torch.any(temperature_k > high):
        raise ValueError(f"Li4SiO4 heat-capacity relation is limited to {low:g}-{high:g} K")
    constant, linear, inverse_square = params.solid_cp_molar_coefficients
    cp_molar = constant + linear * temperature_k + inverse_square / temperature_k.square()
    return cp_molar / params.solid_molar_mass_kg_mol


def li4sio4_sensible_internal_energy(
    temperature_k: torch.Tensor,
    *,
    reference_temperature_k: float = OPENFOAM_TSTD_K,
    parameters: HccbThermophysicalParameters | None = None,
) -> torch.Tensor:
    """P428 sensible internal energy relative to OpenFOAM ``Tstd`` in J/kg.

    The P428 calorimetry expression is the temperature integral of P429.  A
    constant reference shift does not affect the energy equation, but using
    OpenFOAM's 298.15 K reference makes the stored quantity directly comparable
    with ``eIcoTabulated::es``.  The derivative of this function is exactly the
    P429 mass-specific heat capacity used to build the transient solid table.
    """
    _require_torch()
    if not temperature_k.is_floating_point() or torch.any(~torch.isfinite(temperature_k)):
        raise ValueError("temperature must be a finite floating-point tensor")
    params = parameters or load_hccb_thermophysical_parameters()
    low, high = params.solid_cp_temperature_range_k
    if torch.any(temperature_k < low) or torch.any(temperature_k > high):
        raise ValueError(f"Li4SiO4 internal-energy relation is limited to {low:g}-{high:g} K")
    if not low <= reference_temperature_k <= high:
        raise ValueError("reference temperature is outside the Li4SiO4 relation range")

    def molar_energy(value: torch.Tensor) -> torch.Tensor:
        return (
            -17156.0
            + 73.694 * value
            + 0.103210 * value.square()
            - 4163115.0 / value
        )

    reference = temperature_k.new_tensor(reference_temperature_k)
    return (molar_energy(temperature_k) - molar_energy(reference)) / (
        params.solid_molar_mass_kg_mol
    )


def require_target_transient_solid_heat_capacity() -> HccbThermophysicalParameters:
    """Return the source-backed transient parameters or fail explicitly."""
    params = load_hccb_thermophysical_parameters()
    if not params.transient_solid_cp_available:
        raise RuntimeError("target Li4SiO4 transient solid heat capacity is unavailable")
    return params
