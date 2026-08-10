from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_inlet_dimensionless_envelope import (  # noqa: E402
    build_rows,
    build_summary,
    helium_properties,
)


def parameter_rows() -> dict[str, dict[str, str]]:
    values = {
        "P048": "1",
        "P068": "4.002602",
        "P070": "mu=0.4646*T_K^0.66*1e-6",
        "P071": "lambda=0.1448*(T_K/273)^0.68*(1+2.5e-3*p^1.17*T^-1.85)",
        "P073": "8.314472",
        "P388": "5200",
        "P418": "matrix",
        "P426": "0.12",
    }
    return {key: {"value": value} for key, value in values.items()}


def cases() -> list[dict[str, object]]:
    rows = []
    for velocity in (0.05, 0.10, 0.15, 0.20, 0.25):
        for temperature in (300.0, 500.0, 700.0, 900.0):
            for source in (4.85, 6.85, 8.85):
                rows.append(
                    {
                        "condition_id": f"u{velocity}_T{temperature}_q{source}",
                        "inlet_velocity_m_s": velocity,
                        "inlet_temperature_K": temperature,
                        "solid_heat_source_MW_m3": source,
                        "dictionary_values_match_sources": True,
                    }
                )
    return rows


def test_helium_properties_match_registered_correlations() -> None:
    result = helium_properties(
        300.0,
        0.12e6,
        molar_mass_kg_mol=4.002602e-3,
        gas_constant_j_mol_k=8.314472,
        cp_j_kg_k=5200.0,
    )
    assert abs(result["density_kg_m3"] - 0.19256073) < 1.0e-8
    assert abs(result["prandtl"] - 0.67497215) < 1.0e-8


def test_p418_inlet_envelope_and_support_boundary() -> None:
    rows = build_rows(cases(), parameter_rows())
    summary = build_summary(rows)
    assert len(rows) == 60
    assert summary["unique_velocity_temperature_state_count"] == 20
    assert abs(summary["particle_reynolds_inlet"]["minimum"] - 0.0775420) < 1.0e-6
    assert abs(summary["particle_reynolds_inlet"]["maximum"] - 2.401754) < 1.0e-6
    assert summary["case_count_with_inlet_reynolds_at_or_above_1p8"] == 6
    assert summary["unique_velocity_temperature_states_at_or_above_1p8"] == [
        "u=0.20 m/s, T=300 K",
        "u=0.25 m/s, T=300 K",
    ]
    assert summary["new_physical_parameters"] == []


def test_changed_openfoam_dictionary_is_rejected() -> None:
    rows = cases()
    rows[0]["dictionary_values_match_sources"] = False
    try:
        build_rows(rows, parameter_rows())
    except ValueError as error:
        assert "differ" in str(error)
    else:
        raise AssertionError("changed source-backed dictionary was accepted")
