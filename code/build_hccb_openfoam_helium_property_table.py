#!/usr/bin/env python3
"""Build and audit an OpenFOAM table from published HCCB helium properties."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "parameters/literature_parameter_manifest.csv"
DEFAULT_OUT = ROOT / "results/apd006_hccb_openfoam_helium_property_table"


def rows() -> dict[str, dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def helium_rho(p_pa: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    return 480.19 * (p_pa / 1.0e6) / temperature_k


def helium_mu(_p_pa: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    return 0.4646 * temperature_k**0.66 * 1.0e-6


def helium_kappa(p_pa: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    p_mpa = p_pa / 1.0e6
    theta = temperature_k / 273.0
    return 0.1448 * theta**0.68 * (1.0 + 2.5e-3 * p_mpa**1.17 * theta**-1.85)


def p418_temperature_support(value: str) -> np.ndarray:
    """Read the four inlet temperatures from the published P418 matrix."""
    compact = value.replace(" ", "")
    match = re.search(r"T_in=([0-9.,]+)K", compact)
    if not match:
        raise ValueError("P418 inlet-temperature support cannot be parsed")
    temperatures = np.asarray(
        [float(item) for item in match.group(1).split(",")], dtype=np.float64
    )
    if temperatures.tolist() != [300.0, 500.0, 700.0, 900.0]:
        raise ValueError(f"unexpected P418 inlet-temperature support: {temperatures}")
    return temperatures


def foam_matrix(values: np.ndarray) -> str:
    rows_text = []
    for row in values:
        rows_text.append("        (\n" + "\n".join(f"            {value:.15e}" for value in row) + "\n        )")
    return f"{values.shape[0]} {values.shape[1]}\n    (\n" + "\n".join(rows_text) + "\n    )"


def table_block(name: str, p: np.ndarray, temperature: np.ndarray, values: np.ndarray) -> str:
    return f"""    {name}
    {{
        low             ({p[0]:.12g} {temperature[0]:.12g});
        high            ({p[-1]:.12g} {temperature[-1]:.12g});
        values          {foam_matrix(values)};
    }}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--pressure-guard-multiplier",
        type=float,
        default=10.0,
        help="Numerical table half-width as a multiple of the published P391 pressure drop.",
    )
    parser.add_argument(
        "--pressure-support-design-id",
        default=None,
        help=(
            "Registered numerical-design identifier for a non-default pressure "
            "lookup range. The identifier changes no property correlation."
        ),
    )
    args = parser.parse_args()
    if args.pressure_guard_multiplier < 1.0:
        raise ValueError("pressure guard multiplier must be at least 1")
    manifest = rows()

    expected_equations = {
        "P070": "mu=0.4646*T_K^0.66*1e-6",
        "P071": "lambda_f=0.1448*(T_K/273)^0.68*(1+2.5e-3*p_MPa^1.17*(T_K/273)^-1.85)",
        "P388": "5200",
        "P389": "rho_f=480.19*p_MPa/T_K",
    }
    equation_checks = {
        key: manifest[key]["value"].replace(" ", "") == value.replace(" ", "")
        for key, value in expected_equations.items()
    }
    if not all(equation_checks.values()):
        raise RuntimeError(f"registered HCCB helium equations changed: {equation_checks}")

    inlet_temperature_support = p418_temperature_support(manifest["P418"]["value"])
    blanket_temperature_context = np.asarray(
        [float(item) for item in manifest["P424"]["value"].split(" to ")]
    )
    temperature_support = np.concatenate(
        [inlet_temperature_support, blanket_temperature_context]
    )
    endpoint_guard_k = 1.0
    temperature = np.arange(
        float(temperature_support.min()) - endpoint_guard_k,
        float(temperature_support.max()) + endpoint_guard_k + 0.5,
        1.0,
    )
    pressure_center_pa = float(manifest["P426"]["value"]) * 1.0e6
    reference = dict(
        item.split("=", 1) for item in manifest["P391"]["value"].split(";")
    )
    reference_delta_p = float(reference["deltaP"])
    pressure_guard_pa = reference_delta_p * args.pressure_guard_multiplier
    pressure = pressure_center_pa + pressure_guard_pa * np.asarray([-1.0, 0.0, 1.0])
    pressure_support_design_id = args.pressure_support_design_id or (
        "ND043" if args.pressure_guard_multiplier <= 10.0 else "ND047"
    )
    numerical_design_ids = ["ND042", "ND043", "ND044", "ND045", "ND046"]
    if pressure_support_design_id not in numerical_design_ids:
        numerical_design_ids.append(pressure_support_design_id)
    pp, tt = np.meshgrid(pressure, temperature, indexing="ij")
    tables = {
        "rho": helium_rho(pp, tt),
        "mu": helium_mu(pp, tt),
        "kappa": helium_kappa(pp, tt),
    }

    p_mid = 0.5 * (pressure[:-1] + pressure[1:])
    t_mid = 0.5 * (temperature[:-1] + temperature[1:])
    p_eval, t_eval = np.meshgrid(p_mid, t_mid, indexing="ij")
    points = np.column_stack([p_eval.ravel(), t_eval.ravel()])
    exact_functions = {"rho": helium_rho, "mu": helium_mu, "kappa": helium_kappa}
    errors: dict[str, float] = {}
    for name, values in tables.items():
        interpolator = RegularGridInterpolator((pressure, temperature), values, method="linear")
        predicted = interpolator(points)
        exact = exact_functions[name](points[:, 0], points[:, 1])
        errors[name] = float(np.max(np.abs(predicted - exact) / np.maximum(np.abs(exact), 1.0e-30)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "helium_property_table.npz",
        pressure_pa=pressure,
        temperature_k=temperature,
        rho_kg_m3=tables["rho"],
        mu_pa_s=tables["mu"],
        kappa_w_m_k=tables["kappa"],
    )
    include = (
        "// Generated only from P070/P071/P388/P389/P391/P418/P424/P426.\n"
        f"// {pressure_support_design_id} defines only the pressure lookup domain; it is not a physical parameter.\n"
        "equationOfState\n{\n"
        + table_block("rho", pressure, temperature, tables["rho"])
        + "}\n\nthermodynamics\n{\n    Cp 5200;\n    hf 0;\n}\n\ntransport\n{\n"
        + table_block("mu", pressure, temperature, tables["mu"])
        + table_block("kappa", pressure, temperature, tables["kappa"])
        + "}\n"
    )
    (args.output_dir / "helium_tabulated_properties.foam").write_text(include, encoding="ascii")
    openfoam_rr = 8314.46261815324
    effective_mol_weight = openfoam_rr * 480.19 / 1.0e6
    transport_only = (
        "thermodynamics\n{\n    Cp 5200;\n    hf 0;\n}\n\ntransport\n{\n"
        + table_block("mu", pressure, temperature, tables["mu"])
        + table_block("kappa", pressure, temperature, tables["kappa"])
        + "}\n"
    )
    full_dictionary = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Version: 13
    \\  /    A nd           |
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "constant/fluid";
    object      physicalProperties;
}}

thermoType
{{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       tabulated;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}}

mixture
{{
    specie
    {{
        // ND045: algebraically derived from P389, not a physical molar-mass claim.
        molWeight       {effective_mol_weight:.16g};
    }}

{transport_only}
}}

// ************************************************************************* //
"""
    (args.output_dir / "physicalProperties").write_text(full_dictionary, encoding="ascii")

    gate = 1.0e-4
    checks = {
        "registered_equations_match_code": all(equation_checks.values()),
        "temperature_table_contains_P418_and_P424_with_ND046_guard": bool(
            temperature[0] == temperature_support.min() - endpoint_guard_k
            and temperature[-1] == temperature_support.max() + endpoint_guard_k
            and temperature[1] == temperature_support.min()
            and temperature[-2] == temperature_support.max()
        ),
        "pressure_center_matches_P426": bool(pressure[1] == pressure_center_pa),
        "pressure_span_is_derived_from_P391": bool(
            pressure[0] == pressure_center_pa - pressure_guard_pa
            and pressure[-1] == pressure_center_pa + pressure_guard_pa
        ),
        "all_properties_positive": all(bool(np.all(values > 0.0)) for values in tables.values()),
        "maximum_relative_interpolation_error_below_ND044": max(errors.values()) <= gate,
        "perfect_gas_representation_reproduces_P389": math.isclose(
            effective_mol_weight / openfoam_rr,
            480.19 / 1.0e6,
            rel_tol=0.0,
            abs_tol=1.0e-18,
        ),
    }
    passed = all(checks.values())
    payload = {
        "status": "hccb_openfoam_helium_property_table_passed" if passed else "hccb_openfoam_helium_property_table_failed",
        "parameter_ids": [
            "P070", "P071", "P388", "P389", "P391", "P418", "P424", "P426"
        ],
        "source_roles": {
            "P418": "published 5 x 4 x 3 P418 operating matrix; inlet-temperature support",
            "P426": "published P418 absolute working pressure",
            "P070": "published helium dynamic-viscosity correlation",
            "P071": "published helium thermal-conductivity correlation",
            "P388": "published helium constant-pressure heat capacity",
            "P389": "published helium density correlation",
            "P391": "published pore-scale pressure-drop reference used only to span the numerical table",
            "P424": "published blanket temperature context",
        },
        "numerical_design_ids": numerical_design_ids,
        "pressure_support_design_id": pressure_support_design_id,
        "pressure_nodes_pa": pressure.tolist(),
        "pressure_guard_multiplier_of_P391_deltaP": args.pressure_guard_multiplier,
        "temperature_range_K": [float(temperature[0]), float(temperature[-1])],
        "published_inlet_temperature_support_K": [
            float(inlet_temperature_support.min()), float(inlet_temperature_support.max())
        ],
        "published_blanket_temperature_context_K": [
            float(blanket_temperature_context.min()), float(blanket_temperature_context.max())
        ],
        "endpoint_guard_K": endpoint_guard_k,
        "temperature_spacing_K": 1.0,
        "table_shape": [int(len(pressure)), int(len(temperature))],
        "openfoam_equation_of_state": {
            "type": "perfectGas",
            "effective_molWeight_kg_per_kmol": effective_mol_weight,
            "role": "exact algebraic representation of P389, not physical helium molar mass",
        },
        "maximum_relative_interpolation_errors": errors,
        "acceptance_gate": gate,
        "equation_checks": equation_checks,
        "checks": checks,
        "new_fitted_physical_parameters": [],
        "claim_boundary": (
            "This artifact is a numerical representation of the helium-property equations used by the formal P418 calculations. "
            "It is not new property data, an experimental fit, or a CHT validation result."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
