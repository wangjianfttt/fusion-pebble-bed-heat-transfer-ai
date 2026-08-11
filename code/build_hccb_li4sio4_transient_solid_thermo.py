#!/usr/bin/env python3
"""Write an OpenFOAM-13 solid thermo table from pure-Li4SiO4 calorimetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from hccb_source_backed_thermophysical import (
    MANIFEST,
    load_hccb_thermophysical_parameters,
)


PARAMETER_IDS = ("P092", "P403", "P428", "P429", "P430", "P431")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temperature_nodes(step_k: float = 5.0) -> np.ndarray:
    """Return a numerical interpolation grid over the published 298--1300 K range."""
    if not np.isfinite(step_k) or step_k <= 0.0:
        raise ValueError("table step must be positive")
    params = load_hccb_thermophysical_parameters()
    low, high = params.solid_cp_temperature_range_k
    regular = np.arange(low, high + 0.5 * step_k, step_k, dtype=np.float64)
    regular = regular[regular <= high]
    nodes = np.unique(
        np.concatenate(
            [
                regular,
                np.asarray([low, high, *params.solid_transition_temperatures_k]),
            ]
        )
    )
    return nodes


def table_values(step_k: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    nodes = temperature_nodes(step_k)
    cp = specific_heat_capacity_values(nodes)
    return nodes, cp


def specific_heat_capacity_values(temperature_k: np.ndarray) -> np.ndarray:
    """Evaluate the registered P429 relation without a neural-network runtime."""
    temperature = np.asarray(temperature_k, dtype=np.float64)
    if np.any(~np.isfinite(temperature)):
        raise ValueError("temperature must be finite")
    params = load_hccb_thermophysical_parameters()
    low, high = params.solid_cp_temperature_range_k
    if np.any(temperature < low) or np.any(temperature > high):
        raise ValueError(f"Li4SiO4 heat-capacity relation is limited to {low:g}-{high:g} K")
    constant, linear, inverse_square = params.solid_cp_molar_coefficients
    cp_molar = constant + linear * temperature + inverse_square / temperature**2
    return cp_molar / params.solid_molar_mass_kg_mol


def interpolation_error(step_k: float = 5.0) -> float:
    nodes, cp = table_values(step_k)
    dense = np.linspace(nodes[0], nodes[-1], 20001)
    truth = specific_heat_capacity_values(dense)
    interpolated = np.interp(dense, nodes, cp)
    return float(np.max(np.abs(interpolated - truth) / truth))


def physical_properties_text(step_k: float = 5.0) -> str:
    params = load_hccb_thermophysical_parameters()
    nodes, cp = table_values(step_k)
    values = "\n".join(f"                ({t:.12g} {c:.12g})" for t, c in zip(nodes, cp))
    return f'''/*--------------------------------*- C++ -*----------------------------------*\\
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
    location    "constant/solid";
    object      physicalProperties;
}}

// Heat storage: Kleykamp pure-Li4SiO4 calorimetry, P428-P431.
// OpenFOAM's eIcoTabulated solid type names this table Cv.  For the
// incompressible heat-conduction equation it stores the measured mass-specific
// heat capacity used in rho*c*dT/dt; no Cp-Cv correction is introduced.
thermoType
{{
    type            heSolidThermo;
    mixture         pureMixture;
    transport       constIsoSolid;
    thermo          eIcoTabulated;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleInternalEnergy;
}}
mixture
{{
    specie {{ molWeight {params.solid_molar_mass_kg_mol * 1000.0:.12g}; }}
    equationOfState {{ rho {params.solid_density_kg_m3:.12g}; }}
    transport {{ kappa {params.solid_conductivity_w_m_k:.12g}; }}
    thermodynamics
    {{
        hf 0;
        sf 0;
        Cv
        {{
            values
            (
{values}
            );
        }}
    }}
}}
// ************************************************************************* //
'''


def write_transient_solid_physical_properties(
    path: Path,
    *,
    table_step_k: float = 5.0,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    """Atomically replace a possibly hard-linked steady dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(physical_properties_text(table_step_k), encoding="ascii")
    os.replace(temporary, path)
    nodes, cp = table_values(table_step_k)
    payload: dict[str, object] = {
        "status": "pure_li4sio4_transient_solid_thermo_ready",
        "parameter_ids": list(PARAMETER_IDS),
        "temperature_range_K": [float(nodes[0]), float(nodes[-1])],
        "temperature_table_points": int(len(nodes)),
        "numerical_table_step_K": float(table_step_k),
        "maximum_linear_interpolation_relative_error": interpolation_error(table_step_k),
        "heat_capacity_range_J_kg_K": [float(cp.min()), float(cp.max())],
        "second_order_transition_temperatures_K": list(
            load_hccb_thermophysical_parameters().solid_transition_temperatures_k
        ),
        "physical_properties_sha256": sha256(path),
        "parameter_manifest_sha256": sha256(MANIFEST),
        "scope": (
            "Smoothed pure-compound heat capacity for transient storage; "
            "sharp transition anomalies and manufacturing-batch variation are not resolved."
        ),
        "new_fitted_physical_parameters": [],
    }
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--table-step-k", type=float, default=5.0)
    args = parser.parse_args()
    payload = write_transient_solid_physical_properties(
        args.output.resolve(),
        table_step_k=args.table_step_k,
        metadata_path=args.metadata.resolve() if args.metadata else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
