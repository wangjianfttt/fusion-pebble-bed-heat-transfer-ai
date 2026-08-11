#!/usr/bin/env python3
"""Build a literature-parameter steady helium-flow case on a Gmsh fluid mesh."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "parameters/literature_parameter_manifest.csv"


def parameter_rows() -> dict[str, dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="ascii")


def header(object_name: str, class_name: str = "dictionary") -> str:
    return f"""FoamFile
{{
    format      ascii;
    class       {class_name};
    object      {object_name};
}}

"""


def parse_reference(value: str) -> dict[str, float]:
    return {
        key: float(number)
        for key, number in (item.split("=", 1) for item in value.split(";"))
    }


def helium_mu(value: str, temperature_k: float) -> float:
    compact = value.replace(" ", "")
    match = re.fullmatch(r"mu=([0-9.eE+-]+)\*T_K\^([0-9.eE+-]+)\*1e-6", compact)
    if not match:
        raise ValueError(f"cannot parse helium viscosity relation: {value!r}")
    coefficient, exponent = map(float, match.groups())
    return coefficient * temperature_k**exponent * 1.0e-6


def helium_rho(value: str, pressure_mpa: float, temperature_k: float) -> float:
    compact = value.replace(" ", "")
    match = re.fullmatch(r"rho_f=([0-9.eE+-]+)\*p_MPa/T_K", compact)
    if not match:
        raise ValueError(f"cannot parse helium density relation: {value!r}")
    return float(match.group(1)) * pressure_mpa / temperature_k


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flow-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--inlet-patch")
    parser.add_argument("--outlet-patch")
    parser.add_argument("--transverse-patch", action="append", default=[])
    parser.add_argument("--symmetry-patch", action="append", default=[])
    parser.add_argument("--particle-wall-patch", default="fluid_to_solid")
    parser.add_argument("--no-slip-patch", action="append", default=[])
    parser.add_argument("--parallel-subdomains", type=int, default=1)
    parser.add_argument("--end-time", type=int, default=2000)
    parser.add_argument("--write-interval", type=int, default=200)
    args = parser.parse_args()
    if args.parallel_subdomains < 1:
        raise ValueError("parallel-subdomains must be positive")
    if args.end_time <= 0 or args.write_interval <= 0:
        raise ValueError("end-time and write-interval must be positive")

    physical = parameter_rows()
    for parameter_id in ("P054", "P070", "P389", "P391"):
        if physical[parameter_id]["status"] != "extracted":
            raise RuntimeError(f"{parameter_id} is not an extracted literature value")

    reference = parse_reference(physical["P391"]["value"])
    temperature_k = reference["T_in"]
    inlet_velocity = reference["u_in"]
    pressure_mpa = float(physical["P054"]["value"])
    mu = helium_mu(physical["P070"]["value"], temperature_k)
    rho = helium_rho(physical["P389"]["value"], pressure_mpa, temperature_k)
    nu = mu / rho

    axis_index = {"x": 0, "y": 1, "z": 2}[args.flow_axis]
    velocity = [0.0, 0.0, 0.0]
    velocity[axis_index] = inlet_velocity
    inlet = args.inlet_patch or f"{args.flow_axis}min"
    outlet = args.outlet_patch or f"{args.flow_axis}max"
    transverse = args.transverse_patch or [
        f"{axis}{side}"
        for axis in "xyz"
        if axis != args.flow_axis
        for side in ("min", "max")
    ]
    no_slip_patches = set(args.no_slip_patch)
    symmetry_patches = set(args.symmetry_patch)
    invalid_no_slip = no_slip_patches.difference(transverse)
    if invalid_no_slip:
        raise ValueError(f"no-slip patches must be transverse boundaries: {sorted(invalid_no_slip)}")
    invalid_symmetry = symmetry_patches.difference(transverse)
    if invalid_symmetry:
        raise ValueError(
            f"symmetry patches must be transverse boundaries: {sorted(invalid_symmetry)}"
        )
    overlap = no_slip_patches.intersection(symmetry_patches)
    if overlap:
        raise ValueError(f"patches cannot be both no-slip and symmetry: {sorted(overlap)}")
    particle_wall = args.particle_wall_patch
    output = args.output_dir.resolve()

    u_boundaries = []
    p_boundaries = []
    for patch in (inlet, outlet, *transverse, particle_wall):
        if patch == inlet:
            u_body = f"type fixedValue; value uniform ({velocity[0]} {velocity[1]} {velocity[2]});"
            p_body = "type zeroGradient;"
        elif patch == outlet:
            u_body = "type zeroGradient;"
            p_body = "type fixedValue; value uniform 0;"
        elif patch == particle_wall:
            u_body = "type noSlip;"
            p_body = "type zeroGradient;"
        elif patch in symmetry_patches:
            u_body = "type symmetry;"
            p_body = "type symmetry;"
        elif patch in no_slip_patches:
            u_body = "type noSlip;"
            p_body = "type zeroGradient;"
        else:
            u_body = "type slip;"
            p_body = "type zeroGradient;"
        u_boundaries.append(f"    {patch} {{ {u_body} }}")
        p_boundaries.append(f"    {patch} {{ {p_body} }}")

    write(
        output / "0/U",
        header("U", "volVectorField")
        + f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform ({velocity[0]} {velocity[1]} {velocity[2]});
boundaryField
{{
{chr(10).join(u_boundaries)}
}}
""",
    )
    write(
        output / "0/p",
        header("p", "volScalarField")
        + f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
{chr(10).join(p_boundaries)}
}}
""",
    )
    write(
        output / "constant/physicalProperties",
        header("physicalProperties")
        + f"""viscosityModel constant;
nu [0 2 -1 0 0 0 0] {nu:.12g};
""",
    )
    write(
        output / "constant/momentumTransport",
        header("momentumTransport") + "simulationType laminar;\n",
    )
    write(
        output / "system/controlDict",
        header("controlDict")
        + f"""solver incompressibleFluid;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {args.end_time};
deltaT 1;
writeControl timeStep;
writeInterval {args.write_interval};
purgeWrite 2;
writeFormat ascii;
writePrecision 10;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
functions
{{
    inletFlow
    {{
        type surfaceFieldValue;
        libs ("libfieldFunctionObjects.so");
        patch {inlet};
        operation sum;
        log true;
        writeFields false;
        fields (phi);
    }}
    outletFlow
    {{
        type surfaceFieldValue;
        libs ("libfieldFunctionObjects.so");
        patch {outlet};
        operation sum;
        log true;
        writeFields false;
        fields (phi);
    }}
    inletPressure
    {{
        type surfaceFieldValue;
        libs ("libfieldFunctionObjects.so");
        patch {inlet};
        operation areaAverage;
        log true;
        writeFields false;
        fields (p);
    }}
    outletPressure
    {{
        type surfaceFieldValue;
        libs ("libfieldFunctionObjects.so");
        patch {outlet};
        operation areaAverage;
        log true;
        writeFields false;
        fields (p);
    }}
}}
""",
    )
    write(
        output / "system/fvSchemes",
        header("fvSchemes")
        + """ddtSchemes { default steadyState; }
gradSchemes
{
    default Gauss linear;
    grad(U) cellLimited Gauss linear 1;
}
divSchemes
{
    default none;
    div(phi,U) bounded Gauss upwind;
    div(div(phi,U)) Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""",
    )
    write(
        output / "system/fvSolution",
        header("fvSolution")
        + """solvers
{
    Phi
    {
        solver GAMG;
        smoother DIC;
        tolerance 1e-9;
        relTol 0;
    }
    p
    {
        solver GAMG;
        smoother GaussSeidel;
        tolerance 1e-9;
        relTol 0.05;
    }
    pFinal { $p; relTol 0; }
    U
    {
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-10;
        relTol 0.05;
    }
    UFinal { $U; relTol 0; }
}
potentialFlow
{
    nNonOrthogonalCorrectors 10;
}
PIMPLE
{
    momentumPredictor yes;
    nOuterCorrectors 1;
    nNonOrthogonalCorrectors 2;
    residualControl
    {
        p 1e-7;
        U 1e-7;
    }
}
relaxationFactors
{
    fields { p 0.5; }
    equations { U 0.8; }
}
""",
    )
    if args.parallel_subdomains > 1:
        write(
            output / "system/decomposeParDict",
            header("decomposeParDict")
            + f"numberOfSubdomains {args.parallel_subdomains};\nmethod scotch;\n",
        )

    metadata = {
        "status": "steady_helium_flow_case_built",
        "purpose": "small-to-medium mesh stability and mass-conservation calculation; not a full-bed pressure-drop result",
        "flow_axis": args.flow_axis,
        "inlet_patch": inlet,
        "outlet_patch": outlet,
        "particle_wall_patch": particle_wall,
        "no_slip_outer_patches": sorted(no_slip_patches),
        "symmetry_patches": sorted(symmetry_patches),
        "inlet_temperature_K": temperature_k,
        "inlet_velocity_m_s": inlet_velocity,
        "working_pressure_MPa": pressure_mpa,
        "helium_dynamic_viscosity_Pa_s": mu,
        "helium_density_kg_m3": rho,
        "helium_kinematic_viscosity_m2_s": nu,
        "parameter_ids": ["P054", "P070", "P389", "P391"],
        "numerical_settings": {
            "steady_state": True,
            "convection_scheme": "bounded Gauss upwind",
            "momentum_predictor": True,
            "pressure_relaxation": 0.5,
            "velocity_relaxation": 0.8,
            "non_orthogonal_correctors": 2,
            "parallel_subdomains": args.parallel_subdomains,
            "end_time": args.end_time,
            "write_interval": args.write_interval,
        },
        "new_fitted_physical_parameters": [],
    }
    (output / "case_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
