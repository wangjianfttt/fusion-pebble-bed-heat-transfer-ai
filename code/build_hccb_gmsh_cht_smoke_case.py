#!/usr/bin/env python3
"""Build a literature-parameter steady CHT smoke case on a Gmsh two-region mesh."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

import export_openfoam_multiregion_interface_pairs as foam_mesh
from build_hccb_pore_resolved_openfoam_steady_case import (
    HELIUM_PROPERTIES,
    MANIFEST,
    boundary_entries,
    foam_header,
    p406_cp,
    parse_reference,
    rows,
    scalar_field,
    vector_field,
    write,
)
from build_hccb_gmsh_flow_smoke_case import helium_mu, helium_rho
from build_hccb_openfoam_helium_property_table import helium_kappa


def parse_p418_matrix(value: str) -> tuple[list[float], list[float], list[float]]:
    """Read the published 5 x 4 x 3 operating matrix from manifest row P418."""
    compact = value.replace(" ", "")
    match = re.search(
        r"u_in=([0-9.,]+)m/sxT_in=([0-9.,]+)Kxphi=([0-9.,]+)MW/m3",
        compact,
    )
    if not match:
        raise ValueError(f"cannot parse P418 operating matrix: {value}")
    return tuple(
        [float(item) for item in group.split(",")]
        for group in match.groups()
    )  # type: ignore[return-value]


def matrix_condition_id(velocity: float, temperature: float, source_mw_m3: float) -> str:
    """Return a stable, path-safe identifier for one published operating point."""
    velocity_token = f"{velocity:.2f}".replace(".", "p")
    source_token = f"{source_mw_m3:.2f}".replace(".", "p")
    return f"u{velocity_token}_T{temperature:.0f}_q{source_token}"


def select_p418_condition(
    condition_id: str,
    matrix_value: str,
) -> tuple[float, float, float]:
    """Resolve an identifier only when it is one of the 60 published P418 cases."""
    velocities, temperatures, sources = parse_p418_matrix(matrix_value)
    conditions = {
        matrix_condition_id(velocity, temperature, source): (velocity, temperature, source)
        for velocity in velocities
        for temperature in temperatures
        for source in sources
    }
    if condition_id not in conditions:
        allowed = ", ".join(sorted(conditions))
        raise ValueError(
            f"unknown P418 condition {condition_id!r}; use one of the published cases: {allowed}"
        )
    return conditions[condition_id]


def parse_region_patches(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?s)\n\s*\d+\s*\n\s*\(\s*\n(.*?)\n\s*\)\s*\n", text)
    if not match:
        raise ValueError(f"cannot locate the OpenFOAM patch list in {path}")
    return re.findall(r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)\s*\n\s*\{", match.group(1))


def set_patch_type(path: Path, patch_name: str, patch_type: str) -> None:
    """Set one mesh-boundary patch type without changing its faces."""
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"(?ms)(^\s*{re.escape(patch_name)}\s*\n\s*\{{.*?^\s*type\s+)[A-Za-z0-9_]+;"
    )
    updated, count = pattern.subn(rf"\g<1>{patch_type};", text, count=1)
    if count != 1:
        raise ValueError(f"cannot set patch type for {patch_name} in {path}")
    path.write_text(updated, encoding="utf-8")


def patch_area_m2(case: Path, region: str, patch_name: str) -> float:
    """Return the exact polygonal area of one OpenFOAM mesh patch."""
    mesh = foam_mesh.region_mesh(case, region)
    if patch_name not in mesh["boundaries"]:
        raise ValueError(f"missing {region} patch {patch_name!r}")
    patch = mesh["boundaries"][patch_name]
    start = int(patch["startFace"])
    count = int(patch["nFaces"])
    area = sum(
        foam_mesh.face_geometry(mesh["faces"][face], mesh["float_points"])[2]
        for face in range(start, start + count)
    )
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError(f"non-positive {region} patch area for {patch_name!r}: {area}")
    return float(area)


def source_channel_to_pore_velocity(
    source_channel_velocity_m_s: float,
    fluid_inlet_area_m2: float,
    solid_inlet_area_m2: float,
) -> tuple[float, float]:
    """Preserve source-paper volume flow on a particle-cut local inlet."""
    total_area = fluid_inlet_area_m2 + solid_inlet_area_m2
    if source_channel_velocity_m_s <= 0.0 or fluid_inlet_area_m2 <= 0.0 or total_area <= 0.0:
        raise ValueError("velocity and inlet areas must be positive")
    open_fraction = fluid_inlet_area_m2 / total_area
    if not 0.0 < open_fraction <= 1.0:
        raise ValueError(f"invalid inlet open-area fraction: {open_fraction}")
    return source_channel_velocity_m_s / open_fraction, open_fraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--inlet-patch", default="zmin")
    parser.add_argument("--outlet-patch", default="zmax")
    parser.add_argument("--cooling-wall-patch", default="xmin")
    parser.add_argument("--fluid-solid-patch", default="fluid_to_solid")
    parser.add_argument("--solid-fluid-patch", default="solid_to_fluid")
    parser.add_argument("--symmetry-patch", action="append", default=[])
    parser.add_argument("--helium-properties", type=Path, default=HELIUM_PROPERTIES)
    parser.add_argument(
        "--helium-mode",
        choices=("constant-reference", "tabulated"),
        default="constant-reference",
    )
    parser.add_argument("--initial-velocity-field", type=Path)
    parser.add_argument("--initial-velocity-patch-rename", action="append", default=[])
    parser.add_argument(
        "--fluid-inlet-area-m2",
        type=float,
        help="Precomputed open-fluid inlet area for a shared P418 mesh.",
    )
    parser.add_argument(
        "--solid-inlet-area-m2",
        type=float,
        help="Precomputed solid inlet area for a shared P418 mesh.",
    )
    parser.add_argument("--parallel-subdomains", type=int, default=1)
    parser.add_argument("--end-time", type=int, default=800)
    parser.add_argument("--write-interval", type=int, default=25)
    parser.add_argument("--energy-correctors", type=int, default=20)
    parser.add_argument("--solve-flow-during-energy", action="store_true")
    parser.add_argument(
        "--p418-condition-id",
        help=(
            "Use one exact operating point from Wang et al. (2023) Table 2, for example "
            "u0p20_T700_q6p85. Values outside P418 are rejected."
        ),
    )
    args = parser.parse_args()
    if args.parallel_subdomains < 1:
        raise ValueError("parallel-subdomains must be at least one")
    if args.end_time < 1 or args.write_interval < 1 or args.energy_correctors < 1:
        raise ValueError("end-time, write-interval and energy-correctors must be positive")
    if (args.fluid_inlet_area_m2 is None) != (args.solid_inlet_area_m2 is None):
        raise ValueError("fluid and solid inlet areas must be supplied together")
    case = args.case.resolve()

    physical = rows(MANIFEST, "parameter_id")
    parameter_ids = [
        "P053", "P054", "P055", "P070", "P071", "P092", "P388", "P389",
        "P391", "P392", "P403", "P406",
    ]
    for parameter_id in parameter_ids:
        if physical[parameter_id]["status"] != "extracted":
            raise RuntimeError(f"{parameter_id} is not an extracted literature value")

    if args.p418_condition_id:
        parameter_ids = [
            parameter_id
            for parameter_id in parameter_ids
            if parameter_id not in {"P053", "P054", "P055", "P391", "P392"}
        ]
        inlet_velocity, inlet_temperature, source_mw_m3 = select_p418_condition(
            args.p418_condition_id,
            physical["P418"]["value"],
        )
        heat_source = source_mw_m3 * 1.0e6
        parameter_ids.extend(["P418", "P424", "P425", "P426", "P427"])
        operating_condition_source = (
            "Wang et al. (2023), International Journal of Heat and Mass Transfer 213, "
            "124325, Table 2"
        )
        outlet_pressure = float(physical["P426"]["value"]) * 1.0e6
        pressure_mpa = float(physical["P426"]["value"])
        cooling_temperature = float(physical["P425"]["value"])
    else:
        reference = parse_reference(physical["P391"]["value"])
        inlet_temperature = reference["T_in"]
        inlet_velocity = reference["u_in"]
        source_mw_m3 = float(physical["P053"]["value"])
        heat_source = source_mw_m3 * 1.0e6
        operating_condition_source = "P391 reference condition with P053 heat source"
        outlet_pressure = float(physical["P054"]["value"]) * 1.0e6
        pressure_mpa = float(physical["P054"]["value"])
        cooling_temperature = float(physical["P055"]["value"])
    solid_k = float(physical["P092"]["value"])
    solid_rho = float(physical["P403"]["value"])
    solid_cv = p406_cp(physical["P406"]["value"], inlet_temperature)
    helium_cp = float(physical["P388"]["value"])
    helium_dynamic_viscosity = helium_mu(physical["P070"]["value"], inlet_temperature)
    helium_density = helium_rho(physical["P389"]["value"], pressure_mpa, inlet_temperature)
    helium_thermal_conductivity = float(
        helium_kappa(np.asarray(outlet_pressure), np.asarray(inlet_temperature))
    )
    helium_prandtl = helium_cp * helium_dynamic_viscosity / helium_thermal_conductivity

    fluid_boundary = case / "constant/fluid/polyMesh/boundary"
    solid_boundary = case / "constant/solid/polyMesh/boundary"
    set_patch_type(fluid_boundary, args.cooling_wall_patch, "wall")
    fluid_patches = parse_region_patches(fluid_boundary)
    solid_patches = parse_region_patches(solid_boundary)
    required_fluid = {
        args.inlet_patch,
        args.outlet_patch,
        args.cooling_wall_patch,
        args.fluid_solid_patch,
        *args.symmetry_patch,
    }
    required_solid = {args.solid_fluid_patch}
    if missing := required_fluid.difference(fluid_patches):
        raise RuntimeError(f"missing fluid patches {sorted(missing)} from {fluid_patches}")
    if missing := required_solid.difference(solid_patches):
        raise RuntimeError(f"missing solid patches {sorted(missing)} from {solid_patches}")
    symmetry_patches = set(args.symmetry_patch)

    source_channel_velocity = inlet_velocity
    pore_boundary_velocity = inlet_velocity
    inlet_open_area_fraction = 1.0
    fluid_inlet_area_m2 = None
    solid_inlet_area_m2 = None
    if args.p418_condition_id:
        if args.initial_velocity_field:
            raise ValueError(
                "P418 source-flow mapping cannot be combined with an externally supplied "
                "initial velocity field"
            )
        if args.fluid_inlet_area_m2 is None:
            fluid_inlet_area_m2 = patch_area_m2(case, "fluid", args.inlet_patch)
            solid_inlet_area_m2 = patch_area_m2(case, "solid", args.inlet_patch)
        else:
            fluid_inlet_area_m2 = args.fluid_inlet_area_m2
            solid_inlet_area_m2 = args.solid_inlet_area_m2
        pore_boundary_velocity, inlet_open_area_fraction = source_channel_to_pore_velocity(
            source_channel_velocity,
            fluid_inlet_area_m2,
            solid_inlet_area_m2,
        )

    u_entries: dict[str, str] = {}
    t_entries: dict[str, str] = {}
    prgh_entries: dict[str, str] = {}
    p_entries: dict[str, str] = {}
    for patch in fluid_patches:
        if patch == args.inlet_patch:
            u_entries[patch] = f"        type fixedValue;\n        value uniform (0 0 {pore_boundary_velocity:.12g});"
            t_entries[patch] = f"        type fixedValue;\n        value uniform {inlet_temperature:.12g};"
            prgh_entries[patch] = f"        type fixedFluxPressure;\n        value uniform {outlet_pressure:.12g};"
        elif patch == args.outlet_patch:
            u_entries[patch] = "        type pressureInletOutletVelocity;\n        value uniform (0 0 0);"
            t_entries[patch] = f"        type inletOutlet;\n        inletValue uniform {inlet_temperature:.12g};\n        value uniform {inlet_temperature:.12g};"
            prgh_entries[patch] = f"        type fixedValue;\n        value uniform {outlet_pressure:.12g};"
        elif patch == args.cooling_wall_patch:
            u_entries[patch] = "        type noSlip;"
            t_entries[patch] = f"        type fixedValue;\n        value uniform {cooling_temperature:.12g};"
            prgh_entries[patch] = f"        type fixedFluxPressure;\n        value uniform {outlet_pressure:.12g};"
        elif patch == args.fluid_solid_patch:
            u_entries[patch] = "        type noSlip;"
            t_entries[patch] = f"        type coupledTemperature;\n        value uniform {inlet_temperature:.12g};"
            prgh_entries[patch] = f"        type fixedFluxPressure;\n        value uniform {outlet_pressure:.12g};"
        elif patch in symmetry_patches:
            u_entries[patch] = "        type symmetry;"
            t_entries[patch] = "        type symmetry;"
            prgh_entries[patch] = "        type symmetry;"
        else:
            u_entries[patch] = "        type slip;"
            t_entries[patch] = "        type zeroGradient;"
            prgh_entries[patch] = "        type zeroGradient;"
        if patch in symmetry_patches:
            p_entries[patch] = "        type symmetry;"
        else:
            p_entries[patch] = (
                f"        type calculated;\n"
                f"        value uniform {outlet_pressure:.12g};"
            )

    solid_t_entries: dict[str, str] = {}
    for patch in solid_patches:
        if patch == args.solid_fluid_patch:
            solid_t_entries[patch] = (
                f"        type coupledTemperature;\n"
                f"        value uniform {inlet_temperature:.12g};"
            )
        elif patch == args.cooling_wall_patch:
            solid_t_entries[patch] = (
                f"        type fixedValue;\n"
                f"        value uniform {cooling_temperature:.12g};"
            )
        elif patch in symmetry_patches:
            solid_t_entries[patch] = "        type symmetry;"
        else:
            solid_t_entries[patch] = "        type zeroGradient;"
    write(case / "0/fluid/U", vector_field("0/fluid", "U", (0.0, 0.0, pore_boundary_velocity), u_entries))
    if args.initial_velocity_field:
        velocity_text = args.initial_velocity_field.resolve().read_text(
            encoding="utf-8", errors="strict"
        )
        velocity_text, count = re.subn(
            r'(?m)^\s*location\s+"[^"]+"\s*;',
            '    location    "0/fluid";',
            velocity_text,
            count=1,
        )
        if count != 1 or not re.search(r"(?m)^\s*object\s+U\s*;", velocity_text):
            raise ValueError("initial velocity field is not an OpenFOAM U field")
        for rename in args.initial_velocity_patch_rename:
            if "=" not in rename:
                raise ValueError("initial velocity patch rename must use OLD=NEW")
            old, new = rename.split("=", 1)
            velocity_text, count = re.subn(
                rf"(?m)^(\s*){re.escape(old)}\s*$",
                rf"\g<1>{new}",
                velocity_text,
                count=1,
            )
            if count != 1:
                raise ValueError(f"cannot rename velocity patch {old!r} to {new!r}")
        (case / "0/fluid/U").write_text(velocity_text, encoding="utf-8")
    write(case / "0/fluid/p", scalar_field("0/fluid", "p", "[1 -1 -2 0 0 0 0]", outlet_pressure, p_entries))
    write(case / "0/fluid/p_rgh", scalar_field("0/fluid", "p_rgh", "[1 -1 -2 0 0 0 0]", outlet_pressure, prgh_entries))
    write(case / "0/fluid/T", scalar_field("0/fluid", "T", "[0 0 0 1 0 0 0]", inlet_temperature, t_entries))
    write(case / "0/solid/T", scalar_field("0/solid", "T", "[0 0 0 1 0 0 0]", inlet_temperature, solid_t_entries))

    (case / "constant/fluid").mkdir(parents=True, exist_ok=True)
    helium_properties = args.helium_properties.resolve()
    if args.helium_mode == "tabulated":
        shutil.copy2(helium_properties, case / "constant/fluid/physicalProperties")
    else:
        write(
            case / "constant/fluid/physicalProperties",
            foam_header("constant/fluid", "physicalProperties", "dictionary")
            + f"""thermoType
{{
    type heRhoThermo;
    mixture pureMixture;
    transport const;
    thermo hConst;
    equationOfState rhoConst;
    specie specie;
    energy sensibleEnthalpy;
}}
mixture
{{
    specie {{ molWeight 1; }}
    equationOfState {{ rho {helium_density:.12g}; }}
    thermodynamics {{ Cp {helium_cp:.12g}; hf 0; }}
    transport {{ mu {helium_dynamic_viscosity:.12g}; Pr {helium_prandtl:.12g}; }}
}}
""",
        )
    write(
        case / "constant/fluid/momentumTransport",
        foam_header("constant/fluid", "momentumTransport", "dictionary") + "simulationType laminar;\n",
    )
    write(
        case / "constant/fluid/g",
        foam_header("constant/fluid", "g", "uniformDimensionedVectorField")
        + "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 0);\n",
    )
    (case / "constant/fluid/fvConstraints").unlink(missing_ok=True)
    write(
        case / "constant/solid/physicalProperties",
        foam_header("constant/solid", "physicalProperties", "dictionary")
        + f"""thermoType
{{
    type heSolidThermo;
    mixture pureMixture;
    transport constIsoSolid;
    thermo eConst;
    equationOfState rhoConst;
    specie specie;
    energy sensibleInternalEnergy;
}}
mixture
{{
    specie {{ molWeight 1; }}
    equationOfState {{ rho {solid_rho:.12g}; }}
    transport {{ kappa {solid_k:.12g}; }}
    thermodynamics {{ hf 0; Cv {solid_cv:.12g}; }}
}}
""",
    )
    write(
        case / "constant/solid/fvModels",
        foam_header("constant/solid", "fvModels", "dictionary")
        + f"""energySource
{{
    type heatSource;
    cellZone all;
    q {heat_source:.12g};
}}
""",
    )

    control_dict = foam_header("system", "controlDict", "dictionary") + """regionSolvers
{
    fluid fluid;
    solid solid;
}
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
"""
    control_dict = control_dict.replace("{args.end_time}", str(args.end_time)).replace(
        "{args.write_interval}", str(args.write_interval)
    )
    write(case / "system/controlDict", control_dict)
    write(
        case / "system/fluid/fvSchemes",
        foam_header("system/fluid", "fvSchemes", "dictionary")
        + """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss upwind;
    div(phi,K) Gauss linear;
    div(phi,h) bounded Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""",
    )
    write(
        case / "system/solid/fvSchemes",
        foam_header("system/solid", "fvSchemes", "dictionary")
        + """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""",
    )
    fluid_fv_solution = foam_header("system/fluid", "fvSolution", "dictionary") + """solvers
{
    p_rgh { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.02; }
    p_rghFinal { $p_rgh; relTol 0; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.03; }
    UFinal { $U; relTol 0; }
    h { solver PBiCGStab; preconditioner DILU; tolerance 1e-9; relTol 0.03; }
    hFinal { $h; relTol 0; }
}
PIMPLE
{
    flow __FLOW_SWITCH__;
    momentumPredictor __MOMENTUM_SWITCH__;
    nOuterCorrectors 1;
    nNonOrthogonalCorrectors 2;
}
relaxationFactors
{
    fields { p_rgh 0.3; }
    equations { U 0.6; h 1; }
}
"""
    fluid_fv_solution = fluid_fv_solution.replace(
        "__FLOW_SWITCH__", "yes" if args.solve_flow_during_energy else "no"
    ).replace(
        "__MOMENTUM_SWITCH__", "yes" if args.solve_flow_during_energy else "no"
    )
    write(case / "system/fluid/fvSolution", fluid_fv_solution)
    write(
        case / "system/solid/fvSolution",
        foam_header("system/solid", "fvSolution", "dictionary")
        + """solvers
{
    "e.*" { solver PCG; preconditioner DIC; tolerance 1e-9; relTol 0.02; }
}
PIMPLE { nNonOrthogonalCorrectors 2; }
relaxationFactors { equations { e 1; } }
""",
    )
    write(
        case / "system/fvSolution",
        foam_header("system", "fvSolution", "dictionary")
        + f"PIMPLE {{ nEnergyCorrectors {args.energy_correctors}; }}\n",
    )
    if args.parallel_subdomains > 1:
        write(
            case / "system/decomposeParDict",
            foam_header("system", "decomposeParDict", "dictionary")
            + f"numberOfSubdomains {args.parallel_subdomains};\nmethod scotch;\n",
        )

    functions = foam_header("system", "functions", "dictionary") + """inletMassFlow
{
    type surfaceFieldValue;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    patch {args.inlet_patch};
    operation sum;
    log true;
    writeFields false;
    fields (phi);
}
outletMassFlow
{
    type surfaceFieldValue;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    patch {args.outlet_patch};
    operation sum;
    log true;
    writeFields false;
    fields (phi);
}
inletTemperature
{
    type surfaceFieldValue;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    patch {args.inlet_patch};
    operation areaAverage;
    log true;
    writeFields false;
    fields (T);
}
outletTemperature
{
    type surfaceFieldValue;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    patch {args.outlet_patch};
    operation areaAverage;
    log true;
    writeFields false;
    fields (T);
}
inletPressure
{
    type surfaceFieldValue;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    patch {args.inlet_patch};
    operation areaAverage;
    log true;
    writeFields false;
    fields (p);
}
outletPressure
{
    type surfaceFieldValue;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    patch {args.outlet_patch};
    operation areaAverage;
    log true;
    writeFields false;
    fields (p);
}
inletEnthalpyFlow
{
    type surfaceFieldValue;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    patch {args.inlet_patch};
    operation sum;
    weightField phi;
    log true;
    writeFields false;
    fields (h);
}
outletEnthalpyFlow
{
    type surfaceFieldValue;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    patch {args.outlet_patch};
    operation sum;
    weightField phi;
    log true;
    writeFields false;
    fields (h);
}
fluidWallHeatFlux
{
    type wallHeatFlux;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    patches ({args.cooling_wall_patch});
    log true;
    writeFields true;
}
coolingWallPower
{
    type surfaceFieldValue;
    libs ("libfieldFunctionObjects.so");
    region fluid;
    patch {args.cooling_wall_patch};
    operation areaIntegrate;
    log true;
    writeFields false;
    fields (wallHeatFlux);
}
solidTemperatureMaximum
{
    type volFieldValue;
    libs ("libfieldFunctionObjects.so");
    region solid;
    cellZone all;
    operation max;
    log true;
    writeFields false;
    fields (T);
}
"""
    functions = (
        functions.replace("{args.inlet_patch}", args.inlet_patch)
        .replace("{args.outlet_patch}", args.outlet_patch)
        .replace("{args.cooling_wall_patch}", args.cooling_wall_patch)
    )
    write(case / "system/functions", functions)

    metadata = {
        "status": "hccb_gmsh_steady_cht_smoke_case_built",
        "purpose": (
            "small-to-medium coupled flow and CHT calculation; not a full HCCB result"
            if args.solve_flow_during_energy
            else "small-to-medium fixed-flow CHT calculation; not a full HCCB result"
        ),
        "parameter_ids": parameter_ids,
        "operating_condition_id": (
            args.p418_condition_id
            if args.p418_condition_id
            else matrix_condition_id(inlet_velocity, inlet_temperature, source_mw_m3)
        ),
        "operating_condition_source": operating_condition_source,
        "p418_exact_matrix_case": bool(args.p418_condition_id),
        "p418_source_doi": (
            physical["P418"]["source_url_or_doi"] if args.p418_condition_id else None
        ),
        "source_geometry_and_current_geometry_relation": (
            "P427 uses a 12.5dp x 12.5dp x 10dp packed region with 10dp inlet and "
            "outlet extensions. This case retains its own smaller local crop and the same "
            "one-cooling-wall/remaining-symmetry boundary types; it is a new resolved "
            "calculation over the published operating matrix, not a pointwise reproduction."
            if args.p418_condition_id
            else None
        ),
        "inlet_temperature_K": inlet_temperature,
        "inlet_velocity_m_s": source_channel_velocity,
        "source_inlet_channel_velocity_m_s": source_channel_velocity,
        "pore_opening_boundary_velocity_m_s": pore_boundary_velocity,
        "inlet_open_area_fraction": inlet_open_area_fraction,
        "fluid_inlet_area_m2": fluid_inlet_area_m2,
        "solid_inlet_area_m2": solid_inlet_area_m2,
        "source_channel_volume_flow_preserved": bool(args.p418_condition_id),
        "inlet_velocity_mapping": (
            "u_pore = u_in_source * (A_fluid + A_solid) / A_fluid"
            if args.p418_condition_id
            else "no particle-cut source-channel mapping"
        ),
        "outlet_pressure_Pa": outlet_pressure,
        "cooling_wall_temperature_K": cooling_temperature,
        "solid_heat_source_W_m3": heat_source,
        "solid_thermal_conductivity_W_m_K": solid_k,
        "solid_density_kg_m3": solid_rho,
        "solid_Cv_J_kg_K": solid_cv,
        "flow_during_energy_solve": (
            "coupled momentum-pressure-energy correction; PIMPLE flow yes"
            if args.solve_flow_during_energy
            else "fixed OpenFOAM velocity field; PIMPLE flow no"
        ),
        "energy_coupling_correctors": args.energy_correctors,
        "temperature_equation_relaxation": 1.0,
        "parallel_subdomains": args.parallel_subdomains,
        "initial_velocity_field": (
            str(args.initial_velocity_field.resolve()) if args.initial_velocity_field else None
        ),
        "cooling_wall_patch": args.cooling_wall_patch,
        "inlet_patch": args.inlet_patch,
        "outlet_patch": args.outlet_patch,
        "fluid_solid_patch": args.fluid_solid_patch,
        "solid_fluid_patch": args.solid_fluid_patch,
        "symmetry_patches": sorted(symmetry_patches),
        "end_time": args.end_time,
        "steady_iteration_end": args.end_time,
        "solver_time_semantics": "steady_iteration_index",
        "physical_time_s": None,
        "write_interval": args.write_interval,
        "write_interval_semantics": "steady_iterations",
        "new_fitted_physical_parameters": [],
        "helium_properties_file": str(helium_properties),
        "helium_mode": args.helium_mode,
        "helium_density_kg_m3": helium_density,
        "helium_dynamic_viscosity_Pa_s": helium_dynamic_viscosity,
        "helium_thermal_conductivity_W_m_K": helium_thermal_conductivity,
        "helium_specific_heat_J_kg_K": helium_cp,
        "helium_Prandtl_number": helium_prandtl,
    }
    (case / "cht_smoke_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
