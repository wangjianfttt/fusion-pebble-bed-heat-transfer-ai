#!/usr/bin/env python3
"""Verify source-state initialization and target inputs for a coupled step case."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from verify_hccb_p418_step_initialization import (
    MESH_FILES,
    TRANSIENT_THERMO_PARAMETER_IDS,
    foam_value,
    internal_field_values,
    scalar_from_uniform,
    sha256,
    velocity_z,
)


def boundary_patch_block(text: str, patch: str) -> str:
    match = re.search(rf"\b{re.escape(patch)}\s*\{{", text)
    if not match:
        raise ValueError(f"cannot find patch {patch}")
    start = text.index("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise ValueError(f"unclosed patch block {patch}")


def boundary_scalar_values(path: Path, patch: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    block = boundary_patch_block(text, patch)
    nonuniform = re.search(
        r"value\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        block,
        flags=re.S,
    )
    if nonuniform:
        values = np.asarray(
            [float(value) for value in nonuniform.group(2).split()],
            dtype=np.float64,
        )
        if values.size != int(nonuniform.group(1)):
            raise ValueError(f"declared list length mismatch for {patch} in {path}")
    else:
        uniform = re.search(r"value\s+uniform\s+([-+0-9.eE]+)\s*;", block)
        if not uniform:
            raise ValueError(f"cannot find scalar boundary values for {patch} in {path}")
        values = np.asarray([float(uniform.group(1))], dtype=np.float64)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError(f"boundary values are empty or non-finite for {patch} in {path}")
    return values


def inlet_mass_flow_consistency(
    case_phi: Path,
    source_phi: Path,
    target_phi: Path,
) -> dict[str, object]:
    case_mass_flow = float(boundary_scalar_values(case_phi, "inlet").sum())
    source_mass_flow = float(boundary_scalar_values(source_phi, "inlet").sum())
    target_mass_flow = float(boundary_scalar_values(target_phi, "inlet").sum())
    scale = max(abs(target_mass_flow), np.finfo(np.float64).tiny)
    target_relative_difference = abs(case_mass_flow - target_mass_flow) / scale
    return {
        "case_initial_inlet_mass_flow_kg_s": case_mass_flow,
        "source_endpoint_inlet_mass_flow_kg_s": source_mass_flow,
        "target_endpoint_inlet_mass_flow_kg_s": target_mass_flow,
        "case_to_source_ratio": (
            case_mass_flow / source_mass_flow if source_mass_flow != 0.0 else None
        ),
        "target_to_source_ratio": (
            target_mass_flow / source_mass_flow if source_mass_flow != 0.0 else None
        ),
        "case_to_target_relative_difference": target_relative_difference,
        "initial_inlet_phi_matches_target_boundary": bool(
            np.isclose(
                case_mass_flow,
                target_mass_flow,
                rtol=1.0e-6,
                atol=1.0e-14,
            )
        ),
    }


def compare_internal_fields(reference_path: Path, case_path: Path) -> dict[str, object]:
    reference = internal_field_values(reference_path)
    actual = internal_field_values(case_path)
    same_size = reference.shape == actual.shape
    if same_size:
        difference = np.abs(reference - actual)
        maximum_absolute = float(difference.max(initial=0.0))
        scale = np.maximum(np.abs(reference), 1.0e-30)
        maximum_relative = float((difference / scale).max(initial=0.0))
        equal = bool(np.allclose(reference, actual, rtol=5.0e-10, atol=5.0e-12))
    else:
        maximum_absolute = float("inf")
        maximum_relative = float("inf")
        equal = False
    return {
        "reference_path": str(reference_path),
        "reference_value_count": int(reference.size),
        "case_value_count": int(actual.size),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "equal_to_source_write_precision": equal,
    }


def verify(case: Path) -> dict[str, object]:
    case = case.resolve()
    metadata = json.loads(
        (case / "fully_coupled_step_metadata.json").read_text(encoding="utf-8")
    )
    source = Path(metadata["source_case"])
    target = Path(metadata["target_case"])
    source_time = str(metadata["source_final_time_s"])
    target_time = str(metadata["target_final_time_s"])
    target_parameters = metadata["target_parameters"]

    mesh_checks: dict[str, dict[str, bool]] = {}
    for region in ("fluid", "solid"):
        mesh_checks[region] = {
            name: sha256(case / "constant" / region / "polyMesh" / name)
            == sha256(source / "constant" / region / "polyMesh" / name)
            for name in MESH_FILES
        }
    if not all(all(rows.values()) for rows in mesh_checks.values()):
        raise ValueError("fully coupled step mesh differs from the source endpoint mesh")

    field_checks = {
        name: compare_internal_fields(
            source / source_time / name,
            case / "0" / name,
        )
        for name in (
            "fluid/U",
            "fluid/p",
            "fluid/p_rgh",
            "fluid/phi",
            "fluid/T",
            "solid/T",
        )
    }
    if not all(row["equal_to_source_write_precision"] for row in field_checks.values()):
        raise ValueError(f"fully coupled initial state differs from its source endpoint: {field_checks}")

    thermo_path = case / "constant/solid/physicalProperties"
    thermo_metadata = json.loads(
        (case / "transient_solid_thermo.json").read_text(encoding="utf-8")
    )
    recorded_ids = tuple(thermo_metadata.get("parameter_ids", ()))
    declared_ids = tuple(metadata.get("transient_thermo_parameter_ids", ()))
    if recorded_ids != TRANSIENT_THERMO_PARAMETER_IDS or declared_ids != TRANSIENT_THERMO_PARAMETER_IDS:
        raise ValueError("fully coupled transient thermo does not use P092, P403 and P428-P431")
    thermo_text = thermo_path.read_text(encoding="ascii")
    if "eIcoTabulated" not in thermo_text or "P406" in thermo_text:
        raise ValueError("fully coupled case retained the steady solid heat-capacity dictionary")
    if sha256(thermo_path) != thermo_metadata.get("physical_properties_sha256"):
        raise ValueError("fully coupled transient thermo checksum differs from its record")
    if thermo_path.samefile(target / "constant/solid/physicalProperties"):
        raise ValueError("fully coupled transient thermo remains hard-linked to the target endpoint")

    heat_source_exact = sha256(case / "constant/solid/fvModels") == sha256(
        target / "constant/solid/fvModels"
    )
    if not heat_source_exact:
        raise ValueError("fully coupled heat-source dictionary differs from the target endpoint")

    expected_u = float(
        target_parameters.get(
            "pore_opening_boundary_velocity_m_s",
            target_parameters["inlet_velocity_m_s"],
        )
    )
    expected_T = float(target_parameters["inlet_temperature_K"])
    inlet_u = velocity_z(foam_value(case / "0/fluid/U", "boundaryField/inlet/value"))
    inlet_T = scalar_from_uniform(
        foam_value(case / "0/fluid/T", "boundaryField/inlet/value")
    )
    outlet_backflow_T = scalar_from_uniform(
        foam_value(case / "0/fluid/T", "boundaryField/outlet/inletValue")
    )
    if abs(inlet_u - expected_u) > 1.0e-12:
        raise ValueError("fully coupled inlet velocity differs from the target pore-opening velocity")
    if abs(inlet_T - expected_T) > 1.0e-12 or abs(outlet_backflow_T - expected_T) > 1.0e-12:
        raise ValueError("fully coupled inlet or outlet-backflow temperature differs from target")

    inlet_flow = inlet_mass_flow_consistency(
        case / "0/fluid/phi",
        source / source_time / "fluid/phi",
        target / target_time / "fluid/phi",
    )
    if not inlet_flow["initial_inlet_phi_matches_target_boundary"]:
        raise ValueError(
            "fully coupled inlet U uses the target condition but inlet phi does not "
            f"match the target boundary mass flow: {inlet_flow}"
        )

    temperature_range = tuple(float(value) for value in thermo_metadata["temperature_range_K"])
    fluid_T = internal_field_values(case / "0/fluid/T")
    solid_T = internal_field_values(case / "0/solid/T")
    initial_temperature_range = (
        float(min(fluid_T.min(), solid_T.min())),
        float(max(fluid_T.max(), solid_T.max())),
    )
    if initial_temperature_range[0] < temperature_range[0] or initial_temperature_range[1] > temperature_range[1]:
        raise ValueError("fully coupled initial temperature is outside the published thermo range")

    return {
        "status": "verified_p418_fully_coupled_flow_heat_step_initial_field",
        "sequence_id": metadata["sequence_id"],
        "source_condition_id": metadata["source_condition_id"],
        "target_condition_id": metadata["target_condition_id"],
        "mesh_files_identical": mesh_checks,
        "all_initial_internal_fields_from_source_endpoint": field_checks,
        "target_pore_opening_velocity_m_s": inlet_u,
        "initial_inlet_mass_flow_consistency": inlet_flow,
        "target_inlet_temperature_K": inlet_T,
        "target_outlet_backflow_temperature_K": outlet_backflow_T,
        "target_heat_source_dictionary_exact": heat_source_exact,
        "transient_thermo_parameter_ids": list(recorded_ids),
        "transient_thermo_temperature_range_K": list(temperature_range),
        "initial_temperature_range_K": list(initial_temperature_range),
        "new_physical_parameters": [],
        "openfoam_calculation_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--write-record", action="store_true")
    args = parser.parse_args()
    result = verify(args.case)
    if args.write_record:
        (args.case.resolve() / "fully_coupled_initial_field_map_complete.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
