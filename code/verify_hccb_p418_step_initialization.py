#!/usr/bin/env python3
"""Check mapped step fields and exact target boundary values."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np


MESH_FILES = ("points", "faces", "owner", "neighbour", "boundary")
TRANSIENT_THERMO_PARAMETER_IDS = ("P092", "P403", "P428", "P429", "P430", "P431")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def foam_value(path: Path, entry: str) -> str:
    result = subprocess.run(
        ["foamDictionary", str(path), "-entry", entry, "-value"],
        check=True,
        capture_output=True,
        text=True,
    )
    return " ".join(result.stdout.split())


def scalar_from_uniform(value: str) -> float:
    match = re.search(r"uniform\s+([-+0-9.eE]+)", value)
    if not match:
        raise ValueError(f"not a uniform scalar: {value}")
    return float(match.group(1))


def velocity_z(value: str) -> float:
    match = re.search(r"uniform\s*\(\s*[-+0-9.eE]+\s+[-+0-9.eE]+\s+([-+0-9.eE]+)\s*\)", value)
    if not match:
        raise ValueError(f"not a uniform vector: {value}")
    return float(match.group(1))


FLOAT_PATTERN = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def internal_field_values(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"internalField\s+(.*?)\s+boundaryField", text, flags=re.S)
    if not match:
        raise ValueError(f"cannot locate internalField in {path}")
    field = match.group(1).strip()
    nonuniform = re.fullmatch(
        r"nonuniform\s+List<[^>]+>\s+\d+\s*\((.*)\)\s*;?",
        field,
        flags=re.S,
    )
    if nonuniform:
        values_text = nonuniform.group(1)
    else:
        uniform = re.fullmatch(r"uniform\s+(.*?)\s*;?", field, flags=re.S)
        if not uniform:
            raise ValueError(f"unsupported internalField representation in {path}")
        values_text = uniform.group(1)
    values = np.asarray(
        [float(value) for value in FLOAT_PATTERN.findall(values_text)], dtype=np.float64
    )
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError(f"internalField values are empty or non-finite in {path}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--write-record", action="store_true")
    args = parser.parse_args()
    case = args.case.resolve()
    metadata = json.loads((case / "step_case_metadata.json").read_text(encoding="utf-8"))
    source = Path(metadata["source_case"])
    target_case = Path(metadata["target_case"])
    source_time = str(metadata["source_final_time_s"])
    target_time = str(metadata["target_final_time_s"])
    target = metadata["target_parameters"]

    thermo_path = case / "constant/solid/physicalProperties"
    thermo_metadata_path = case / "transient_solid_thermo.json"
    if not thermo_metadata_path.is_file():
        raise ValueError("transient solid-thermo metadata is missing")
    thermo_metadata = json.loads(thermo_metadata_path.read_text(encoding="utf-8"))
    declared_ids = tuple(metadata.get("transient_thermo_parameter_ids", ()))
    recorded_ids = tuple(thermo_metadata.get("parameter_ids", ()))
    if declared_ids != TRANSIENT_THERMO_PARAMETER_IDS or recorded_ids != TRANSIENT_THERMO_PARAMETER_IDS:
        raise ValueError(
            "transient Li4SiO4 thermo must use P092, P403 and P428-P431 exactly; "
            f"case={declared_ids}, thermo={recorded_ids}"
        )
    thermo_text = thermo_path.read_text(encoding="ascii")
    if "eIcoTabulated" not in thermo_text or "P406" in thermo_text:
        raise ValueError("transient solid thermo did not replace the steady P406 dictionary")
    if sha256(thermo_path) != thermo_metadata.get("physical_properties_sha256"):
        raise ValueError("transient solid-thermo checksum differs from its build record")
    if thermo_path.samefile(target_case / "constant/solid/physicalProperties"):
        raise ValueError("transient solid thermo is still hard-linked to the steady endpoint")

    mesh_checks: dict[str, dict[str, bool]] = {}
    for region in ("fluid", "solid"):
        region_checks = {}
        for name in MESH_FILES:
            region_checks[name] = sha256(case / "constant" / region / "polyMesh" / name) == sha256(
                source / "constant" / region / "polyMesh" / name
            )
        mesh_checks[region] = region_checks
    if not all(all(values.values()) for values in mesh_checks.values()):
        raise ValueError("source and target step meshes are not identical")

    field_pairs = {
        "fluid/U": ("target_hydrodynamic", target_case / target_time / "fluid/U", case / "0/fluid/U"),
        "fluid/T": ("source_thermal", source / source_time / "fluid/T", case / "0/fluid/T"),
        "fluid/p": ("target_hydrodynamic", target_case / target_time / "fluid/p", case / "0/fluid/p"),
        "fluid/p_rgh": ("target_hydrodynamic", target_case / target_time / "fluid/p_rgh", case / "0/fluid/p_rgh"),
        "fluid/phi": ("target_mass_flux", target_case / target_time / "fluid/phi", case / "0/fluid/phi"),
        "solid/T": ("source_thermal", source / source_time / "solid/T", case / "0/solid/T"),
    }
    internal_field_checks = {}
    for name, (origin, reference_path, step_path) in field_pairs.items():
        reference_values = internal_field_values(reference_path)
        step_values = internal_field_values(step_path)
        same_size = reference_values.shape == step_values.shape
        if same_size:
            absolute = np.abs(reference_values - step_values)
            maximum_absolute_difference = float(absolute.max(initial=0.0))
            scale = np.maximum(np.abs(reference_values), 1.0e-30)
            maximum_relative_difference = float((absolute / scale).max(initial=0.0))
            values_equal_to_file_precision = bool(
                np.allclose(reference_values, step_values, rtol=5.0e-10, atol=5.0e-12)
            )
        else:
            maximum_absolute_difference = float("inf")
            maximum_relative_difference = float("inf")
            values_equal_to_file_precision = False
        internal_field_checks[name] = {
            "origin": origin,
            "reference_path": str(reference_path),
            "reference_value_count": int(reference_values.size),
            "step_value_count": int(step_values.size),
            "maximum_absolute_difference": maximum_absolute_difference,
            "maximum_relative_difference": maximum_relative_difference,
            "equal_to_reference_write_precision": values_equal_to_file_precision,
        }
    if not all(row["equal_to_reference_write_precision"] for row in internal_field_checks.values()):
        raise ValueError(f"step internal fields differ from their declared origins: {internal_field_checks}")

    solid_initial_temperature = internal_field_values(case / "0/solid/T")
    fluid_initial_temperature = internal_field_values(case / "0/fluid/T")
    temperature_range = tuple(float(value) for value in thermo_metadata["temperature_range_K"])
    initial_temperature_range = (
        float(min(fluid_initial_temperature.min(), solid_initial_temperature.min())),
        float(max(fluid_initial_temperature.max(), solid_initial_temperature.max())),
    )
    if initial_temperature_range[0] < temperature_range[0] or initial_temperature_range[1] > temperature_range[1]:
        raise ValueError(
            f"initial temperature range {initial_temperature_range} is outside the published "
            f"Li4SiO4 heat-capacity range {temperature_range}"
        )
    target_heat_source_file_exact = sha256(case / "constant/solid/fvModels") == sha256(
        target_case / "constant/solid/fvModels"
    )
    if not target_heat_source_file_exact:
        raise ValueError("step heat-source dictionary differs from the target P418 case")

    required = [
        case / "0/fluid/U",
        case / "0/fluid/T",
        case / "0/fluid/p",
        case / "0/fluid/p_rgh",
        case / "0/fluid/phi",
        case / "0/solid/T",
    ]
    nonuniform = {str(path.relative_to(case)): "nonuniform" in path.read_text(encoding="utf-8", errors="replace") for path in required}
    if not all(nonuniform.values()):
        raise ValueError(f"mapped internal fields are incomplete: {nonuniform}")

    inlet_u = velocity_z(foam_value(case / "0/fluid/U", "boundaryField/inlet/value"))
    inlet_T = scalar_from_uniform(foam_value(case / "0/fluid/T", "boundaryField/inlet/value"))
    outlet_backflow_T = scalar_from_uniform(
        foam_value(case / "0/fluid/T", "boundaryField/outlet/inletValue")
    )
    if abs(inlet_u - float(target["inlet_velocity_m_s"])) > 1e-12:
        raise ValueError("mapped inlet velocity does not match target P418 condition")
    if abs(inlet_T - float(target["inlet_temperature_K"])) > 1e-12:
        raise ValueError("mapped inlet temperature does not match target P418 condition")
    if abs(outlet_backflow_T - float(target["inlet_temperature_K"])) > 1e-12:
        raise ValueError("outlet backflow temperature does not match target P418 condition")

    record = {
        "status": "verified_p418_quasi_steady_flow_thermal_step_initial_field",
        "sequence_id": metadata["sequence_id"],
        "source_condition_id": metadata["source_condition_id"],
        "target_condition_id": metadata["target_condition_id"],
        "mesh_files_identical": mesh_checks,
        "declared_initial_fields_equal_to_write_precision": internal_field_checks,
        "hydrodynamic_field_origin": "target converged U, p, p_rgh and conservative face mass flux phi",
        "thermal_field_origin": "source converged fluid and solid T",
        "target_heat_source_dictionary_exact": target_heat_source_file_exact,
        "mapped_fields_nonuniform": nonuniform,
        "target_inlet_velocity_m_s": inlet_u,
        "target_inlet_temperature_K": inlet_T,
        "target_outlet_backflow_temperature_K": outlet_backflow_T,
        "transient_thermo_parameter_ids": list(recorded_ids),
        "transient_thermo_temperature_range_K": list(temperature_range),
        "initial_temperature_range_K": list(initial_temperature_range),
        "transient_thermo_checksum_verified": True,
        "steady_P406_absent_from_transient_thermo": True,
        "transient_thermo_hardlink_broken": True,
        "new_physical_parameters": [],
    }
    if args.write_record:
        (case / "initial_field_map_complete.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
