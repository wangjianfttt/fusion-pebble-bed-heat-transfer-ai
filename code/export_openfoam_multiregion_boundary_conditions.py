#!/usr/bin/env python3
"""Export OpenFOAM multiregion boundary conditions as face-wise ML tensors.

The exporter reads the actual ``0/<region>`` field dictionaries.  It does not
infer inlet/outlet roles from patch names.  Generic patches are classified only
when the boundary-condition combination and the oriented face geometry provide
the required evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import export_openfoam_multiregion_boundary_faces as boundary_faces


ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_TYPES = {
    "U": {
        "fixedValue",
        "noSlip",
        "pressureInletOutletVelocity",
        "symmetry",
        "symmetryPlane",
        "empty",
        "cyclic",
        "processor",
        "wedge",
    },
    "T": {
        "fixedValue",
        "inletOutlet",
        "zeroGradient",
        "coupledTemperature",
        "symmetry",
        "symmetryPlane",
        "empty",
        "cyclic",
        "processor",
        "wedge",
    },
    "p_rgh": {
        "fixedValue",
        "fixedFluxPressure",
        "zeroGradient",
        "symmetry",
        "symmetryPlane",
        "empty",
        "cyclic",
        "processor",
        "wedge",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_case(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def balanced_block(text: str, start: int, opening: str = "{", closing: str = "}") -> tuple[str, int]:
    if start >= len(text) or text[start] != opening:
        raise ValueError(f"expected {opening!r} at character {start}")
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError(f"unclosed {opening}{closing} block")


def keyword_block(text: str, keyword: str) -> str:
    match = re.search(rf"\b{re.escape(keyword)}\b", text)
    if not match:
        raise ValueError(f"missing {keyword} block")
    brace = text.find("{", match.end())
    if brace < 0:
        raise ValueError(f"missing opening brace after {keyword}")
    return balanced_block(text, brace)[0]


def read_token(text: str, start: int) -> tuple[str, bool, int]:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return "", False, index
    if text[index] in {'"', "'"}:
        quote = text[index]
        index += 1
        begin = index
        escaped = False
        chars: list[str] = []
        while index < len(text):
            char = text[index]
            if escaped:
                chars.append(char)
                escaped = False
            elif char == "\\":
                chars.append(char)
                escaped = True
            elif char == quote:
                return "".join(chars), True, index + 1
            else:
                chars.append(char)
            index += 1
        raise ValueError(f"unclosed quoted token beginning at {begin}")
    begin = index
    while index < len(text) and not text[index].isspace() and text[index] not in "{};":
        index += 1
    return text[begin:index], False, index


def split_statements(block: str) -> dict[str, str]:
    statements: dict[str, str] = {}
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    pairs = {")": "(", "]": "[", "}": "{"}
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(block):
        char = block[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char in depths:
            depths[char] += 1
        elif char in pairs:
            depths[pairs[char]] -= 1
            if depths[pairs[char]] < 0:
                raise ValueError("unbalanced field-entry delimiters")
        elif char == ";" and all(value == 0 for value in depths.values()):
            statement = block[start:index].strip()
            start = index + 1
            if statement and not statement.startswith("#"):
                parts = statement.split(None, 1)
                if len(parts) != 2:
                    raise ValueError(f"cannot parse field entry {statement!r}")
                key, value = parts
                statements[key] = value.strip()
        index += 1
    if block[start:].strip() and not block[start:].strip().startswith("#"):
        raise ValueError(f"unterminated field entry: {block[start:].strip()!r}")
    return statements


@dataclass(frozen=True)
class SelectorBlock:
    selector: str
    is_regex: bool
    entries: dict[str, str]


def parse_selector_blocks(boundary_block: str) -> list[SelectorBlock]:
    blocks: list[SelectorBlock] = []
    index = 0
    while index < len(boundary_block):
        while index < len(boundary_block) and boundary_block[index].isspace():
            index += 1
        if index >= len(boundary_block):
            break
        if boundary_block[index] == "#":
            newline = boundary_block.find("\n", index)
            index = len(boundary_block) if newline < 0 else newline + 1
            continue
        token, quoted, after_token = read_token(boundary_block, index)
        if not token:
            raise ValueError(f"cannot read boundary selector near character {index}")
        index = after_token
        while index < len(boundary_block) and boundary_block[index].isspace():
            index += 1
        if index >= len(boundary_block) or boundary_block[index] != "{":
            raise ValueError(f"selector {token!r} is not followed by a dictionary")
        body, index = balanced_block(boundary_block, index)
        blocks.append(SelectorBlock(token, quoted, split_statements(body)))
    return blocks


def parse_number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite field value {value!r}")
    return result


def parse_uniform_value(expression: str, internal: np.ndarray) -> np.ndarray:
    expression = expression.strip()
    if expression == "$internalField":
        return internal.copy()
    if not expression.startswith("uniform "):
        raise ValueError(f"only uniform values or $internalField are admitted, got {expression!r}")
    payload = expression[len("uniform ") :].strip()
    if payload.startswith("(") and payload.endswith(")"):
        values = np.asarray([parse_number(item) for item in payload[1:-1].split()], dtype=np.float64)
    else:
        values = np.asarray([parse_number(payload)], dtype=np.float64)
    if values.shape != internal.shape:
        raise ValueError(f"field value has shape {values.shape}, expected {internal.shape}")
    return values


def expand_selector(selector: SelectorBlock, patch_names: list[str]) -> list[str]:
    if not selector.is_regex:
        return [selector.selector] if selector.selector in patch_names else []
    try:
        pattern = re.compile(selector.selector)
    except re.error as exc:
        raise ValueError(f"invalid OpenFOAM patch regex {selector.selector!r}") from exc
    return [name for name in patch_names if pattern.fullmatch(name)]


def mesh_constraint_type(patch_type: str) -> str | None:
    if patch_type in {"empty", "cyclic", "processor", "wedge"}:
        return patch_type
    if patch_type in {"symmetry", "symmetryPlane"}:
        return patch_type
    return None


def field_object(text: str) -> str:
    match = re.search(r"(?m)^\s*object\s+([^;]+);", text)
    if not match:
        raise ValueError("field file does not declare an object")
    return match.group(1).strip()


def parse_field(
    path: Path,
    object_name: str,
    vector: bool,
    patches: list[dict],
    require_coverage: bool = True,
) -> dict:
    text = strip_comments(path.read_text(encoding="utf-8", errors="strict"))
    if field_object(text) != object_name:
        raise ValueError(f"{path} does not declare object {object_name}")
    internal_match = re.search(r"\binternalField\s+([^;]+);", text)
    if not internal_match:
        raise ValueError(f"{path} misses internalField")
    shape = (3,) if vector else (1,)
    internal = parse_uniform_value(internal_match.group(1), np.zeros(shape, dtype=np.float64))
    patch_names = [record["patch_name"] for record in patches]
    assignments: dict[str, dict[str, str]] = {}
    selector_table: list[dict] = []
    for selector in parse_selector_blocks(keyword_block(text, "boundaryField")):
        matched = expand_selector(selector, patch_names)
        selector_table.append(
            {
                "selector": selector.selector,
                "is_regex": selector.is_regex,
                "matched_patches": matched,
            }
        )
        if not matched:
            raise ValueError(f"field selector {selector.selector!r} matches no mesh patch")
        for name in matched:
            if name in assignments:
                raise ValueError(f"field {object_name} assigns patch {name!r} more than once")
            assignments[name] = dict(selector.entries)

    for record in patches:
        name = record["patch_name"]
        if name not in assignments:
            constraint = mesh_constraint_type(record["patch_type"])
            if constraint:
                assignments[name] = {"type": constraint}

    missing = sorted(set(patch_names) - set(assignments))
    if require_coverage and missing:
        raise ValueError(f"field {object_name} misses mesh patches: {missing}")
    extra = sorted(set(assignments) - set(patch_names))
    if extra:
        raise ValueError(f"field {object_name} contains unknown mesh patches: {extra}")

    parsed: dict[str, dict] = {}
    for name, entries in assignments.items():
        if "type" not in entries:
            raise ValueError(f"field {object_name} patch {name!r} misses type")
        bc_type = entries["type"].strip()
        if bc_type not in SUPPORTED_TYPES[object_name]:
            raise ValueError(f"unsupported {object_name} boundary type {bc_type!r} on {name}")
        record = {"type": bc_type, "entries": entries}
        for key in ("value", "inletValue"):
            if key in entries:
                record[key] = parse_uniform_value(entries[key], internal)
        parsed[name] = record
    return {
        "path": path,
        "internal": internal,
        "patches": parsed,
        "missing_patches": missing,
        "selectors": selector_table,
    }


def load_boundary_contract(summary_path: Path, case: Path) -> tuple[dict, dict[str, np.ndarray], Path]:
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "openfoam_multiregion_boundary_faces_passed":
        raise RuntimeError("boundary-face summary has not passed")
    if resolved_case(summary["case"]) != case:
        raise RuntimeError("boundary-face summary belongs to a different OpenFOAM case")
    npz_path = summary_path.parent / "multiregion_boundary_faces.npz"
    if not npz_path.is_file() or sha256(npz_path) != summary.get("boundary_npz_sha256"):
        raise RuntimeError("boundary-face NPZ is missing or does not match its summary")
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    required = {
        "patch_index",
        "region_type",
        "physical_role_id",
        "interface_pair_index",
        "outward_area_vector_m2",
    }
    if not required.issubset(arrays):
        raise RuntimeError(f"boundary-face NPZ misses arrays: {sorted(required - set(arrays))}")
    return summary, arrays, npz_path


def field_path(case: Path, time_name: str, region: str, name: str, override: Path | None) -> Path:
    return override.resolve() if override else (case / time_name / region / name).resolve()


def assign_type_ids(field_data: dict[str, dict], patch_table: list[dict], patch_index: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    types = sorted({record["type"] for record in field_data.values()})
    table = [{"bc_type_id": index, "bc_type": name} for index, name in enumerate(types)]
    mapping = {record["bc_type"]: record["bc_type_id"] for record in table}
    per_patch = {record["patch_index"]: mapping[field_data[record["patch_name"]]["type"]] for record in patch_table}
    return np.asarray([per_patch[int(index)] for index in patch_index], dtype=np.int16), table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--boundary-summary", type=Path, required=True)
    parser.add_argument("--time", default="0")
    parser.add_argument("--fluid-region", default="fluid")
    parser.add_argument("--solid-region", default="solid")
    parser.add_argument("--fluid-u", type=Path)
    parser.add_argument("--fluid-t", type=Path)
    parser.add_argument("--fluid-p-rgh", type=Path)
    parser.add_argument("--solid-t", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    case = args.case.resolve()
    summary, boundary, boundary_npz = load_boundary_contract(args.boundary_summary, case)
    if summary.get("fluid_region") != args.fluid_region or summary.get("solid_region") != args.solid_region:
        raise RuntimeError("requested regions do not match the boundary-face contract")
    output = (args.output_dir or (case / "boundary_conditions")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    patch_table = summary["patch_table"]
    fluid_patches = [record for record in patch_table if record["region"] == args.fluid_region]
    solid_patches = [record for record in patch_table if record["region"] == args.solid_region]
    paths = {
        "fluid_U": field_path(case, args.time, args.fluid_region, "U", args.fluid_u),
        "fluid_T": field_path(case, args.time, args.fluid_region, "T", args.fluid_t),
        "fluid_p_rgh": field_path(case, args.time, args.fluid_region, "p_rgh", args.fluid_p_rgh),
        "solid_T": field_path(case, args.time, args.solid_region, "T", args.solid_t),
    }
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"missing field dictionaries: {missing_paths}")

    fields = {
        "fluid_U": parse_field(paths["fluid_U"], "U", True, fluid_patches),
        "fluid_T": parse_field(paths["fluid_T"], "T", False, fluid_patches),
        "fluid_p_rgh": parse_field(paths["fluid_p_rgh"], "p_rgh", False, fluid_patches),
        "solid_T": parse_field(paths["solid_T"], "T", False, solid_patches),
    }
    n_faces = len(boundary["patch_index"])
    patch_index = np.asarray(boundary["patch_index"], dtype=np.int64)
    region_type = np.asarray(boundary["region_type"], dtype=np.int8)
    area_vector = np.asarray(boundary["outward_area_vector_m2"], dtype=np.float64)
    role_id = np.asarray(boundary["physical_role_id"], dtype=np.int8).copy()
    role_basis = {record["patch_index"]: record["role_basis"] for record in patch_table}

    all_u: dict[str, dict] = {}
    all_t: dict[str, dict] = {}
    all_p: dict[str, dict] = {}
    for record in patch_table:
        name = record["patch_name"]
        if record["region"] == args.fluid_region:
            all_u[name] = fields["fluid_U"]["patches"][name]
            all_t[name] = fields["fluid_T"]["patches"][name]
            all_p[name] = fields["fluid_p_rgh"]["patches"][name]
        else:
            all_t[name] = fields["solid_T"]["patches"][name]

    u_type_id = np.full(n_faces, -1, dtype=np.int16)
    t_type_id = np.full(n_faces, -1, dtype=np.int16)
    p_type_id = np.full(n_faces, -1, dtype=np.int16)
    u_types = sorted({record["type"] for record in fields["fluid_U"]["patches"].values()})
    t_types = sorted({record["type"] for record in all_t.values()})
    p_types = sorted({record["type"] for record in fields["fluid_p_rgh"]["patches"].values()})
    u_type_map = {name: index for index, name in enumerate(u_types)}
    t_type_map = {name: index for index, name in enumerate(t_types)}
    p_type_map = {name: index for index, name in enumerate(p_types)}

    u_value = np.full((n_faces, 3), np.nan, dtype=np.float64)
    u_value_mask = np.zeros(n_faces, dtype=bool)
    u_fixed_mask = np.zeros(n_faces, dtype=bool)
    u_no_slip_mask = np.zeros(n_faces, dtype=bool)
    u_pressure_inlet_outlet_mask = np.zeros(n_faces, dtype=bool)
    t_value = np.full(n_faces, np.nan, dtype=np.float64)
    t_value_mask = np.zeros(n_faces, dtype=bool)
    t_fixed_mask = np.zeros(n_faces, dtype=bool)
    t_inlet_value = np.full(n_faces, np.nan, dtype=np.float64)
    t_inlet_value_mask = np.zeros(n_faces, dtype=bool)
    t_inlet_outlet_mask = np.zeros(n_faces, dtype=bool)
    t_zero_gradient_mask = np.zeros(n_faces, dtype=bool)
    t_coupled_mask = np.zeros(n_faces, dtype=bool)
    p_value = np.full(n_faces, np.nan, dtype=np.float64)
    p_value_mask = np.zeros(n_faces, dtype=bool)
    p_fixed_mask = np.zeros(n_faces, dtype=bool)
    p_fixed_flux_mask = np.zeros(n_faces, dtype=bool)
    symmetry_mask = np.zeros(n_faces, dtype=bool)
    empty_mask = np.zeros(n_faces, dtype=bool)

    role_codes = summary["physical_role_codes"]
    updated_patch_table: list[dict] = []
    for record in patch_table:
        patch = dict(record)
        index = int(record["patch_index"])
        mask = patch_index == index
        name = record["patch_name"]
        t_record = all_t[name]
        t_type_id[mask] = t_type_map[t_record["type"]]
        if "value" in t_record:
            t_value[mask] = float(t_record["value"][0])
            t_value_mask[mask] = True
        if "inletValue" in t_record:
            t_inlet_value[mask] = float(t_record["inletValue"][0])
            t_inlet_value_mask[mask] = True
        t_fixed_mask[mask] = t_record["type"] == "fixedValue"
        t_inlet_outlet_mask[mask] = t_record["type"] == "inletOutlet"
        t_zero_gradient_mask[mask] = t_record["type"] == "zeroGradient"
        t_coupled_mask[mask] = t_record["type"] == "coupledTemperature"
        symmetry_mask[mask] |= t_record["type"] in {"symmetry", "symmetryPlane"}
        empty_mask[mask] |= t_record["type"] == "empty"
        patch["T_type"] = t_record["type"]

        if record["region"] == args.fluid_region:
            u_record = all_u[name]
            p_record = all_p[name]
            u_type_id[mask] = u_type_map[u_record["type"]]
            p_type_id[mask] = p_type_map[p_record["type"]]
            if "value" in u_record:
                u_value[mask] = u_record["value"]
                u_value_mask[mask] = True
            if "value" in p_record:
                p_value[mask] = float(p_record["value"][0])
                p_value_mask[mask] = True
            u_fixed_mask[mask] = u_record["type"] in {"fixedValue", "noSlip"}
            u_no_slip_mask[mask] = u_record["type"] == "noSlip"
            if u_record["type"] == "noSlip":
                u_value[mask] = 0.0
                u_value_mask[mask] = True
            u_pressure_inlet_outlet_mask[mask] = u_record["type"] == "pressureInletOutletVelocity"
            p_fixed_mask[mask] = p_record["type"] == "fixedValue"
            p_fixed_flux_mask[mask] = p_record["type"] == "fixedFluxPressure"
            symmetry_mask[mask] |= u_record["type"] in {"symmetry", "symmetryPlane"}
            empty_mask[mask] |= u_record["type"] == "empty"
            patch["U_type"] = u_record["type"]
            patch["p_rgh_type"] = p_record["type"]

            if record["physical_role"] == "unresolved":
                if (
                    u_record["type"] == "fixedValue"
                    and t_record["type"] == "fixedValue"
                    and p_record["type"] == "fixedFluxPressure"
                    and "value" in u_record
                ):
                    signed_flux = area_vector[mask] @ u_record["value"]
                    tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(
                        1.0, np.linalg.norm(area_vector[mask], axis=1) * np.linalg.norm(u_record["value"])
                    )
                    if np.all(signed_flux <= tolerance) and np.any(signed_flux < -tolerance):
                        role_id[mask] = role_codes["inlet"]
                        patch["physical_role"] = "inlet"
                        patch["role_basis"] = "fixed inflow U/T, fixedFluxPressure and inward oriented face flux"
                elif (
                    u_record["type"] == "pressureInletOutletVelocity"
                    and t_record["type"] == "inletOutlet"
                    and p_record["type"] == "fixedValue"
                ):
                    role_id[mask] = role_codes["outlet"]
                    patch["physical_role"] = "outlet"
                    patch["role_basis"] = "pressureInletOutletVelocity, inletOutlet T and fixed p_rgh"
                elif u_record["type"] == "noSlip":
                    role_id[mask] = role_codes["wall"]
                    patch["physical_role"] = "wall"
                    patch["role_basis"] = "noSlip velocity boundary condition"
        updated_patch_table.append(patch)

    exact_interface = np.asarray(boundary["interface_pair_index"], dtype=np.int64) >= 0
    fluid_exact = exact_interface & (region_type == 0)
    solid_exact = exact_interface & (region_type == 1)
    checks = {
        "all_boundary_faces_receive_temperature_bc": bool(np.all(t_type_id >= 0)),
        "all_fluid_boundary_faces_receive_velocity_bc": bool(np.all(u_type_id[region_type == 0] >= 0)),
        "all_fluid_boundary_faces_receive_pressure_bc": bool(np.all(p_type_id[region_type == 0] >= 0)),
        "solid_faces_do_not_receive_fluid_bc": bool(
            np.all(u_type_id[region_type == 1] == -1) and np.all(p_type_id[region_type == 1] == -1)
        ),
        "all_exact_interface_faces_use_coupled_temperature": bool(
            np.all(t_coupled_mask[fluid_exact]) and np.all(t_coupled_mask[solid_exact])
        ),
        "exact_fluid_interface_faces_are_no_slip": bool(np.all(u_no_slip_mask[fluid_exact])),
        "exact_fluid_interface_faces_use_fixed_flux_pressure": bool(np.all(p_fixed_flux_mask[fluid_exact])),
        "all_numeric_boundary_values_are_finite": bool(
            np.all(np.isfinite(u_value[u_value_mask]))
            and np.all(np.isfinite(t_value[t_value_mask]))
            and np.all(np.isfinite(t_inlet_value[t_inlet_value_mask]))
            and np.all(np.isfinite(p_value[p_value_mask]))
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"boundary-condition tensor checks failed: {checks}")

    arrays = {
        "patch_index": patch_index,
        "region_type": region_type,
        "physical_role_id": role_id,
        "U_bc_type_id": u_type_id,
        "T_bc_type_id": t_type_id,
        "p_rgh_bc_type_id": p_type_id,
        "U_reference_value_m_s": u_value,
        "U_reference_value_mask": u_value_mask,
        "U_fixed_value_mask": u_fixed_mask,
        "U_no_slip_mask": u_no_slip_mask,
        "U_pressure_inlet_outlet_mask": u_pressure_inlet_outlet_mask,
        "T_reference_value_K": t_value,
        "T_reference_value_mask": t_value_mask,
        "T_fixed_value_mask": t_fixed_mask,
        "T_inlet_value_K": t_inlet_value,
        "T_inlet_value_mask": t_inlet_value_mask,
        "T_inlet_outlet_mask": t_inlet_outlet_mask,
        "T_zero_gradient_mask": t_zero_gradient_mask,
        "T_coupled_temperature_mask": t_coupled_mask,
        "p_rgh_reference_value_Pa": p_value,
        "p_rgh_reference_value_mask": p_value_mask,
        "p_rgh_fixed_value_mask": p_fixed_mask,
        "p_rgh_fixed_flux_mask": p_fixed_flux_mask,
        "symmetry_mask": symmetry_mask,
        "empty_mask": empty_mask,
    }
    npz_path = output / "multiregion_boundary_conditions.npz"
    np.savez_compressed(npz_path, **arrays)

    unresolved_fluid = [
        record for record in updated_patch_table
        if record["region"] == args.fluid_region and record["physical_role"] == "unresolved"
    ]
    unpaired_coupled = [
        record for record in updated_patch_table
        if record["physical_role"] == "mapped_coupled_unpaired" and record["T_type"] == "coupledTemperature"
    ]
    payload = {
        "status": "openfoam_multiregion_boundary_conditions_passed",
        "case": str(case.relative_to(ROOT)) if case.is_relative_to(ROOT) else str(case),
        "time": args.time,
        "fluid_region": args.fluid_region,
        "solid_region": args.solid_region,
        "counts": {
            "boundary_faces": n_faces,
            "fluid_faces": int(np.count_nonzero(region_type == 0)),
            "solid_faces": int(np.count_nonzero(region_type == 1)),
            "fixed_velocity_faces": int(np.count_nonzero(u_fixed_mask)),
            "fixed_temperature_faces": int(np.count_nonzero(t_fixed_mask)),
            "coupled_temperature_faces": int(np.count_nonzero(t_coupled_mask)),
            "resolved_inlet_faces": int(np.count_nonzero(role_id == role_codes["inlet"])),
            "resolved_outlet_faces": int(np.count_nonzero(role_id == role_codes["outlet"])),
        },
        "checks": checks,
        "field_paths": {
            name: str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            for name, path in paths.items()
        },
        "field_sha256": {name: sha256(path) for name, path in paths.items()},
        "internal_fields": {
            name: fields[name]["internal"].tolist() for name in fields
        },
        "selector_expansion": {
            name: fields[name]["selectors"] for name in fields
        },
        "U_bc_type_table": [
            {"bc_type_id": index, "bc_type": name} for name, index in u_type_map.items()
        ],
        "T_bc_type_table": [
            {"bc_type_id": index, "bc_type": name} for name, index in t_type_map.items()
        ],
        "p_rgh_bc_type_table": [
            {"bc_type_id": index, "bc_type": name} for name, index in p_type_map.items()
        ],
        "physical_role_codes": role_codes,
        "patch_table": updated_patch_table,
        "unresolved_fluid_physical_role_patches": [
            f"{record['region']}/{record['patch_name']}" for record in unresolved_fluid
        ],
        "unpaired_coupled_temperature_patches": [
            f"{record['region']}/{record['patch_name']}" for record in unpaired_coupled
        ],
        "boundary_face_summary_sha256": sha256(args.boundary_summary.resolve()),
        "boundary_face_npz_sha256": sha256(boundary_npz),
        "boundary_condition_npz_sha256": sha256(npz_path),
        "new_fitted_physical_parameters": [],
        "neural_training_allowed": not unresolved_fluid and not unpaired_coupled,
        "claim_boundary": (
            "Boundary-condition semantics and sourced uniform values only. This artifact does not reconstruct "
            "cell-to-face fluxes and is not a heat-transfer solution."
        ),
    }
    summary_out = output / "summary.json"
    summary_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
