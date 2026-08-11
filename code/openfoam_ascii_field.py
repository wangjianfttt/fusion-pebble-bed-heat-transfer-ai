#!/usr/bin/env python3
"""Read ASCII OpenFOAM scalar and vector volume/surface fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from export_openfoam_multiregion_boundary_conditions import (
    keyword_block,
    parse_selector_blocks,
    strip_comments,
)


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


@dataclass(frozen=True)
class OpenFoamAsciiField:
    field_class: str
    object_name: str
    internal: np.ndarray
    boundary_type: dict[str, str]
    boundary_value: dict[str, np.ndarray | None]


def _field_header_value(text: str, key: str) -> str:
    foam_file = keyword_block(text, "FoamFile")
    match = re.search(rf"\b{re.escape(key)}\s+([^;\s]+)\s*;", foam_file)
    if not match:
        raise ValueError(f"missing FoamFile/{key}")
    return match.group(1).strip('"')


def _parse_value_expression(
    expression: str,
    *,
    expected_count: int,
    expected_components: int,
) -> np.ndarray:
    expression = expression.strip()
    uniform = re.fullmatch(r"uniform\s+(.+)", expression, flags=re.DOTALL)
    if uniform:
        payload = uniform.group(1).strip()
        if payload.startswith("(") and payload.endswith(")"):
            values = [float(value) for value in re.findall(NUMBER, payload)]
        else:
            values = [float(payload)]
        if len(values) != expected_components:
            raise ValueError(
                f"uniform field has {len(values)} components, expected {expected_components}"
            )
        row = np.asarray(values, dtype=np.float64)
        return np.repeat(row[None, :], expected_count, axis=0)

    nonuniform = re.fullmatch(
        r"nonuniform\s+List<(scalar|vector)>\s+(\d+)\s*\((.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if not nonuniform:
        raise ValueError(f"unsupported OpenFOAM field expression: {expression[:80]!r}")
    kind, count_text, body = nonuniform.groups()
    count = int(count_text)
    if count != expected_count:
        raise ValueError(f"field declares {count} values, expected {expected_count}")
    components = 1 if kind == "scalar" else 3
    if components != expected_components:
        raise ValueError(
            f"field declares {components} components, expected {expected_components}"
        )
    if components == 1:
        values = [float(value) for value in re.findall(NUMBER, body)]
        array = np.asarray(values, dtype=np.float64)[:, None]
    else:
        records = re.findall(
            rf"\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)", body
        )
        array = np.asarray(records, dtype=np.float64)
    if array.shape != (expected_count, expected_components):
        raise ValueError(
            f"parsed field shape {array.shape}, expected {(expected_count, expected_components)}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("field contains a non-finite value")
    return array


def _internal_expression(text: str) -> str:
    match = re.search(
        r"\binternalField\b\s+(.*?)\s*;\s*\bboundaryField\b",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("cannot locate internalField before boundaryField")
    return match.group(1)


def read_openfoam_ascii_field(
    path: Path,
    *,
    internal_count: int,
    patch_sizes: dict[str, int],
) -> OpenFoamAsciiField:
    """Read one scalar/vector field and expand uniform patch values."""
    text = strip_comments(path.read_text(encoding="utf-8", errors="strict"))
    if re.search(r"\bformat\s+binary\s*;", text):
        raise ValueError(f"binary field is not supported: {path}")
    field_class = _field_header_value(text, "class")
    object_name = _field_header_value(text, "object")
    components = 3 if "Vector" in field_class else 1
    internal = _parse_value_expression(
        _internal_expression(text),
        expected_count=internal_count,
        expected_components=components,
    )
    selectors = parse_selector_blocks(keyword_block(text, "boundaryField"))
    entries_by_name = {selector.selector: selector.entries for selector in selectors}
    missing = set(patch_sizes) - set(entries_by_name)
    extra = set(entries_by_name) - set(patch_sizes)
    if missing or extra:
        raise ValueError(
            f"field/mesh patch mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    boundary_type: dict[str, str] = {}
    boundary_value: dict[str, np.ndarray | None] = {}
    for name, count in patch_sizes.items():
        entries = entries_by_name[name]
        if "type" not in entries:
            raise ValueError(f"patch {name} has no boundary type")
        boundary_type[name] = entries["type"]
        value = entries.get("value")
        boundary_value[name] = None if value is None else _parse_value_expression(
            value,
            expected_count=count,
            expected_components=components,
        )
    return OpenFoamAsciiField(
        field_class=field_class,
        object_name=object_name,
        internal=internal,
        boundary_type=boundary_type,
        boundary_value=boundary_value,
    )

