#!/usr/bin/env python3
"""Verify the completed fine-mesh case used by the P418 mesh comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"fine-mesh comparison input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--completion", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--mesh-manifest", required=True, type=Path)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--end-time", required=True, type=float)
    args = parser.parse_args()

    metadata = load(args.metadata)
    completion = load(args.completion)
    result = load(args.result)
    manifest = load(args.mesh_manifest)
    physical = result.get("physical_conditions", {})

    source = metadata.get("source_inlet_channel_velocity_m_s")
    pore = metadata.get("pore_opening_boundary_velocity_m_s")
    fraction = metadata.get("inlet_open_area_fraction")
    finite_velocities = False
    flow_mapping = False
    try:
        values = tuple(float(value) for value in (source, pore, fraction))
        finite_velocities = all(math.isfinite(value) for value in values)
        flow_mapping = finite_velocities and math.isclose(
            values[1] * values[2], values[0], rel_tol=1.0e-12
        )
    except (TypeError, ValueError):
        pass

    checks = {
        "condition": metadata.get("operating_condition_id") == args.condition_id,
        "fine_mesh_label": metadata.get("mesh_resolution_label") == "fine",
        "packing": metadata.get("mesh_source_packing_sha256")
        == manifest.get("source_packing_sha256"),
        "source_flow": metadata.get("source_channel_volume_flow_preserved") is True,
        "finite_velocities": finite_velocities,
        "flow_mapping": flow_mapping,
        "metadata_end_time": close(metadata.get("end_time"), args.end_time),
        "completed": completion.get("solver_finished") is True,
        "completion_condition": completion.get("condition_id") == args.condition_id,
        "completion_time": close(completion.get("time"), args.end_time),
        "result_velocity": close(physical.get("inlet_velocity_m_s"), source),
        "result_temperature": close(
            physical.get("inlet_temperature_K"), metadata.get("inlet_temperature_K")
        ),
        "result_heat_source": close(
            physical.get("solid_heat_source_W_m3"),
            metadata.get("solid_heat_source_W_m3"),
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("fine-mesh comparison input failed: " + ", ".join(failed))
    print(
        json.dumps(
            {
                "status": "verified_p418_mesh_fine_reference",
                "condition_id": args.condition_id,
                "end_time_s": args.end_time,
                "source_packing_sha256": manifest["source_packing_sha256"],
                "checks": checks,
                "new_physical_parameters": [],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
