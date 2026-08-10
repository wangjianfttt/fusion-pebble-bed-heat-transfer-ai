#!/usr/bin/env python3
"""Store one shared mesh and separate physical fields for completed P418 cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from summarize_hccb_p418_formal_steady_tail import verified_recorded_summary


TOPOLOGY_KEYS = (
    "fluid_cell_centroid_m",
    "fluid_cell_volume_m3",
    "fluid_internal_face_owner",
    "fluid_internal_face_neighbour",
    "solid_cell_centroid_m",
    "solid_cell_volume_m3",
    "solid_internal_face_owner",
    "solid_internal_face_neighbour",
    "fluid_boundary_face_owner",
    "fluid_boundary_face_patch",
    "fluid_boundary_face_centroid_m",
    "fluid_boundary_face_area_vector_outward_m2",
    "fluid_boundary_face_area_m2",
    "fluid_boundary_velocity_value_mask",
    "fluid_boundary_pressure_value_mask",
    "fluid_boundary_temperature_value_mask",
    "fluid_boundary_density_value_mask",
    "fluid_boundary_mass_flow_value_mask",
    "solid_boundary_face_owner",
    "solid_boundary_face_patch",
    "solid_boundary_face_centroid_m",
    "solid_boundary_face_area_vector_outward_m2",
    "solid_boundary_face_area_m2",
    "solid_boundary_temperature_value_mask",
    "interface_fluid_cell",
    "interface_solid_cell",
    "interface_face_centroid_m",
    "interface_area_vector_fluid_outward_m2",
    "interface_face_area_m2",
)
FIELD_KEYS = (
    "fluid_velocity_m_s",
    "fluid_pressure_Pa",
    "fluid_temperature_K",
    "fluid_density_kg_m3",
    "fluid_internal_face_mass_flow_kg_s",
    "fluid_boundary_velocity_m_s",
    "fluid_boundary_pressure_Pa",
    "fluid_boundary_temperature_K",
    "fluid_boundary_density_kg_m3",
    "fluid_boundary_face_mass_flow_kg_s",
    "solid_temperature_K",
    "solid_boundary_temperature_K",
)
POSITIVE_KEYS = (
    "fluid_cell_volume_m3",
    "solid_cell_volume_m3",
    "fluid_boundary_face_area_m2",
    "solid_boundary_face_area_m2",
    "interface_face_area_m2",
    "fluid_temperature_K",
    "fluid_density_kg_m3",
    "solid_temperature_K",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sample_arrays(
    arrays: dict[str, np.ndarray], condition_id: str, patch_names: dict[str, list[str]]
) -> None:
    expected_keys = set(TOPOLOGY_KEYS + FIELD_KEYS)
    actual_keys = set(arrays)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"{condition_id}: schema-v3 keys differ; missing={missing}, unexpected={unexpected}"
        )
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"{condition_id}: {key} contains non-finite values")
    for key in POSITIVE_KEYS:
        if np.any(arrays[key] <= 0.0):
            raise ValueError(f"{condition_id}: {key} must be positive")

    n_fluid = len(arrays["fluid_cell_volume_m3"])
    n_solid = len(arrays["solid_cell_volume_m3"])
    n_fluid_internal = len(arrays["fluid_internal_face_owner"])
    n_solid_internal = len(arrays["solid_internal_face_owner"])
    n_fluid_boundary = len(arrays["fluid_boundary_face_owner"])
    n_solid_boundary = len(arrays["solid_boundary_face_owner"])
    n_interface = len(arrays["interface_fluid_cell"])
    expected_shapes = {
        "fluid_cell_centroid_m": (n_fluid, 3),
        "fluid_internal_face_neighbour": (n_fluid_internal,),
        "solid_cell_centroid_m": (n_solid, 3),
        "solid_internal_face_neighbour": (n_solid_internal,),
        "fluid_boundary_face_patch": (n_fluid_boundary,),
        "fluid_boundary_face_centroid_m": (n_fluid_boundary, 3),
        "fluid_boundary_face_area_vector_outward_m2": (n_fluid_boundary, 3),
        "fluid_boundary_face_area_m2": (n_fluid_boundary,),
        "solid_boundary_face_patch": (n_solid_boundary,),
        "solid_boundary_face_centroid_m": (n_solid_boundary, 3),
        "solid_boundary_face_area_vector_outward_m2": (n_solid_boundary, 3),
        "solid_boundary_face_area_m2": (n_solid_boundary,),
        "interface_solid_cell": (n_interface,),
        "interface_face_centroid_m": (n_interface, 3),
        "interface_area_vector_fluid_outward_m2": (n_interface, 3),
        "interface_face_area_m2": (n_interface,),
        "fluid_velocity_m_s": (n_fluid, 3),
        "fluid_pressure_Pa": (n_fluid,),
        "fluid_temperature_K": (n_fluid,),
        "fluid_density_kg_m3": (n_fluid,),
        "fluid_internal_face_mass_flow_kg_s": (n_fluid_internal,),
        "fluid_boundary_velocity_m_s": (n_fluid_boundary, 3),
        "fluid_boundary_pressure_Pa": (n_fluid_boundary,),
        "fluid_boundary_temperature_K": (n_fluid_boundary,),
        "fluid_boundary_density_kg_m3": (n_fluid_boundary,),
        "fluid_boundary_face_mass_flow_kg_s": (n_fluid_boundary,),
        "solid_temperature_K": (n_solid,),
        "solid_boundary_temperature_K": (n_solid_boundary,),
    }
    for key in (
        "fluid_boundary_velocity_value_mask",
        "fluid_boundary_pressure_value_mask",
        "fluid_boundary_temperature_value_mask",
        "fluid_boundary_density_value_mask",
        "fluid_boundary_mass_flow_value_mask",
    ):
        expected_shapes[key] = (n_fluid_boundary,)
    expected_shapes["solid_boundary_temperature_value_mask"] = (n_solid_boundary,)
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(
                f"{condition_id}: {key} shape {arrays[key].shape} != {expected}"
            )

    index_limits = {
        "fluid_internal_face_owner": n_fluid,
        "fluid_internal_face_neighbour": n_fluid,
        "solid_internal_face_owner": n_solid,
        "solid_internal_face_neighbour": n_solid,
        "fluid_boundary_face_owner": n_fluid,
        "solid_boundary_face_owner": n_solid,
        "interface_fluid_cell": n_fluid,
        "interface_solid_cell": n_solid,
        "fluid_boundary_face_patch": len(patch_names["fluid"]),
        "solid_boundary_face_patch": len(patch_names["solid"]),
    }
    for key, upper in index_limits.items():
        value = arrays[key]
        if len(value) and (np.any(value < 0) or np.any(value >= upper)):
            raise ValueError(f"{condition_id}: {key} contains an out-of-range index")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-directory-name",
        default="training_sample_300",
        help="exact per-case sample directory, allowing old and revised samples to coexist",
    )
    parser.add_argument(
        "--sample-paths-from-completion-markers",
        action="store_true",
        help="read each case's schema-v3 sample path from formal_sample_complete.json",
    )
    parser.add_argument("--expected-case-count", type=int)
    parser.add_argument("--require-completion-markers", action="store_true")
    parser.add_argument(
        "--require-sourceflow-mapping",
        action="store_true",
        help="require the verified source-channel to pore-opening velocity mapping",
    )
    parser.add_argument(
        "--require-steady-final-window",
        action="store_true",
        help="require each completion marker's measured final 25 steady-iteration change summary",
    )
    args = parser.parse_args()
    if args.require_steady_final_window and not args.require_completion_markers:
        parser.error("--require-steady-final-window requires --require-completion-markers")

    matrix_root = args.matrix_root.resolve()
    output = args.output_dir.resolve()
    field_dir = output / "fields"
    field_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_paths_from_completion_markers:
        samples = []
        for marker_path in sorted(matrix_root.glob("*/formal_sample_complete.json")):
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            sample = Path(str(marker["training_sample"]))
            if not sample.is_absolute():
                sample = marker_path.parent / sample
            samples.append(sample.resolve())
    else:
        samples = sorted(
            matrix_root.glob(f"*/{args.sample_directory_name}/fields_and_topology.npz")
        )
    if not samples:
        raise FileNotFoundError(f"no completed training samples in {matrix_root}")
    if args.expected_case_count is not None and len(samples) != args.expected_case_count:
        raise ValueError(
            f"training sample count {len(samples)} != expected {args.expected_case_count}"
        )

    topology: dict[str, np.ndarray] | None = None
    patch_names: dict[str, list[str]] | None = None
    records: list[dict[str, object]] = []
    field_ranges = {
        key: {"minimum": float("inf"), "maximum": float("-inf")} for key in FIELD_KEYS
    }
    for sample in samples:
        case = sample.parents[1]
        metadata = json.loads((case / "cht_smoke_metadata.json").read_text(encoding="utf-8"))
        condition_id = metadata["operating_condition_id"]
        mapping_keys = (
            "pore_opening_boundary_velocity_m_s",
            "inlet_open_area_fraction",
            "source_channel_volume_flow_preserved",
        )
        missing_mapping = [key for key in mapping_keys if key not in metadata]
        if args.require_sourceflow_mapping and missing_mapping:
            raise ValueError(
                f"{condition_id}: source-flow mapping is missing {missing_mapping}"
            )
        if not missing_mapping:
            source_velocity = float(metadata["inlet_velocity_m_s"])
            pore_velocity = float(metadata["pore_opening_boundary_velocity_m_s"])
            open_fraction = float(metadata["inlet_open_area_fraction"])
            if not bool(metadata["source_channel_volume_flow_preserved"]):
                raise ValueError(f"{condition_id}: source-channel flow is not preserved")
            if not math.isclose(
                pore_velocity * open_fraction,
                source_velocity,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"{condition_id}: pore velocity and open fraction do not preserve source flow"
                )
        else:
            pore_velocity = None
            open_fraction = None
        sample_metadata_path = sample.parent / "metadata.json"
        sample_metadata = json.loads(sample_metadata_path.read_text(encoding="utf-8"))
        if sample_metadata.get("schema_version") != 3:
            raise ValueError(
                f"{condition_id}: training metadata schema_version is "
                f"{sample_metadata.get('schema_version')}, expected 3"
            )
        current_patch_names = {
            "fluid": list(sample_metadata["fluid_patch_names"]),
            "solid": list(sample_metadata["solid_patch_names"]),
        }
        if patch_names is None:
            patch_names = current_patch_names
        elif patch_names != current_patch_names:
            raise ValueError(f"boundary patch ordering differs in {condition_id}")
        with np.load(sample, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
            validate_sample_arrays(arrays, condition_id, current_patch_names)
            if topology is None:
                topology = {key: arrays[key] for key in TOPOLOGY_KEYS}
            else:
                for key in TOPOLOGY_KEYS:
                    if not np.array_equal(topology[key], arrays[key]):
                        raise ValueError(f"shared-mesh array {key} differs in {condition_id}")
            fields = {key: arrays[key] for key in FIELD_KEYS}
        for key, value in fields.items():
            if value.size == 0:
                continue
            field_ranges[key]["minimum"] = min(
                field_ranges[key]["minimum"], float(np.min(value))
            )
            field_ranges[key]["maximum"] = max(
                field_ranges[key]["maximum"], float(np.max(value))
            )

        steady_summary_path = None
        steady_full_field_available = None
        if args.require_completion_markers:
            marker_path = case / "formal_sample_complete.json"
            if not marker_path.is_file():
                raise ValueError(f"{condition_id}: formal completion marker is missing")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            sample_digest = sha256(sample)
            if marker.get("training_sample_schema_version") != 3:
                raise ValueError(f"{condition_id}: completion marker is not schema version 3")
            if marker.get("training_sample_sha256") != sample_digest:
                raise ValueError(f"{condition_id}: completion marker checksum differs")
            if args.require_steady_final_window:
                steady_summary_path, steady_document = verified_recorded_summary(case, marker)
                steady_full_field_available = bool(steady_document["full_field_available"])

        field_path = field_dir / f"{condition_id}.npz"
        np.savez_compressed(field_path, **fields)
        records.append(
            {
                "condition_id": condition_id,
                "inlet_velocity_m_s": metadata["inlet_velocity_m_s"],
                "source_inlet_channel_velocity_m_s": metadata["inlet_velocity_m_s"],
                "pore_opening_boundary_velocity_m_s": pore_velocity,
                "inlet_open_area_fraction": open_fraction,
                "inlet_temperature_K": metadata["inlet_temperature_K"],
                "solid_heat_source_W_m3": metadata["solid_heat_source_W_m3"],
                "outlet_pressure_Pa": metadata["outlet_pressure_Pa"],
                "cooling_wall_temperature_K": metadata["cooling_wall_temperature_K"],
                "field_file": str(field_path.relative_to(output)),
                "field_sha256": sha256(field_path),
                "source_sample": str(sample),
                "steady_final_window_summary": (
                    str(steady_summary_path) if steady_summary_path is not None else None
                ),
                "steady_final_window_full_field_available": steady_full_field_available,
            }
        )

    condition_ids = [str(record["condition_id"]) for record in records]
    if len(condition_ids) != len(set(condition_ids)):
        repeated = sorted(
            condition_id
            for condition_id in set(condition_ids)
            if condition_ids.count(condition_id) > 1
        )
        raise ValueError(f"duplicate operating-condition identifiers: {repeated}")

    assert topology is not None
    topology_path = output / "shared_mesh_topology.npz"
    np.savez_compressed(topology_path, **topology)
    serializable_ranges = {
        key: {
            bound: (None if not np.isfinite(value) else value)
            for bound, value in limits.items()
        }
        for key, limits in field_ranges.items()
    }
    payload = {
        "schema_version": 3,
        "status": "shared_mesh_multicondition_dataset",
        "case_count": len(records),
        "shared_topology_file": topology_path.name,
        "shared_topology_sha256": sha256(topology_path),
        "topology_shapes": {key: list(value.shape) for key, value in topology.items()},
        "field_keys": list(FIELD_KEYS),
        "field_ranges": serializable_ranges,
        "boundary_patch_names": patch_names,
        "conditions": records,
        "condition_coordinate_definitions": {
            "inlet_velocity_m_s": (
                "Published P418 full inlet-channel cross-section velocity used as the "
                "machine-learning operating-condition coordinate."
            ),
            "pore_opening_boundary_velocity_m_s": (
                "OpenFOAM velocity applied only on the resolved fluid opening."
            ),
            "velocity_mapping": (
                "u_pore * A_fluid = u_source * A_total; the open-area fraction is "
                "A_fluid/A_total and is computed from the fixed mesh."
            ),
        },
        "sourceflow_mapping_required": bool(args.require_sourceflow_mapping),
        "steady_final_window_required": bool(args.require_steady_final_window),
        "physical_use": (
            "Steady three-dimensional fluid-solid fields on one fixed pebble-bed mesh. "
            "Operating points are exact combinations from P418."
        ),
    }
    (output / "dataset_index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
