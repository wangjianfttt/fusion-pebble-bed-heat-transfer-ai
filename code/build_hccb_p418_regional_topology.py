#!/usr/bin/env python3
"""Build a scalable fluid-solid regional hierarchy on the fixed P418 mesh.

The hierarchy follows native finite-volume connections.  Fluid and solid cells
are never mixed inside one region, and fluid-solid interface faces remain a
separate edge type.  No solution value is used while constructing the mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def unique_native_edges(native: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    source = native["edge_source_global"].astype(np.int64)
    target = native["edge_target_global"].astype(np.int64)
    keep = source < target
    if np.count_nonzero(keep) * 2 != len(source):
        raise ValueError("native graph must contain one reciprocal pair per face")
    return {
        "source": source[keep],
        "target": target[keep],
        "kind": native["edge_kind"][keep].astype(np.int8),
        "count": np.ones(np.count_nonzero(keep), dtype=np.int64),
        "area": native["edge_area_m2"][keep].astype(np.float64),
        "area_vector": native["edge_area_vector_m2"][keep].astype(np.float64),
        "centroid": native["edge_face_centroid_m"][keep].astype(np.float64),
    }


def csr_adjacency(
    node_count: int, source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    row = np.concatenate((source, target))
    column = np.concatenate((target, source))
    order = np.argsort(row, kind="stable")
    count = np.bincount(row, minlength=node_count)
    pointer = np.empty(node_count + 1, dtype=np.int64)
    pointer[0] = 0
    np.cumsum(count, out=pointer[1:])
    return pointer, column[order]


def connected_chunks(
    node_type: np.ndarray,
    edges: dict[str, np.ndarray],
    target_size: int,
) -> np.ndarray:
    if target_size < 2:
        raise ValueError("regional subsample factor must be at least two")
    same_type = edges["kind"] != 2
    source = edges["source"][same_type]
    target = edges["target"][same_type]
    if np.any(node_type[source] != node_type[target]):
        raise ValueError("a non-interface face crosses fluid and solid types")
    pointer, neighbours = csr_adjacency(len(node_type), source, target)
    parent = np.full(len(node_type), -1, dtype=np.int64)
    region = 0
    for region_type in (0, 1):
        candidates = np.flatnonzero(node_type == region_type)
        for start in candidates:
            start = int(start)
            if parent[start] >= 0:
                continue
            members = [start]
            parent[start] = region
            queue_index = 0
            while queue_index < len(members) and len(members) < target_size:
                node = members[queue_index]
                queue_index += 1
                for neighbour in neighbours[pointer[node] : pointer[node + 1]]:
                    neighbour = int(neighbour)
                    if parent[neighbour] >= 0 or node_type[neighbour] != region_type:
                        continue
                    parent[neighbour] = region
                    members.append(neighbour)
                    if len(members) == target_size:
                        break
            region += 1
    if np.any(parent < 0):
        raise RuntimeError("at least one cell was not assigned to a region")
    return parent


def aggregate_nodes(
    centroid: np.ndarray,
    volume: np.ndarray,
    node_type: np.ndarray,
    parent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    region_count = int(parent.max()) + 1
    region_volume = np.bincount(parent, weights=volume, minlength=region_count)
    if np.any(region_volume <= 0.0):
        raise RuntimeError("regional volume must be positive")
    moment = np.column_stack(
        [np.bincount(parent, weights=volume * centroid[:, axis], minlength=region_count)
         for axis in range(3)]
    )
    region_type = np.full(region_count, -1, dtype=np.int8)
    region_type[parent[node_type == 0]] = 0
    solid_region = np.unique(parent[node_type == 1])
    if np.any(region_type[solid_region] == 0):
        raise RuntimeError("one region contains both fluid and solid cells")
    region_type[solid_region] = 1
    if np.any(region_type < 0):
        raise RuntimeError("regional type is incomplete")
    member_count = np.bincount(parent, minlength=region_count).astype(np.int64)
    return moment / region_volume[:, None], region_volume, region_type, member_count


def aggregate_edges(
    edges: dict[str, np.ndarray], parent: np.ndarray
) -> dict[str, np.ndarray]:
    mapped_source = parent[edges["source"]]
    mapped_target = parent[edges["target"]]
    keep = mapped_source != mapped_target
    left = mapped_source[keep]
    right = mapped_target[keep]
    source = np.minimum(left, right)
    target = np.maximum(left, right)
    orientation = np.where(left <= right, 1.0, -1.0)
    kind = edges["kind"][keep]
    area = edges["area"][keep]
    area_vector = edges["area_vector"][keep] * orientation[:, None]
    centroid = edges["centroid"][keep]
    fine_count = edges["count"][keep]

    order = np.lexsort((target, source, kind))
    source, target, kind = source[order], target[order], kind[order]
    area, area_vector, centroid = area[order], area_vector[order], centroid[order]
    fine_count = fine_count[order]
    start = np.r_[0, 1 + np.flatnonzero(
        (source[1:] != source[:-1])
        | (target[1:] != target[:-1])
        | (kind[1:] != kind[:-1])
    )]
    total_area = np.add.reduceat(area, start)
    weighted_centroid = np.column_stack(
        [np.add.reduceat(centroid[:, axis] * area, start) for axis in range(3)]
    ) / total_area[:, None]
    return {
        "source": source[start],
        "target": target[start],
        "kind": kind[start],
        "count": np.add.reduceat(fine_count, start),
        "area": total_area,
        "area_vector": np.column_stack(
            [np.add.reduceat(area_vector[:, axis], start) for axis in range(3)]
        ),
        "centroid": weighted_centroid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-topology", type=Path, required=True)
    parser.add_argument("--native-graph", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--subsample-factor", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.levels < 1:
        raise ValueError("at least one regional level is required")

    topology_path = args.shared_topology.resolve()
    native_path = args.native_graph.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    topology = load_npz(topology_path)
    native = load_npz(native_path)

    centroid = np.concatenate(
        (topology["fluid_cell_centroid_m"], topology["solid_cell_centroid_m"]), axis=0
    ).astype(np.float64)
    volume = np.concatenate(
        (topology["fluid_cell_volume_m3"], topology["solid_cell_volume_m3"])
    ).astype(np.float64)
    node_type = native["node_region_type"].astype(np.int8)
    if len(node_type) != len(centroid):
        raise ValueError("shared topology and native graph have different cell counts")
    expected_type = np.concatenate(
        (
            np.zeros(len(topology["fluid_cell_volume_m3"]), dtype=np.int8),
            np.ones(len(topology["solid_cell_volume_m3"]), dtype=np.int8),
        )
    )
    if not np.array_equal(node_type, expected_type):
        raise ValueError("shared topology and native graph disagree on cell types")
    if np.any(volume <= 0.0) or not np.all(np.isfinite(centroid)):
        raise ValueError("cell geometry is non-finite or has non-positive volume")

    arrays: dict[str, np.ndarray] = {
        "fine_node_centroid_m": centroid,
        "fine_node_volume_m3": volume,
        "fine_node_type": node_type,
    }
    edges = unique_native_edges(native)
    initial_volume = {
        int(kind): float(volume[node_type == kind].sum()) for kind in (0, 1)
    }
    level_records: list[dict[str, object]] = []
    current_centroid, current_volume, current_type = centroid, volume, node_type
    for level in range(args.levels):
        parent = connected_chunks(current_type, edges, args.subsample_factor)
        region_centroid, region_volume, region_type, member_count = aggregate_nodes(
            current_centroid, current_volume, current_type, parent
        )
        regional_edges = aggregate_edges(edges, parent)
        prefix = f"level_{level}"
        arrays[f"{prefix}_parent_from_finer"] = parent
        arrays[f"{prefix}_centroid_m"] = region_centroid
        arrays[f"{prefix}_volume_m3"] = region_volume
        arrays[f"{prefix}_node_type"] = region_type
        arrays[f"{prefix}_member_count"] = member_count
        arrays[f"{prefix}_edge_source"] = np.concatenate(
            (regional_edges["source"], regional_edges["target"])
        )
        arrays[f"{prefix}_edge_target"] = np.concatenate(
            (regional_edges["target"], regional_edges["source"])
        )
        arrays[f"{prefix}_edge_kind"] = np.concatenate(
            (regional_edges["kind"], regional_edges["kind"])
        )
        arrays[f"{prefix}_edge_finer_face_count"] = np.concatenate(
            (regional_edges["count"], regional_edges["count"])
        )
        arrays[f"{prefix}_edge_area_m2"] = np.concatenate(
            (regional_edges["area"], regional_edges["area"])
        )
        arrays[f"{prefix}_edge_area_vector_m2"] = np.concatenate(
            (regional_edges["area_vector"], -regional_edges["area_vector"]), axis=0
        )
        arrays[f"{prefix}_edge_centroid_m"] = np.concatenate(
            (regional_edges["centroid"], regional_edges["centroid"]), axis=0
        )
        by_type = {
            int(kind): float(region_volume[region_type == kind].sum())
            for kind in (0, 1)
        }
        level_records.append(
            {
                "level": level,
                "finer_nodes": int(len(current_type)),
                "regional_nodes": int(len(region_type)),
                "fluid_regions": int(np.count_nonzero(region_type == 0)),
                "solid_regions": int(np.count_nonzero(region_type == 1)),
                "directed_regional_edges": int(2 * len(regional_edges["source"])),
                "largest_membership": int(member_count.max()),
                "fluid_volume_relative_difference": abs(by_type[0] - initial_volume[0]) / initial_volume[0],
                "solid_volume_relative_difference": abs(by_type[1] - initial_volume[1]) / initial_volume[1],
            }
        )
        current_centroid, current_volume, current_type = (
            region_centroid,
            region_volume,
            region_type,
        )
        edges = regional_edges
        if len(region_type) < 4:
            break

    hierarchy_path = output / "regional_topology.npz"
    np.savez_compressed(hierarchy_path, **arrays)
    checks = {
        "fluid_and_solid_are_separate": all(
            np.all(
                arrays[f"level_{level}_node_type"]
                [arrays[f"level_{level}_parent_from_finer"]]
                == (node_type if level == 0 else arrays[f"level_{level - 1}_node_type"])
            )
            for level in range(len(level_records))
        ),
        "regional_memberships_do_not_exceed_subsample_factor": all(
            item["largest_membership"] <= args.subsample_factor for item in level_records
        ),
        "fluid_volume_is_conserved": all(
            item["fluid_volume_relative_difference"] < 1.0e-12 for item in level_records
        ),
        "solid_volume_is_conserved": all(
            item["solid_volume_relative_difference"] < 1.0e-12 for item in level_records
        ),
        "hierarchy_has_multiple_levels": len(level_records) >= min(2, args.levels),
        "only_native_finite_volume_faces_are_used": True,
        "solution_fields_are_not_used_to_build_regions": True,
    }
    summary = {
        "status": "p418_multiregion_regional_topology_ready" if all(checks.values()) else "failed",
        "algorithm": {
            "method": "deterministic connected breadth-first regional aggregation",
            "subsample_factor": args.subsample_factor,
            "requested_levels": args.levels,
            "active_levels": len(level_records),
            "physical_parameter_added": False,
            "architecture_reference": "RIGNO-style recursive regional mesh; NeurIPS 2025",
        },
        "levels": level_records,
        "checks": checks,
        "source": {
            "shared_topology": str(topology_path),
            "shared_topology_sha256": sha256(topology_path),
            "native_graph": str(native_path),
            "native_graph_sha256": sha256(native_path),
        },
        "regional_topology_file": hierarchy_path.name,
        "regional_topology_sha256": sha256(hierarchy_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
