#!/usr/bin/env python3
"""Measure full-bed overlap, wall clearance and P390 crop geometry."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@lru_cache(maxsize=None)
def gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache quadrature nodes because crop/profile calculations reuse them."""
    return np.polynomial.legendre.leggauss(order)


def read_last_custom_dump(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="ascii").splitlines()
    starts = [i for i, line in enumerate(lines) if line == "ITEM: TIMESTEP"]
    if not starts:
        raise ValueError(f"no LAMMPS custom-dump snapshot in {path}")
    start = starts[-1]
    try:
        atoms_header = next(
            i for i in range(start, len(lines)) if lines[i].startswith("ITEM: ATOMS ")
        )
    except StopIteration as exc:
        raise ValueError(f"last snapshot has no atom table in {path}") from exc
    fields = lines[atoms_header].split()[2:]
    required = ["id", "x", "y", "z"]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"missing dump fields: {missing}")
    count_line = next(i for i in range(start, atoms_header) if lines[i] == "ITEM: NUMBER OF ATOMS")
    count = int(lines[count_line + 1])
    table = np.loadtxt(lines[atoms_header + 1 : atoms_header + 1 + count], ndmin=2)
    ids = table[:, fields.index("id")].astype(np.int64)
    centres = np.column_stack([table[:, fields.index(axis)] for axis in ("x", "y", "z")])
    order = np.argsort(ids)
    return ids[order], centres[order]


def sphere_box_intersection_volume(
    centre: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray, radius: float
) -> tuple[float, float]:
    """Return one sphere's box-clipped volume and its 128-to-256 point change."""
    lo = np.maximum(box_lo - centre, -radius)
    hi = np.minimum(box_hi - centre, radius)
    if np.any(lo >= hi):
        return 0.0, 0.0
    if np.all(lo <= -radius) and np.all(hi >= radius):
        return 4.0 * np.pi * radius**3 / 3.0, 0.0

    def integrate(order: int) -> float:
        nodes, weights = gauss_legendre(order)
        xmid = 0.5 * (lo[0] + hi[0])
        xhalf = 0.5 * (hi[0] - lo[0])
        total = 0.0
        for node, weight in zip(nodes, weights):
            x = xmid + xhalf * node
            rho2 = max(0.0, radius**2 - x**2)
            rho = np.sqrt(rho2)
            ylo = max(float(lo[1]), -rho)
            yhi = min(float(hi[1]), rho)
            if ylo >= yhi:
                continue
            ymid = 0.5 * (ylo + yhi)
            yhalf = 0.5 * (yhi - ylo)
            y = ymid + yhalf * nodes
            zlim = np.sqrt(np.maximum(0.0, rho2 - y**2))
            zlength = np.maximum(
                0.0, np.minimum(float(hi[2]), zlim) - np.maximum(float(lo[2]), -zlim)
            )
            total += weight * yhalf * float(np.dot(weights, zlength))
        return float(xhalf * total)

    volume_128 = integrate(128)
    volume_256 = integrate(256)
    return volume_256, abs(volume_256 - volume_128)


def crop_bounds(
    placement: str, large_box: np.ndarray, crop_box: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[str, str], tuple[str, str]]:
    """Return crop bounds plus physical-wall and symmetry-face names."""
    if not np.isclose(crop_box[2], large_box[2]):
        raise ValueError("the published P390 crop must retain the full flow direction")
    placements = {
        "xlo_ycentre": (
            (0.0, 0.5 * (large_box[1] - crop_box[1])),
            ("xlo",),
            ("xhi", "ylo", "yhi"),
        ),
        "xhi_ycentre": (
            (large_box[0] - crop_box[0], 0.5 * (large_box[1] - crop_box[1])),
            ("xhi",),
            ("xlo", "ylo", "yhi"),
        ),
        "xcentre_ylo": (
            (0.5 * (large_box[0] - crop_box[0]), 0.0),
            ("ylo",),
            ("yhi", "xlo", "xhi"),
        ),
        "xcentre_yhi": (
            (0.5 * (large_box[0] - crop_box[0]), large_box[1] - crop_box[1]),
            ("yhi",),
            ("ylo", "xlo", "xhi"),
        ),
        "xlo_ylo": ((0.0, 0.0), ("xlo", "ylo"), ("xhi", "yhi")),
        "xlo_yhi": ((0.0, large_box[1] - crop_box[1]), ("xlo", "yhi"), ("xhi", "ylo")),
        "xhi_ylo": ((large_box[0] - crop_box[0], 0.0), ("xhi", "ylo"), ("xlo", "yhi")),
        "xhi_yhi": (
            (large_box[0] - crop_box[0], large_box[1] - crop_box[1]),
            ("xhi", "yhi"),
            ("xlo", "ylo"),
        ),
        "centre": (
            (
                0.5 * (large_box[0] - crop_box[0]),
                0.5 * (large_box[1] - crop_box[1]),
            ),
            (),
            ("xlo", "xhi", "ylo", "yhi"),
        ),
    }
    if placement not in placements:
        raise ValueError(f"unknown crop placement {placement!r}")
    (xlo, ylo), physical_walls, symmetry_faces = placements[placement]
    lo = np.array([xlo, ylo, 0.0], dtype=float)
    hi = lo + crop_box
    return lo, hi, tuple(physical_walls), tuple(symmetry_faces)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--crop-placement",
        choices=(
            "manifest",
            "xlo_ycentre",
            "xhi_ycentre",
            "xcentre_ylo",
            "xcentre_yhi",
            "xlo_ylo",
            "xlo_yhi",
            "xhi_ylo",
            "xhi_yhi",
            "centre",
        ),
        default="manifest",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ids, centres = read_last_custom_dump(args.dump)
    expected = int(manifest["particle_count_large_box"])
    if len(ids) != expected:
        raise ValueError(f"particle count {len(ids)} does not match manifest {expected}")

    box = np.asarray(manifest["large_box_dp"], dtype=float)
    crop_box = np.asarray(manifest["crop_box_dp"], dtype=float)
    placement = args.crop_placement
    if placement == "manifest":
        crop_lo = np.asarray(manifest["crop_bounds_dp"]["lo"], dtype=float)
        crop_hi = np.asarray(manifest["crop_bounds_dp"]["hi"], dtype=float)
        placement = manifest.get("crop_placement_id", "legacy_manifest")
        physical_walls = tuple(manifest.get("cooled_wall_faces", ()))
        symmetry_faces = tuple(manifest.get("symmetry_faces", ()))
    else:
        crop_lo, crop_hi, physical_walls, symmetry_faces = crop_bounds(
            placement, box, crop_box
        )
    packing_radius = 0.5
    meshed_diameter = float(manifest["meshed_diameter_dp"])
    mesh_radius = meshed_diameter / 2.0

    tree = cKDTree(centres)
    distances, neighbours = tree.query(centres, k=2)
    nearest = distances[:, 1]
    min_index = int(np.argmin(nearest))
    min_partner = int(neighbours[min_index, 1])
    overlap_pairs = tree.query_pairs(r=1.0, output_type="ndarray")
    if overlap_pairs.size:
        pair_distances = np.linalg.norm(
            centres[overlap_pairs[:, 0]] - centres[overlap_pairs[:, 1]], axis=1
        )
        strict_overlap_pairs = overlap_pairs[pair_distances < 1.0]
    else:
        strict_overlap_pairs = np.empty((0, 2), dtype=np.int64)

    mesh_overlap_pairs = tree.query_pairs(r=meshed_diameter, output_type="ndarray")
    if mesh_overlap_pairs.size:
        mesh_pair_distances = np.linalg.norm(
            centres[mesh_overlap_pairs[:, 0]] - centres[mesh_overlap_pairs[:, 1]], axis=1
        )
        mesh_overlap_pairs = mesh_overlap_pairs[mesh_pair_distances < meshed_diameter]
    else:
        mesh_overlap_pairs = np.empty((0, 2), dtype=np.int64)

    packing_face_clearance = np.column_stack(
        [
            centres[:, 0] - packing_radius,
            box[0] - packing_radius - centres[:, 0],
            centres[:, 1] - packing_radius,
            box[1] - packing_radius - centres[:, 1],
            centres[:, 2] - packing_radius,
            box[2] - packing_radius - centres[:, 2],
        ]
    )
    mesh_face_clearance = np.column_stack(
        [
            centres[:, 0] - mesh_radius,
            box[0] - mesh_radius - centres[:, 0],
            centres[:, 1] - mesh_radius,
            box[1] - mesh_radius - centres[:, 1],
            centres[:, 2] - mesh_radius,
            box[2] - mesh_radius - centres[:, 2],
        ]
    )
    face_names = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")

    centre_inside = np.all((centres >= crop_lo) & (centres <= crop_hi), axis=1)
    intersects = np.all(
        (centres + packing_radius >= crop_lo) & (centres - packing_radius <= crop_hi), axis=1
    )
    fully_inside = np.all(
        (centres - packing_radius >= crop_lo) & (centres + packing_radius <= crop_hi), axis=1
    )
    clipped = intersects & ~fully_inside
    mesh_intersects = np.all(
        (centres + mesh_radius >= crop_lo) & (centres - mesh_radius <= crop_hi), axis=1
    )
    mesh_fully_inside = np.all(
        (centres - mesh_radius >= crop_lo) & (centres + mesh_radius <= crop_hi), axis=1
    )
    mesh_clipped = mesh_intersects & ~mesh_fully_inside
    crop_tree = cKDTree(centres[mesh_intersects])
    crop_mesh_overlap_pairs = crop_tree.query_pairs(r=meshed_diameter, output_type="ndarray")
    if crop_mesh_overlap_pairs.size:
        crop_centres = centres[mesh_intersects]
        crop_mesh_pair_distances = np.linalg.norm(
            crop_centres[crop_mesh_overlap_pairs[:, 0]]
            - crop_centres[crop_mesh_overlap_pairs[:, 1]],
            axis=1,
        )
        crop_mesh_overlap_pairs = crop_mesh_overlap_pairs[
            crop_mesh_pair_distances < meshed_diameter
        ]
    else:
        crop_mesh_overlap_pairs = np.empty((0, 2), dtype=np.int64)

    crop_solid_volume_dp3: float | None = None
    crop_porosity: float | None = None
    crop_volume_error_dp3: float | None = None
    if len(crop_mesh_overlap_pairs) == 0:
        full_sphere_volume = 4.0 * np.pi * mesh_radius**3 / 3.0
        crop_solid_volume_dp3 = float(np.count_nonzero(mesh_fully_inside)) * full_sphere_volume
        crop_volume_error_dp3 = 0.0
        for centre in centres[mesh_clipped]:
            clipped_volume, integration_error = sphere_box_intersection_volume(
                centre, crop_lo, crop_hi, mesh_radius
            )
            crop_solid_volume_dp3 += clipped_volume
            crop_volume_error_dp3 += integration_error
        crop_volume_dp3 = float(np.prod(crop_hi - crop_lo))
        crop_porosity = 1.0 - crop_solid_volume_dp3 / crop_volume_dp3

    result = {
        "status": "geometry_measured",
        "crop_placement_id": placement,
        "crop_bounds_dp": {"lo": crop_lo.tolist(), "hi": crop_hi.tolist()},
        "cooled_wall_faces": list(physical_walls),
        "symmetry_faces": list(symmetry_faces),
        "seed": manifest["seed"],
        "diameter_growth_stages": manifest["diameter_growth_stages"],
        "full_diameter_final_relaxation_blocks": manifest[
            "full_diameter_final_relaxation_blocks"
        ],
        "particle_count_large_box": len(ids),
        "minimum_centre_distance_dp": float(nearest[min_index]),
        "minimum_pair_ids": [int(ids[min_index]), int(ids[min_partner])],
        "strict_overlap_pair_count": int(len(strict_overlap_pairs)),
        "meshed_diameter_dp": meshed_diameter,
        "mesh_strict_overlap_pair_count_large_box": int(len(mesh_overlap_pairs)),
        "mesh_strict_overlap_pair_count_crop": int(len(crop_mesh_overlap_pairs)),
        "minimum_final_radius_wall_clearance_dp": float(np.min(packing_face_clearance)),
        "minimum_meshed_radius_wall_clearance_dp": float(np.min(mesh_face_clearance)),
        "meshed_wall_clearance_by_face_dp": {
            name: float(np.min(mesh_face_clearance[:, index]))
            for index, name in enumerate(face_names)
        },
        "minimum_crop_flow_face_clearance_meshed_dp": float(
            np.min(mesh_face_clearance[mesh_intersects][:, [4, 5]])
        ),
        "centre_inside_crop_count": int(np.count_nonzero(centre_inside)),
        "sphere_intersects_crop_count": int(np.count_nonzero(intersects)),
        "sphere_fully_inside_crop_count": int(np.count_nonzero(fully_inside)),
        "sphere_clipped_by_crop_boundary_count": int(np.count_nonzero(clipped)),
        "meshed_sphere_intersects_crop_count": int(np.count_nonzero(mesh_intersects)),
        "meshed_sphere_fully_inside_crop_count": int(np.count_nonzero(mesh_fully_inside)),
        "meshed_sphere_clipped_by_crop_boundary_count": int(np.count_nonzero(mesh_clipped)),
        "crop_solid_volume_dp3": crop_solid_volume_dp3,
        "crop_porosity_geometric": crop_porosity,
        "crop_volume_quadrature_128_to_256_change_dp3": crop_volume_error_dp3,
        "crop_porosity_status": (
            "computed by summing exact full-sphere volumes and adaptively integrated clipped-sphere volumes"
            if crop_porosity is not None
            else "not computed because meshed spheres overlap inside the crop"
        ),
        "crop_porosity_basis": "P404 one-percent diameter reduction before meshing",
        "physical_interpretation": (
            "Spheres intersecting a crop plane are retained for later geometric clipping. "
            "The P390 porosity uses sphere volume inside the crop rather than a centre-count approximation."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "large_and_crop_centres.npz",
        particle_id=ids,
        centre_dp=centres,
        centre_inside_crop=centre_inside,
        sphere_intersects_crop=intersects,
        sphere_fully_inside_crop=fully_inside,
        sphere_clipped_by_crop_boundary=clipped,
        meshed_sphere_intersects_crop=mesh_intersects,
        meshed_sphere_fully_inside_crop=mesh_fully_inside,
        meshed_sphere_clipped_by_crop_boundary=mesh_clipped,
        crop_lo_dp=crop_lo,
        crop_hi_dp=crop_hi,
    )
    (args.output_dir / "geometry_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
