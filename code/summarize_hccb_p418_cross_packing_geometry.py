#!/usr/bin/env python3
"""Summarize geometric differences among the three P418 packing realizations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKING_SET = (
    ROOT / "data/apd006_hccb_source_sequence_target_packings/packing_set_summary.json"
)
DEFAULT_PLAN = ROOT / "parameters/hccb_p418_cross_packing_plan.json"
DEFAULT_LOCAL_MANIFESTS = {
    101: ROOT / "runs/hccb_dense_snappy_g2_nativezone_r2/case_manifest.json",
    202: ROOT
    / "results/hccb_p418_cross_packing_surface_preflight/seed202/case_manifest.json",
    303: ROOT
    / "results/hccb_p418_cross_packing_surface_preflight/seed303/case_manifest.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    result = np.quantile(values, [0.05, 0.5, 0.95])
    return tuple(float(value) for value in result)


def load_physical_directions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["condition_id"]: row["cooling_wall_heat_direction"]
            for row in csv.DictReader(handle)
        }


def summarize_packings(
    root: Path,
    packing_set_path: Path,
    local_manifests: dict[int, Path],
    plan_path: Path,
    physical_csv: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records = json.loads(packing_set_path.read_text(encoding="utf-8"))
    if [int(record["seed"]) for record in records] != [101, 202, 303]:
        raise ValueError("packing set must contain seeds 101, 202 and 303 in order")

    rows: list[dict[str, object]] = []
    for record in records:
        seed = int(record["seed"])
        if not all(record.get("checks", {}).values()):
            raise ValueError(f"packing seed {seed} has a failed geometry check")
        relative_packing = (
            Path("data/apd006_hccb_source_sequence_target_packings")
            / f"seed{seed}_s80_xlo_ycentre/packing.npz"
        )
        packing_path = root / relative_packing
        if sha256(packing_path) != record["packing_npz_sha256"]:
            raise ValueError(f"packing checksum differs for seed {seed}")
        with np.load(packing_path) as data:
            centres = np.asarray(data["centres_m"], dtype=float)
            physical_radius = float(np.asarray(data["physical_radius_m"]))
            meshing_radius = float(np.asarray(data["meshing_radius_m"]))
            particle_diameter = 2.0 * physical_radius
        nearest = cKDTree(centres).query(centres, k=2)[0][:, 1] / particle_diameter
        cooled_wall_clearance = (centres[:, 0] - physical_radius) / particle_diameter
        nn_q05, nn_median, nn_q95 = quantiles(nearest)
        wall_q05, wall_median, wall_q95 = quantiles(cooled_wall_clearance)

        manifest_path = local_manifests[seed]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_packing_sha256"] != record["packing_npz_sha256"]:
            raise ValueError(f"local manifest uses another packing for seed {seed}")
        if manifest.get("new_physical_parameters"):
            raise ValueError(f"local manifest adds physical parameters for seed {seed}")
        crop_lower = np.asarray(manifest["crop_lower_m"], dtype=float)
        crop_upper = np.asarray(manifest["crop_upper_m"], dtype=float)
        intersects_local_crop = np.all(
            (centres + meshing_radius >= crop_lower)
            & (centres - meshing_radius <= crop_upper),
            axis=1,
        )
        local_centre_x_dp = (
            centres[intersects_local_crop, 0] - crop_lower[0]
        ) / particle_diameter
        first_diameter_count = int(np.count_nonzero(local_centre_x_dp <= 1.0))
        intersecting_count = int(manifest["intersecting_particle_count"])
        if int(np.count_nonzero(intersects_local_crop)) != intersecting_count:
            raise ValueError(f"local intersecting count differs for seed {seed}")
        rows.append(
            {
                "seed": seed,
                "full_crop_particle_count": int(record["particle_count"]),
                "full_crop_geometric_porosity": float(
                    record["crop_porosity_geometric"]
                ),
                "nearest_neighbour_q05_dp": nn_q05,
                "nearest_neighbour_median_dp": nn_median,
                "nearest_neighbour_q95_dp": nn_q95,
                "cooled_wall_clearance_q05_dp": wall_q05,
                "cooled_wall_clearance_median_dp": wall_median,
                "cooled_wall_clearance_q95_dp": wall_q95,
                "local_intersecting_particle_count": intersecting_count,
                "local_first_diameter_particle_count": first_diameter_count,
                "local_first_diameter_particle_fraction": (
                    first_diameter_count / intersecting_count
                ),
                "local_retained_fragment_count": int(
                    manifest["retained_particle_fragment_count"]
                ),
                "local_triangulated_porosity": float(
                    manifest["triangulated_porosity"]
                ),
                "local_omitted_solid_volume_m3": float(
                    manifest["omitted_solid_volume_m3"]
                ),
                "local_porosity_change_from_omission": float(
                    manifest["porosity_change_from_omission"]
                ),
                "packing_sha256": record["packing_npz_sha256"],
                "local_manifest": str(manifest_path.resolve()),
            }
        )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    screening_ids = [
        str(condition["condition_id"])
        for condition in plan["screening_design"]["conditions"]
    ]
    if len(screening_ids) != 9 or len(set(screening_ids)) != 9:
        raise ValueError("cross-packing plan must contain nine unique conditions")
    directions = load_physical_directions(physical_csv)
    known_directions = {
        identifier: directions[identifier]
        for identifier in screening_ids
        if identifier in directions
    }
    unknown_ids = [identifier for identifier in screening_ids if identifier not in directions]
    full_porosity = [float(row["full_crop_geometric_porosity"]) for row in rows]
    local_porosity = [float(row["local_triangulated_porosity"]) for row in rows]
    fragments = [int(row["local_retained_fragment_count"]) for row in rows]
    summary: dict[str, object] = {
        "status": "hccb_p418_cross_packing_geometry_summarized",
        "seeds": [101, 202, 303],
        "full_crop_geometric_porosity_range": [min(full_porosity), max(full_porosity)],
        "full_crop_geometric_porosity_span": max(full_porosity) - min(full_porosity),
        "local_triangulated_porosity_range": [min(local_porosity), max(local_porosity)],
        "local_triangulated_porosity_span": max(local_porosity) - min(local_porosity),
        "local_retained_fragment_count_range": [min(fragments), max(fragments)],
        "screening_condition_count": len(screening_ids),
        "screening_known_physical_direction_count": len(known_directions),
        "screening_wall_to_fluid_count": sum(
            value == "wall_to_fluid" for value in known_directions.values()
        ),
        "screening_fluid_to_wall_count": sum(
            value == "fluid_to_wall" for value in known_directions.values()
        ),
        "screening_unknown_condition_ids": unknown_ids,
        "new_physical_parameters": [],
        "interpretation": (
            "The three packings share the same literature-defined particle size, box "
            "and boundary families but differ in particle arrangement. Geometry "
            "statistics do not substitute for the later flow and heat-transfer results."
        ),
    }
    return rows, summary


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "packing_geometry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P418三套颗粒排列的几何差异",
        "",
        "三套球床使用相同的颗粒直径、计算域和边界类型，只改变随机装填。",
        "",
        "| 装填 | 完整裁剪区颗粒数 | 完整裁剪区孔隙率 | 精细局部域颗粒片段数 | 精细局部域孔隙率 | 最近邻中位数 ($d_p$) | 局部冷却壁1个粒径内颗粒数 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {count} | {porosity:.5f} | {fragments} | {local:.5f} | {nn:.4f} | {wall} |".format(
                seed=row["seed"],
                count=row["full_crop_particle_count"],
                porosity=row["full_crop_geometric_porosity"],
                fragments=row["local_retained_fragment_count"],
                local=row["local_triangulated_porosity"],
                nn=row["nearest_neighbour_median_dp"],
                wall=row["local_first_diameter_particle_count"],
            )
        )
    lines.extend(
        [
            "",
            "完整裁剪区孔隙率跨度为"
            f"`{100.0 * float(summary['full_crop_geometric_porosity_span']):.3f}`个百分点；"
            "进入当前精细局部域后，孔隙率跨度为"
            f"`{100.0 * float(summary['local_triangulated_porosity_span']):.3f}`个百分点。",
            "这说明三套装填的总体固相率接近，但局部流道和壁面邻近结构并不相同。"
            "是否导致压降、热点和壁面热量变化，必须由后续OpenFOAM结果判断。",
            "",
            "跨装填计算选取P418范围的8个角点和1个内部工况。当前seed101中已有"
            f"`{summary['screening_known_physical_direction_count']}/9`个完成，"
            f"其中壁面向流体`{summary['screening_wall_to_fluid_count']}`个、"
            f"流体向壁面`{summary['screening_fluid_to_wall_count']}`个。",
            "",
        ]
    )
    (output_dir / "P418_三套颗粒排列几何差异_CN.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def render_latex_table(rows: list[dict[str, object]]) -> str:
    """Render a compact manuscript table proving that the three packings differ."""
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Geometric comparison of the three independently generated pebble arrangements. All arrangements use the same literature-defined particle diameter, source packing procedure, local crop and boundary orientation. $N_{\mathrm{full}}$ and $\phi_{\mathrm{full}}$ denote the particle count and geometric porosity of the full crop; $N_{\mathrm{local}}$ and $\phi_{\mathrm{local}}$ denote retained fragments and triangulated porosity in the local domain. $N_{\mathrm{wall}}$ is the number of local particle centres within one particle diameter of the cooling wall.}",
        r"\label{tab:cross_packing_geometry}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Seed & $N_{\mathrm{full}}$ & $\phi_{\mathrm{full}}$ & $N_{\mathrm{local}}$ & $\phi_{\mathrm{local}}$ & med. $d_{nn}/d_p$ & $N_{\mathrm{wall}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {:.5f} & {} & {:.5f} & {:.4f} & {} \\\\".format(
                int(row["seed"]),
                int(row["full_crop_particle_count"]),
                float(row["full_crop_geometric_porosity"]),
                int(row["local_retained_fragment_count"]),
                float(row["local_triangulated_porosity"]),
                float(row["nearest_neighbour_median_dp"]),
                int(row["local_first_diameter_particle_count"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--packing-set", type=Path, default=DEFAULT_PACKING_SET)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--physical-csv", type=Path)
    parser.add_argument("--manifest-seed101", type=Path)
    parser.add_argument("--manifest-seed202", type=Path)
    parser.add_argument("--manifest-seed303", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    local_manifests = {
        101: (
            args.manifest_seed101.resolve()
            if args.manifest_seed101
            else DEFAULT_LOCAL_MANIFESTS[101].resolve()
        ),
        202: (
            args.manifest_seed202.resolve()
            if args.manifest_seed202
            else DEFAULT_LOCAL_MANIFESTS[202].resolve()
        ),
        303: (
            args.manifest_seed303.resolve()
            if args.manifest_seed303
            else DEFAULT_LOCAL_MANIFESTS[303].resolve()
        ),
    }
    rows, summary = summarize_packings(
        root,
        args.packing_set.resolve(),
        local_manifests,
        args.plan.resolve(),
        args.physical_csv.resolve() if args.physical_csv else None,
    )
    write_outputs(args.output_dir.resolve(), rows, summary)
    if args.tex_output is not None:
        tex_output = args.tex_output.resolve()
        tex_output.parent.mkdir(parents=True, exist_ok=True)
        tex_output.write_text(render_latex_table(rows), encoding="utf-8")
    print(args.output_dir.resolve() / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
