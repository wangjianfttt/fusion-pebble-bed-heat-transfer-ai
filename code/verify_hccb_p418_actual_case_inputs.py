#!/usr/bin/env python3
"""Verify that every formal P418 case uses the registered literature inputs.

The check reads both ``cht_smoke_metadata.json`` and the OpenFOAM dictionaries.
Numerical controls such as end time, write interval and solver correctors are
reported separately; they are not treated as pebble-bed physical parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from build_hccb_gmsh_cht_smoke_case import (
    matrix_condition_id,
    parse_p418_matrix,
)
from build_hccb_pore_resolved_openfoam_steady_case import p406_cp
from hccb_p418_source_contract import CASE_PHYSICS_PARAMETER_IDS
from hccb_p418_source_contract import MESH_GEOMETRY_SOURCE_PARAMETER_IDS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "hccb_dense_cht_p418_60_sourceflow_r3"
DEFAULT_SOURCES = ROOT / "parameters/hccb_p418_physical_parameter_sources.csv"
DEFAULT_CANONICAL_HELIUM = (
    ROOT / "results/apd006_hccb_openfoam_helium_property_table/physicalProperties"
)
DEFAULT_PACKING_INPUT = (
    ROOT
    / "results/apd006_hccb_source_sequence_lammps/sweep/seed101_s80/packing_input_manifest.json"
)
DEFAULT_PACKING_SUMMARY = (
    ROOT
    / "data/apd006_hccb_source_sequence_target_packings/seed101_s80_xlo_ycentre/summary.json"
)
DEFAULT_MESH_MANIFEST = next(
    (
        path
        for path in (
            ROOT / "runs/hccb_dense_snappy_g2_nativezone_r2/case_manifest.json",
            ROOT / "hccb_dense_snappy_g2_nativezone_r2/case_manifest.json",
        )
        if path.is_file()
    ),
    ROOT / "runs/hccb_dense_snappy_g2_nativezone_r2/case_manifest.json",
)


def source_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {row["parameter_id"]: row for row in csv.DictReader(stream)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_from_foam_value(value: str) -> float:
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
    if len(numbers) != 1:
        raise ValueError(f"expected one scalar in OpenFOAM value, found {value!r}")
    return float(numbers[0])


def vector_from_foam_value(value: str) -> tuple[float, float, float]:
    match = re.search(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
        value,
    )
    if not match:
        raise ValueError(f"expected one three-component vector, found {value!r}")
    return tuple(float(item) for item in match.groups())  # type: ignore[return-value]


def foam_reader(openfoam_bashrc: Path) -> Callable[[Path, str], str]:
    executable = shutil.which("foamDictionary")
    if executable is None and not openfoam_bashrc.is_file():
        raise FileNotFoundError(openfoam_bashrc)

    def read(path: Path, entry: str) -> str:
        if executable is not None:
            command = [
                executable,
                str(path),
                "-writePrecision",
                "14",
                "-entry",
                entry,
                "-value",
            ]
        else:
            shell_command = (
                'source "$1" >/dev/null 2>&1; '
                'foamDictionary "$2" -writePrecision 14 -entry "$3" -value'
            )
            command = [
                "bash",
                "-c",
                shell_command,
                "_",
                str(openfoam_bashrc),
                str(path),
                entry,
            ]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        return completed.stdout.strip()

    return read


def published_conditions(p418_value: str) -> dict[str, tuple[float, float, float]]:
    velocities, temperatures, sources = parse_p418_matrix(p418_value)
    return {
        matrix_condition_id(velocity, temperature, source): (
            velocity,
            temperature,
            source,
        )
        for velocity in velocities
        for temperature in temperatures
        for source in sources
    }


def dp_dimensions(value: str) -> tuple[float, float, float]:
    """Read the first three ``dp`` dimensions from a literature value."""
    match = re.search(
        r"([-+0-9.]+)\s*d?p\s*x\s*([-+0-9.]+)\s*d?p\s*x\s*([-+0-9.]+)\s*d?p",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"cannot parse three dp dimensions from {value!r}")
    return tuple(float(item) for item in match.groups())  # type: ignore[return-value]


def close(actual: float, expected: float, name: str, *, rtol: float = 1.0e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=max(1.0e-12, abs(expected) * rtol)):
        raise ValueError(f"{name}: actual {actual:.12g} != expected {expected:.12g}")


def verify_geometry_sources(
    rows: dict[str, dict[str, str]],
    packing_input_path: Path,
    packing_summary_path: Path,
    mesh_manifest_path: Path,
) -> dict[str, object]:
    packing_input = json.loads(packing_input_path.read_text(encoding="utf-8"))
    packing_summary = json.loads(packing_summary_path.read_text(encoding="utf-8"))
    mesh = json.loads(mesh_manifest_path.read_text(encoding="utf-8"))

    physical_diameter_m = float(rows["P048"]["采用值或关系式"]) * 1.0e-3
    target_porosity = float(rows["P049"]["采用值或关系式"])
    large_box_dp = dp_dimensions(rows["P050"]["采用值或关系式"])
    published_crop_dp = dp_dimensions(rows["P390"]["采用值或关系式"])
    diameter_reduction = float(rows["P404"]["采用值或关系式"]) / 100.0
    expected_mesh_diameter_m = physical_diameter_m * (1.0 - diameter_reduction)

    close(float(packing_input["particle_diameter_m"]), physical_diameter_m, "packing P048")
    close(float(packing_input["porosity"]), target_porosity, "packing P049")
    for index, expected in enumerate(large_box_dp):
        close(float(packing_input["large_box_dp"][index]), expected, f"packing P050 axis {index}")
    for index, expected in enumerate(published_crop_dp):
        close(float(packing_input["crop_box_dp"][index]), expected, f"packing P390 axis {index}")
        close(float(packing_summary["box_lengths_m"][index]), expected * physical_diameter_m, f"exported P390 axis {index}")
    close(
        float(packing_input["mesh_diameter_reduction_fraction"]),
        diameter_reduction,
        "packing P404 reduction",
    )
    close(
        float(packing_summary["meshing_particle_diameter_m"]),
        expected_mesh_diameter_m,
        "exported P404 diameter",
    )
    close(
        float(mesh["physical_particle_diameter_m"]),
        physical_diameter_m,
        "fine mesh physical diameter",
    )
    close(
        float(mesh["meshing_particle_diameter_m"]),
        expected_mesh_diameter_m,
        "fine mesh P404 diameter",
    )

    expected_packing_ids = set(MESH_GEOMETRY_SOURCE_PARAMETER_IDS).difference({"P423"})
    if not expected_packing_ids.issubset(set(packing_input["physical_parameter_ids"])):
        raise ValueError("packing input does not contain the P048/P049/P050/P390/P404 source rows")
    if packing_summary["crop_placement_id"] != "xlo_ycentre":
        raise ValueError("P423 wall-adjacent, laterally centred crop is not active")
    if packing_summary["cooled_wall_face"] != "xlo":
        raise ValueError("P423 cooled-wall face is not xlo")
    if set(packing_summary["symmetry_faces"]) != {"xhi", "ylo", "yhi"}:
        raise ValueError("P423 transverse symmetry faces differ")
    if mesh["source_packing_sha256"] != packing_summary["packing_npz_sha256"]:
        raise ValueError("fine mesh source packing differs from the registered seed101 packing")

    fine_bounds = [float(value) for value in mesh["crop_box_dp"]]
    fine_lengths = [
        fine_bounds[1] - fine_bounds[0],
        fine_bounds[3] - fine_bounds[2],
        fine_bounds[5] - fine_bounds[4],
    ]
    return {
        "source_parameter_ids": list(MESH_GEOMETRY_SOURCE_PARAMETER_IDS),
        "physical_particle_diameter_m": physical_diameter_m,
        "target_packing_porosity": target_porosity,
        "published_large_box_dp": list(large_box_dp),
        "published_crop_box_dp": list(published_crop_dp),
        "meshing_particle_diameter_m": expected_mesh_diameter_m,
        "published_crop_placement": packing_summary["crop_placement_id"],
        "fine_local_crop_bounds_dp": fine_bounds,
        "fine_local_crop_lengths_dp": fine_lengths,
        "fine_local_retained_particle_fragments": mesh["retained_particle_fragment_count"],
        "fine_local_triangulated_porosity": mesh["triangulated_porosity"],
        "source_packing_sha256": mesh["source_packing_sha256"],
        "all_published_geometry_and_meshing_inputs_match": True,
        "fine_local_crop_is_a_computed_geometry_result": True,
    }


def verify_case(
    case: Path,
    expected: tuple[float, float, float],
    rows: dict[str, dict[str, str]],
    read_foam: Callable[[Path, str], str],
    canonical_helium_hash: str,
) -> dict[str, object]:
    velocity, temperature, source_mw_m3 = expected
    metadata_path = case / "cht_smoke_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("operating_condition_id") != case.name:
        raise ValueError(f"{case.name}: metadata condition id differs")
    if tuple(metadata.get("parameter_ids", ())) != CASE_PHYSICS_PARAMETER_IDS:
        raise ValueError(f"{case.name}: physical parameter list differs")
    if metadata.get("new_fitted_physical_parameters") != []:
        raise ValueError(f"{case.name}: unexpected fitted physical parameters")

    pressure_pa = float(rows["P426"]["采用值或关系式"]) * 1.0e6
    wall_temperature = float(rows["P425"]["采用值或关系式"])
    solid_conductivity = float(rows["P092"]["采用值或关系式"])
    solid_density = float(rows["P403"]["采用值或关系式"])
    helium_cp = float(rows["P388"]["采用值或关系式"])
    steady_solid_cv = p406_cp(rows["P406"]["采用值或关系式"], temperature)
    source_w_m3 = source_mw_m3 * 1.0e6

    metadata_expected = {
        "inlet_velocity_m_s": velocity,
        "inlet_temperature_K": temperature,
        "solid_heat_source_W_m3": source_w_m3,
        "outlet_pressure_Pa": pressure_pa,
        "cooling_wall_temperature_K": wall_temperature,
        "solid_thermal_conductivity_W_m_K": solid_conductivity,
        "solid_density_kg_m3": solid_density,
        "solid_Cv_J_kg_K": steady_solid_cv,
        "helium_specific_heat_J_kg_K": helium_cp,
    }
    for name, expected_value in metadata_expected.items():
        close(float(metadata[name]), expected_value, f"{case.name} metadata {name}")
    if metadata.get("source_channel_volume_flow_preserved") is not True:
        raise ValueError(f"{case.name}: source inlet-channel volume flow is not preserved")
    open_fraction = float(metadata["inlet_open_area_fraction"])
    pore_velocity = float(metadata["pore_opening_boundary_velocity_m_s"])
    close(
        pore_velocity * open_fraction,
        velocity,
        f"{case.name} source-channel/pore-opening velocity mapping",
    )

    u_in = vector_from_foam_value(
        read_foam(case / "0/fluid/U", "boundaryField/inlet/value")
    )
    close(u_in[0], 0.0, f"{case.name} inlet Ux")
    close(u_in[1], 0.0, f"{case.name} inlet Uy")
    close(u_in[2], pore_velocity, f"{case.name} inlet Uz")
    dictionary_scalars = {
        "fluid inlet temperature": (
            case / "0/fluid/T",
            "boundaryField/inlet/value",
            temperature,
        ),
        "fluid cooling-wall temperature": (
            case / "0/fluid/T",
            "boundaryField/coolingWall/value",
            wall_temperature,
        ),
        "solid cooling-wall temperature": (
            case / "0/solid/T",
            "boundaryField/coolingWall/value",
            wall_temperature,
        ),
        "absolute pressure": (case / "0/fluid/p", "internalField", pressure_pa),
        "solid density": (
            case / "constant/solid/physicalProperties",
            "mixture/equationOfState/rho",
            solid_density,
        ),
        "solid conductivity": (
            case / "constant/solid/physicalProperties",
            "mixture/transport/kappa",
            solid_conductivity,
        ),
        "steady solid heat capacity": (
            case / "constant/solid/physicalProperties",
            "mixture/thermodynamics/Cv",
            steady_solid_cv,
        ),
        "volumetric heat source": (
            case / "constant/solid/fvModels",
            "energySource/q",
            source_w_m3,
        ),
        "helium heat capacity": (
            case / "constant/fluid/physicalProperties",
            "mixture/thermodynamics/Cp",
            helium_cp,
        ),
    }
    for name, (path, entry, expected_value) in dictionary_scalars.items():
        actual = scalar_from_foam_value(read_foam(path, entry))
        close(actual, expected_value, f"{case.name} {name}")

    helium_hash = sha256(case / "constant/fluid/physicalProperties")
    if helium_hash != canonical_helium_hash:
        raise ValueError(f"{case.name}: helium table differs from the registered table")
    return {
        "condition_id": case.name,
        "inlet_velocity_m_s": velocity,
        "pore_opening_boundary_velocity_m_s": pore_velocity,
        "inlet_open_area_fraction": open_fraction,
        "source_channel_volume_flow_preserved": True,
        "inlet_temperature_K": temperature,
        "solid_heat_source_MW_m3": source_mw_m3,
        "dictionary_values_match_sources": True,
        "helium_property_table_sha256": helium_hash,
        "mesh_crop_box_dp": metadata["mesh_crop_box_dp"],
        "mesh_triangulated_porosity": metadata["mesh_triangulated_porosity"],
        "numerical_controls": {
            "end_time_s": metadata["end_time"],
            "write_interval_s": metadata["write_interval"],
            "energy_coupling_correctors": metadata["energy_coupling_correctors"],
            "parallel_subdomains": metadata["parallel_subdomains"],
        },
    }


def verify_matrix(
    matrix_root: Path,
    parameter_source: Path,
    canonical_helium: Path,
    openfoam_bashrc: Path,
    packing_input: Path,
    packing_summary: Path,
    mesh_manifest: Path,
) -> dict[str, object]:
    rows = source_rows(parameter_source)
    required = set(CASE_PHYSICS_PARAMETER_IDS) | set(MESH_GEOMETRY_SOURCE_PARAMETER_IDS)
    missing = sorted(required.difference(rows))
    if missing:
        raise ValueError(f"source table is missing {missing}")
    if any(not rows[item]["链接或DOI"].strip() for item in required):
        raise ValueError("one or more physical inputs lack a literature link")
    conditions = published_conditions(rows["P418"]["采用值或关系式"])
    if len(conditions) != 60:
        raise ValueError(f"P418 should define 60 cases, found {len(conditions)}")
    actual_names = {path.name for path in matrix_root.glob("u*_T*_q*") if path.is_dir()}
    if actual_names != set(conditions):
        raise ValueError(
            f"matrix condition set differs; missing={sorted(set(conditions)-actual_names)}, "
            f"extra={sorted(actual_names-set(conditions))}"
        )
    canonical_hash = sha256(canonical_helium)
    geometry = verify_geometry_sources(rows, packing_input, packing_summary, mesh_manifest)
    read_foam = foam_reader(openfoam_bashrc)
    cases = [
        verify_case(
            matrix_root / condition_id,
            conditions[condition_id],
            rows,
            read_foam,
            canonical_hash,
        )
        for condition_id in sorted(conditions)
    ]
    mesh_hashes = {
        json.dumps(
            [row["mesh_crop_box_dp"], row["mesh_triangulated_porosity"]],
            sort_keys=True,
        )
        for row in cases
    }
    if len(mesh_hashes) != 1:
        raise ValueError("the 60 cases do not share one fixed mesh geometry")
    return {
        "status": "hccb_p418_60_actual_case_inputs_verified",
        "case_count": len(cases),
        "physical_parameter_ids": sorted(required),
        "all_operating_points_are_exact_P418_values": True,
        "all_openfoam_dictionary_values_match_registered_sources": True,
        "all_cases_share_one_fixed_mesh": True,
        "realized_mesh_porosity_is_a_computed_geometry_result": True,
        "new_fitted_physical_parameters": [],
        "canonical_helium_property_table_sha256": canonical_hash,
        "geometry_sources": geometry,
        "numerical_controls_are_reported_separately": True,
        "cases": cases,
    }


def write_chinese_summary(result: dict[str, object], output: Path) -> None:
    geometry = result["geometry_sources"]
    lines = [
        "# P418正式换热算例的文献参数与实际输入对应结果",
        "",
        f"已检查全部 **{result['case_count']}** 个正式稳态算例。文献入口通道速度通过实际入口开口面积换算为局部孔道边界速度，因此体积流量保持一致；入口温度、颗粒发热率、冷却壁温度、压力、氦气物性、颗粒导热系数、密度和稳态字典比热均与参数表一致。",
        "",
        "## 颗粒床和网格",
        "",
        f"- 物理颗粒直径：{geometry['physical_particle_diameter_m']:.6g} m（P048）。",
        f"- 大球床尺寸：{' x '.join(str(v) for v in geometry['published_large_box_dp'])} dp（P050）。",
        f"- 文献截取区域：{' x '.join(str(v) for v in geometry['published_crop_box_dp'])} dp（P390）。",
        f"- 网格颗粒直径：{geometry['meshing_particle_diameter_m']:.6g} m，即按P404缩小1%。",
        f"- 当前精细局部域尺寸：{' x '.join(f'{v:.4g}' for v in geometry['fine_local_crop_lengths_dp'])} dp，保留{geometry['fine_local_retained_particle_fragments']}个颗粒片段，三角网格孔隙率为{geometry['fine_local_triangulated_porosity']:.6f}。这些是本次网格的计算结果，不是新增文献参数。",
        "",
        "## 对后续神经网络的含义",
        "",
        "PINN、Transformer和扩散模型读取的60组稳态数据都来自上述同一套实际OpenFOAM输入。颗粒排列、物性和边界条件没有在神经网络阶段重新拟合或另行指定。",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--parameter-source", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--canonical-helium", type=Path, default=DEFAULT_CANONICAL_HELIUM)
    parser.add_argument("--packing-input", type=Path, default=DEFAULT_PACKING_INPUT)
    parser.add_argument("--packing-summary", type=Path, default=DEFAULT_PACKING_SUMMARY)
    parser.add_argument("--mesh-manifest", type=Path, default=DEFAULT_MESH_MANIFEST)
    parser.add_argument(
        "--openfoam-bashrc", type=Path, default=Path("/opt/openfoam13/etc/bashrc")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/hccb_p418_60_actual_case_input_check/summary.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "results/hccb_p418_60_actual_case_input_check/P418_正式算例参数对应_CN.md",
    )
    args = parser.parse_args()
    result = verify_matrix(
        args.matrix_root.resolve(),
        args.parameter_source.resolve(),
        args.canonical_helium.resolve(),
        args.openfoam_bashrc.resolve(),
        args.packing_input.resolve(),
        args.packing_summary.resolve(),
        args.mesh_manifest.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_chinese_summary(result, args.markdown_output.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
