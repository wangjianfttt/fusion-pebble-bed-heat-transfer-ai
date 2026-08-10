#!/usr/bin/env python3
"""Write manuscript macros from completed P418 result artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def corrected_matrix_progress(project_root: Path) -> tuple[int, int]:
    """Read progress only from the corrected source-channel-flow matrix."""
    coverage_path = (
        project_root
        / "results/hccb_p418_training_data_coverage_partial/summary.json"
    )
    if coverage_path.is_file():
        coverage = load(coverage_path)
        completed = int(coverage["completed_case_count"])
        total = int(coverage["expected_case_count"])
        if completed > total:
            raise ValueError(
                f"verified training samples exceed total: {completed}/{total}"
            )
        return completed, total

    matrix = project_root / "hccb_dense_cht_p418_60_sourceflow_r3"
    case_dirs = sorted(path for path in matrix.glob("u*_T*_q*") if path.is_dir())
    if case_dirs:
        completed = sum(
            (case / "formal_sample_complete.json").is_file() for case in case_dirs
        )
        return completed, len(case_dirs)

    progress_path = project_root / "results/hccb_p418_sourceflow_runtime_progress.json"
    progress = load(progress_path)
    if progress.get("status") != "P418 matrix runtime progress":
        raise ValueError("corrected source-flow progress file has an unexpected status")
    return int(progress["completed_cases"]), int(progress["total_cases"])


def corrected_input_summary(project_root: Path) -> dict[str, object]:
    candidates = (
        project_root
        / "results/hccb_p418_sourceflow_preflight/formal_60_input_summary.json",
        project_root / "results/hccb_p418_60_actual_case_input_check/summary.json",
    )
    for path in candidates:
        if path.is_file():
            result = load(path)
            if result.get("status") == "hccb_p418_60_actual_case_inputs_verified":
                return result
    raise FileNotFoundError("no verified corrected source-flow input summary is available")


def scientific_tex(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return f"{mantissa:.{digits - 1}f}\\times 10^{{{exponent}}}"


def build(project_root: Path) -> dict[str, str]:
    result_root = project_root / "results"
    graph = load(
        result_root
        / "hccb_p418_actual_spatiotemporal_operator_56time_gpu_factorized"
        / "summary.json"
    )
    if graph.get("status") != (
        "formal_actual_graph_model_and_transient_physics_backward_passed"
    ):
        raise ValueError("manuscript graph values do not come from the formal 56-time physics model")
    if int(graph.get("time_points", -1)) != 56:
        raise ValueError("manuscript graph values do not use the formal 56-time schedule")
    interface = load(result_root / "hccb_p418_actual_interface_coupling" / "summary.json")
    time_scale = load(result_root / "hccb_p418_velocity_step_time_scales" / "summary.json")
    input_check = corrected_input_summary(project_root)

    input_rows = input_check.get("conditions") or input_check.get("cases")
    if not isinstance(input_rows, list) or not input_rows:
        raise ValueError("actual-case input summary has no condition rows")
    first_input = input_rows[0]
    interface_all = interface["all_conditions"]
    particle_scale = time_scale["particle_radial_conduction_scale_s"]
    if time_scale.get("velocity_basis") != (
        "source_channel_area_preserving_pore_boundary_velocity"
    ):
        raise ValueError("time-scale result does not use the corrected pore-boundary velocity")
    local_crossing = time_scale["resolved_local_crop_crossing_times_s"]

    completed, total = corrected_matrix_progress(project_root)
    if completed > total:
        raise ValueError(f"completed steady cases exceed total: {completed}/{total}")
    return {
        "CompletedSteadyCases": str(completed),
        "TotalSteadyCases": str(total),
        "SteadyProgressText": f"{completed}/{total}",
        "RegionalNodes": str(int(graph["nodes"])),
        "RegionalEdges": str(int(graph["edges"])),
        "TransientTimePoints": str(int(graph["time_points"])),
        "TransientModelParameters": str(int(graph["model_parameter_count"])),
        "TransientPeakGpuGB": f"{float(graph['peak_gpu_GB']):.2f}",
        "InterfacePairCount": str(int(interface_all["interface_pair_count"])),
        "InterfaceFluxRelativeMismatch": scientific_tex(
            float(interface_all["maximum_flux_sum_over_global_interface_flux"])
        ),
        "MeshPorosity": f"{float(first_input['mesh_triangulated_porosity']):.6f}",
        "LocalFlowCrossingMin": f"{min(map(float, local_crossing)):.4f}",
        "LocalFlowCrossingMax": f"{max(map(float, local_crossing)):.4f}",
        "ParticleConductionMin": f"{float(particle_scale['minimum']):.3f}",
        "ParticleConductionMax": f"{float(particle_scale['maximum']):.3f}",
    }


def write_macros(values: dict[str, str], output: Path) -> None:
    lines = ["% Generated by code/build_hccb_p418_manuscript_values.py"]
    lines.extend(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in values.items())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_macros(build(args.project_root.resolve()), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
