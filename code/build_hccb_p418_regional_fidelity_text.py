#!/usr/bin/env python3
"""Write the regional-representation result directly from completed fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path, status: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise ValueError(f"unexpected result status in {path}: {payload.get('status')}")
    if payload.get("new_physical_parameters") != []:
        raise ValueError(f"result unexpectedly introduces physical parameters: {path}")
    return payload


def number(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--reconstruction-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    representation = load(
        args.representation_summary.resolve(),
        "regional_representation_fidelity_ready",
    )
    reconstruction = load(
        args.reconstruction_summary.resolve(),
        "native_reconstruction_comparison_ready",
    )
    counts = representation["counts"]
    case_count = int(counts["cases"])
    if case_count != int(reconstruction["case_count"]):
        raise ValueError("representation and reconstruction case counts differ")
    if representation["exact_saved_regional_state_matches_direct_volume_average"] is not True:
        raise ValueError("stored regional state does not match direct volume averaging")

    rm = representation["metrics"]
    nm = reconstruction["metrics"]
    compression = representation["compression_ratio"]
    lines = [
        (
            f"Across the {case_count} completed fields, the fixed regional graph contains "
            f"{int(counts['regional_fluid_nodes']):,} fluid and "
            f"{int(counts['regional_solid_nodes']):,} solid nodes, representing "
            f"{int(counts['native_fluid_cells']):,} fluid and "
            f"{int(counts['native_solid_cells']):,} solid native cells, respectively. "
            f"One regional node therefore represents {number(compression['total_native_cells_per_regional_node'])} "
            "native cells on average. Direct volume averaging reproduces every stored regional "
            "temperature and preserves each phase volume-mean temperature to numerical precision."
        ),
        "",
        (
            f"The mean representation RMSE is \\SI{{{number(rm['fluid_volume_weighted_rmse_K']['mean'])}}}{{K}} "
            f"for the fluid and \\SI{{{number(rm['solid_volume_weighted_rmse_K']['mean'])}}}{{K}} "
            f"for the solid, corresponding to {number(rm['fluid_rmse_over_native_range_percent']['mean'])}\\% "
            f"and {number(rm['solid_rmse_over_native_range_percent']['mean'])}\\% of the native phase-temperature "
            f"ranges. The regional solid maximum is within \\SI{{{number(rm['solid_hotspot_temperature_loss_K']['mean'])}}}{{K}} "
            f"of the native maximum on average. The hottest regional node contains the hottest native cell in "
            f"{number(100.0 * representation['hottest_native_cell_region_match_fraction'])}\\% of the cases, and "
            f"the nearest native cell belonging to that node is displaced by "
            f"{number(rm['solid_hotspot_nearest_cell_distance_dp']['mean'])} $d_p$ on average. "
            "The regional representation therefore retains hotspot magnitude more accurately than exact "
            "native-cell location."
        ),
        "",
        (
            f"Unrestricted affine reconstruction reduces the fluid and solid RMSEs to "
            f"\\SI{{{number(nm['fluid_affine_volume_weighted_rmse_K']['mean'])}}}{{K}} and "
            f"\\SI{{{number(nm['solid_affine_volume_weighted_rmse_K']['mean'])}}}{{K}}, but gives a mean "
            f"solid-maximum error of \\SI{{{number(nm['solid_affine_max_temperature_error_K']['mean'])}}}{{K}} "
            f"and a hotspot displacement of {number(nm['solid_affine_hotspot_distance_dp']['mean'])} $d_p$. "
            "It is therefore not a physically admissible improvement."
        ),
        "",
        (
            f"The parameter-free limited reconstruction gives fluid and solid RMSEs of "
            f"\\SI{{{number(nm['fluid_limited_volume_weighted_rmse_K']['mean'])}}}{{K}} and "
            f"\\SI{{{number(nm['solid_limited_volume_weighted_rmse_K']['mean'])}}}{{K}}, removes the overshoot, "
            f"and reduces the mean solid-hotspot displacement from "
            f"{number(nm['solid_constant_hotspot_distance_dp']['mean'])} to "
            f"{number(nm['solid_limited_hotspot_distance_dp']['mean'])} $d_p$. Relative to piecewise-constant "
            f"prolongation, it removes {number(nm['fluid_limited_variance_reduction_percent']['mean'])}\\% of the "
            f"fluid and {number(nm['solid_limited_variance_reduction_percent']['mean'])}\\% of the solid squared "
            "representation error. Learned native-cell predictions are consequently compared with this bounded "
            "deterministic reconstruction. The diffusion branch remains assigned to the temporal residual after "
            "regional prediction."
        ),
        "",
    ]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "status": "complete_p418_regional_fidelity_manuscript_text",
        "case_count": case_count,
        "representation_summary": str(args.representation_summary.resolve()),
        "reconstruction_summary": str(args.reconstruction_summary.resolve()),
        "tex": str(output),
        "new_physical_parameters": [],
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
