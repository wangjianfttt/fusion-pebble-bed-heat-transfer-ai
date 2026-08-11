#!/usr/bin/env python3
"""Write the full-matrix P417/P419 comparison into the manuscript."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_pair(payload: dict, key: str) -> tuple[float, float]:
    values = payload.get(key)
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{key} must contain two values")
    low, high = map(float, values)
    if not all(math.isfinite(value) for value in (low, high)) or low > high:
        raise ValueError(f"invalid {key}: {values}")
    return low, high


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-summary", required=True, type=Path)
    parser.add_argument("--pressure-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    source = args.input_summary.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "p418_dimensionless_heat_transfer_comparison_complete":
        raise ValueError("dimensionless heat-transfer result is incomplete")
    case_count = int(payload["case_count"])
    if case_count != 60:
        raise ValueError("formal manuscript text requires all 60 P418 conditions")
    parameter_ids = set(map(str, payload.get("parameter_ids", [])))
    if not {"P417", "P419"}.issubset(parameter_ids):
        raise ValueError("the P417 correlation and P419 definition are not registered")

    comparable = int(payload["p417_p419_in_range_comparable_case_count"])
    within = int(payload["p417_reference_within_source_30_percent_case_count"])
    nonpositive = int(payload["p419_nonpositive_phase_difference_case_count"])
    positive = int(payload["p419_positive_phase_difference_case_count"])
    if positive + nonpositive != case_count or not 0 <= within <= comparable <= positive:
        raise ValueError("inconsistent P417/P419 case counts")
    fraction = float(payload["p417_reference_within_source_30_percent_fraction"])
    if not math.isclose(fraction, within / comparable, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("reported P417 fraction differs from the case counts")
    maximum_difference = float(
        payload["maximum_absolute_in_range_correlation_difference_percent"]
    )
    if not math.isfinite(maximum_difference) or maximum_difference < 0.0:
        raise ValueError("invalid maximum correlation difference")
    re_low, re_high = finite_pair(payload, "reynolds_axial_throughflow_range")
    pr_low, pr_high = finite_pair(payload, "prandtl_mean_properties_range")
    interface_nu_low, interface_nu_high = finite_pair(
        payload, "openfoam_interface_flux_nusselt_range"
    )
    interface_case_count = int(payload["openfoam_interface_flux_case_count"])
    sign_consistent_count = int(
        payload["openfoam_interface_flux_sign_consistent_case_count"]
    )
    if interface_case_count != case_count or not 0 <= sign_consistent_count <= case_count:
        raise ValueError("resolved interface-flux case count is incomplete")
    solid_partition_error = float(
        payload[
            "maximum_absolute_openfoam_solid_energy_partition_error_over_generated"
        ]
    )
    if not math.isfinite(solid_partition_error) or solid_partition_error < 0.0:
        raise ValueError("invalid resolved solid-energy partition difference")
    pressure_source = args.pressure_summary.resolve()
    pressure = json.loads(pressure_source.read_text(encoding="utf-8"))
    if pressure.get("status") != "p418_local_crop_pressure_correlation_complete":
        raise ValueError("local-crop pressure comparison is incomplete")
    if int(pressure["case_count"]) != case_count:
        raise ValueError("pressure and heat-transfer case counts differ")
    pressure_median = float(pressure["median_absolute_difference_percent"])
    pressure_maximum = float(pressure["maximum_absolute_difference_percent"])
    pressure_inside = int(pressure["inside_source_P422_case_count"])

    text = (
        "The 60 local-crop fields span "
        f"$Re_{{p,\\mathrm{{ax}}}}={re_low:.3f}$--${re_high:.3f}$ and "
        f"$Pr={pr_low:.3f}$--${pr_high:.3f}$. Of the {comparable} cases for which "
        "$\\overline{T}_s>\\overline{T}_f$ and the declared $Re_p<1.8$ range both hold, "
        f"{within} ({100.0 * fraction:.1f}\\%) lie within $\\pm30\\%$ of the published "
        "whole-bed P417 Nusselt correlation; the largest absolute difference is "
        f"{maximum_difference:.1f}\\%. The remaining {nonpositive} conditions have "
        "$\\overline{T}_s\\leq\\overline{T}_f$ in the wall-adjacent crop and are not "
        "forced into the positive phase-temperature-difference definition of P419. "
        "This difference from the source paper's whole-bed statistic is not removed by "
        "refitting: the present domain is a smaller wall-adjacent crop, and the source "
        "paper does not publish the spatial averaging operation used for "
        "$Re_{p,\\mathrm{AVE}}$ and $Pr_{\\mathrm{AVE}}$. The correlation is therefore "
        "retained as an aggregate literature reference rather than a local-field label "
        "or a neural-network loss. Direct integration of the finite-volume fluid--solid "
        f"interface gives $Nu_{{sf}}={interface_nu_low:.3f}$--${interface_nu_high:.3f}$; "
        f"its heat-flow direction agrees with $\\overline{{T}}_s-\\overline{{T}}_f$ in "
        f"{sign_consistent_count} of {case_count} cases. The maximum solid energy-partition "
        f"difference is {100.0 * solid_partition_error:.3f}\\% of generated power. This "
        "resolved interface quantity, rather than the aggregate P419 generated-power "
        "definition, supplies the local finite-volume heat-transfer target used by the "
        "neural models.\n"
        "Using the resolved crop porosity and length together with the superficial "
        "velocity through the complete inlet section, the P420/P421 modified pressure "
        f"relation differs from the resolved pressure drop by {pressure_median:.2f}\\% "
        f"at the median and {pressure_maximum:.1f}\\% at most; {pressure_inside} of "
        f"{case_count} cases also lie inside the source full-domain P422 value. The "
        "published inlet-channel volume flow is preserved by area-correcting the velocity "
        "on the open fluid part of the cropped inlet; that pore-opening velocity is not "
        "inserted directly into the packed-bed relation.\n"
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")

    record = {
        "status": "complete_p418_same_source_correlation_manuscript_text",
        "input_summary": str(source),
        "pressure_summary": str(pressure_source),
        "case_count": case_count,
        "positive_phase_difference_case_count": positive,
        "nonpositive_phase_difference_case_count": nonpositive,
        "comparable_case_count": comparable,
        "within_30_percent_case_count": within,
        "within_30_percent_fraction": fraction,
        "maximum_absolute_in_range_difference_percent": maximum_difference,
        "reynolds_axial_throughflow_range": [re_low, re_high],
        "prandtl_range": [pr_low, pr_high],
        "openfoam_interface_flux_nusselt_range": [
            interface_nu_low,
            interface_nu_high,
        ],
        "openfoam_interface_flux_sign_consistent_case_count": sign_consistent_count,
        "maximum_absolute_openfoam_solid_energy_partition_error_over_generated": (
            solid_partition_error
        ),
        "pressure_median_absolute_difference_percent": pressure_median,
        "pressure_maximum_absolute_difference_percent": pressure_maximum,
        "pressure_inside_source_P422_case_count": pressure_inside,
        "tex": str(output),
        "new_physical_parameters": [],
    }
    summary = args.summary.resolve()
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
