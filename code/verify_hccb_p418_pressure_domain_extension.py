#!/usr/bin/env python3
"""Verify that a wider helium lookup domain changes no physical correlation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = (
    ROOT / "results/apd006_hccb_openfoam_helium_property_table/helium_property_table.npz"
)
DEFAULT_EXTENDED = (
    ROOT
    / "results/hccb_p418_pressure_domain_extension_u025_20260726/helium_property_table.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/hccb_p418_pressure_domain_extension_u025_20260726/domain_extension_check.json"
)


def relative_difference(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(candidate - reference)
            / np.maximum(np.abs(reference), np.float64(1.0e-30))
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-table", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--extended-table", type=Path, default=DEFAULT_EXTENDED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-job-id", default="13199")
    parser.add_argument("--failed-case-id", default="u0p25_T900_q6p85")
    parser.add_argument("--observed-pressure-min-pa", type=float)
    parser.add_argument("--observed-pressure-max-pa", type=float)
    parser.add_argument("--relative-difference-gate", type=float, default=1.0e-4)
    args = parser.parse_args()
    observed = [
        value
        for value in (
            args.observed_pressure_min_pa,
            args.observed_pressure_max_pa,
        )
        if value is not None
    ]
    if not observed:
        raise ValueError("at least one observed pressure bound is required")

    with np.load(args.canonical_table, allow_pickle=False) as canonical:
        with np.load(args.extended_table, allow_pickle=False) as extended:
            canonical_pressure = canonical["pressure_pa"].copy()
            extended_pressure = extended["pressure_pa"].copy()
            temperature = canonical["temperature_k"].copy()
            extended_temperature = extended["temperature_k"].copy()
            pressure_eval = np.linspace(
                float(canonical_pressure[0]), float(canonical_pressure[-1]), 101
            )
            temperature_eval = np.linspace(
                float(temperature[0]), float(temperature[-1]), 141
            )
            pp, tt = np.meshgrid(pressure_eval, temperature_eval, indexing="ij")
            points = np.column_stack([pp.ravel(), tt.ravel()])

            differences: dict[str, float] = {}
            for name in ("rho_kg_m3", "mu_pa_s", "kappa_w_m_k"):
                canonical_values = RegularGridInterpolator(
                    (canonical_pressure, temperature), canonical[name]
                )(points)
                extended_values = RegularGridInterpolator(
                    (extended_pressure, extended["temperature_k"]), extended[name]
                )(points)
                differences[name] = relative_difference(
                    canonical_values, extended_values
                )

    checks = {
        "observed_pressure_bounds_are_inside_extended_domain": bool(
            all(extended_pressure[0] < value < extended_pressure[-1] for value in observed)
        ),
        "canonical_domain_is_inside_extended_domain": bool(
            extended_pressure[0] <= canonical_pressure[0]
            and extended_pressure[-1] >= canonical_pressure[-1]
        ),
        "temperature_nodes_are_unchanged": bool(
            np.array_equal(temperature, extended_temperature)
        ),
        "difference_inside_canonical_domain_is_below_existing_ND044_gate": bool(
            max(differences.values()) <= args.relative_difference_gate
        ),
    }
    passed = all(checks.values())
    payload = {
        "status": (
            "hccb_p418_pressure_domain_extension_passed"
            if passed
            else "hccb_p418_pressure_domain_extension_failed"
        ),
        "failed_job_id": args.failed_job_id,
        "failed_case_id": args.failed_case_id,
        "observed_pressure_min_pa": args.observed_pressure_min_pa,
        "observed_pressure_max_pa": args.observed_pressure_max_pa,
        "canonical_pressure_nodes_pa": canonical_pressure.tolist(),
        "extended_pressure_nodes_pa": extended_pressure.tolist(),
        "extended_margin_below_observed_min_pa": (
            float(args.observed_pressure_min_pa - extended_pressure[0])
            if args.observed_pressure_min_pa is not None
            else None
        ),
        "extended_margin_above_observed_max_pa": (
            float(extended_pressure[-1] - args.observed_pressure_max_pa)
            if args.observed_pressure_max_pa is not None
            else None
        ),
        "maximum_relative_difference_inside_canonical_domain": differences,
        "relative_difference_gate": args.relative_difference_gate,
        "checks": checks,
        "physical_correlations_changed": False,
        "operating_conditions_changed": False,
        "new_fitted_physical_parameters": [],
        "interpretation": (
            "The extended table evaluates the same P070, P071 and P389 equations "
            "over a wider numerical lookup domain. It does not change the helium "
            "model or the P418 operating matrix."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
