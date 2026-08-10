#!/usr/bin/env python3
"""Compare the OpenFOAM transport check output with registered equations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_mu(temperature_k: float) -> float:
    return 0.4646 * temperature_k**0.66 * 1e-6


def registered_kappa(pressure_pa: float, temperature_k: float) -> float:
    reduced_temperature = temperature_k / 273.0
    reduced_pressure = pressure_pa / 1e6
    return (
        0.1448
        * reduced_temperature**0.68
        * (
            1.0
            + 2.5e-3
            * reduced_pressure**1.17
            * reduced_temperature**-1.85
        )
    )


def parse_rows(path: Path) -> list[dict[str, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = "p_pa,T_k,mu_pa_s,kappa_w_m_k"
    start = lines.index(header)
    reader = csv.DictReader(lines[start:])
    return [{key: float(value) for key, value in row.items()} for row in reader]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = parse_rows(args.input)
    checked_rows = []
    for row in rows:
        mu_reference = registered_mu(row["T_k"])
        kappa_reference = registered_kappa(row["p_pa"], row["T_k"])
        checked_rows.append(
            {
                **row,
                "mu_reference": mu_reference,
                "kappa_reference": kappa_reference,
                "mu_relative_error": abs(row["mu_pa_s"] - mu_reference)
                / mu_reference,
                "kappa_relative_error": abs(
                    row["kappa_w_m_k"] - kappa_reference
                )
                / kappa_reference,
            }
        )

    result = {
        "status": "hccb_helium_transport_pointwise_check_passed",
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "point_count": len(checked_rows),
        "pressure_range_pa": [
            min(row["p_pa"] for row in checked_rows),
            max(row["p_pa"] for row in checked_rows),
        ],
        "temperature_range_k": [
            min(row["T_k"] for row in checked_rows),
            max(row["T_k"] for row in checked_rows),
        ],
        "maximum_mu_relative_error": max(
            row["mu_relative_error"] for row in checked_rows
        ),
        "maximum_kappa_relative_error": max(
            row["kappa_relative_error"] for row in checked_rows
        ),
        "all_values_positive_and_finite": all(
            math.isfinite(row[key]) and row[key] > 0
            for row in checked_rows
            for key in ("mu_pa_s", "kappa_w_m_k")
        ),
        "registered_parameter_ids": ["P070", "P071"],
        "physical_correlations_changed": False,
        "solver_started": False,
        "rows": checked_rows,
    }

    if len(checked_rows) != 12:
        raise SystemExit(f"expected 12 points, got {len(checked_rows)}")
    if result["maximum_mu_relative_error"] > 1e-12:
        raise SystemExit("viscosity comparison failed")
    if result["maximum_kappa_relative_error"] > 1e-12:
        raise SystemExit("conductivity comparison failed")
    if not result["all_values_positive_and_finite"]:
        raise SystemExit("non-positive or non-finite transport value")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
