#!/usr/bin/env python3
"""Verify that recovered P418 three-mesh files are the formal GCI results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


FORMAL_SUMMARY_SHA256 = (
    "8944934cdd01157dcaf31835d9e0d9ecc44c3191f4f552a24bda4fbde007fe26"
)
FORMAL_GCI_SHA256 = (
    "9c42e564ed862530cf5476e242ce33d0e4ac26ca1c05c339f7e470497da01bf2"
)
FORMAL_CELLS = {
    "coarse": (160989, 200162),
    "medium": (432384, 515540),
    "fine": (858419, 1011645),
}
FORMAL_METRICS = {
    "pressure_drop_Pa",
    "outlet_temperature_change_K",
    "solid_maximum_temperature_change_K",
    "cooling_wall_heat_fraction",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"three-mesh result is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"three-mesh JSON must contain an object: {path}")
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"three-mesh result is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"three-mesh CSV contains no rows: {path}")
    return rows


def finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def csv_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def validate(
    root: Path,
    *,
    expected_summary_sha256: str,
    expected_gci_sha256: str,
) -> dict[str, object]:
    summary_path = root / "summary.json"
    engineering_path = root / "engineering_observables.csv"
    gci_path = root / "mesh_gci.csv"
    summary = load_json(summary_path)
    engineering = load_csv(engineering_path)
    gci_rows = load_csv(gci_path)

    file_hashes = {
        "summary.json": file_sha256(summary_path),
        "engineering_observables.csv": file_sha256(engineering_path),
        "mesh_gci.csv": file_sha256(gci_path),
    }
    checks: dict[str, bool] = {
        "formal_summary_sha256": (
            file_hashes["summary.json"] == expected_summary_sha256
        ),
        "formal_gci_sha256": file_hashes["mesh_gci.csv"] == expected_gci_sha256,
        "status": (
            summary.get("status") == "completed_three_mesh_p418_cht_comparison"
        ),
        "no_new_physical_parameters": summary.get("new_physical_parameters") == [],
    }

    levels = summary.get("mesh_levels")
    if not isinstance(levels, list) or len(levels) != 3:
        raise ValueError("formal three-mesh summary must contain three mesh levels")
    level_by_name = {
        str(row.get("mesh_level")): row for row in levels if isinstance(row, dict)
    }
    checks["three_named_levels"] = set(level_by_name) == set(FORMAL_CELLS)
    for name, (fluid_cells, solid_cells) in FORMAL_CELLS.items():
        row = level_by_name.get(name, {})
        checks[f"{name}_formal_cell_counts"] = bool(
            row.get("fluid_cells") == fluid_cells
            and row.get("solid_cells") == solid_cells
            and row.get("total_cells") == fluid_cells + solid_cells
        )
        checks[f"{name}_basic_mesh_checks"] = bool(
            row.get("fluid_basic_check_passes") is True
            and row.get("solid_basic_check_passes") is True
        )
        checks[f"{name}_finite_observables"] = all(
            finite_number(row.get(metric)) for metric in FORMAL_METRICS
        )

    engineering_by_name = {row.get("mesh_level", ""): row for row in engineering}
    checks["engineering_three_named_levels"] = set(engineering_by_name) == set(
        FORMAL_CELLS
    )
    for name, (fluid_cells, solid_cells) in FORMAL_CELLS.items():
        row = engineering_by_name.get(name, {})
        summary_row = level_by_name.get(name, {})
        checks[f"{name}_engineering_matches_summary"] = bool(
            row
            and int(row.get("fluid_cells", -1)) == fluid_cells
            and int(row.get("solid_cells", -1)) == solid_cells
            and int(row.get("total_cells", -1)) == fluid_cells + solid_cells
            and csv_bool(row.get("fluid_basic_check_passes", ""))
            and csv_bool(row.get("solid_basic_check_passes", ""))
            and all(
                finite_number(row.get(metric))
                and math.isclose(
                    float(row[metric]),
                    float(summary_row[metric]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                for metric in FORMAL_METRICS
            )
        )

    convergence = summary.get("grid_convergence")
    if not isinstance(convergence, list):
        raise ValueError("formal three-mesh summary has no grid-convergence records")
    convergence_by_metric = {
        str(row.get("metric")): row
        for row in convergence
        if isinstance(row, dict)
    }
    gci_by_metric = {row.get("metric", ""): row for row in gci_rows}
    checks["four_formal_gci_metrics"] = (
        set(convergence_by_metric) == FORMAL_METRICS
        and set(gci_by_metric) == FORMAL_METRICS
    )
    comparable_fields = {
        "coarse_value",
        "medium_value",
        "fine_value",
        "coarse_to_medium_refinement_ratio",
        "medium_to_fine_refinement_ratio",
        "observed_order",
        "richardson_extrapolated_value",
        "fine_gci_fraction",
        "fine_gci_absolute",
    }
    for metric in FORMAL_METRICS:
        summary_row = convergence_by_metric.get(metric, {})
        csv_row = gci_by_metric.get(metric, {})
        matching = bool(
            summary_row
            and csv_row
            and csv_row.get("convergence_status")
            == str(summary_row.get("convergence_status"))
        )
        for field in comparable_fields:
            left = summary_row.get(field)
            right = csv_row.get(field, "")
            if left is None:
                matching = matching and right in {"", "None"}
            else:
                matching = matching and finite_number(left) and finite_number(right)
                if matching:
                    matching = math.isclose(
                        float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12
                    )
        checks[f"{metric}_gci_csv_matches_summary"] = matching

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("formal three-mesh recovery failed: " + ", ".join(failed))
    return {
        "status": "verified_formal_p418_three_mesh_recovery",
        "result_root": str(root.resolve()),
        "formal_cell_counts": {
            name: {
                "fluid": cells[0],
                "solid": cells[1],
                "total": sum(cells),
            }
            for name, cells in FORMAL_CELLS.items()
        },
        "file_sha256": file_hashes,
        "checks": checks,
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected-summary-sha256", default=FORMAL_SUMMARY_SHA256
    )
    parser.add_argument("--expected-gci-sha256", default=FORMAL_GCI_SHA256)
    args = parser.parse_args()
    payload = validate(
        args.root,
        expected_summary_sha256=args.expected_summary_sha256,
        expected_gci_sha256=args.expected_gci_sha256,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
