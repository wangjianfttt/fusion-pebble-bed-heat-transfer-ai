#!/usr/bin/env python3
"""Compare integral CHT responses for common seed101 and seed202 conditions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


METRICS = {
    "outlet_temperature_K": "outlet_temperature_K",
    "maximum_solid_temperature_K": "solid_maximum_temperature_K",
    "pressure_drop_Pa": "pressure_drop_Pa",
}


def read_seed101(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row["condition_id"]
        if case_id in indexed:
            raise ValueError(f"duplicate seed101 condition: {case_id}")
        indexed[case_id] = row
    return indexed


def finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value}")
    return number


def relative_percent(delta: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("zero reference prevents relative comparison")
    return 100.0 * delta / abs(reference)


def bounds(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed101-csv", required=True, type=Path)
    parser.add_argument("--seed202-status", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-accepted", type=int, default=9)
    parser.add_argument("--expected-failed", type=int, default=0)
    parser.add_argument("--registered-case-count", type=int, default=9)
    args = parser.parse_args()

    seed101_path = args.seed101_csv.resolve()
    seed202_path = args.seed202_status.resolve()
    seed101 = read_seed101(seed101_path)
    seed202 = json.loads(seed202_path.read_text(encoding="utf-8"))
    accepted = seed202.get("accepted_cases", [])
    failed = seed202.get("failed_cases", [])
    if len(accepted) != args.expected_accepted:
        raise ValueError(
            f"expected {args.expected_accepted} accepted seed202 cases, "
            f"found {len(accepted)}"
        )
    if len(failed) != args.expected_failed:
        raise ValueError(
            f"expected {args.expected_failed} failed seed202 cases, found {len(failed)}"
        )

    seen: set[str] = set()
    comparison_rows: list[dict[str, object]] = []
    for case in accepted:
        case_id = str(case["case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate seed202 condition: {case_id}")
        seen.add(case_id)
        if case_id not in seed101:
            raise ValueError(f"seed101 reference is missing {case_id}")
        reference = seed101[case_id]
        row: dict[str, object] = {
            "condition_id": case_id,
            "seed202_job": case["job"],
            "seed202_summary_sha256": case["summary_sha256"],
            "seed202_mass_relative_difference": finite(
                case["mass_relative_difference"], f"{case_id} seed202 mass difference"
            ),
            "seed202_energy_relative_difference": finite(
                case["energy_relative_difference"],
                f"{case_id} seed202 energy difference",
            ),
            "seed101_mass_relative_difference": finite(
                reference["relative_mass_difference"],
                f"{case_id} seed101 mass difference",
            ),
            "seed101_energy_relative_difference": finite(
                reference["relative_energy_difference"],
                f"{case_id} seed101 energy difference",
            ),
        }
        for output_name, reference_name in METRICS.items():
            seed202_value = finite(case[output_name], f"{case_id} seed202 {output_name}")
            seed101_value = finite(
                reference[reference_name], f"{case_id} seed101 {reference_name}"
            )
            delta = seed202_value - seed101_value
            row[f"seed101_{output_name}"] = seed101_value
            row[f"seed202_{output_name}"] = seed202_value
            row[f"seed202_minus_seed101_{output_name}"] = delta
            row[f"relative_change_{output_name}_percent"] = relative_percent(
                delta, seed101_value
            )
        comparison_rows.append(row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paired_integral_differences.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    metric_summary: dict[str, object] = {}
    for output_name in METRICS:
        delta_values = [
            float(row[f"seed202_minus_seed101_{output_name}"])
            for row in comparison_rows
        ]
        relative_values = [
            float(row[f"relative_change_{output_name}_percent"])
            for row in comparison_rows
        ]
        absolute_relative = [abs(value) for value in relative_values]
        metric_summary[output_name] = {
            "signed_difference_range": bounds(delta_values),
            "relative_change_percent_range": bounds(relative_values),
            "mean_absolute_relative_change_percent": mean(absolute_relative),
            "median_absolute_relative_change_percent": median(absolute_relative),
            "maximum_absolute_relative_change_percent": max(absolute_relative),
        }

    complete = (
        len(comparison_rows) == args.registered_case_count and len(failed) == 0
    )
    summary = {
        "status": (
            "completed_seed101_seed202_integral_response_comparison"
            if complete
            else "partial_seed101_seed202_integral_response_comparison"
        ),
        "accepted_common_case_count": len(comparison_rows),
        "failed_seed202_case_count": len(failed),
        "registered_case_count": args.registered_case_count,
        "complete_nine_case_comparison": complete,
        "seed101_source": str(seed101_path),
        "seed202_source": str(seed202_path),
        "failed_seed202_cases": [case["case_id"] for case in failed],
        "metric_summary": metric_summary,
        "new_physical_parameters": [],
        "manuscript_use": (
            "complete nine-condition independent-packing physical comparison"
            if complete
            else "diagnostic only until all registered seed202 conditions are valid"
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tout = metric_summary["outlet_temperature_K"]
    tmax = metric_summary["maximum_solid_temperature_K"]
    dp = metric_summary["pressure_drop_Pa"]
    completion_text = (
        f"- 两套颗粒排列的 `{len(comparison_rows)}` 个登记工况均已完成，"
        "这里是完整九工况比较。\n"
        if complete
        else (
            f"- 当前只比较两套颗粒排列共同完成的 `{len(comparison_rows)}` 个工况。\n"
            f"- seed202另有 `{len(failed)}` 个工况失败，因此这里不是完整九工况结论。\n"
        )
    )
    chinese = (
        "# seed101与seed202共同工况的整体物理量比较\n\n"
        + completion_text
        +
        f"- 出口温度的平均绝对相对变化为 "
        f"`{tout['mean_absolute_relative_change_percent']:.3f}%`，最大为 "
        f"`{tout['maximum_absolute_relative_change_percent']:.3f}%`。\n"
        f"- 颗粒最高温度的平均绝对相对变化为 "
        f"`{tmax['mean_absolute_relative_change_percent']:.3f}%`，最大为 "
        f"`{tmax['maximum_absolute_relative_change_percent']:.3f}%`。\n"
        f"- 压降的平均绝对相对变化为 "
        f"`{dp['mean_absolute_relative_change_percent']:.3f}%`，最大为 "
        f"`{dp['maximum_absolute_relative_change_percent']:.3f}%`。\n"
        "- 逐工况原值、差值、相对变化和质量/能量闭合量见"
        "`paired_integral_differences.csv`。\n"
    )
    (output_dir / "比较说明_CN.md").write_text(chinese, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
