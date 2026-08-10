#!/usr/bin/env python3
"""Join P418 learning-curve errors with measured OpenFOAM solver time."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from report_hccb_p418_runtime_progress import accumulated_solver_clock_time


PARALLEL_RANKS = 32


def summarize(
    *,
    comparison_csv: Path,
    split_file: Path,
    matrix_root: Path,
    output_dir: Path,
    expected_training_counts: list[int] | None = None,
    expected_architectures: list[str] | None = None,
    expected_split_names: list[str] | None = None,
    expected_validation_count: int | None = None,
    expected_test_count: int | None = None,
) -> dict[str, object]:
    splits = json.loads(split_file.read_text(encoding="utf-8"))["splits"]
    with comparison_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("learning-curve comparison table is empty")
    required_metrics = (
        "test_state_normalized_rmse",
        "test_fluid_temperature_normalized_rmse",
        "test_solid_temperature_normalized_rmse",
    )
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        pair = (str(row["architecture"]), str(row["split"]))
        if pair in seen_pairs:
            raise ValueError(f"duplicate learning-curve result: {pair}")
        seen_pairs.add(pair)
        for metric in required_metrics:
            if metric not in row or row[metric] == "":
                raise ValueError(f"learning-curve result lacks {metric} for {pair}")
            value = float(row[metric])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid learning-curve metric {metric} for {pair}")

    runtime_by_condition: dict[str, float] = {}
    for case in matrix_root.glob("u*_T*_q*"):
        runtime = accumulated_solver_clock_time(case / "log.foamMultiRun.formal")
        if runtime is not None:
            runtime_by_condition[case.name] = float(runtime)

    output_rows: list[dict[str, object]] = []
    for row in rows:
        split_name = row["split"]
        if split_name not in splits:
            raise ValueError(f"comparison contains unknown split {split_name}")
        training_ids = [str(value) for value in splits[split_name]["train"]]
        missing = [item for item in training_ids if item not in runtime_by_condition]
        if missing:
            raise ValueError(
                f"OpenFOAM clock time is missing for {split_name}: {missing}"
            )
        recorded_count = int(row["train_case_count"])
        if recorded_count != len(training_ids):
            raise ValueError(
                f"comparison train count differs from split {split_name}: "
                f"comparison={recorded_count}, split={len(training_ids)}"
            )
        solver_seconds = sum(runtime_by_condition[item] for item in training_ids)
        output_rows.append(
            {
                **row,
                "openfoam_training_solver_hours": solver_seconds / 3600.0,
                "openfoam_training_core_hours_32ranks": (
                    solver_seconds * PARALLEL_RANKS / 3600.0
                ),
                "training_data_fraction_of_36": len(training_ids) / 36.0,
            }
        )

    output_rows.sort(key=lambda item: (str(item["architecture"]), int(item["train_case_count"])))
    training_counts = sorted({int(row["train_case_count"]) for row in output_rows})
    architectures = sorted({str(row["architecture"]) for row in output_rows})
    split_names = sorted({str(row["split"]) for row in output_rows})
    validation_counts = sorted(
        {int(row["validation_case_count"]) for row in output_rows}
    )
    test_counts = sorted({int(row["test_case_count"]) for row in output_rows})
    if expected_training_counts is not None and training_counts != sorted(
        expected_training_counts
    ):
        raise ValueError(
            f"learning-curve training counts differ: {training_counts}"
        )
    if expected_architectures is not None and architectures != sorted(
        expected_architectures
    ):
        raise ValueError(f"learning-curve architectures differ: {architectures}")
    if expected_split_names is not None and split_names != sorted(expected_split_names):
        raise ValueError(f"learning-curve splits differ: {split_names}")
    if expected_validation_count is not None and validation_counts != [
        expected_validation_count
    ]:
        raise ValueError(
            f"validation count is not fixed at {expected_validation_count}: "
            f"{validation_counts}"
        )
    if expected_test_count is not None and test_counts != [expected_test_count]:
        raise ValueError(
            f"test count is not fixed at {expected_test_count}: {test_counts}"
        )
    if expected_architectures is not None and expected_split_names is not None:
        expected_pairs = {
            (architecture, split_name)
            for architecture in expected_architectures
            for split_name in expected_split_names
        }
        if seen_pairs != expected_pairs:
            missing = sorted(expected_pairs - seen_pairs)
            extra = sorted(seen_pairs - expected_pairs)
            raise ValueError(
                f"learning-curve model/split table is incomplete: "
                f"missing={missing}, extra={extra}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "learning_curve_efficiency.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    payload = {
        "status": "p418_steady_learning_curve_complete",
        "training_condition_counts": training_counts,
        "architectures": architectures,
        "split_names": split_names,
        "fixed_validation_condition_count": validation_counts,
        "fixed_test_condition_count": test_counts,
        "parallel_ranks_per_openfoam_case": PARALLEL_RANKS,
        "runtime_definition": (
            "sum of the maximum OpenFOAM ClockTime from every restart segment; "
            "core-hours multiply that case time by 32 MPI ranks"
        ),
        "table": output_csv.name,
        "new_physical_parameters": [],
    }
    (output_dir / "learning_curve_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-training-counts", type=int, nargs="+")
    parser.add_argument("--expected-architectures", nargs="+")
    parser.add_argument("--expected-split-names", nargs="+")
    parser.add_argument("--expected-validation-count", type=int)
    parser.add_argument("--expected-test-count", type=int)
    args = parser.parse_args()
    payload = summarize(
        comparison_csv=args.comparison_csv.resolve(),
        split_file=args.split_file.resolve(),
        matrix_root=args.matrix_root.resolve(),
        output_dir=args.output_dir.resolve(),
        expected_training_counts=args.expected_training_counts,
        expected_architectures=args.expected_architectures,
        expected_split_names=args.expected_split_names,
        expected_validation_count=args.expected_validation_count,
        expected_test_count=args.expected_test_count,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
