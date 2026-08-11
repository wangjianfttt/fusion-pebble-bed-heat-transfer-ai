#!/usr/bin/env python3
"""Summarize the registered P418 complete-trajectory learning curve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


RUNS = (
    ("transient_learning_n03_up", "up"),
    ("transient_learning_n03_down", "down"),
    ("transient_learning_n06_both", "both"),
)
ROLES = ("train", "validation", "test")


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def finite_nonnegative(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def exact_split(summary: dict, expected: dict[str, list[str]], name: str) -> None:
    recorded = summary.get("split_case_ids")
    if not isinstance(recorded, dict):
        raise ValueError(f"{name} does not record complete-trajectory split identifiers")
    for role in ROLES:
        actual = [str(value) for value in recorded.get(role, [])]
        if actual != expected[role] or len(actual) != len(set(actual)):
            raise ValueError(f"{name} {role} trajectories differ from the registered split")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        (
            "\\caption{Sensitivity to the number and direction of complete OpenFOAM "
            "training trajectories. Validation and independent-test trajectories are "
            "fixed; saved time points are not counted as independent samples.}"
        ),
        "\\label{tab:transient_learning_curve}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Training trajectories & Direction & Solid-$T$ RMSE (K) & Train time (h) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['training_trajectory_count']} & {row['training_direction']} & "
            f"{row['test_solid_temperature_RMSE_K']:.3g} & "
            f"{row['training_seconds'] / 3600.0:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(result_root: Path, split_file: Path) -> tuple[dict, list[dict]]:
    split_payload = load_json(split_file)
    registered = split_payload.get("splits")
    if not isinstance(registered, dict):
        raise ValueError("learning-curve split file lacks splits")

    rows: list[dict] = []
    seeds: set[int] = set()
    for split_name, direction in RUNS:
        if split_name not in registered:
            raise ValueError(f"missing registered split {split_name}")
        expected = {
            role: [str(value) for value in registered[split_name][role]]
            for role in ROLES
        }
        source = result_root / split_name / "summary.json"
        if not source.is_file():
            raise FileNotFoundError(f"missing transient learning-curve result: {source}")
        summary = load_json(source)
        if summary.get("status") != "completed_p418_spatiotemporal_regional_operator":
            raise ValueError(f"{split_name} is unfinished or has the wrong result type")
        if summary.get("split_name") != split_name:
            raise ValueError(f"{split_name} summary records a different split")
        if summary.get("new_physical_parameters") != []:
            raise ValueError(f"{split_name} introduces unregistered physical parameters")
        if summary.get("physics_mode") != "energy_and_flux":
            raise ValueError(f"{split_name} does not use the selected physics objective")
        architecture = summary.get("architecture")
        if not isinstance(architecture, dict) or architecture.get(
            "spatial_temporal_mode"
        ) != "factorized_static_spatial":
            raise ValueError(f"{split_name} is not the registered factorized model")
        exact_split(summary, expected, split_name)
        seed = int(summary.get("seed", -1))
        if seed < 0:
            raise ValueError(f"{split_name} lacks the training seed")
        seeds.add(seed)
        metrics = summary.get("metrics", {}).get("test", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"{split_name} lacks independent-test metrics")
        rows.append(
            {
                "split_name": split_name,
                "training_trajectory_count": len(expected["train"]),
                "training_direction": direction,
                "validation_trajectory_count": len(expected["validation"]),
                "test_trajectory_count": len(expected["test"]),
                "test_solid_temperature_RMSE_K": finite_nonnegative(
                    metrics.get("solid_temperature_RMSE_K"),
                    f"{split_name} solid-temperature RMSE",
                ),
                "test_fluid_temperature_RMSE_K": finite_nonnegative(
                    metrics.get("fluid_temperature_RMSE_K"),
                    f"{split_name} fluid-temperature RMSE",
                ),
                "test_projection_energy_normalized_RMSE": finite_nonnegative(
                    metrics.get("projection_aware_energy_normalized_RMSE"),
                    f"{split_name} projected-energy RMSE",
                ),
                "training_seconds": finite_nonnegative(
                    summary.get("training_seconds"), f"{split_name} training time"
                ),
                "selected_epoch": int(summary.get("selected_epoch", -1)),
                "seed": seed,
                "source_summary": str(source.resolve()),
                "source_summary_sha256": sha256(source),
            }
        )
        if rows[-1]["selected_epoch"] <= 0:
            raise ValueError(f"{split_name} lacks the validation-selected epoch")

    if len(seeds) != 1:
        raise ValueError("learning-curve runs use different training seeds")
    counts = sorted({int(row["training_trajectory_count"]) for row in rows})
    if counts != [3, 6]:
        raise ValueError(f"unexpected training-trajectory counts: {counts}")
    if [row["training_direction"] for row in rows] != ["up", "down", "both"]:
        raise ValueError("learning-curve direction design is incomplete")
    if {int(row["validation_trajectory_count"]) for row in rows} != {2}:
        raise ValueError("validation trajectory count is not fixed at two")
    if {int(row["test_trajectory_count"]) for row in rows} != {4}:
        raise ValueError("independent-test trajectory count is not fixed at four")

    summary = {
        "status": "completed_p418_transient_learning_curve",
        "training_trajectory_counts": counts,
        "fixed_validation_trajectory_count": 2,
        "fixed_test_trajectory_count": 4,
        "seed": next(iter(seeds)),
        "model": "physics_constrained_factorized_graph_transformer",
        "runs": rows,
        "new_physical_parameters": [],
        "scientific_scope": (
            "Sensitivity to three or six complete OpenFOAM trajectories. Saved times "
            "within one trajectory are not treated as independent training cases."
        ),
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = summarize(args.result_root.resolve(), args.splits.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "transient_learning_curve.csv", rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_tex(args.tex_output.resolve(), rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
