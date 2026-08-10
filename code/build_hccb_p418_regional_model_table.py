#!/usr/bin/env python3
"""Build the strict like-for-like regional temperature-field comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SPLIT = "pair_disjoint_stress_test"
MODELS = (
    ("Initial-field persistence", f"regional_persistence_{SPLIT}", None),
    (
        "Continuous-time regional DMDc",
        f"regional_dmdc_{SPLIT}",
        "selected_rank",
    ),
    (
        "Data-only graph--Transformer",
        f"regional_graph_transformer_bounded_data_only_{SPLIT}",
        "selected_epoch",
    ),
    (
        "Physics-constrained graph--Transformer",
        f"regional_graph_transformer_bounded_physics_{SPLIT}",
        "selected_epoch",
    ),
    (
        "Factorized physics-constrained graph--Transformer",
        f"regional_graph_transformer_bounded_factorized_{SPLIT}",
        "selected_epoch",
    ),
)
METRICS = (
    ("fluid_temperature_RMSE_K", "fluid_temperature_RMSE_K"),
    ("solid_temperature_RMSE_K", "solid_temperature_RMSE_K"),
    (
        "solid_maximum_temperature_history_RMSE_K",
        "maximum_solid_temperature_history_RMSE_K",
    ),
    (
        "solid_regional_hotspot_location_p95_error_m",
        "hotspot_location_p95_error_m",
    ),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_summary(
    summary: dict,
    *,
    model: str,
    split_ids: dict | None,
    selection_key: str | None,
) -> dict:
    if summary.get("split_name") != SPLIT:
        raise ValueError(f"{model} does not report the registered strict split")
    expected_selection = "validation" if selection_key is not None else "not_applicable"
    if summary.get("selection_split") != expected_selection:
        raise ValueError(
            f"{model} has an invalid model-selection declaration"
        )
    if selection_key is None and (
        int(summary.get("model_parameter_count", -1)) != 0
        or float(summary.get("training_seconds", math.nan)) != 0.0
    ):
        raise ValueError(f"{model} must contain no fitted parameters")
    current_ids = summary.get("split_case_ids")
    if not isinstance(current_ids, dict):
        raise ValueError(f"{model} does not record complete trajectory IDs")
    if split_ids is not None and current_ids != split_ids:
        raise ValueError(f"{model} uses a different train/validation/test split")
    if summary.get("temperature_metric_definition") != (
        "regional-volume-weighted RMSE, reported separately for fluid and solid"
    ):
        raise ValueError(f"{model} does not use the registered volume-weighted metric")
    test = summary.get("metrics", {}).get("test", {})
    for source, _ in METRICS:
        value = float(test[source])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{model} has an invalid test metric: {source}")
    return current_ids


def completed_rows(result_root: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    split_ids: dict | None = None
    for model, directory, selection_key in MODELS:
        path = result_root / directory / "summary.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        summary = load(path)
        split_ids = require_summary(
            summary,
            model=model,
            split_ids=split_ids,
            selection_key=selection_key,
        )
        test = summary["metrics"]["test"]
        row: dict[str, object] = {
            "split_name": SPLIT,
            "model": model,
            "validation_selected_rank_or_epoch": (
                int(summary[selection_key]) if selection_key is not None else "n/a"
            ),
            "training_seconds": float(summary["training_seconds"]),
            "compute_device": str(summary["compute_device"]),
            "source_summary": str(path.resolve()),
            "source_sha256": sha256(path),
        }
        for source, output in METRICS:
            row[output] = float(test[source])
        rows.append(row)
    return rows, missing


def latex_table(rows: list[dict[str, object]]) -> str:
    if len(rows) != len(MODELS):
        raise ValueError(
            f"regional manuscript table requires {len(MODELS)} models, found {len(rows)}"
        )
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Like-for-like prediction of regional temperature fields for the endpoint-pair holdout. All models use the same six training, two validation and four independent trajectories, the same 46\,089-node regional representation and the original 56 physical times. Rank or epoch is selected using validation trajectories only. Temperature errors are regional-volume weighted; hotspot error is the 95th percentile distance between predicted and reference hottest solid regional nodes.}",
        r"\label{tab:regional_dynamics}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Model & Selected & Fluid $T$ RMSE (K) & Solid $T$ RMSE (K) & $T_{s,\max}$ RMSE (K) & Hotspot p95 (mm) \\",
        r"\midrule",
    ]
    for row in rows:
        selected = row["validation_selected_rank_or_epoch"]
        lines.append(
            f"{row['model']} & {selected} & "
            f"{float(row['fluid_temperature_RMSE_K']):.2f} & "
            f"{float(row['solid_temperature_RMSE_K']):.2f} & "
            f"{float(row['maximum_solid_temperature_history_RMSE_K']):.2f} & "
            f"{1000.0 * float(row['hotspot_location_p95_error_m']):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def result_text(rows: list[dict[str, object]]) -> str:
    if len(rows) != len(MODELS):
        raise ValueError(
            f"regional manuscript text requires {len(MODELS)} models, found {len(rows)}"
        )
    by_model = {str(row["model"]): row for row in rows}
    persistence = by_model["Initial-field persistence"]
    dmdc = by_model["Continuous-time regional DMDc"]
    lowest_solid = min(rows, key=lambda row: float(row["solid_temperature_RMSE_K"]))
    lowest_hotspot = min(rows, key=lambda row: float(row["hotspot_location_p95_error_m"]))
    return (
        "On the strict endpoint-pair holdout, repeating the initial temperature "
        "field gave fluid- and solid-temperature RMSE values of "
        f"{float(persistence['fluid_temperature_RMSE_K']):.2f} and "
        f"{float(persistence['solid_temperature_RMSE_K']):.2f}\\,K, whereas "
        "continuous-time regional DMDc gave "
        f"{float(dmdc['fluid_temperature_RMSE_K']):.2f} and "
        f"{float(dmdc['solid_temperature_RMSE_K']):.2f}\\,K. "
        f"The lowest solid-field error was obtained by "
        f"{lowest_solid['model']} "
        f"({float(lowest_solid['solid_temperature_RMSE_K']):.2f}\\,K), whereas "
        f"the lowest hotspot p95 displacement was obtained by "
        f"{lowest_hotspot['model']} "
        f"({1000.0 * float(lowest_hotspot['hotspot_location_p95_error_m']):.2f}\\,mm). "
        "These quantities are reported separately because a lower field-average "
        "temperature error does not by itself prove more accurate hotspot localization.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tex", type=Path)
    parser.add_argument("--text", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    rows, missing = completed_rows(args.result_root.resolve())
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(
            "regional field comparison is incomplete: " + "; ".join(missing)
        )

    fields = [
        "split_name",
        "model",
        "validation_selected_rank_or_epoch",
        "fluid_temperature_RMSE_K",
        "solid_temperature_RMSE_K",
        "maximum_solid_temperature_history_RMSE_K",
        "hotspot_location_p95_error_m",
        "training_seconds",
        "compute_device",
        "source_summary",
        "source_sha256",
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    complete = not missing
    if complete and args.tex:
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        args.tex.write_text(latex_table(rows), encoding="utf-8")
    if complete and args.text:
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(result_text(rows), encoding="utf-8")
    payload = {
        "status": (
            "complete_p418_like_for_like_regional_model_comparison"
            if complete
            else "incomplete_p418_like_for_like_regional_model_comparison"
        ),
        "split_name": SPLIT,
        "completed_models": [str(row["model"]) for row in rows],
        "missing_summaries": missing,
        "csv": str(args.csv.resolve()),
        "tex": str(args.tex.resolve()) if complete and args.tex else None,
        "text": str(args.text.resolve()) if complete and args.text else None,
        "new_physical_parameters": [],
        "scientific_scope": (
            "Direct comparison on identical volume-weighted regional temperature "
            "fields and complete endpoint-pair holdout trajectories."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
