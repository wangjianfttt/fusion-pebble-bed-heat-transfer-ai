#!/usr/bin/env python3
"""Compare P418 thermal-step histories from successive numerical time steps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def gci_triplet(
    coarse: float,
    medium: float,
    fine: float,
    refinement_ratio: float,
    safety_factor: float,
) -> dict[str, float | str | None]:
    """Celik et al. three-resolution GCI for an equal refinement ratio."""
    if refinement_ratio <= 1.0 or safety_factor <= 0.0:
        raise ValueError("GCI refinement ratio and safety factor must be positive")
    epsilon_32 = medium - coarse
    epsilon_21 = fine - medium
    scale = max(abs(coarse), abs(medium), abs(fine), 1.0)
    tiny = np.finfo(float).eps * scale
    result: dict[str, float | str | None] = {
        "coarse_value": float(coarse),
        "medium_value": float(medium),
        "fine_value": float(fine),
        "observed_order": None,
        "richardson_extrapolated_value": None,
        "fine_gci_fraction": None,
        "fine_gci_absolute": None,
    }
    if abs(epsilon_32) <= tiny and abs(epsilon_21) <= tiny:
        result.update(
            {
                "convergence_status": "identical_within_float64_resolution",
                "fine_gci_fraction": 0.0,
                "fine_gci_absolute": 0.0,
                "richardson_extrapolated_value": float(fine),
            }
        )
        return result
    if abs(epsilon_21) <= tiny:
        result.update(
            {
                "convergence_status": "fine_pair_identical",
                "fine_gci_fraction": 0.0,
                "fine_gci_absolute": 0.0,
                "richardson_extrapolated_value": float(fine),
            }
        )
        return result
    if epsilon_32 * epsilon_21 <= 0.0:
        result["convergence_status"] = "oscillatory_no_gci_reported"
        return result
    ratio = abs(epsilon_32 / epsilon_21)
    observed_order = float(np.log(ratio) / np.log(refinement_ratio))
    if not np.isfinite(observed_order) or observed_order <= 0.0:
        result["convergence_status"] = "not_monotonically_reducing_no_gci_reported"
        result["observed_order"] = observed_order
        return result
    denominator = refinement_ratio**observed_order - 1.0
    extrapolated = fine + (fine - medium) / denominator
    absolute_gci = safety_factor * abs(fine - medium) / denominator
    result.update(
        {
            "convergence_status": "monotonic_gci_reported",
            "observed_order": observed_order,
            "richardson_extrapolated_value": float(extrapolated),
            "fine_gci_fraction": (
                float(absolute_gci / abs(fine)) if abs(fine) > tiny else None
            ),
            "fine_gci_absolute": float(absolute_gci),
        }
    )
    return result


def load_curve(path: Path, requested: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    if len(data["case_id"]) != 1:
        raise ValueError(f"expected one sensitivity sequence in {path}")
    names = [str(value) for value in data["signal_names"]]
    missing = sorted(set(requested).difference(names))
    if missing:
        raise ValueError(f"missing comparison signals {missing} in {path}")
    mask = data["time_mask"][0].astype(bool)
    time = data["time_s"][0, mask].astype(np.float64)
    values = data["values"][0, mask].astype(np.float64)
    curves = {name: values[:, names.index(name)] for name in requested}
    for name, curve in curves.items():
        if not np.isfinite(curve).all():
            raise ValueError(f"non-finite {name} values in {path}")
    return time, curves


def compare_pair(
    coarse_time: np.ndarray,
    coarse: dict[str, np.ndarray],
    fine_time: np.ndarray,
    fine: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for signal, coarse_curve in coarse.items():
        fine_curve = np.interp(coarse_time, fine_time, fine[signal])
        difference = coarse_curve - fine_curve
        denominator = max(float(np.linalg.norm(fine_curve)), 1.0e-30)
        endpoint_scale = max(abs(float(fine_curve[-1])), 1.0e-30)
        response_span = float(np.max(fine_curve) - np.min(fine_curve))
        response_scale = max(
            response_span,
            abs(float(fine_curve[-1] - fine_curve[0])),
            1.0e-30,
        )
        rows.append(
            {
                "signal": signal,
                "relative_curve_l2": float(np.linalg.norm(difference) / denominator),
                "maximum_absolute_difference": float(np.max(np.abs(difference))),
                "endpoint_absolute_difference": float(abs(difference[-1])),
                "endpoint_relative_difference": float(abs(difference[-1]) / endpoint_scale),
                "maximum_difference_over_response_span": float(
                    np.max(np.abs(difference)) / response_scale
                ),
                "endpoint_difference_over_response_span": float(
                    abs(difference[-1]) / response_scale
                ),
            }
        )
    return rows


def label(delta_t: float) -> str:
    value = f"{delta_t:.12g}".replace("-", "m").replace("+", "").replace(".", "p")
    return f"dt_{value}"


def select_finest_declared_step(delta_t: list[float]) -> float:
    """Use the finest completed preregistered resolution without a fitted threshold."""
    if len(delta_t) < 2 or any(value <= 0 for value in delta_t):
        raise ValueError("at least two positive time steps are required")
    return min(delta_t)


def refinement_trend(
    rows: list[dict[str, float | str]], delta_t: list[float]
) -> dict[str, dict[str, float | bool]]:
    """Report whether successive differences decrease; this does not select a step."""
    signals = sorted({str(row["signal"]) for row in rows})
    trends: dict[str, dict[str, float | bool]] = {}
    for signal in signals:
        signal_rows = [row for row in rows if row["signal"] == signal]
        signal_rows.sort(key=lambda row: float(row["coarse_delta_t_s"]), reverse=True)
        if len(signal_rows) != len(delta_t) - 1:
            raise ValueError(f"incomplete refinement chain for {signal}")
        coarse = float(signal_rows[0]["maximum_difference_over_response_span"])
        finest = float(signal_rows[-1]["maximum_difference_over_response_span"])
        trends[signal] = {
            "coarse_pair_maximum_difference_over_response_span": coarse,
            "finest_pair_maximum_difference_over_response_span": finest,
            "successive_maximum_difference_decreases": finest <= coarse,
        }
    return trends


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-kind",
        choices=("fixed_hydrodynamics_thermal", "fully_coupled_flow_heat"),
        default="fixed_hydrodynamics_thermal",
    )
    args = parser.parse_args()

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    requested = [str(value) for value in config["comparison_quantities"]]
    delta_t = sorted((float(value) for value in config["delta_t_s"]), reverse=True)
    loaded = {}
    for value in delta_t:
        path = (
            args.result_root.resolve()
            / label(value)
            / "hccb_p418_transient_observables.npz"
        )
        loaded[value] = load_curve(path, requested)

    rows = []
    for coarse_dt, fine_dt in zip(delta_t[:-1], delta_t[1:]):
        coarse_time, coarse = loaded[coarse_dt]
        fine_time, fine = loaded[fine_dt]
        for row in compare_pair(coarse_time, coarse, fine_time, fine):
            rows.append({"coarse_delta_t_s": coarse_dt, "fine_delta_t_s": fine_dt, **row})

    if config.get("formal_selection_rule") != "finest_completed_predeclared_step":
        raise ValueError("unsupported formal time-step selection rule")
    selected_delta_t = select_finest_declared_step(delta_t)
    formal_schedule = [
        {
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "delta_t_s": float(row["delta_t_s"]),
        }
        for row in config["formal_time_step_schedule"]
    ]
    if formal_schedule[0]["delta_t_s"] != selected_delta_t:
        raise ValueError("formal staged schedule does not start with the finest declared step")
    trends = refinement_trend(rows, delta_t)
    gci_config = config["discretization_uncertainty_method"]
    refinement_ratio = float(gci_config["refinement_ratio"])
    safety_factor = float(gci_config["safety_factor"])
    if not np.allclose(
        [delta_t[0] / delta_t[1], delta_t[1] / delta_t[2]],
        refinement_ratio,
    ):
        raise ValueError("declared GCI ratio does not match the three time steps")
    gci_rows = []
    coarse_curves = loaded[delta_t[0]][1]
    medium_curves = loaded[delta_t[1]][1]
    fine_curves = loaded[delta_t[2]][1]
    for signal in requested:
        for quantity, reducer in (
            ("endpoint", lambda values: float(values[-1])),
            ("curve_maximum", lambda values: float(np.max(values))),
        ):
            gci_rows.append(
                {
                    "signal": signal,
                    "quantity": quantity,
                    **gci_triplet(
                        reducer(coarse_curves[signal]),
                        reducer(medium_curves[signal]),
                        reducer(fine_curves[signal]),
                        refinement_ratio,
                        safety_factor,
                    ),
                }
            )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.analysis_kind == "fully_coupled_flow_heat":
        file_prefix = "fully_coupled_timestep"
        status = "completed_p418_fully_coupled_timestep_sensitivity"
    else:
        file_prefix = "thermal_timestep"
        status = "completed_p418_thermal_timestep_sensitivity"
    csv_path = output / f"{file_prefix}_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gci_csv_path = output / f"{file_prefix}_gci.csv"
    with gci_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gci_rows[0]))
        writer.writeheader()
        writer.writerows(gci_rows)
    summary = {
        "status": status,
        "analysis_kind": args.analysis_kind,
        "sequence_id": config["sequence_id"],
        "config": str(args.config.resolve()),
        "delta_t_s": delta_t,
        "comparisons": rows,
        "interpretation_rule": config["comparison_rule"],
        "formal_selection_rule": config["formal_selection_rule"],
        "selected_delta_t_s": selected_delta_t,
        "selected_time_step_schedule": formal_schedule,
        "refinement_trend": trends,
        "discretization_uncertainty_method": gci_config,
        "gci_results": gci_rows,
        "selection_scope": (
            "The formal histories use the finest completed predeclared staged numerical "
            "schedule. Successive differences are reported, not compared with an "
            "unsourced acceptance percentage. No material property or operating "
            "condition is changed."
        ),
        "new_physical_parameters": [],
        "artifact": str(csv_path),
        "gci_artifact": str(gci_csv_path),
    }
    (output / f"{file_prefix}_sensitivity.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
