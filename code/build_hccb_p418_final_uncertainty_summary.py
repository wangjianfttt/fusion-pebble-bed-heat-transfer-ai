#!/usr/bin/env python3
"""Assemble numerical, training, packing and predictive uncertainty results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


DIFFUSION_METRICS = {
    "coverage": (
        "diffusion_unobserved_dynamic_solid_90pct_interval_coverage_fraction"
    ),
    "width": "diffusion_unobserved_dynamic_solid_90pct_interval_mean_width_K",
    "crps": "diffusion_unobserved_dynamic_solid_CRPS_K",
}


def load_json(path: Path, expected_status: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing formal result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != expected_status:
        raise ValueError(
            f"unexpected status in {path}: {payload.get('status')}; "
            f"expected {expected_status}"
        )
    return payload


def finite_nonnegative(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def gci_components(
    rows: list[dict],
    *,
    source_kind: str,
    quantity_keys: tuple[str, ...],
) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError(f"{source_kind} has no GCI rows")
    output = []
    finite_rows = []
    for row in rows:
        label = " / ".join(str(row[key]) for key in quantity_keys)
        value = row.get("fine_gci_fraction")
        status = str(row.get("convergence_status", ""))
        triplet = [
            row.get("coarse_value"),
            row.get("medium_value"),
            row.get("fine_value"),
        ]
        crosses_zero = all(item is not None for item in triplet) and (
            min(float(item) for item in triplet) <= 0.0
            <= max(float(item) for item in triplet)
        )
        if crosses_zero:
            value = None
            status = "zero_crossing_no_gci_reported"
        if value is None:
            output.append(
                {
                    "source_kind": source_kind,
                    "quantity": label,
                    "metric": "fine_GCI",
                    "value": "",
                    "unit": "fraction",
                    "result_status": status or "GCI_not_available",
                    "interpretation": (
                        "The three-resolution sequence is reported without forcing "
                        "a GCI value."
                    ),
                }
            )
            continue
        number = finite_nonnegative(value, f"{source_kind} {label} fine GCI")
        finite_rows.append((label, number, status))
        output.append(
            {
                "source_kind": source_kind,
                "quantity": label,
                "metric": "fine_GCI",
                "value": number,
                "unit": "fraction",
                "result_status": status,
                "interpretation": "Celik-type fine-resolution GCI.",
            }
        )
    if not finite_rows:
        headline = {
            "available_count": 0,
            "unavailable_count": len(rows),
            "largest_finite_fraction": None,
            "largest_finite_quantity": None,
        }
    else:
        label, number, _ = max(finite_rows, key=lambda item: item[1])
        headline = {
            "available_count": len(finite_rows),
            "unavailable_count": len(rows) - len(finite_rows),
            "largest_finite_fraction": number,
            "largest_finite_quantity": label,
        }
    return output, headline


def seed_components(
    rows: list[dict],
    *,
    source_kind: str,
    model_key: str,
    mean_key: str,
    standard_deviation_key: str,
    unit_key: str | None,
    fixed_unit: str | None,
) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError(f"{source_kind} has no seed-robustness rows")
    output = []
    largest = None
    for row in rows:
        unit = fixed_unit if fixed_unit is not None else str(row.get(unit_key or "", ""))
        mean = finite_nonnegative(row[mean_key], f"{source_kind} mean")
        standard_deviation = finite_nonnegative(
            row[standard_deviation_key], f"{source_kind} sample standard deviation"
        )
        model = str(row[model_key])
        metric = str(row["metric"])
        output.append(
            {
                "source_kind": source_kind,
                "quantity": f"{model} / {metric}",
                "metric": "three_seed_mean",
                "value": mean,
                "unit": unit,
                "result_status": "reported",
                "interpretation": "Mean over three independent training initializations.",
            }
        )
        output.append(
            {
                "source_kind": source_kind,
                "quantity": f"{model} / {metric}",
                "metric": "three_seed_sample_standard_deviation",
                "value": standard_deviation,
                "unit": unit,
                "result_status": "reported",
                "interpretation": (
                    "Training variability with unchanged fields, split and normalization."
                ),
            }
        )
        if largest is None or standard_deviation > largest["sample_standard_deviation"]:
            largest = {
                "model": model,
                "metric": metric,
                "sample_standard_deviation": standard_deviation,
                "unit": unit,
            }
    return output, largest


def fully_coupled_scope_components(scope: dict) -> tuple[list[dict], dict]:
    records = [
        row
        for row in scope.get("records", [])
        if str(row.get("filename", "")).startswith("maxCo_")
    ]
    if len(records) != 3:
        raise ValueError("scope summary must contain the three formal maxCo runs")
    output = []
    stop_times = []
    max_co_values = []
    for row in records:
        meaning = str(row.get("scientific_meaning", ""))
        time_match = re.search(r"stopped at ([0-9.eE+-]+) s", meaning)
        co_match = re.search(r"maxCo=([0-9.]+)", meaning)
        if not time_match or not co_match:
            raise ValueError("fully coupled scope record lacks maxCo or stop time")
        stop_time = finite_nonnegative(time_match.group(1), "fully coupled stop time")
        max_co = finite_nonnegative(co_match.group(1), "fully coupled maxCo")
        if row.get("status") != "failed_solver_exit_propagated":
            raise ValueError("formal fully coupled record is not the retained failure")
        stop_times.append(stop_time)
        max_co_values.append(max_co)
        output.append(
            {
                "source_kind": "fully_coupled_applicability",
                "quantity": f"maxCo={max_co:g}",
                "metric": "property_range_stop_time",
                "value": stop_time,
                "unit": "s",
                "result_status": "outside_registered_helium_property_range",
                "interpretation": (
                    "The startup left the registered helium-property range; no "
                    "time-step convergence value is reported."
                ),
            }
        )
    return output, {
        "run_count": len(records),
        "max_co_values": sorted(max_co_values),
        "earliest_stop_s": min(stop_times),
        "latest_stop_s": max(stop_times),
        "gci_reported": False,
        "conclusion": "outside_registered_helium_property_range",
    }


def packing_components(packing: dict) -> tuple[list[dict], dict]:
    if packing.get("complete_nine_case_comparison") is not True:
        raise ValueError("independent-packing comparison is incomplete")
    if int(packing.get("accepted_common_case_count", -1)) != 9:
        raise ValueError("independent-packing comparison must contain nine cases")
    if int(packing.get("failed_seed202_case_count", -1)) != 0:
        raise ValueError("independent-packing comparison contains failed cases")
    metrics = packing.get("metric_summary", {})
    required = (
        "outlet_temperature_K",
        "maximum_solid_temperature_K",
        "pressure_drop_Pa",
    )
    if set(required) - set(metrics):
        raise ValueError("independent-packing summary lacks a required metric")
    output = []
    maximum_changes = {}
    for metric in required:
        values = metrics[metric]
        for key, label in (
            ("maximum_absolute_relative_change_percent", "maximum_absolute_change"),
            ("mean_absolute_relative_change_percent", "mean_absolute_change"),
            ("median_absolute_relative_change_percent", "median_absolute_change"),
        ):
            number = finite_nonnegative(values[key], f"packing {metric} {key}")
            output.append(
                {
                    "source_kind": "packing_realization",
                    "quantity": metric,
                    "metric": label,
                    "value": number,
                    "unit": "%",
                    "result_status": "reported",
                    "interpretation": (
                        "Direct change across nine matched seed101 and seed202 CHT "
                        "calculations."
                    ),
                }
            )
            if key == "maximum_absolute_relative_change_percent":
                maximum_changes[metric] = number
    largest_metric = max(maximum_changes, key=maximum_changes.get)
    return output, {
        "case_count": 9,
        "maximum_absolute_relative_change_percent": maximum_changes,
        "largest_metric": largest_metric,
        "largest_change_percent": maximum_changes[largest_metric],
    }


def load_diffusion_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"missing transient metric table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row.get("split_name") == "pair_disjoint_stress_test"
        and row.get("model") == "diffusion_residual_correction"
        and row.get("data_role") == "test"
    ]
    output = {}
    for key, metric in DIFFUSION_METRICS.items():
        matches = [row for row in selected if row.get("metric") == metric]
        if len(matches) != 1:
            raise ValueError(
                f"expected one strict-split diffusion metric {metric}, found {len(matches)}"
            )
        output[key] = finite_nonnegative(matches[0]["value"], metric)
    if output["coverage"] > 1.0:
        raise ValueError("diffusion interval coverage must be a fraction")
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 1000.0:
        return f"{value:.2e}"
    return f"{value:.3g}"


def tex_escape(value: object) -> str:
    return str(value).replace("_", "\\_")


def gci_tex(headline: dict) -> str:
    value = headline["largest_finite_fraction"]
    if value is None:
        return (
            f"no forced GCI ({headline['unavailable_count']} unresolved "
            "three-resolution triplets)"
        )
    unresolved = headline["unavailable_count"]
    suffix = f"; {unresolved} unresolved" if unresolved else ""
    return (
        f"{100.0 * value:.2f}\\% for "
        f"\\texttt{{{tex_escape(headline['largest_finite_quantity'])}}}{suffix}"
    )


def gci_plain(headline: dict) -> str:
    value = headline["largest_finite_fraction"]
    if value is None:
        return (
            f"没有强行给出GCI（{headline['unavailable_count']}组"
            "三分辨率结果无法可靠计算GCI）"
        )
    unresolved = headline["unavailable_count"]
    suffix = f"，另有{unresolved}组无法可靠计算GCI" if unresolved else ""
    return (
        f"最大的可计算细分辨率GCI为{100.0 * value:.2f}%"
        f"（{headline['largest_finite_quantity']}）{suffix}"
    )


def _write_tex_table_legacy(path: Path, headline: dict) -> None:
    """Retained only for old bundles; the current paper uses ``write_tex``."""
    steady = headline["steady_training_seeds"]
    transient = headline["transient_training_seeds"]
    diffusion = headline["diffusion_ensemble"]
    packing = headline["packing_realization"]
    lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\caption{Separate contributions to numerical and model uncertainty. "
        "The entries are not combined into one scalar because they describe different "
        "physical quantities and statistical sources. The largest value within each "
        "numerical-resolution family is shown conservatively; complete rows are supplied "
        "in the generated data table.}",
        "\\label{tab:final_uncertainty}",
        "\\begin{tabular}{p{0.22\\textwidth}p{0.30\\textwidth}p{0.40\\textwidth}}",
        "\\toprule",
        "Source & Reported result & Interpretation \\\\",
        "\\midrule",
        f"Spatial discretization & {gci_tex(headline['mesh'])} & "
        "Fine-grid GCI for monotonically converging engineering quantities; oscillatory "
        "triplets are retained without an invented GCI. \\\\",
        f"Fixed-flow time step & {gci_tex(headline['fixed_timestep'])} & "
        "Thermal response with the hydrodynamic field fixed. \\\\",
        "Fully coupled startup & not reported & "
        "Flow and heat transfer advanced together. \\\\",
        (
            "Steady network initialization & "
            f"largest sample standard deviation {fmt(steady['sample_standard_deviation'])} "
            f"{steady['unit']} ({tex_escape(steady['model'])}, "
            f"{tex_escape(steady['metric'])}) & "
            "Three independent training seeds with identical fields and data split. \\\\"
        ),
        (
            "Transient network initialization & "
            f"largest sample standard deviation {fmt(transient['sample_standard_deviation'])} "
            f"K ({tex_escape(transient['model'])}) & "
            "Three independent training seeds on the strict complete-trajectory split. \\\\"
        ),
        (
            "Pebble arrangement & "
            "direct seed101/seed202 matched-condition changes & "
            "The architecture and weights are frozen before the third packing is read. \\\\"
        ),
        (
            "Diffusion prediction interval & "
            f"coverage {100.0 * diffusion['coverage']:.1f}\\%, mean width "
            f"{fmt(diffusion['width_K'])} K, CRPS {fmt(diffusion['crps_K'])} K & "
            "Independent unobserved solid-temperature samples on the strict split. \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tex(path: Path, headline: dict) -> None:
    """Write one compact main-text paragraph instead of a supplementary table."""
    steady = headline["steady_training_seeds"]
    transient = headline["transient_training_seeds"]
    diffusion = headline["diffusion_ensemble"]
    packing = headline["packing_realization"]
    coupled = headline["fully_coupled_scope"]
    changes = packing["maximum_absolute_relative_change_percent"]
    lines = [
        "\\paragraph{Numerical and model sensitivity.}",
        "The separate checks are not collapsed into one scalar. The largest available "
        f"fine-grid GCI is {gci_tex(headline['mesh'])}, and the corresponding fixed-flow "
        f"time-step result is {gci_tex(headline['fixed_timestep'])}. No fully coupled GCI "
        f"is reported: all {coupled['run_count']} registered maxCo starts left the helium-"
        f"property range between {1000.0 * coupled['earliest_stop_s']:.3f} and "
        f"{1000.0 * coupled['latest_stop_s']:.3f} ms. Across three initializations, the "
        f"largest sample standard deviations are {fmt(steady['sample_standard_deviation'])} "
        f"{steady['unit']} for the steady models and "
        f"{fmt(transient['sample_standard_deviation'])} K for the transient models. "
        f"The independent packing changes outlet and maximum-solid temperatures by at "
        f"most {fmt(changes['outlet_temperature_K'])}\\% and "
        f"{fmt(changes['maximum_solid_temperature_K'])}\\%, respectively, but pressure "
        f"drop by {fmt(changes['pressure_drop_Pa'])}\\%. For the diffusion refiner, the "
        f"strict-split 90\\% interval has {100.0 * diffusion['coverage']:.1f}\\% coverage, "
        f"mean width {fmt(diffusion['width_K'])} K and CRPS "
        f"{fmt(diffusion['crps_K'])} K.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chinese(path: Path, headline: dict, external: dict) -> None:
    diffusion = headline["diffusion_ensemble"]
    lines = [
        "# P418最终结果中的误差和波动怎样理解",
        "",
        "本项目没有把不同单位的误差硬合并成一个“总不确定度”。网格、时间步、网络重复训练、颗粒装填变化和扩散模型区间回答的是不同问题，应分别报告。",
        "",
        "## 数值离散",
        "",
        f"- 三套网格：{gci_plain(headline['mesh'])}。",
        f"- 固定流场时间步：{gci_plain(headline['fixed_timestep'])}。",
        (
            f"- 全耦合启动：{headline['fully_coupled_scope']['run_count']}组maxCo计算"
            f"均在{1000.0 * headline['fully_coupled_scope']['earliest_stop_s']:.3f}--"
            f"{1000.0 * headline['fully_coupled_scope']['latest_stop_s']:.3f} ms离开已登记"
            "氦物性范围，因此不报告全耦合GCI。"
        ),
        "- 如果三组结果出现振荡变化，程序只报告“无法可靠给出GCI”，不会强行算一个百分数。",
        "",
        "## 网络重复训练",
        "",
        f"- 稳态模型中最大的三随机种子标准差为 {fmt(headline['steady_training_seeds']['sample_standard_deviation'])} {headline['steady_training_seeds']['unit']}。",
        f"- 瞬态模型中最大的三随机种子标准差为 {fmt(headline['transient_training_seeds']['sample_standard_deviation'])} K。",
        "",
        "## 颗粒装填与概率预测",
        "",
        (
            "- seed101与seed202九个相同工况直接计算表明：出口温度、"
            f"最高固体温度和压降的最大绝对相对变化分别为 "
            f"{fmt(headline['packing_realization']['maximum_absolute_relative_change_percent']['outlet_temperature_K'])}%、"
            f"{fmt(headline['packing_realization']['maximum_absolute_relative_change_percent']['maximum_solid_temperature_K'])}%和"
            f"{fmt(headline['packing_realization']['maximum_absolute_relative_change_percent']['pressure_drop_Pa'])}%。"
        ),
        f"- 扩散模型90%区间在未观测固相温度上的覆盖率为 {100.0 * diffusion['coverage']:.1f}%，平均区间宽度为 {fmt(diffusion['width_K'])} K，CRPS为 {fmt(diffusion['crps_K'])} K。",
        "",
        "## 为什么没有随意做“所有参数±5%”",
        "",
        "P418正式计算所用的颗粒尺寸、孔隙率、材料热物性和边界条件都有文献来源，但这些来源没有给出一套彼此一致、可以直接抽样的概率分布。随意规定统一的±5%会制造没有文献根据的新参数。因此本文只传播有实际计算依据的网格、时间步、训练种子、装填和概率预测波动。材料参数的概率传播要等具体实验批次给出测量均值、标准差和相关性后再做。",
        "",
        "## 外部实验",
        "",
        f"- 外部比较共保留 {external['comparison_count']} 组：PREMUX内部温度、TESOMEX瞬态温度、HELOKA换热和1 mm固定床压降。",
        "- 它们只用于检查温度、换热和压降量级，不参加P418网络训练，也不被混入上述数值误差。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    mesh = load_json(
        args.mesh_summary.resolve(), "completed_three_mesh_p418_cht_comparison"
    )
    fixed = load_json(
        args.fixed_timestep_summary.resolve(),
        "completed_p418_thermal_timestep_sensitivity",
    )
    coupled_scope = load_json(
        args.scope_limit_summary.resolve(),
        "P418_SCOPE_LIMITS_EVIDENCE_SYNCED",
    )
    steady = load_json(
        args.steady_seed_summary.resolve(),
        "completed_p418_main_steady_split_seed_robustness",
    )
    transient = load_json(
        args.transient_seed_summary.resolve(),
        "completed_p418_strict_split_seed_robustness",
    )
    packing = load_json(
        args.cross_packing_summary.resolve(),
        "completed_seed101_seed202_integral_response_comparison",
    )
    external = load_json(
        args.external_summary.resolve(), "external_thermal_hydraulic_comparison_complete"
    )
    if external.get("use_in_p418_training") is not False:
        raise ValueError("external comparisons must remain outside P418 training")

    components = []
    rows, mesh_head = gci_components(
        mesh["grid_convergence"],
        source_kind="spatial_mesh",
        quantity_keys=("metric",),
    )
    components.extend(rows)
    rows, fixed_head = gci_components(
        fixed["gci_results"],
        source_kind="fixed_flow_timestep",
        quantity_keys=("signal", "quantity"),
    )
    components.extend(rows)
    rows, coupled_head = fully_coupled_scope_components(coupled_scope)
    components.extend(rows)
    rows, steady_head = seed_components(
        steady["metrics"],
        source_kind="steady_training_seed",
        model_key="architecture",
        mean_key="mean",
        standard_deviation_key="sample_std",
        unit_key="unit",
        fixed_unit=None,
    )
    components.extend(rows)
    rows, transient_head = seed_components(
        transient["metrics"],
        source_kind="transient_training_seed",
        model_key="model",
        mean_key="mean_K",
        standard_deviation_key="sample_std_K",
        unit_key=None,
        fixed_unit="K",
    )
    components.extend(rows)

    rows, packing_head = packing_components(packing)
    components.extend(rows)

    diffusion = load_diffusion_metrics(args.transient_metrics.resolve())
    for key, unit in (("coverage", "fraction"), ("width", "K"), ("crps", "K")):
        components.append(
            {
                "source_kind": "diffusion_predictive_interval",
                "quantity": "unobserved_dynamic_solid_temperature",
                "metric": key,
                "value": diffusion[key],
                "unit": unit,
                "result_status": "reported",
                "interpretation": (
                    "Strict-split independent prediction; the interval is not calibrated "
                    "with repeated experimental trajectories."
                ),
            }
        )

    external_metrics_path = args.external_metrics.resolve()
    with external_metrics_path.open(newline="", encoding="utf-8") as handle:
        external_rows = list(csv.DictReader(handle))
    if len(external_rows) != 5:
        raise ValueError("external comparison table must retain the five declared rows")
    external_head = {
        "comparison_count": len({row["experiment"] for row in external_rows}),
        "metric_row_count": len(external_rows),
        "used_in_training": False,
    }
    headline = {
        "mesh": mesh_head,
        "fixed_timestep": fixed_head,
        "fully_coupled_scope": coupled_head,
        "steady_training_seeds": steady_head,
        "transient_training_seeds": transient_head,
        "packing_realization": packing_head,
        "diffusion_ensemble": {
            "coverage": diffusion["coverage"],
            "width_K": diffusion["width"],
            "crps_K": diffusion["crps"],
        },
        "external_comparisons": external_head,
    }
    summary = {
        "status": "completed_p418_final_uncertainty_summary",
        "headline_results": headline,
        "component_count": len(components),
        "component_table": str(
            (args.output_dir.resolve() / "uncertainty_components.csv")
        ),
        "external_comparison_metrics": str(external_metrics_path),
        "material_parameter_probability_propagation": {
            "performed": False,
            "reason": (
                "The source-backed P418 parameter set does not provide a consistent "
                "joint probability distribution. No arbitrary common percentage range "
                "is introduced."
            ),
        },
        "combination_rule": (
            "Spatial, temporal, training, packing and predictive components are reported "
            "separately because their units and physical meanings differ."
        ),
        "new_physical_parameters": [],
    }
    return summary, components


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-summary", type=Path, required=True)
    parser.add_argument("--fixed-timestep-summary", type=Path, required=True)
    parser.add_argument("--scope-limit-summary", type=Path, required=True)
    parser.add_argument("--steady-seed-summary", type=Path, required=True)
    parser.add_argument("--transient-seed-summary", type=Path, required=True)
    parser.add_argument("--transient-metrics", type=Path, required=True)
    parser.add_argument("--cross-packing-summary", type=Path, required=True)
    parser.add_argument("--external-summary", type=Path, required=True)
    parser.add_argument("--external-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    args = parser.parse_args()

    summary, components = build_summary(args)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "uncertainty_components.csv", components)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_tex(args.tex_output.resolve(), summary["headline_results"])
    write_chinese(
        output / "P418_不确定性结果怎样理解_CN.md",
        summary["headline_results"],
        summary["headline_results"]["external_comparisons"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
