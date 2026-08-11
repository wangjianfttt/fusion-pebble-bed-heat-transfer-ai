#!/usr/bin/env python3
"""Verify the shared P418 steady/transient model-comparison protocol."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_comparison_contract import STEADY_METRIC_CONTRACT  # noqa: E402
from summarize_hccb_p418_60_model_comparison import (  # noqa: E402
    ARCHITECTURES as STEADY_ARCHITECTURES,
    SPLITS as STEADY_SPLITS,
)
from summarize_hccb_p418_step_model_comparison import (  # noqa: E402
    COMMON_HOTSPOT_METRICS,
    PRIMARY_ENERGY_METRIC,
    SPLITS as TRANSIENT_SPLITS,
)


TRANSIENT_MODELS = (
    "dmdc",
    "graph_transformer_data_only",
    "graph_transformer_energy_flux",
    "graph_transformer_factorized_energy_flux",
    "low_rank_residual_correction",
    "diffusion_residual_correction",
)

FORMAL_JOB_MANIFEST = (
    "results/hccb_p418_public_data_release_preflight/formal_training_manifest_public.json"
)
CANONICAL_SPLIT_FILE = "parameters/hccb_p418_step_response_splits.json"
CANONICAL_OBSERVABLE_DATA = (
    "results/hccb_p418_physical_steps_12/hccb_p418_transient_observables.npz"
)
CANONICAL_REGIONAL_DATA = (
    "results/hccb_p418_physical_steps_12/regional_sequences/dataset_index.json"
)


def load_json(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_exact_partition(
    *, available: set[str], split_name: str, roles: dict[str, list[str]]
) -> None:
    seen: set[str] = set()
    for role in ("train", "validation", "test"):
        values = [str(value) for value in roles.get(role, [])]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{split_name} has an empty or repeated {role} list")
        overlap = seen.intersection(values)
        if overlap:
            raise ValueError(f"{split_name} reuses entries across roles: {sorted(overlap)}")
        seen.update(values)
    if seen != available:
        raise ValueError(
            f"{split_name} does not partition the declared data: "
            f"missing={sorted(available-seen)}, extra={sorted(seen-available)}"
        )


def command_options(command: str) -> tuple[str, dict[str, str | bool]]:
    tokens = shlex.split(command)
    if len(tokens) < 2:
        raise ValueError(f"invalid formal command: {command}")
    options: dict[str, str | bool] = {}
    index = 2
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            options[token] = tokens[index + 1]
            index += 2
        else:
            options[token] = True
            index += 1
    return Path(tokens[1]).name, options


def ends_with(value: object, relative: str) -> bool:
    return str(value).replace("\\", "/").endswith(relative)


def verify_formal_job_fairness(split_names: tuple[str, ...]) -> dict[str, object]:
    manifest = load_json(FORMAL_JOB_MANIFEST)
    jobs = manifest.get("jobs", [])
    if len(jobs) != int(manifest.get("job_count", -1)) or not jobs:
        raise ValueError("formal training manifest job count is inconsistent")
    by_id = {str(job["job_id"]): job for job in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("formal training manifest repeats job IDs")

    model_stages = {"independent_training", "random_seed_repeat", "dependent_correction"}
    model_jobs = [job for job in jobs if job["stage"] in model_stages]
    direct_split_jobs = 0
    inherited_split_jobs = 0
    for job in model_jobs:
        job_id = str(job["job_id"])
        split_name = str(job.get("split_name"))
        if split_name not in split_names:
            raise ValueError(f"{job_id} uses an undeclared split: {split_name}")
        _, options = command_options(str(job["command"]))
        if job["stage"] == "dependent_correction":
            dependencies = [str(value) for value in job.get("depends_on", [])]
            if len(dependencies) != 1 or dependencies[0] not in by_id:
                raise ValueError(f"{job_id} must inherit one registered upstream model")
            upstream = by_id[dependencies[0]]
            if upstream.get("split_name") != split_name or upstream.get("seed") != job.get("seed"):
                raise ValueError(f"{job_id} does not inherit its upstream split and seed")
            if str(options.get("--prediction-dir")) != str(upstream["output_dir"]):
                raise ValueError(f"{job_id} prediction directory differs from its dependency")
            if "--split-name" in options and options["--split-name"] != split_name:
                raise ValueError(f"{job_id} command split differs from its metadata")
            inherited_split_jobs += 1
            continue

        if options.get("--split-name") != split_name:
            raise ValueError(f"{job_id} command split differs from its metadata")
        if not ends_with(options.get("--splits"), CANONICAL_SPLIT_FILE):
            raise ValueError(f"{job_id} does not use the common complete-curve split file")
        data_ok = ends_with(options.get("--data"), CANONICAL_OBSERVABLE_DATA) or ends_with(
            options.get("--dataset-index"), CANONICAL_REGIONAL_DATA
        )
        if not data_ok:
            raise ValueError(f"{job_id} does not use a registered 12-curve data source")
        if job.get("seed") is not None and str(options.get("--seed")) != str(job["seed"]):
            raise ValueError(f"{job_id} command seed differs from its metadata")
        direct_split_jobs += 1

    energy_jobs = [job for job in jobs if job["stage"] == "energy_evaluation"]
    for job in energy_jobs:
        dependencies = [str(value) for value in job.get("depends_on", [])]
        if len(dependencies) != 1 or dependencies[0] not in by_id:
            raise ValueError(f"{job['job_id']} lacks one registered model dependency")
        upstream = by_id[dependencies[0]]
        _, options = command_options(str(job["command"]))
        if str(options.get("--model-summary")) != str(upstream["completion_file"]):
            raise ValueError(f"{job['job_id']} evaluates a different model summary")
        if not ends_with(options.get("--dataset-index"), CANONICAL_REGIONAL_DATA):
            raise ValueError(f"{job['job_id']} does not use the common regional data")

    source_requirements = {
        "code/train_hccb_p418_transient_observable_transformer.py": (
            "train_idx = split[\"train\"]",
            "target_train = targets[train_idx][train_mask]",
            "maximum_time = float(np.nanmax(time_s[train_idx]))",
        ),
        "code/train_hccb_p418_spatiotemporal_regional_operator.py": (
            "training_statistics(",
            "split[\"train\"]",
            "independent_test_read",
            "test_evaluated",
        ),
        "code/train_hccb_p418_regional_dmdc.py": (
            "training = [load_sequence(root, source_records[value]) for value in split[\"train\"]]",
        ),
        "code/train_hccb_p418_observable_dmdc.py": ("split[\"train\"]",),
        "code/train_hccb_p418_low_rank_temperature_residual.py": ('data["train"]',),
        "code/train_hccb_p418_temporal_temperature_diffusion.py": ('splits["train"]',),
    }
    checked_sources = 0
    for relative, snippets in source_requirements.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        missing = [snippet for snippet in snippets if snippet not in text]
        if missing:
            raise ValueError(f"{relative} no longer proves train-only fitting: {missing}")
        checked_sources += 1

    return {
        "formal_manifest_job_count": len(jobs),
        "direct_common_split_job_count": direct_split_jobs,
        "upstream_inherited_split_job_count": inherited_split_jobs,
        "common_energy_evaluation_job_count": len(energy_jobs),
        "train_only_source_program_count": checked_sources,
    }


def verify(protocol_path: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.resolve().read_text(encoding="utf-8"))
    if protocol["physical_parameter_rule"].get("new_physical_parameters") != []:
        raise ValueError("comparison protocol introduces new physical parameters")

    for relative in (
        protocol["physical_parameter_rule"]["registry"],
        protocol["physical_parameter_rule"]["evidence_registry"],
        protocol["physical_parameter_rule"]["equation_map"],
        protocol["steady_comparison"]["split_file"],
        protocol["physical_transient_comparison"]["plan_file"],
        protocol["physical_transient_comparison"]["split_file"],
        protocol["numerical_resolution_checks"]["thermal_time_step_file"],
        protocol["packing_generalization"]["plan_file"],
        protocol["verification_script"],
    ):
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)

    with (ROOT / protocol["physical_parameter_rule"]["registry"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        parameter_rows = list(csv.DictReader(handle))
    if len(parameter_rows) != 22:
        raise ValueError(f"expected 22 physical parameters, found {len(parameter_rows)}")

    expected_channels = ["Ux_m_s", "Uy_m_s", "Uz_m_s", "pressure_Pa", "temperature_K"]
    if protocol["common_quantities"]["state_channel_order"] != expected_channels:
        raise ValueError("common state-channel order differs from the fused model")

    steady = protocol["steady_comparison"]
    steady_splits = load_json(steady["split_file"])
    conditions = {str(item["condition_id"]) for item in steady_splits["conditions"]}
    if len(conditions) != int(steady["required_condition_count"]) or len(conditions) != 60:
        raise ValueError("steady comparison does not contain the complete 60-condition matrix")
    if tuple(steady["models"]) != tuple(STEADY_ARCHITECTURES):
        raise ValueError("steady model list differs from the formal summary program")
    if tuple(steady["split_names"]) != tuple(STEADY_SPLITS):
        raise ValueError("steady split list differs from the formal summary program")
    for name in steady["split_names"]:
        require_exact_partition(
            available=conditions,
            split_name=name,
            roles=steady_splits["splits"][name],
        )
    engineering = set(steady["engineering_metrics"])
    registered_engineering = set(STEADY_METRIC_CONTRACT["engineering_errors"])
    if engineering != registered_engineering:
        raise ValueError("steady engineering metrics differ from the implemented definitions")

    transient = protocol["physical_transient_comparison"]
    step_plan = load_json(transient["plan_file"])
    sequence_ids = {str(item["sequence_id"]) for item in step_plan["sequences"]}
    if len(sequence_ids) != int(transient["required_sequence_count"]) or len(sequence_ids) != 12:
        raise ValueError("transient comparison does not contain the 12 declared step curves")
    if step_plan.get("new_physical_parameters") != []:
        raise ValueError("physical-step plan introduces new physical parameters")
    if tuple(transient["models"]) != TRANSIENT_MODELS:
        raise ValueError("transient model list differs from the formal summary program")
    if tuple(transient["split_names"]) != tuple(TRANSIENT_SPLITS):
        raise ValueError("transient split list differs from the formal summary program")
    step_splits = load_json(transient["split_file"])["splits"]
    for name in transient["split_names"]:
        require_exact_partition(available=sequence_ids, split_name=name, roles=step_splits[name])
    if transient["physical_consistency_metric"] != PRIMARY_ENERGY_METRIC:
        raise ValueError("transient energy metric differs from the common implementation")
    if not set(transient["hotspot_metrics"]).issubset(COMMON_HOTSPOT_METRICS):
        raise ValueError("protocol contains an unimplemented hotspot metric")

    time_step = load_json(protocol["numerical_resolution_checks"]["thermal_time_step_file"])
    if len(time_step["delta_t_s"]) != 3 or time_step.get("new_physical_parameters") != []:
        raise ValueError("time-step comparison must retain the three declared schedules")

    packing = load_json(protocol["packing_generalization"]["plan_file"])
    by_seed = {int(item["seed"]): item for item in packing["packing_realisations"]}
    if set(by_seed) != {101, 202, 303}:
        raise ValueError("packing comparison must contain seed101, seed202 and seed303")
    if int(packing["screening_design"]["case_count_per_new_packing"]) != 9:
        raise ValueError("each independent packing must retain the declared nine cases")
    if packing.get("new_physical_parameter_values_added") != []:
        raise ValueError("packing comparison introduces new physical parameters")

    architecture_registry = load_json(steady["repeat_rule_source"])
    repeat = architecture_registry["steady_training_repeat_rule"]
    transient_repeat = architecture_registry["training_repeat_rule"]
    if repeat["seeds"] != [20260717, 20260718, 20260719]:
        raise ValueError("steady repeated-initialization seeds changed")
    if transient_repeat["strict_split"] != "pair_disjoint_stress_test":
        raise ValueError("transient repeat rule no longer uses the strict split")

    fairness = verify_formal_job_fairness(tuple(transient["split_names"]))

    return {
        "status": "p418_common_model_comparison_protocol_verified",
        "physical_parameter_count": len(parameter_rows),
        "steady_condition_count": len(conditions),
        "steady_model_count": len(steady["models"]),
        "steady_split_count": len(steady["split_names"]),
        "transient_sequence_count": len(sequence_ids),
        "transient_output_time_count": int(transient["required_output_time_count"]),
        "transient_model_count": len(transient["models"]),
        "transient_split_count": len(transient["split_names"]),
        "packing_seeds": sorted(by_seed),
        "independent_packing_case_count_each": 9,
        "steady_repeat_seed_count": len(repeat["seeds"]),
        "transient_repeat_seed_count": len(transient_repeat["strict_split_seeds"]),
        **fairness,
        "same_physical_inputs_for_all_models": True,
        "train_only_normalization": True,
        "complete_curve_splitting": True,
        "separate_temperature_heat_transfer_conservation_and_cost_metrics": True,
        "new_physical_parameters": []
    }


def write_chinese_summary(summary: dict[str, object], output: Path) -> None:
    text = f"""# P418球床换热模型统一比较方案

## 比较对象

- 稳态：响应面、纯数据坐标网络、带质量/能量方程的PINN、区域图神经算子和Transolver，共`{summary['steady_model_count']}`类方法。
- 瞬态：DMDc、纯数据图--Transformer、带能量和面热流约束的图--Transformer、因子化图--Transformer、POD低秩修正和扩散剩余误差修正，共`{summary['transient_model_count']}`类方法。
- 三维数据：稳态使用完整`{summary['steady_condition_count']}`工况；瞬态使用`{summary['transient_sequence_count']}`条物理热阶跃，每条保存`{summary['transient_output_time_count']}`个三维时刻。

## 怎样保证比较公平

1. 所有方法读取相同的入口速度、入口温度、颗粒发热率、出口压力和冷却壁温度，物理量全部来自已经登记的`{summary['physical_parameter_count']}`项文献参数。
2. 稳态五种工况划分和瞬态三种整曲线划分对所有模型完全相同；归一化只使用训练工况或训练曲线。
3. 同一条热阶跃的不同时间点不会被拆到训练和测试两边。
4. 稳态同时比较流体/颗粒温度场、压降、出口温度、颗粒最高温度、冷却壁换热、流固界面换热、质量和能量收支以及计算时间。
5. 瞬态同时比较流体/颗粒温度、最高温度过程、热点位置、共同能量方程误差以及计算时间。
6. 不把这些量压成一个人为总分。某个模型即使温度RMSE最低，如果壁面换热或能量收支变差，也必须分别写出来。

## 扩散模型怎样判断是否有用

扩散模型只修正物理约束图--Transformer留下的温度误差。它的模型大小、训练时间和预测时间必须包含上游图--Transformer。测试曲线上只有在颗粒温度误差降低、同时共同能量方程误差不增加时，才说明扩散修正确实同时改善了温度和物理一致性；其他结果也照常保留，不能删去。

## 不同颗粒排列

seed101用于60工况训练，seed202用于比较模型结构，seed303用于固定模型后的零样本预测。seed303共有`{summary['independent_packing_case_count_each']}`个文献工况，在读取其三维结果前必须冻结模型结构、权重和数据处理方式。

## 当前能说到哪一步

这份方案和程序检查已经完成，但正式准确率仍必须等待60个稳态工况和12条物理热阶跃全部算完。一轮或少量工况结果只说明程序能运行，不能作为论文中的模型性能结论。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "parameters/hccb_p418_model_comparison_protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/hccb_p418_model_comparison_protocol/summary.json",
    )
    parser.add_argument(
        "--chinese-summary",
        type=Path,
        default=ROOT / "parameters/HCCB_P418_MODEL_COMPARISON_PROTOCOL_CN.md",
    )
    args = parser.parse_args()
    summary = verify(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_chinese_summary(summary, args.chinese_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
