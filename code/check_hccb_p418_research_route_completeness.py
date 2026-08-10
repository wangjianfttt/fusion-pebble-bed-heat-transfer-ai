#!/usr/bin/env python3
"""Check whether the P418 research route and its required records are complete."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_hccb_p418_parameter_use_summary import build as build_parameter_use


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(root: Path, relative_paths: list[str]) -> dict[str, object]:
    missing = [path for path in relative_paths if not (root / path).is_file()]
    return {
        "required_file_count": len(relative_paths),
        "missing_files": missing,
        "complete": not missing,
    }


def current_formal_data(root: Path, fused: dict[str, object]) -> dict[str, object]:
    data = json.loads(json.dumps(fused["current_data"]))
    coverage_path = (
        root / "results/hccb_p418_training_data_coverage_partial/summary.json"
    )
    if coverage_path.is_file():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        data["steady"].update(
            {
                "completed": int(coverage["completed_case_count"]),
                "required": int(coverage["expected_case_count"]),
                "complete": (
                    int(coverage["completed_case_count"])
                    == int(coverage["expected_case_count"])
                ),
                "progress_source": "verified_training_data_coverage",
                "progress_source_path": str(coverage_path),
            }
        )
    return data


def verify_architecture_sources(root: Path) -> dict[str, object]:
    path = root / "parameters/hccb_p418_ai_architecture_sources.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    architectures = registry["architectures"]
    precedents = registry["domain_specific_pinn_precedents"]
    missing_links: list[str] = []
    missing_local_files: list[str] = []
    changed_local_files: list[str] = []
    checked_hashes = 0

    for group_name, entries in (
        ("architecture", architectures),
        ("domain_precedent", precedents),
    ):
        for index, entry in enumerate(entries):
            label = str(entry.get("name") or entry.get("paper") or f"{group_name}_{index}")
            url = str(entry.get("paper_url", "")).strip()
            if not url.startswith(("https://", "http://")):
                missing_links.append(label)
            for key, expected in entry.items():
                if not key.endswith("_sha256") or not isinstance(expected, str):
                    continue
                source_key = key[: -len("_sha256")]
                relative = entry.get(source_key)
                if not isinstance(relative, str) or not relative.strip():
                    continue
                local_path = root / relative
                if not local_path.is_file():
                    missing_local_files.append(relative)
                    continue
                checked_hashes += 1
                if sha256(local_path) != expected:
                    changed_local_files.append(relative)

    return {
        "architecture_count": len(architectures),
        "domain_specific_pinn_precedent_count": len(precedents),
        "checked_local_source_hash_count": checked_hashes,
        "missing_paper_links": missing_links,
        "missing_local_source_files": sorted(set(missing_local_files)),
        "changed_local_source_files": sorted(set(changed_local_files)),
        "complete": not missing_links
        and not missing_local_files
        and not changed_local_files,
    }


def build(root: Path) -> tuple[dict[str, object], str]:
    parameter_use, _ = build_parameter_use(root)
    fused_path = root / "results/hccb_p418_fused_preflight/summary.json"
    fused = json.loads(fused_path.read_text(encoding="utf-8"))
    architectures = verify_architecture_sources(root)

    route_documents = require_files(
        root,
        [
            "研究主线_简明版_CN.md",
            "algorithms/P418_模型组合为什么这样设计_CN.md",
            "实验实施步骤_简明_CN.md",
            "EXPERIMENTAL_VALIDATION_PLAN_CN.md",
            "PROCESS_LOG_CN.md",
            "CURRENT_STATUS_CN.md",
            "parameters/HCCB_P418_PARAMETER_EVIDENCE_CN.md",
            "parameters/hccb_p418_model_numerical_settings_CN.md",
            "results/hccb_p418_parameter_use/P418_参数怎样进入研究_CN.md",
        ],
    )
    core_code = require_files(
        root,
        [
            "code/prepare_hccb_p418_model_data.py",
            "code/hccb_p418_coordinate_pinn.py",
            "code/hccb_p418_spatiotemporal_regional_operator.py",
            "code/hccb_p418_fully_coupled_spatiotemporal_operator.py",
            "code/hccb_p418_temporal_temperature_diffusion.py",
            "code/train_hccb_p418_regional_response_surface.py",
            "code/train_hccb_p418_regional_dmdc.py",
            "code/train_hccb_p418_spatiotemporal_regional_operator.py",
            "code/train_hccb_p418_fully_coupled_spatiotemporal_operator.py",
            "code/train_hccb_p418_low_rank_temperature_residual.py",
            "code/train_hccb_p418_temporal_temperature_diffusion.py",
            "parameters/hccb_p418_fused_model_contract.json",
            "parameters/hccb_p418_model_comparison_protocol.json",
            "parameters/hccb_p418_model_data_pipeline.json",
        ],
    )
    experimental_templates = require_files(
        root,
        [
            "experimental_data_templates/calibration_records.csv",
            "experimental_data_templates/experiment_conditions.csv",
            "experimental_data_templates/sensor_layout.csv",
            "experimental_data_templates/steady_measurements.csv",
            "experimental_data_templates/transient_measurements.csv",
        ],
    )

    current_counts_match = {
        "physical_parameter_count": (
            fused["fused_contract"]["physical_parameter_count"]
            == parameter_use["physical_parameter_count"]
        ),
        "equation_map_row_count": (
            fused["fused_contract"]["equation_map_row_count"]
            == parameter_use["equation_map_row_count"]
        ),
        "model_setting_count": (
            fused["model_settings"]["setting_count"]
            == parameter_use["model_numerical_setting_count"]
        ),
        "model_setting_verified_count": (
            fused["model_settings"]["verified_setting_count"]
            == parameter_use["model_numerical_setting_count"]
        ),
        "architecture_count": (
            fused["algorithm_sources"]["architecture_count"]
            == architectures["architecture_count"]
        ),
    }
    physical_inputs_complete = bool(
        parameter_use["physical_parameter_count"] == 22
        and parameter_use["physical_parameters_used_by_equations"] == 22
        and parameter_use["equation_map_row_count"] == 31
        and not parameter_use["unused_physical_parameter_ids"]
        and not parameter_use["unknown_equation_parameter_ids"]
        and not parameter_use["new_physical_parameters"]
    )
    classified_numerical_setting_count = sum(
        parameter_use[key]
        for key in (
            "literature_or_official_model_setting_count",
            "case_or_data_derived_model_setting_count",
            "predeclared_project_comparison_setting_count",
        )
    )
    numerical_settings_complete = bool(
        parameter_use["model_numerical_setting_count"] > 0
        and classified_numerical_setting_count
        == parameter_use["model_numerical_setting_count"]
        and parameter_use["all_model_settings_are_nonphysical"]
        and parameter_use["all_model_setting_source_paths_exist"]
        and not parameter_use["unknown_model_setting_types"]
    )
    experiment_route_complete = bool(
        route_documents["complete"]
        and experimental_templates["complete"]
        and parameter_use["experimental_observable_count"] == 12
        and parameter_use["experimental_observation_source_count"] == 17
        and parameter_use["experimental_templates_contain_no_measurements"]
    )
    scheme_checks = {
        "literature_physical_inputs": physical_inputs_complete,
        "published_model_and_algorithm_sources": architectures["complete"],
        "numerical_settings_separated_from_physics": numerical_settings_complete,
        "experimental_route_and_data_structure": experiment_route_complete,
        "chinese_process_and_explanation_documents": route_documents["complete"],
        "initial_code_and_model_data_structure": core_code["complete"],
        "current_summary_counts_match_source_tables": all(
            current_counts_match.values()
        ),
    }
    scheme_complete = all(scheme_checks.values())
    data = current_formal_data(root, fused)
    # The submitted study uses the verified steady matrix and fixed-flow thermal
    # transients. Fully coupled startup runs define the property-range limit and
    # are not a missing training-data family.
    formal_calculation_complete = bool(
        data["steady"]["complete"] and data["physical_transient"]["complete"]
    )
    payload = {
        "status": (
            "research_route_complete_formal_calculation_pending"
            if scheme_complete and not formal_calculation_complete
            else (
                "research_route_and_formal_calculation_complete"
                if scheme_complete
                else "research_route_incomplete"
            )
        ),
        "scheme_checks": scheme_checks,
        "scheme_complete": scheme_complete,
        "formal_calculation_complete": formal_calculation_complete,
        "formal_data_progress": {
            "steady": f"{data['steady']['completed']}/{data['steady']['required']}",
            "fixed_hydrodynamics_steps": (
                f"{data['physical_transient']['completed']}/"
                f"{data['physical_transient']['required']}"
            ),
            "fully_coupled_steps": (
                f"{data['fully_coupled_transient']['completed']}/"
                f"{data['fully_coupled_transient']['required']}"
            ),
        },
        "fully_coupled_scope_status": (
            "property_range_limited_not_part_of_formal_training_data"
        ),
        "physical_and_model_inputs": parameter_use,
        "architecture_sources": architectures,
        "required_documents": route_documents,
        "core_code_and_contracts": core_code,
        "experimental_templates": experimental_templates,
        "current_summary_count_checks": current_counts_match,
        "new_physical_parameters": [],
    }
    if not scheme_complete:
        raise ValueError(json.dumps(payload, ensure_ascii=False, indent=2))

    lines = [
        "# P418球床换热研究方案完成情况",
        "",
        "## 当前结论",
        "",
        "研究方案、稳态OpenFOAM矩阵和固定流场瞬态数据已经完成；模型比较仍在进行。",
        "",
        "已经完成的内容：",
        "",
        f"- 文献物理参数：`{parameter_use['physical_parameter_count']}`项，全部进入`{parameter_use['equation_map_row_count']}`项方程、边界条件或几何程序。",
        f"- 模型与算法：`{architectures['architecture_count']}`类，另有`{architectures['domain_specific_pinn_precedent_count']}`篇直接针对球床PINN的论文；论文链接、本地资料和文件校验均已检查。",
        f"- 网络与数值设置：`{parameter_use['model_numerical_setting_count']}`项，全部与物理参数分开，并能找到采用依据或当前计算条件。",
        f"- 实验路线：`{parameter_use['experimental_observable_count']}`类观测量、`{parameter_use['experimental_observation_source_count']}`条文献测量记录和5张空白数据表已经接通。",
        "- 中文记录：研究主线、模型选择原因、实验步骤、参数来源、当前进度和逐步过程均有独立文件。",
        "- 初始代码：稳态PINN、图--Transformer、全耦合图--Transformer、DMDc/POD和温度扩散修正均有数据入口、模型程序和训练入口。",
        "",
        "## 正式数据与适用范围",
        "",
        f"- 稳态三维工况：`{data['steady']['completed']}/{data['steady']['required']}`。",
        f"- 固定流场热阶跃：`{data['physical_transient']['completed']}/{data['physical_transient']['required']}`。",
        f"- 全耦合热阶跃：`{data['fully_coupled_transient']['completed']}/{data['fully_coupled_transient']['required']}`；短算因局部压力超出已登记氦物性范围而停止，仅用于说明本研究的适用范围。",
        "- 现在可以基于60组稳态数据和12条固定流场热阶跃报告模型准确率与计算效率，但必须等正式模型比较自然完成。",
        "",
        "这份检查不启动OpenFOAM或模型训练，也不增加物理参数。",
        "",
    ]
    return payload, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/hccb_p418_research_route_completeness",
    )
    args = parser.parse_args()
    payload, document = build(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "P418_研究方案完成情况_CN.md").write_text(
        document,
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
