#!/usr/bin/env python3
"""Build a checksum manifest for the P418 paper source and reproducibility files."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import time
from pathlib import Path


REQUIRED_FILES = (
    "CITATION.cff",
    "LICENSE",
    "DATA_LICENSE.md",
    "Makefile",
    "requirements-p418.txt",
    "reproducibility/README.md",
    "reproducibility/README_CN.md",
    "reproducibility/p418_environment.json",
    "reproducibility/repository_release_metadata_draft.json",
    "scripts/reproduce_p418_paper.sh",
    "scripts/test_p418_public_package.sh",
    "code/run_hccb_p418_formal_calculations.sh",
    "code/run_hccb_p418_60_postprocess.sh",
    "code/run_hccb_p418_manuscript_refresh.sh",
    "code/run_hccb_p418_post_manifest_manuscript_finalization.sh",
    "code/check_hccb_p418_final_scientific_requirements.py",
    "code/check_hccb_p418_ijhmt_submission.py",
    "code/ijhmt_figure_style.py",
    "parameters/hccb_p418_physical_parameter_sources.csv",
    "parameters/hccb_p418_equation_input_map.csv",
    "parameters/hccb_p418_model_numerical_settings.csv",
    "parameters/hccb_p418_model_splits.json",
    "parameters/hccb_p418_transient_step_plan.json",
    "parameters/hccb_p418_ai_architecture_sources.json",
    "parameters/literature_parameter_manifest.csv",
    "results/hccb_p418_three_mesh_cht_sensitivity/summary.json",
    "results/hccb_p418_three_mesh_cht_sensitivity/engineering_observables.csv",
    "results/hccb_p418_three_mesh_cht_sensitivity/mesh_gci.csv",
    "results/hccb_p418_public_figure_data/README.md",
    "results/hccb_p418_public_figure_data/physical_response_60.csv",
    "results/hccb_p418_public_figure_data/seed202_integral_comparison_9.csv",
    "results/hccb_p418_public_figure_data/seed202_integral_summary.json",
    "results/hccb_p418_public_figure_data/steady_model_comparison_5x5.csv",
    "results/hccb_p418_public_figure_data/summary.json",
    "results/hccb_p418_public_figure_data/direct_transport_scope_limit.json",
    "results/hccb_p418_public_data_release_preflight/README.md",
    "results/hccb_p418_public_data_release_preflight/summary.json",
    "results/hccb_p418_public_data_release_preflight/formal_training_manifest_public.json",
    "results/hccb_p418_60_steady_seed_robustness_100epoch/summary.json",
    "results/hccb_p418_60_steady_seed_robustness_100epoch/steady_seed_metrics.csv",
    "results/hccb_p418_60_steady_seed_robustness_100epoch/steady_seed_summary.csv",
    "results/hccb_p418_60_steady_seed_robustness_100epoch/README_CN.md",
    "data/apd006_hccb_source_sequence_target_packings/seed101_s80_xlo_ycentre/packing.npz",
    "runs/hccb_dense_snappy_g2_nativezone_r2/geometry/packing_crop.npz",
    "results/hccb_p418_actual_spatiotemporal_operator_37time_gpu_data_only/regional_sequence_geometry.npz",
    "results/hccb_p418_60_sourceflow_r3_model_geometry/model_geometry.npz",
    "manuscript/main.tex",
    "manuscript/methods_condensed.tex",
    "manuscript/results_condensed.tex",
    "manuscript/result_source_map.csv",
    "manuscript/supplement.tex",
    "manuscript/supplement_condensed_body.tex",
    "manuscript/references.bib",
    "manuscript/P418_论文中文便读版.md",
    "manuscript/elsarticle.cls",
    "manuscript/elsarticle-num.bst",
    "submission/cover_letter_IJHMT.md",
    "submission/highlights.txt",
    "submission/data_release_license_choice.json",
    "submission/data_release_repository_record.json",
    "results/hccb_p418_fixed_flow_runtime_recovery_checks/summary.json",
)

SOURCE_PATTERNS = (
    "code/*hccb_p418*.py",
    "code/*hccb_p418*.sh",
    "code/hccb_p418*.py",
    "solver_extensions/hccbHeliumTransport/**/*",
    "tests/test_*hccb_p418*.py",
    "parameters/*p418*",
    "parameters/*P418*",
    "experimental_data_templates/*",
    "reproducibility/*",
    "submission/*",
)

PUBLIC_SOURCE_EXCLUDES = {
    "parameters/hccb_p418_formal_training_jobs.json",
    # Machine-local smoke-test bookkeeping; it is not a formal split and
    # contains absolute source paths by design.
    "parameters/hccb_p418_pilot_smoke_splits.json",
}

OPTIONAL_FINAL_OUTPUTS = (
    "manuscript/main.pdf",
    "manuscript/supplement.pdf",
    "manuscript/P418_论文中文便读版.md",
    "figures/hccb_p418_physical_model_domain.pdf",
    "figures/hccb_p418_physical_response.pdf",
    "figures/hccb_heat_ai_external_evidence.pdf",
    "figures/hccb_p418_seed202_integral_9.pdf",
    "figures/hccb_p418_steady_model_comparison.pdf",
    "figures/hccb_p418_transient_model_comparison.pdf",
    "figures/hccb_p418_openfoam_model_field_comparison.pdf",
    "manuscript/generated_steady_model_comparison_validated.tex",
    "manuscript/generated_steady_seed_robustness_text.tex",
    "manuscript/generated_transient_model_comparison_validated.tex",
    "manuscript/generated_openfoam_model_field_comparison_validated.tex",
    "manuscript/generated_final_abstract.tex",
    "manuscript/generated_final_discussion.tex",
    "manuscript/generated_final_conclusions.tex",
    "results/hccb_p418_final_manuscript_narrative.json",
    "results/hccb_p418_ijhmt_submission_check_current/summary.json",
    "results/hccb_p418_final_scientific_requirements_current/summary.json",
)

REQUIRED_FINAL_JSON_STATUS = {
    "results/hccb_p418_final_manuscript_narrative.json": (
        "complete_p418_final_manuscript_narrative"
    ),
    "results/hccb_p418_ijhmt_submission_check_current/summary.json": (
        "completed_p418_ijhmt_submission_check"
    ),
    "results/hccb_p418_final_scientific_requirements_current/summary.json": (
        "completed_p418_final_scientific_requirements"
    ),
}

EXCLUDED_LARGE_CONTENT = (
    "raw OpenFOAM time directories and processor decompositions",
    "model checkpoints and training caches",
    "cloud migration tar archives",
    "legacy project branches unrelated to the P418 manuscript",
)

TRAINING_COVERAGE_SUMMARY = (
    "results/hccb_p418_training_data_coverage_partial/summary.json"
)


def sha256(path: Path, attempts: int = 2) -> str:
    """Hash a file, reopening once after a transient cloud-storage timeout."""
    for attempt in range(attempts):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.2)
    raise RuntimeError("unreachable")


def category_for(path: str) -> str:
    if path.startswith("parameters/"):
        return "physical_and_model_inputs"
    if path.startswith("tests/"):
        return "tests"
    if (
        path.startswith("code/")
        or path.startswith("scripts/")
        or path.startswith("solver_extensions/")
    ):
        return "programs"
    if path.startswith("manuscript/"):
        return "manuscript"
    if path.startswith("cloud_migration/"):
        return "compute_environment_and_migration"
    if path.startswith("experimental_data_templates/"):
        return "experimental_templates"
    if path.startswith("reproducibility/") or path.startswith("requirements"):
        return "reproducibility"
    if path.startswith("literature/"):
        return "literature_evidence"
    if path.startswith("results/"):
        return "small_processed_results"
    return "project_entry"


def add_local_python_dependencies(root: Path, paths: set[Path]) -> set[Path]:
    """Include project modules imported by the selected Python sources."""
    expanded = set(paths)
    pending = [path for path in expanded if path.suffix == ".py"]
    inspected: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in inspected:
            continue
        inspected.add(source)
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        for module in modules:
            relative = Path(*module.split("."))
            candidates = (
                root / "code" / relative.with_suffix(".py"),
                root / relative.with_suffix(".py"),
            )
            dependency = next((path for path in candidates if path.is_file()), None)
            if dependency is not None and dependency not in expanded:
                expanded.add(dependency)
                pending.append(dependency)
    return expanded


def collect_source_paths(root: Path) -> tuple[list[Path], list[str]]:
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    paths = {root / path for path in REQUIRED_FILES if (root / path).is_file()}
    for pattern in SOURCE_PATTERNS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    paths = add_local_python_dependencies(root, paths)
    paths = {
        path
        for path in paths
        if path.relative_to(root).as_posix() not in PUBLIC_SOURCE_EXCLUDES
    }
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix()), missing


def build_manifest(root: Path) -> dict:
    source_paths, missing = collect_source_paths(root)
    rows = []
    for path in source_paths:
        relative = path.relative_to(root).as_posix()
        try:
            file_sha256 = sha256(path)
        except OSError as exc:
            raise OSError(f"cannot read reproducibility source: {relative}") from exc
        rows.append(
            {
                "category": category_for(relative),
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256,
                "required": relative in REQUIRED_FILES,
                "present": True,
            }
        )
    final_outputs = []
    for relative in OPTIONAL_FINAL_OUTPUTS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        required_status = REQUIRED_FINAL_JSON_STATUS.get(relative)
        actual_status = None
        if present and required_status:
            try:
                actual_status = json.loads(path.read_text(encoding="utf-8")).get(
                    "status"
                )
            except (json.JSONDecodeError, OSError):
                actual_status = None
            present = actual_status == required_status
        final_outputs.append(
            {
                "path": relative,
                "present": present,
                "required_status": required_status,
                "actual_status": actual_status,
            }
        )
    source_ready = not missing
    final_ready = all(row["present"] for row in final_outputs)
    coverage_path = root / TRAINING_COVERAGE_SUMMARY
    training_data_progress = None
    if coverage_path.is_file():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        training_data_progress = {
            "completed_case_count": int(coverage["completed_case_count"]),
            "expected_case_count": int(coverage["expected_case_count"]),
            "missing_case_count": len(coverage["missing_condition_ids"]),
            "problem_count": len(coverage["problems"]),
            "solver_time_semantics": coverage.get(
                "solver_time_semantics", "steady_iteration_index"
            ),
            "physical_time_s": coverage.get("physical_time_s"),
            "steady_iteration_column": coverage.get(
                "steady_iteration_column", "steady_iteration"
            ),
            "summary_path": TRAINING_COVERAGE_SUMMARY,
            "summary_sha256": sha256(coverage_path),
        }
    if source_ready and final_ready:
        status = "completed_p418_reproducibility_package"
    elif source_ready:
        status = "p418_reproducibility_source_ready_final_outputs_pending"
    else:
        status = "p418_reproducibility_source_incomplete"
    return {
        "status": status,
        "manifest_version": 2,
        "project_root": ".",
        "source_package_ready": source_ready,
        "final_outputs_ready": final_ready,
        "source_file_count": len(rows),
        "source_bytes": sum(int(row["size_bytes"]) for row in rows),
        "missing_required_files": missing,
        "files": rows,
        "optional_final_outputs": final_outputs,
        "excluded_large_content": list(EXCLUDED_LARGE_CONTENT),
        "training_data_progress": training_data_progress,
        "formal_solver_started_by_reproduction_script": False,
        "new_physical_parameters": [],
    }


def write_outputs(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "category",
                "path",
                "size_bytes",
                "sha256",
                "required",
                "present",
            ),
        )
        writer.writeheader()
        writer.writerows(payload["files"])
    ready = "齐全" if payload["source_package_ready"] else "不齐全"
    final_count = sum(row["present"] for row in payload["optional_final_outputs"])
    lines = [
        "# P418论文复现文件说明",
        "",
        f"- 小型代码与说明文件：{payload['source_file_count']} 个。",
        f"- 文件总量：{payload['source_bytes'] / 1024 / 1024:.2f} MiB。",
        f"- 必需源文件：{ready}。",
        f"- 最终论文结果文件：{final_count}/{len(payload['optional_final_outputs'])}。",
        "- 复现脚本默认不会启动OpenFOAM或模型训练。",
    ]
    progress = payload.get("training_data_progress")
    if progress:
        lines.extend(
            [
                (
                    "- 正式三维稳态求解与schema-3样本："
                    f"{progress['completed_case_count']}/"
                    f"{progress['expected_case_count']}；"
                    f"剩余{progress['missing_case_count']}组；"
                    f"样本问题数{progress['problem_count']}。"
                ),
                (
                    "- 训练数据覆盖汇总SHA-256："
                    f"`{progress['summary_sha256']}`。"
                ),
                (
                    "- 稳态结果编号按非线性迭代记录，不是物理时间；"
                    f"CSV字段为`{progress['steady_iteration_column']}`，"
                    "`physical_time_s`为空。"
                ),
            ]
        )
    lines.extend(["", "## 还未进入小型代码包的大文件", ""])
    lines.extend(f"- {item}" for item in payload["excluded_large_content"])
    lines.extend(["", "## 最终结果文件", ""])
    for row in payload["optional_final_outputs"]:
        mark = "已有" if row["present"] else "未完成"
        lines.append(f"- {mark}：`{row['path']}`")
    if payload["missing_required_files"]:
        lines.extend(["", "## 缺少的必需文件", ""])
        lines.extend(f"- `{path}`" for path in payload["missing_required_files"])
    (output_dir / "P418_复现文件说明_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-source-complete", action="store_true")
    parser.add_argument("--require-final-complete", action="store_true")
    args = parser.parse_args()
    payload = build_manifest(args.project_root.resolve())
    write_outputs(payload, args.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "source_package_ready": payload["source_package_ready"],
                "final_outputs_ready": payload["final_outputs_ready"],
                "source_file_count": payload["source_file_count"],
                "source_bytes": payload["source_bytes"],
                "missing_required_files": payload["missing_required_files"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.require_source_complete and not payload["source_package_ready"]:
        raise SystemExit("P418 reproducibility source package is incomplete")
    if args.require_final_complete and not payload["final_outputs_ready"]:
        raise SystemExit("P418 final reproducibility outputs are incomplete")


if __name__ == "__main__":
    main()
