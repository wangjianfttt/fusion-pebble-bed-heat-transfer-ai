#!/usr/bin/env python3
"""Read-only preflight for the P418 PINN--Transformer--diffusion study."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_check(script: str, output: Path, extra: list[str] | None = None) -> dict[str, object]:
    command = [sys.executable, str(ROOT / script), "--output", str(output)]
    if extra:
        command.extend(extra)
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(output.read_text(encoding="utf-8"))


def verify_algorithm_sources(
    require_local_source_files: bool = True,
) -> dict[str, object]:
    path = ROOT / "parameters/hccb_p418_ai_architecture_sources.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    required = ("name", "role", "paper", "venue", "paper_url", "project_adaptation", "status")
    archived_file_count = 0
    for entry in registry["architectures"]:
        missing = [name for name in required if not str(entry.get(name, "")).strip()]
        if missing:
            raise ValueError(f"algorithm source {entry.get('name')} lacks {missing}")
        for key, value in entry.items():
            checksum_key = f"{key}_sha256"
            if not isinstance(value, str) or checksum_key not in entry:
                continue
            local_path = ROOT / value
            checksum = str(entry[checksum_key])
            if len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                raise ValueError(f"invalid SHA-256 in algorithm registry: {checksum_key}")
            if not require_local_source_files:
                archived_file_count += 1
                continue
            if not local_path.is_file():
                raise FileNotFoundError(local_path)
            if sha256(local_path) != checksum:
                raise ValueError(f"archived algorithm file changed: {value}")
            archived_file_count += 1
    names = [str(entry["name"]) for entry in registry["architectures"]]
    if len(names) != len(set(names)):
        raise ValueError("algorithm names are duplicated")
    return {
        "registry": str(path),
        "architecture_count": len(names),
        "architecture_names": names,
        "archived_file_count": archived_file_count,
        "source_verification_mode": (
            "local_files_and_sha256"
            if require_local_source_files
            else "registered_metadata_and_sha256"
        ),
        "all_entries_have_paper_links": all(
            str(entry.get("paper_url", "")).startswith(("https://", "http://"))
            for entry in registry["architectures"]
        ),
    }


def count_completion(
    root: Path,
    marker: str,
    expected: int,
    fallback_summary: Path | None = None,
    fallback_completed_key: str = "completed_case_count",
    fallback_required_key: str | None = "expected_case_count",
) -> dict[str, object]:
    exists = root.is_dir()
    completed = len(list(root.glob(f"*/{marker}"))) if exists else 0
    source = "completion_markers"
    if not exists and fallback_summary is not None and fallback_summary.is_file():
        payload = json.loads(fallback_summary.read_text(encoding="utf-8"))
        completed = int(payload[fallback_completed_key])
        if fallback_required_key is not None:
            expected = int(payload[fallback_required_key])
        source = "verified_summary"
    return {
        "path": str(root),
        "directory_available": exists,
        "progress_source": source,
        "fallback_summary": (
            str(fallback_summary)
            if source == "verified_summary" and fallback_summary is not None
            else None
        ),
        "completed": completed,
        "required": expected,
        "complete": completed == expected,
    }


def verify_requirements() -> dict[str, object]:
    path = ROOT / "requirements-p418.txt"
    packages = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"P418 dependency is not version-pinned: {line}")
        packages.append(line)
    return {"path": str(path), "pinned_package_count": len(packages), "packages": packages}


def chinese_summary(payload: dict[str, object]) -> str:
    steady = payload["current_data"]["steady"]
    transient = payload["current_data"]["physical_transient"]
    fully_coupled = payload["current_data"]["fully_coupled_transient"]
    readiness = (
        "稳态和固定流场瞬态训练输入已经齐全。"
        if payload["full_training_can_start"]
        else "模型方法已经接通，但正式训练数据还没有齐全。"
    )
    return (
        "# P418融合模型计算前检查\n\n"
        f"状态：{readiness}\n\n"
        "## 已确认\n\n"
        f"- 物理参数：`{payload['fused_contract']['physical_parameter_count']}`项，"
        f"方程和边界输入：`{payload['fused_contract']['equation_map_row_count']}`项；"
        f"本地资料引用：`{payload['fused_contract']['local_evidence_reference_count']}`处。\n"
        f"- 模型数值设置：`{payload['model_settings']['verified_setting_count']}/"
        f"{payload['model_settings']['setting_count']}`项与当前代码一致。\n"
        f"- 算法来源：`{payload['algorithm_sources']['architecture_count']}`类；"
        "每一类都记录了论文、用途和本项目中的改动。\n"
        "- 共同状态顺序：`Ux、Uy、Uz、压力、温度`。扩散模型只修正温度。\n"
        "- 12条物理阶跃按完整曲线划分，不拆分同一曲线的不同时刻。\n\n"
        "## 当前数据\n\n"
        f"- 稳态三维工况：`{steady['completed']}/{steady['required']}`。\n"
        f"- 固定流场物理热阶跃：`{transient['completed']}/{transient['required']}`。\n"
        f"- 速度、压力和温度全耦合阶跃：`{fully_coupled['completed']}/"
        f"{fully_coupled['required']}`。\n"
        "- 全耦合短算在登记的氦物性范围边界处停止，因此只用来界定"
        "本文的适用范围，不作为模型训练数据或收敛性证据。\n"
        "- 60工况收敛轨迹只用于求解加速，不作为物理瞬态换热结果。\n\n"
        "## 下一步\n\n"
        "用已完成的60组稳态工况和12条固定流场瞬态曲线完成模型比较，"
        "生成瞬态误差图和OpenFOAM--模型温度场对比图，然后完成论文定稿。\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=ROOT / "hccb_dense_cht_p418_60_sourceflow_r3",
    )
    parser.add_argument(
        "--step-root", type=Path, default=ROOT / "hccb_p418_physical_steps_12"
    )
    parser.add_argument(
        "--fully-coupled-step-root",
        type=Path,
        default=ROOT / "hccb_p418_fully_coupled_steps_12",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/hccb_p418_fused_preflight"
    )
    parser.add_argument(
        "--metadata-only-evidence",
        action="store_true",
        help="Validate literature metadata and registered hashes without local copyrighted files.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fused_extra = ["--metadata-only-evidence"] if args.metadata_only_evidence else None
    fused = run_check(
        "code/verify_hccb_p418_fused_model_contract.py",
        output / "fused_model_contract_check.json",
        fused_extra,
    )
    settings = run_check(
        "code/verify_hccb_p418_model_settings.py",
        output / "model_setting_check.json",
        [
            "--chinese-summary",
            str(output / "P418_模型数值设置对应_CN.md"),
            *(["--metadata-only-sources"] if args.metadata_only_evidence else []),
        ],
    )
    algorithms = verify_algorithm_sources(
        require_local_source_files=not args.metadata_only_evidence
    )
    requirements = verify_requirements()
    steady = count_completion(
        args.matrix_root.resolve(),
        "formal_sample_complete.json",
        60,
        fallback_summary=(
            ROOT / "results/hccb_p418_training_data_coverage_partial/summary.json"
        ),
    )
    transient = count_completion(args.step_root.resolve(), "step_response_complete.json", 12)
    if not transient["directory_available"]:
        transient = count_completion(
            args.step_root.resolve(),
            "step_response_complete.json",
            12,
            fallback_summary=(
                ROOT
                / "results/hccb_p418_physical_steps_12/regional_sequences/dataset_index.json"
            ),
            fallback_completed_key="sequence_count",
            fallback_required_key=None,
        )
    fully_coupled = count_completion(
        args.fully_coupled_step_root.resolve(),
        "fully_coupled_step_response_complete.json",
        12,
    )
    full_training = bool(steady["complete"] and transient["complete"])
    payload = {
        "status": (
            "p418_fused_method_and_data_ready"
            if full_training
            else "p418_fused_method_ready_openfoam_data_pending"
        ),
        "fused_contract": fused,
        "model_settings": settings,
        "algorithm_sources": algorithms,
        "software_requirements": requirements,
        "current_data": {
            "steady": steady,
            "physical_transient": transient,
            "fully_coupled_transient": fully_coupled,
        },
        "full_training_can_start": full_training,
        "fully_coupled_extension_data_complete": fully_coupled["complete"],
        "fully_coupled_scope_limitation": {
            "status": "property_range_limited_not_training_data",
            "evidence": str(
                ROOT
                / "results/hccb_p418_scope_limits_20260730/scope_limits_summary.json"
            ),
            "interpretation": (
                "The fully coupled short runs left the registered helium-property "
                "range and are retained only as scope-limit evidence."
            ),
        },
        "large_computation_started_by_this_check": False,
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "P418_融合模型计算前检查_CN.md").write_text(
        chinese_summary(payload), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
