#!/usr/bin/env python3
"""Build one verified handoff sheet for the four P418 cloud packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "cloud_migration/cloud_package_specs.json"
DEFAULT_TASKS = ROOT / "cloud_migration/cpu_task_candidates.csv"
DEFAULT_PACKAGE_DIR = ROOT / "cloud_migration_build"
DEFAULT_JSON = ROOT / "cloud_migration/cloud_submission_manifest.json"
DEFAULT_CN = ROOT / "cloud_migration/易算云提交清单_简明_CN.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_declared_sha(path: Path) -> str:
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) < 2 or len(fields[0]) != 64:
        raise ValueError(f"invalid checksum file: {path}")
    return fields[0].lower()


def read_tasks(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    tasks = {row["task_id"]: row for row in rows}
    if len(tasks) != len(rows):
        raise ValueError("task table contains duplicate task identifiers")
    return tasks


def build_manifest(
    spec_path: Path,
    task_path: Path,
    package_dir: Path,
) -> tuple[dict[str, object], bool]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    tasks = read_tasks(task_path)
    package_rows: list[dict[str, object]] = []
    all_checks_passed = True

    for package in sorted(
        spec["packages"], key=lambda item: int(item["submission_order"])
    ):
        filename = str(package["filename"])
        archive = package_dir / filename
        checksum_file = package_dir / f"{filename}.sha256"
        checks = {
            "archive_exists": archive.is_file(),
            "checksum_file_exists": checksum_file.is_file(),
            "all_task_ids_exist": all(
                task_id in tasks for task_id in package["task_ids"]
            ),
            "tasks_do_not_repeat_workstation_cases": all(
                tasks.get(task_id, {}).get("duplicate_with_workstation") == "否"
                for task_id in package["task_ids"]
            ),
        }
        declared_sha = None
        actual_sha = None
        size_bytes = None
        if archive.is_file():
            size_bytes = archive.stat().st_size
            actual_sha = sha256(archive)
        if checksum_file.is_file():
            declared_sha = read_declared_sha(checksum_file)
        checks["checksum_matches"] = (
            actual_sha is not None
            and declared_sha is not None
            and actual_sha == declared_sha
        )

        package_task_paths = [
            tasks[task_id]["path_or_package"]
            for task_id in package["task_ids"]
            if task_id in tasks
            and tasks[task_id]["path_or_package"].strip().endswith(".tar.zst")
        ]
        checks["task_table_package_name_matches"] = all(
            Path(path).name == filename for path in package_task_paths
        )
        passed = all(checks.values())
        all_checks_passed = all_checks_passed and passed
        package_rows.append(
            {
                **package,
                "authoritative_workstation_path": (
                    Path(spec["authoritative_workstation_package_dir"]) / filename
                ).as_posix(),
                "size_bytes": size_bytes,
                "sha256": actual_sha,
                "checks": checks,
                "ready_for_transfer": passed,
                "task_summaries": [
                    {
                        "task_id": task_id,
                        "task_name": tasks.get(task_id, {}).get("task_name"),
                        "readiness": tasks.get(task_id, {}).get("readiness"),
                        "submit_rule": tasks.get(task_id, {}).get("submit_rule"),
                    }
                    for task_id in package["task_ids"]
                ],
            }
        )

    payload = {
        "status": (
            "all_four_cloud_packages_verified"
            if all_checks_passed
            else "cloud_package_problem_found"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_id": spec["project_id"],
        "authoritative_workstation_package_dir": spec[
            "authoritative_workstation_package_dir"
        ],
        "workstation_project_root": spec["workstation_project_root"],
        "output_location_rule": spec["output_location_rule"],
        "package_count": len(package_rows),
        "all_checks_passed": all_checks_passed,
        "packages": package_rows,
        "important_note_cn": (
            "只使用本清单中的cloud_migration_build目录。项目根目录若有早期同名副本，"
            "不作为当前提交文件。"
        ),
    }
    return payload, all_checks_passed


def render_cn(payload: dict[str, object]) -> str:
    lines = [
        "# 易算云提交清单（简明版）",
        "",
        f"生成时间（UTC）：`{payload['generated_at_utc']}`",
        "",
        "## 唯一使用位置",
        "",
        f"`{payload['authoritative_workstation_package_dir']}`",
        "",
        payload["important_note_cn"],
        "",
        "所有计算输出写入易算云项目盘或机械盘，不写入`/home`。",
        "",
        "## 四个文件",
        "",
        "| 顺序 | 文件 | 大小 | SHA-256 | 用途 | 当前状态 |",
        "|---:|---|---:|---|---|---|",
    ]
    for package in payload["packages"]:
        size = (
            str(package["size_bytes"])
            if package["size_bytes"] is not None
            else "缺失"
        )
        digest = package["sha256"] or "缺失"
        state = "可以传输" if package["ready_for_transfer"] else "需要处理"
        lines.append(
            f"| {package['submission_order']} | `{package['filename']}` | "
            f"{size} B | `{digest}` | {package['purpose_cn']} | {state} |"
        )

    lines.extend(["", "## 提交顺序", ""])
    for package in payload["packages"]:
        lines.extend(
            [
                f"### {package['submission_order']}. {package['package_id']}",
                "",
                f"- 前提：{package['precondition_cn']}",
                f"- 工作站文件：`{package['authoritative_workstation_path']}`",
                f"- 解压后目录：`{package['unpacked_directory']}`",
                "- 首次命令：",
                "",
            ]
        )
        lines.extend(f"  - `{command}`" for command in package["first_commands"])
        lines.append("")

    lines.extend(
        [
            "## 提交后首先返回的结果",
            "",
            "1. CPU01的1次稳态迭代运行日志和`cht_result_summary_1.json`；",
            "2. CPU01完整200次稳态迭代的`cloud_runtime_resources.json`；",
            "3. 与工作站结果逐项比较的`cloud_reference_comparison.json`；",
            "4. CPU03两套网格的流体区和固体区`checkMesh`结果。",
            "",
            "得到CPU01的实测时间和峰值内存后，再填写CPU02的数组任务时限和内存。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-cn", type=Path, default=DEFAULT_CN)
    args = parser.parse_args()

    payload, passed = build_manifest(
        args.spec.resolve(), args.tasks.resolve(), args.package_dir.resolve()
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_cn.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_cn.write_text(render_cn(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
