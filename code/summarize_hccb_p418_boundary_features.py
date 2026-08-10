#!/usr/bin/env python3
"""Summarize boundary-adjacency features on a P418 regional graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_feature_key(keys: list[str], requested: str | None) -> str:
    if requested:
        if requested not in keys:
            raise ValueError(f"geometry does not contain {requested}")
        return requested
    candidates: list[tuple[int, str]] = []
    for key in keys:
        match = re.fullmatch(r"level_(\d+)_boundary_volume_fraction", key)
        if match:
            candidates.append((int(match.group(1)), key))
    if not candidates:
        raise ValueError("geometry contains no regional boundary-volume feature")
    return max(candidates)[1]


def summarize(
    geometry_path: Path,
    roles_path: Path,
    feature_key: str | None = None,
) -> dict[str, object]:
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    expected_names = [str(name) for name in roles["role_order"]]
    with np.load(geometry_path, allow_pickle=False) as loaded:
        selected_key = select_feature_key(list(loaded.files), feature_key)
        values = np.asarray(loaded[selected_key], dtype=np.float64)
        stored_names = (
            [str(name) for name in loaded["boundary_role_names"].tolist()]
            if "boundary_role_names" in loaded.files
            else expected_names
        )

    if values.ndim != 2:
        raise ValueError("boundary feature array must have two dimensions")
    if values.shape[1] != len(expected_names):
        raise ValueError("boundary feature columns do not match the role table")
    if stored_names != expected_names:
        raise ValueError("stored boundary role order differs from the role table")
    if not np.isfinite(values).all():
        raise ValueError("boundary features contain non-finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("boundary features must lie between zero and one")

    role_rows: list[dict[str, object]] = []
    for column, name in enumerate(expected_names):
        feature = values[:, column]
        role_rows.append(
            {
                "role": name,
                "nonzero_regional_nodes": int(np.count_nonzero(feature > 0.0)),
                "regional_coverage_fraction": float(np.mean(feature > 0.0)),
                "fully_boundary_adjacent_regional_nodes": int(
                    np.count_nonzero(feature >= 1.0 - 1.0e-7)
                ),
                "mean_boundary_adjacent_volume_fraction": float(feature.mean()),
                "maximum_boundary_adjacent_volume_fraction": float(feature.max()),
            }
        )

    checks = {
        "five_registered_roles_are_present": len(expected_names) == 5,
        "all_roles_have_nonzero_regional_support": all(
            int(row["nonzero_regional_nodes"]) > 0 for row in role_rows
        ),
        "all_values_are_finite": True,
        "all_values_are_between_zero_and_one": True,
    }
    nodes_with_any = np.any(values > 0.0, axis=1)
    nodes_with_multiple = np.count_nonzero(values > 0.0, axis=1) > 1
    return {
        "status": (
            "p418_boundary_features_present_on_regional_graph"
            if all(checks.values())
            else "p418_boundary_feature_problem"
        ),
        "definition": (
            "For each regional node and boundary role, the feature is the "
            "volume-weighted fraction of constituent fine cells adjacent to "
            "that boundary. It is not a boundary-area fraction."
        ),
        "geometry_file": str(geometry_path),
        "geometry_sha256": sha256(geometry_path),
        "boundary_role_file": str(roles_path),
        "boundary_role_sha256": sha256(roles_path),
        "feature_key": selected_key,
        "regional_node_count": int(values.shape[0]),
        "role_count": int(values.shape[1]),
        "regional_nodes_with_any_boundary": int(np.count_nonzero(nodes_with_any)),
        "regional_nodes_with_multiple_boundary_roles": int(
            np.count_nonzero(nodes_with_multiple)
        ),
        "roles": role_rows,
        "checks": checks,
    }


def render_cn(payload: dict[str, object]) -> str:
    lines = [
        "# P418真实区域图边界位置检查",
        "",
        f"区域节点数：`{payload['regional_node_count']}`。",
        "",
        "这里的数值是一个区域中与某类边界相邻的细网格体积占比，"
        "不是边界面积比例，也不是拟合参数。",
        "",
        "| 边界类型 | 非零区域节点 | 区域覆盖比例 | 完全由边界相邻细网格组成的区域 | 平均体积占比 | 最大体积占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["roles"]:
        lines.append(
            f"| {row['role']} | {row['nonzero_regional_nodes']} | "
            f"{100.0 * row['regional_coverage_fraction']:.2f}% | "
            f"{row['fully_boundary_adjacent_regional_nodes']} | "
            f"{row['mean_boundary_adjacent_volume_fraction']:.5f} | "
            f"{row['maximum_boundary_adjacent_volume_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"至少邻接一种边界的区域节点：`{payload['regional_nodes_with_any_boundary']}`。",
            f"同时邻接多类边界的区域节点：`{payload['regional_nodes_with_multiple_boundary_roles']}`。",
            "",
            f"程序结果：`{payload['status']}`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument(
        "--roles",
        type=Path,
        default=Path("parameters/hccb_dense_cht_boundary_roles.json"),
    )
    parser.add_argument("--feature-key")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chinese-output", type=Path)
    args = parser.parse_args()

    payload = summarize(
        args.geometry.resolve(),
        args.roles.resolve(),
        args.feature_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.chinese_output:
        args.chinese_output.parent.mkdir(parents=True, exist_ok=True)
        args.chinese_output.write_text(render_cn(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(payload["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
