#!/usr/bin/env python3
"""Describe how the published P418 thermal-step curves cover condition space."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


CONDITION = re.compile(r"^u(?P<u>[0-9]+p[0-9]+)_T(?P<T>[0-9]+)_q(?P<q>[0-9]+p[0-9]+)$")
FEATURE_NAMES = (
    "source_velocity",
    "source_temperature",
    "source_heat_source",
    "target_velocity",
    "target_temperature",
    "target_heat_source",
)
DIMENSIONLESS_FEATURE_NAMES = (
    "source_inlet_particle_reynolds",
    "source_inlet_prandtl",
    "source_inlet_particle_peclet",
    "target_inlet_particle_reynolds",
    "target_inlet_prandtl",
    "target_inlet_particle_peclet",
)


def condition_values(identifier: str) -> np.ndarray:
    match = CONDITION.fullmatch(identifier)
    if match is None:
        raise ValueError(f"cannot parse P418 condition: {identifier}")
    return np.asarray(
        [
            float(match.group("u").replace("p", ".")),
            float(match.group("T")),
            float(match.group("q").replace("p", ".")),
        ],
        dtype=float,
    )


def step_vector(row: dict[str, object]) -> np.ndarray:
    return np.concatenate(
        [
            condition_values(str(row["source_condition_id"])),
            condition_values(str(row["target_condition_id"])),
        ]
    )


def read_dimensionless_conditions(path: Path) -> dict[str, np.ndarray]:
    conditions: dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            identifier = row["condition_id"]
            values = np.asarray(
                [
                    float(row["particle_reynolds_inlet"]),
                    float(row["prandtl"]),
                    float(row["particle_peclet_inlet"]),
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"invalid inlet dimensionless state: {identifier}")
            if identifier in conditions and not np.allclose(
                conditions[identifier], values, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(f"inconsistent inlet dimensionless state: {identifier}")
            conditions[identifier] = values
    if not conditions:
        raise ValueError("no inlet dimensionless conditions")
    return conditions


def step_dimensionless_vector(
    row: dict[str, object], conditions: dict[str, np.ndarray]
) -> np.ndarray:
    source = str(row["source_condition_id"])
    target = str(row["target_condition_id"])
    missing = [identifier for identifier in (source, target) if identifier not in conditions]
    if missing:
        raise ValueError(f"missing inlet dimensionless conditions: {missing}")
    return np.concatenate([conditions[source], conditions[target]])


def endpoint_dimensionless_ranges(matrix: np.ndarray) -> dict[str, list[float]]:
    if matrix.ndim != 2 or matrix.shape[1] != len(DIMENSIONLESS_FEATURE_NAMES):
        raise ValueError("invalid dimensionless step matrix")
    return {
        "particle_reynolds_inlet": [
            float(matrix[:, [0, 3]].min()),
            float(matrix[:, [0, 3]].max()),
        ],
        "prandtl_inlet": [
            float(matrix[:, [1, 4]].min()),
            float(matrix[:, [1, 4]].max()),
        ],
        "particle_peclet_inlet": [
            float(matrix[:, [2, 5]].min()),
            float(matrix[:, [2, 5]].max()),
        ],
    }


def endpoint_pair(row: dict[str, object]) -> tuple[str, str]:
    return tuple(sorted((str(row["source_condition_id"]), str(row["target_condition_id"]))))


def reverse_id(row: dict[str, object], sequences: dict[str, dict[str, object]]) -> str | None:
    source = str(row["source_condition_id"])
    target = str(row["target_condition_id"])
    for identifier, candidate in sequences.items():
        if (
            str(candidate["source_condition_id"]) == target
            and str(candidate["target_condition_id"]) == source
        ):
            return identifier
    return None


def analyze(
    plan: dict,
    split_payload: dict,
    dimensionless_conditions: dict[str, np.ndarray],
) -> tuple[dict, list[dict[str, object]]]:
    sequences = {str(row["sequence_id"]): row for row in plan["sequences"]}
    all_vectors = np.stack([step_vector(row) for row in sequences.values()])
    feature_min = all_vectors.min(axis=0)
    feature_max = all_vectors.max(axis=0)
    scale = feature_max - feature_min
    scale[scale < 1.0e-12] = 1.0
    normalized = {
        identifier: (step_vector(row) - feature_min) / scale
        for identifier, row in sequences.items()
    }
    all_dimensionless_vectors = np.stack(
        [
            step_dimensionless_vector(row, dimensionless_conditions)
            for row in sequences.values()
        ]
    )
    dimensionless_min = all_dimensionless_vectors.min(axis=0)
    dimensionless_max = all_dimensionless_vectors.max(axis=0)
    dimensionless_scale = dimensionless_max - dimensionless_min
    dimensionless_scale[dimensionless_scale < 1.0e-12] = 1.0
    dimensionless_vectors = {
        identifier: (
            step_dimensionless_vector(row, dimensionless_conditions)
            - dimensionless_min
        )
        / dimensionless_scale
        for identifier, row in sequences.items()
    }

    split_summaries: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for split_name, roles in split_payload["splits"].items():
        role_sets = {role: set(map(str, roles[role])) for role in ("train", "validation", "test")}
        if any(role_sets[left] & role_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError(f"overlapping curve roles in {split_name}")
        if set().union(*role_sets.values()) != set(sequences):
            raise ValueError(f"split does not cover all planned curves: {split_name}")

        train_ids = list(map(str, roles["train"]))
        train_matrix = np.stack([normalized[identifier] for identifier in train_ids])
        train_dimensionless_matrix = np.stack(
            [dimensionless_vectors[identifier] for identifier in train_ids]
        )
        design_rank = int(np.linalg.matrix_rank(np.column_stack([np.ones(len(train_ids)), train_matrix])))
        train_min = train_matrix.min(axis=0)
        train_max = train_matrix.max(axis=0)
        train_dimensionless_min = train_dimensionless_matrix.min(axis=0)
        train_dimensionless_max = train_dimensionless_matrix.max(axis=0)
        pair_sets = {
            role: {endpoint_pair(sequences[identifier]) for identifier in identifiers}
            for role, identifiers in role_sets.items()
        }
        pair_overlap_count = sum(
            len(pair_sets[left] & pair_sets[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        )

        for role in ("validation", "test"):
            for identifier in map(str, roles[role]):
                vector = normalized[identifier]
                distances = np.linalg.norm(train_matrix - vector[None, :], axis=1)
                nearest_index = int(np.argmin(distances))
                dimensionless_vector = dimensionless_vectors[identifier]
                dimensionless_distances = np.linalg.norm(
                    train_dimensionless_matrix - dimensionless_vector[None, :], axis=1
                )
                dimensionless_nearest_index = int(np.argmin(dimensionless_distances))
                outside = [
                    FEATURE_NAMES[index]
                    for index, value in enumerate(vector)
                    if value < train_min[index] - 1.0e-12 or value > train_max[index] + 1.0e-12
                ]
                outside_dimensionless = [
                    DIMENSIONLESS_FEATURE_NAMES[index]
                    for index, value in enumerate(dimensionless_vector)
                    if value < train_dimensionless_min[index] - 1.0e-12
                    or value > train_dimensionless_max[index] + 1.0e-12
                ]
                reverse = reverse_id(sequences[identifier], sequences)
                reverse_role = next(
                    (candidate for candidate, values in role_sets.items() if reverse in values),
                    "absent",
                )
                rows.append(
                    {
                        "split_name": split_name,
                        "data_role": role,
                        "sequence_id": identifier,
                        "family": sequences[identifier]["family"],
                        "nearest_training_sequence": train_ids[nearest_index],
                        "normalized_nearest_training_distance": float(distances[nearest_index]),
                        "outside_training_feature_count": len(outside),
                        "outside_training_features": ";".join(outside),
                        "nearest_training_sequence_dimensionless": train_ids[
                            dimensionless_nearest_index
                        ],
                        "normalized_nearest_training_dimensionless_distance": float(
                            dimensionless_distances[dimensionless_nearest_index]
                        ),
                        "outside_training_dimensionless_feature_count": len(
                            outside_dimensionless
                        ),
                        "outside_training_dimensionless_features": ";".join(
                            outside_dimensionless
                        ),
                        "reverse_sequence_id": reverse or "",
                        "reverse_sequence_role": reverse_role,
                    }
                )

        test_rows = [row for row in rows if row["split_name"] == split_name and row["data_role"] == "test"]
        role_dimensionless_ranges = {
            role: endpoint_dimensionless_ranges(
                np.stack(
                    [
                        step_dimensionless_vector(
                            sequences[identifier], dimensionless_conditions
                        )
                        for identifier in map(str, roles[role])
                    ]
                )
            )
            for role in ("train", "validation", "test")
        }
        split_summaries[split_name] = {
            "curve_counts": {role: len(values) for role, values in role_sets.items()},
            "training_affine_design_rank": design_rank,
            "maximum_possible_affine_design_rank": min(len(train_ids), len(FEATURE_NAMES) + 1),
            "endpoint_pair_overlap_count_across_roles": pair_overlap_count,
            "test_reverse_curve_roles": {
                str(row["sequence_id"]): str(row["reverse_sequence_role"]) for row in test_rows
            },
            "test_nearest_training_distance_range": [
                float(min(row["normalized_nearest_training_distance"] for row in test_rows)),
                float(max(row["normalized_nearest_training_distance"] for row in test_rows)),
            ],
            "test_curves_outside_training_axis_range": sum(
                int(row["outside_training_feature_count"] > 0) for row in test_rows
            ),
            "test_nearest_training_dimensionless_distance_range": [
                float(
                    min(
                        row["normalized_nearest_training_dimensionless_distance"]
                        for row in test_rows
                    )
                ),
                float(
                    max(
                        row["normalized_nearest_training_dimensionless_distance"]
                        for row in test_rows
                    )
                ),
            ],
            "test_curves_outside_training_dimensionless_range": sum(
                int(row["outside_training_dimensionless_feature_count"] > 0)
                for row in test_rows
            ),
            "endpoint_dimensionless_ranges_by_role": role_dimensionless_ranges,
        }

    return {
        "status": "completed_p418_step_split_coverage",
        "source_parameter_id": plan["source_parameter_id"],
        "source_doi": plan["source_doi"],
        "feature_names": list(FEATURE_NAMES),
        "feature_min": feature_min.tolist(),
        "feature_max": feature_max.tolist(),
        "distance_definition": "Euclidean distance after scaling each source/target P418 variable to the range present in the 12 published-endpoint step curves",
        "dimensionless_feature_names": list(DIMENSIONLESS_FEATURE_NAMES),
        "dimensionless_distance_definition": (
            "Euclidean distance after scaling source/target inlet particle Reynolds, "
            "Prandtl and particle Peclet numbers to the range present in the same "
            "12 published-endpoint step curves"
        ),
        "dimensionless_parameter_ids": [
            "P048",
            "P068",
            "P070",
            "P071",
            "P073",
            "P388",
            "P418",
            "P426",
        ],
        "split_summaries": split_summaries,
        "new_physical_parameters": [],
    }, rows


def write_chinese(summary: dict, path: Path) -> None:
    lines = [
        "# P418热阶跃训练与测试范围",
        "",
        "本文件只分析12条文献端点热阶跃怎样分配，不使用OpenFOAM预测结果，也不增加物理参数。",
        "",
    ]
    for name, row in summary["split_summaries"].items():
        counts = row["curve_counts"]
        distance = row["test_nearest_training_distance_range"]
        dimensionless_distance = row[
            "test_nearest_training_dimensionless_distance_range"
        ]
        training_dimensionless = row["endpoint_dimensionless_ranges_by_role"]["train"]
        test_dimensionless = row["endpoint_dimensionless_ranges_by_role"]["test"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- 训练/检查/测试曲线：`{counts['train']}/{counts['validation']}/{counts['test']}`；",
                f"- 不同数据组之间重复的稳态端点对：`{row['endpoint_pair_overlap_count_across_roles']}`；",
                f"- 测试曲线到最近训练曲线的归一化距离：`{distance[0]:.3f}--{distance[1]:.3f}`；",
                f"- 超出训练逐变量范围的测试曲线：`{row['test_curves_outside_training_axis_range']}/{counts['test']}`。",
                f"- 训练端点入口颗粒Re范围：`{training_dimensionless['particle_reynolds_inlet'][0]:.3f}--{training_dimensionless['particle_reynolds_inlet'][1]:.3f}`；测试端点为`{test_dimensionless['particle_reynolds_inlet'][0]:.3f}--{test_dimensionless['particle_reynolds_inlet'][1]:.3f}`；",
                f"- 训练端点入口颗粒Pe范围：`{training_dimensionless['particle_peclet_inlet'][0]:.3f}--{training_dimensionless['particle_peclet_inlet'][1]:.3f}`；测试端点为`{test_dimensionless['particle_peclet_inlet'][0]:.3f}--{test_dimensionless['particle_peclet_inlet'][1]:.3f}`；",
                f"- 测试曲线到最近训练曲线的无量纲物理距离：`{dimensionless_distance[0]:.3f}--{dimensionless_distance[1]:.3f}`；",
                f"- 超出训练Re/Pr/Pe逐变量范围的测试曲线：`{row['test_curves_outside_training_dimensionless_range']}/{counts['test']}`。",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释",
            "",
            "两个方向测试用于比较升高和降低过程，但测试曲线的反向过程位于检查集，因此不能单独作为最严格的独立性证据。`pair_disjoint_stress_test`把同一对稳态端点的正向和反向曲线放在同一组，训练、检查和测试之间没有重复端点对。最终结果应同时报告方向测试和端点对分离测试。",
            "",
            "入口Re、Pr和Pe由已经登记的1 mm颗粒直径、0.12 MPa压力、氦气物性关系及P418工况计算。它们用于说明数据划分覆盖的低Re换热范围，不替代三维场内的局部Re，也不作为新增物理参数。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--dimensionless-conditions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    dimensionless_conditions = read_dimensionless_conditions(
        args.dimensionless_conditions
    )
    summary, rows = analyze(plan, splits, dimensionless_conditions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "curve_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_chinese(summary, args.output_dir / "P418_热阶跃训练测试范围_CN.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
