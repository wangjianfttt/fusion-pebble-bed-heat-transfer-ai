#!/usr/bin/env python3
"""Confirm that regional step export uses the completed 60-case steady data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(
    summary_path: Path,
    dataset_index_path: Path,
    topology_path: Path,
    *,
    expected_cases: int = 60,
) -> dict[str, object]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing steady post-processing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "p418_60_training_data_ready":
        raise ValueError("steady fields have not completed data preparation")
    if int(summary.get("expected_case_count", -1)) != expected_cases:
        raise ValueError(
            f"post-processing summary expects {summary.get('expected_case_count')} cases, "
            f"required {expected_cases}"
        )
    if not dataset_index_path.is_file():
        raise FileNotFoundError(f"missing steady data index: {dataset_index_path}")
    if not topology_path.is_file():
        raise FileNotFoundError(f"missing shared mesh topology: {topology_path}")
    dataset = json.loads(dataset_index_path.read_text(encoding="utf-8"))
    actual = int(dataset.get("case_count", -1))
    if actual != expected_cases:
        raise ValueError(f"steady data set contains {actual} conditions, expected {expected_cases}")
    if dataset.get("sourceflow_mapping_required") is not True:
        raise ValueError("steady data set does not require the corrected source-flow mapping")
    if dataset.get("steady_final_window_required") is not True:
        raise ValueError("steady data set does not require measured final-window changes")
    conditions = dataset.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != expected_cases:
        raise ValueError("steady data index does not contain every source-flow condition")
    for condition in conditions:
        if condition.get("pore_opening_boundary_velocity_m_s") is None:
            raise ValueError("steady data condition misses the pore-opening velocity")
        if condition.get("inlet_open_area_fraction") is None:
            raise ValueError("steady data condition misses the inlet open-area fraction")
    return {
        "status": "p418_60_steady_dataset_ready_for_step_export",
        "case_count": actual,
        "dataset_index": str(dataset_index_path.resolve()),
        "shared_mesh_topology": str(topology_path.resolve()),
        "sourceflow_mapping_required": True,
        "steady_final_window_required": True,
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-summary", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--shared-topology", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=60)
    args = parser.parse_args()
    result = validate(
        args.postprocess_summary.resolve(),
        args.dataset_index.resolve(),
        args.shared_topology.resolve(),
        expected_cases=args.expected_cases,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
