#!/usr/bin/env python3
"""Verify that formal P418 diffusion does not invent sparse observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_hccb_p418_temporal_temperature_diffusion import (
    observation_masks,
    unobserved_dynamic_selection,
    validate_observation_input,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hccb_p418_diffusion_observation_scope"),
    )
    args = parser.parse_args()
    runner = (ROOT / "code/run_hccb_p418_step_responses.sh").read_text(
        encoding="utf-8"
    )
    route = (ROOT / "parameters/apd006_tdem_diffusion_route_contract.yaml").read_text(
        encoding="utf-8"
    )
    split_values = {
        "train": {
            "baseline_temperature_normalized": np.zeros(
                (2, 4, 6, 1), dtype=np.float32
            )
        }
    }
    empty_mask = observation_masks(None, split_values, "none")["train"]
    validate_observation_input("computed_residual_benchmark", None, "none")
    trainer = (
        ROOT / "code/train_hccb_p418_temporal_temperature_diffusion.py"
    ).read_text(encoding="utf-8")
    test_mask = np.zeros((4, 6, 1), dtype=bool)
    test_mask[2, 3, 0] = True
    node_type = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    selected = unobserved_dynamic_selection(test_mask, node_type)
    fluid_selected = unobserved_dynamic_selection(test_mask, node_type, material=0)
    required_missing_sources = (
        "no_machine_readable_TESOMEX_3d_sensor_coordinate_table",
        "no_source_backed_thermocouple_noise_and_dynamic_response_model",
        "no_repeated_experimental_transient_ensemble_for_posterior_calibration",
        "no_calibrated_observation_operator_between_TESOMEX_sensors_and_particle_nodes",
    )
    checks = {
        "formal_runner_uses_computed_residual_benchmark": (
            "--run-role computed_residual_benchmark" in runner
        ),
        "formal_runner_has_no_observation_mask_argument": (
            "--observation-mask" not in runner
        ),
        "formal_runner_has_no_observation_source_argument": (
            "--observation-source" not in runner
        ),
        "trainer_requires_explicit_observation_source": (
            "--observation-source" in trainer
            and "computed_openfoam_target" in trainer
            and "external experiments cannot be replaced" in trainer
        ),
        "absent_observation_file_produces_empty_mask": not bool(empty_mask.any()),
        "initial_time_excluded_from_unobserved_metrics": not bool(selected[0].any()),
        "supplied_observation_excluded_from_unobserved_metrics": not bool(
            selected[2, 3]
        ),
        "fluid_metric_excludes_solid_nodes": not bool(
            fluid_selected[:, node_type == 1].any()
        ),
        "sparse_experimental_reconstruction_remains_unavailable": (
            "status: diffusion_training_locked_until_sparse_observation_contract_is_complete"
            in route
            and all(item in route for item in required_missing_sources)
        ),
        "computed_and_experimental_sparse_values_are_separated": (
            "computed_sparse_mask_contract:" in route
            and "source_kind: computed_openfoam_target" in route
            and "external_experiment_exact_conditioning_allowed: false" in route
        ),
    }
    summary = {
        "status": "completed_p418_diffusion_observation_scope_verification",
        "formal_diffusion_role": "full_field_computed_temperature_residual_correction",
        "formal_observation_count": int(empty_mask.sum()),
        "formal_observation_mask_file": None,
        "formal_observation_source": "none",
        "openfoam_fields_are_measurements": False,
        "sparse_experimental_reconstruction_status": (
            "not_started_until_source_defined_sensor_coordinates_and_response_data_exist"
        ),
        "missing_source_items": list(required_missing_sources),
        "unobserved_metric_definition": (
            "dynamic locations excluding t=0 and every supplied observation; fluid and "
            "solid regions are also reported separately"
        ),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "interpretation": (
            "The formal P418 diffusion comparison is a computed full-field residual test, "
            "not a sparse-thermocouple reconstruction or experimental posterior."
        ),
    }
    if not summary["all_checks_passed"]:
        raise RuntimeError(json.dumps(summary, ensure_ascii=False, indent=2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README_CN.md").write_text(
        "# 扩散模型温度观测范围说明\n\n"
        "当前P418正式扩散模型不读取温度测点文件，观测点数量为`0`。它比较的是"
        "确定性图--Transformer与OpenFOAM全场之间的温度剩余误差，不是从少量热电偶"
        "重建三维温度场。\n\n"
        "如果今后加入真实温度观测，误差只在初始时刻以外、并且没有给定观测的区域"
        "计算；流体和颗粒区域还会分别报告。\n\n"
        "TESOMEX目前缺少可直接读取的三维热电偶坐标表、传感器动态响应、重复实验原始"
        "曲线以及传感器到P418网格的对应关系。因此稀疏实验重建仍未启动，也不会用"
        "人工选取的测点代替这些资料。\n\n"
        "可选的数值稀疏试验现在必须把来源写成`computed_openfoam_target`，并在掩码"
        "文件中声明点值确实来自OpenFOAM参考场。真实实验数据被标为"
        "`external_experiment`，程序禁止把它作为无误差节点值强行写回三维场。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
