#!/usr/bin/env python3
"""Verify the physical-state boundaries of the P418 diffusion correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hccb_p418_temporal_temperature_diffusion import (
    sample_temporal_temperature_residual,
)
from summarize_hccb_p418_step_model_comparison import (
    diffusion_temperature_energy_decision,
)
from train_hccb_p418_temporal_temperature_diffusion import (
    physical_temperature_state,
)


class ZeroVelocityModel(torch.nn.Module):
    """Minimal deterministic model used only to exercise reverse conditioning."""

    def forward(self, baseline, noised_residual, *unused):  # type: ignore[override]
        return torch.zeros_like(noised_residual)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hccb_p418_diffusion_physical_state"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    batch, times, nodes = 2, 4, 6
    baseline = torch.zeros((batch, times, nodes, 1), device=device)
    observed_residual = torch.zeros_like(baseline)
    observation_mask = torch.zeros_like(baseline, dtype=torch.bool)
    observed_residual[:, 0, 1, 0] = 9.0
    observation_mask[:, 0, 1, 0] = True
    observed_residual[:, 2, 4, 0] = 0.35
    observation_mask[:, 2, 4, 0] = True
    residual = sample_temporal_temperature_residual(
        ZeroVelocityModel().to(device).eval(),
        baseline,
        torch.zeros((batch, 8), device=device),
        torch.zeros((nodes, 6), device=device),
        torch.linspace(0.0, 1.0, times, device=device),
        observed_residual,
        observation_mask,
        initial_noise=torch.ones_like(baseline),
    )
    initial_residual_max = float(residual[:, 0].abs().max().cpu())
    dynamic_observation_error = float(
        (residual[:, 2, 4, 0] - observed_residual[:, 2, 4, 0]).abs().max().cpu()
    )

    fixed_hydrodynamics = np.arange(nodes * 4, dtype=np.float32).reshape(nodes, 4)
    node_type = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    physical_state = physical_temperature_state(
        np.zeros((times, nodes, 1), dtype=np.float32),
        fixed_hydrodynamics,
        node_type,
        np.asarray([300.0, 600.0]),
        np.asarray([10.0, 20.0]),
    )
    expected_hydrodynamics = np.broadcast_to(
        fixed_hydrodynamics, (times, nodes, 4)
    )
    hydrodynamic_change_max = float(
        np.max(np.abs(physical_state[..., :4] - expected_hydrodynamics))
    )

    energy_worse = diffusion_temperature_energy_decision(3.0, 2.0, 0.20, 0.30)
    both_improved = diffusion_temperature_energy_decision(3.0, 2.0, 0.20, 0.15)
    checks = {
        "initial_temperature_residual_exactly_zero": initial_residual_max == 0.0,
        "dynamic_temperature_observation_imposed_exactly": (
            dynamic_observation_error == 0.0
        ),
        "velocity_and_pressure_unchanged": hydrodynamic_change_max == 0.0,
        "lower_temperature_error_with_worse_energy_not_joint_improvement": (
            energy_worse["held_out_outcome"] == "not_joint_improvement"
        ),
        "lower_temperature_error_with_nonworse_energy_is_joint_improvement": (
            both_improved["held_out_outcome"] == "joint_improvement"
        ),
        "held_out_outcome_not_used_for_model_selection": (
            energy_worse["model_selection_uses_this_outcome"] is False
            and both_improved["model_selection_uses_this_outcome"] is False
        ),
    }
    summary = {
        "status": "completed_p418_diffusion_physical_state_verification",
        "device": str(device),
        "torch_version": torch.__version__,
        "corrected_state_channels": ["temperature"],
        "fixed_state_channels": [
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "pressure",
        ],
        "maximum_absolute_initial_temperature_residual": initial_residual_max,
        "maximum_absolute_dynamic_observation_error": dynamic_observation_error,
        "maximum_absolute_fixed_hydrodynamic_change": hydrodynamic_change_max,
        "temperature_better_energy_worse_case": energy_worse,
        "temperature_and_energy_better_case": both_improved,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "interpretation": (
            "This verifies state preservation and the held-out temperature--energy description. "
            "The description does not select or update the model. "
            "It is not a prediction-accuracy result; accuracy requires the completed "
            "OpenFOAM thermal-step histories."
        ),
    }
    if not summary["all_checks_passed"]:
        raise RuntimeError(json.dumps(summary, ensure_ascii=False, indent=2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README_CN.md").write_text(
        "# 扩散温度修正的物理范围检查\n\n"
        "本检查使用实际扩散采样函数和模型比较函数，确认以下四点：\n\n"
        "1. 扩散模型只修正温度，不改变速度和压力；\n"
        "2. 初始时刻的温度修正严格为零，即使输入观测与初始状态冲突；\n"
        "3. 指定的后续温度观测会被严格满足；\n"
        "4. 如果温度误差降低但能量方程误差增加，程序不会采用该修正结果。\n\n"
        f"本次运行的最大初始温度残差为 `{initial_residual_max:.3e}`，"
        f"速度和压力最大改变量为 `{hydrodynamic_change_max:.3e}`，"
        f"指定温度观测最大偏差为 `{dynamic_observation_error:.3e}`。\n\n"
        "这一步证明的是程序没有越过预先规定的物理范围，不代表模型已经达到预测精度；"
        "正式精度仍需等待12组三维热阶跃数据。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
