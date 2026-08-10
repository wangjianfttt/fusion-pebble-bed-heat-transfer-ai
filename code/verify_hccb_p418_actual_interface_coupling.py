#!/usr/bin/env python3
"""Verify conjugate heat transfer on the actual P418 regional interface.

The check uses the same interface map, conductivity functions and coupled-face
formula as the regional PINN/graph-operator physics term.  It does not train a
network and it does not assess prediction accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hccb_p418_regional_cht_adapter import load_p418_subface_geometry
from hccb_source_backed_thermophysical import (
    helium_thermal_conductivity,
    load_hccb_thermophysical_parameters,
    steady_li4sio4_conductivity_like,
)
from openfoam13_face_flux_reconstruction import coupled_temperature_interface


def interface_metrics(
    *,
    fluid_owner_temperature_k: torch.Tensor,
    solid_owner_temperature_k: torch.Tensor,
    interface_temperature_k: torch.Tensor,
    fluid_outward_heat_w: torch.Tensor,
    solid_outward_heat_w: torch.Tensor,
) -> dict[str, float | int]:
    """Return direct continuity, reciprocity and heat-direction checks."""
    tensors = (
        fluid_owner_temperature_k,
        solid_owner_temperature_k,
        interface_temperature_k,
        fluid_outward_heat_w,
        solid_outward_heat_w,
    )
    if any(value.ndim != 2 for value in tensors):
        raise ValueError("all interface arrays must have [condition,pair] shape")
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("all interface arrays must have the same shape")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("interface arrays contain non-finite values")

    flux_sum = fluid_outward_heat_w + solid_outward_heat_w
    flux_magnitude = torch.maximum(
        fluid_outward_heat_w.abs(), solid_outward_heat_w.abs()
    )
    global_flux_scale = flux_magnitude.max().clamp_min(
        torch.finfo(flux_sum.dtype).tiny
    )
    lower = torch.minimum(fluid_owner_temperature_k, solid_owner_temperature_k)
    upper = torch.maximum(fluid_owner_temperature_k, solid_owner_temperature_k)
    interval_violation = torch.maximum(
        lower - interface_temperature_k,
        interface_temperature_k - upper,
    ).clamp_min(0.0)
    fluid_direction = fluid_outward_heat_w * (
        fluid_owner_temperature_k - solid_owner_temperature_k
    )
    solid_direction = solid_outward_heat_w * (
        solid_owner_temperature_k - fluid_owner_temperature_k
    )
    temperature_difference = solid_owner_temperature_k - fluid_owner_temperature_k
    zero_flux_scale = torch.sqrt(
        torch.as_tensor(torch.finfo(flux_sum.dtype).eps, device=flux_sum.device)
    ) * global_flux_scale

    return {
        "condition_count": int(flux_sum.shape[0]),
        "interface_pair_count": int(flux_sum.shape[1]),
        "maximum_absolute_flux_sum_W": float(flux_sum.abs().max().cpu()),
        "maximum_flux_sum_over_global_interface_flux": float(
            (flux_sum.abs().max() / global_flux_scale).cpu()
        ),
        "maximum_absolute_interface_heat_per_face_W": float(global_flux_scale.cpu()),
        "maximum_interface_temperature_interval_violation_K": float(
            interval_violation.max().cpu()
        ),
        "minimum_fluid_heat_direction_product_W_K": float(fluid_direction.min().cpu()),
        "minimum_solid_heat_direction_product_W_K": float(solid_direction.min().cpu()),
        "mean_absolute_interface_heat_per_face_W": float(
            fluid_outward_heat_w.abs().mean().cpu()
        ),
        "maximum_absolute_owner_temperature_difference_K": float(
            temperature_difference.abs().max().cpu()
        ),
        "fluid_hotter_pair_count": int((temperature_difference < 0.0).sum().cpu()),
        "solid_hotter_pair_count": int((temperature_difference > 0.0).sum().cpu()),
        "near_zero_heat_pair_count": int(
            (fluid_outward_heat_w.abs() <= zero_flux_scale).sum().cpu()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subface-geometry", type=Path, required=True)
    parser.add_argument("--regional-state-targets", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    dtype = torch.float64

    dataset = json.loads(args.dataset_index.resolve().read_text(encoding="utf-8"))
    geometry = load_p418_subface_geometry(
        args.subface_geometry.resolve(),
        fluid_patch_names=dataset["boundary_patch_names"]["fluid"],
        solid_patch_names=dataset["boundary_patch_names"]["solid"],
        device=device,
        dtype=dtype,
    )
    with np.load(args.regional_state_targets.resolve(), allow_pickle=False) as loaded:
        condition_ids = loaded["condition_id"].astype(str).tolist()
        state = torch.as_tensor(loaded["state_physical"], dtype=dtype, device=device)
        fluid_global = torch.as_tensor(
            loaded["fluid_global_region"], dtype=torch.long, device=device
        )
        solid_global = torch.as_tensor(
            loaded["solid_global_region"], dtype=torch.long, device=device
        )

    fluid_state = state[:, fluid_global]
    solid_state = state[:, solid_global]
    fluid_owner = geometry.fluid_mesh.boundary_owner[
        geometry.interface.fluid_boundary_face
    ]
    solid_owner = geometry.solid_mesh.boundary_owner[
        geometry.interface.solid_boundary_face
    ]
    fluid_temperature = fluid_state[:, fluid_owner, 4]
    solid_temperature = solid_state[:, solid_owner, 4]
    fluid_pressure = fluid_state[:, fluid_owner, 3]
    parameters = load_hccb_thermophysical_parameters()
    fluid_conductivity = helium_thermal_conductivity(
        fluid_pressure, fluid_temperature
    )
    solid_conductivity = steady_li4sio4_conductivity_like(
        solid_temperature, parameters=parameters
    )
    interface_temperature, fluid_heat, solid_heat = coupled_temperature_interface(
        fluid_cell_temperature=fluid_temperature,
        solid_cell_temperature=solid_temperature,
        fluid_conductivity=fluid_conductivity,
        solid_conductivity=solid_conductivity,
        fluid_cell_centroid=geometry.fluid_mesh.cell_centroid[fluid_owner],
        solid_cell_centroid=geometry.solid_mesh.cell_centroid[solid_owner],
        face_centroid=geometry.fluid_mesh.boundary_face_centroid[
            geometry.interface.fluid_boundary_face
        ],
        fluid_outward_area_vector=geometry.fluid_mesh.boundary_area_vector[
            geometry.interface.fluid_boundary_face
        ],
    )
    metrics = interface_metrics(
        fluid_owner_temperature_k=fluid_temperature,
        solid_owner_temperature_k=solid_temperature,
        interface_temperature_k=interface_temperature,
        fluid_outward_heat_w=fluid_heat,
        solid_outward_heat_w=solid_heat,
    )

    per_condition = []
    for index, condition_id in enumerate(condition_ids):
        item = interface_metrics(
            fluid_owner_temperature_k=fluid_temperature[index : index + 1],
            solid_owner_temperature_k=solid_temperature[index : index + 1],
            interface_temperature_k=interface_temperature[index : index + 1],
            fluid_outward_heat_w=fluid_heat[index : index + 1],
            solid_outward_heat_w=solid_heat[index : index + 1],
        )
        item["condition_id"] = condition_id
        per_condition.append(item)

    passed = (
        metrics["maximum_flux_sum_over_global_interface_flux"] <= 1.0e-12
        and metrics["maximum_interface_temperature_interval_violation_K"] <= 1.0e-12
        and metrics["minimum_fluid_heat_direction_product_W_K"] >= -1.0e-12
        and metrics["minimum_solid_heat_direction_product_W_K"] >= -1.0e-12
    )
    summary = {
        "status": (
            "actual_regional_interface_temperature_and_heat_flux_consistent"
            if passed
            else "actual_regional_interface_consistency_failed"
        ),
        "inputs": {
            "subface_geometry": str(args.subface_geometry.resolve()),
            "regional_state_targets": str(args.regional_state_targets.resolve()),
            "dataset_index": str(args.dataset_index.resolve()),
            "device": str(device),
            "dtype": str(dtype),
        },
        "equation_use": (
            "The same coupled_temperature_interface function is called by the "
            "steady and transient regional energy equations."
        ),
        "all_conditions": metrics,
        "per_condition": per_condition,
        "physical_parameter_ids": ["P388", "P389", "P403", "P418"],
        "new_physical_parameters": [],
        "scientific_scope": (
            "Direct equation and actual-geometry verification. It proves common "
            "interface temperature, reciprocal heat flux and correct heat direction "
            "for the supplied regional states; it is not a neural prediction-accuracy result."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = [
        "# P418实际区域图流固界面换热检查",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 三维工况数：`{metrics['condition_count']}`",
        f"- 流固界面配对数：`{metrics['interface_pair_count']}`",
        f"- 两侧热流和的最大绝对值：`{metrics['maximum_absolute_flux_sum_W']:.6e} W`",
        f"- 两侧热流和相对整组最大界面热流的比例：`{metrics['maximum_flux_sum_over_global_interface_flux']:.6e}`",
        f"- 界面温度超出两侧单元温度范围的最大值：`{metrics['maximum_interface_temperature_interval_violation_K']:.6e} K`",
        "",
        "程序在真实区域图和已完成的三维温度场上逐个检查流固界面。流体侧和颗粒侧使用同一个界面温度；一侧流出的热量等于另一侧流入的热量；热量方向与两侧温差一致。",
        "",
        "这项结果说明网络训练所用的流固界面方程在实际区域图上满足共轭换热关系。它不表示网络已经预测准确，也不替代OpenFOAM结果与实验数据的比较。",
    ]
    (args.output_dir / "README_CN.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
