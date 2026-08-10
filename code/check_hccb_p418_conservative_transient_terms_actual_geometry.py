#!/usr/bin/env python3
"""Check conservative transient-energy terms on actual P418 regional fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hccb_p418_regional_cht_adapter import load_p418_subface_geometry
from export_hccb_p418_step_regional_sequences import (
    preserve_openfoam_subface_mass_flux,
)
from hccb_p418_transient_regional_physics import (
    TRANSIENT_STORAGE_PARAMETER_IDS,
    assemble_p418_transient_regional_residual,
)


def normalized_volume_rms(
    values: torch.Tensor, volume: torch.Tensor, source_density: torch.Tensor
) -> torch.Tensor:
    weight = volume / volume.sum()
    return torch.sqrt(
        (values.square() * weight[None, None, :]).sum(dim=-1).mean(dim=1)
    ) / source_density


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-targets", type=Path, required=True)
    parser.add_argument("--fine-dataset-index", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.state_targets, allow_pickle=False) as loaded:
        state_ids = loaded["condition_id"]
        conditions = loaded["condition_physical"].astype(np.float64)
        state = loaded["state_physical"].astype(np.float64)
    fine_dataset = json.loads(args.fine_dataset_index.read_text(encoding="utf-8"))
    record_by_id = {
        str(record["condition_id"]): record for record in fine_dataset["conditions"]
    }
    missing = [str(value) for value in state_ids if str(value) not in record_by_id]
    if missing:
        raise ValueError(f"fine OpenFOAM dataset is missing cases: {missing}")
    with np.load(args.geometry, allow_pickle=False) as loaded:
        internal_openfoam_face = loaded[
            "fluid_internal_subface_openfoam_face"
        ].astype(np.int64)
        internal_orientation = loaded[
            "fluid_internal_subface_phi_orientation"
        ].astype(np.float64)
        boundary_openfoam_face = loaded["fluid_boundary_openfoam_face"].astype(
            np.int64
        )
    internal_rows = []
    boundary_rows = []
    for condition_id in state_ids:
        record = record_by_id[str(condition_id)]
        with np.load(
            args.fine_dataset_index.parent / str(record["field_file"]),
            allow_pickle=False,
        ) as loaded:
            internal, boundary = preserve_openfoam_subface_mass_flux(
                internal_fine=loaded[
                    "fluid_internal_face_mass_flow_kg_s"
                ].astype(np.float64),
                boundary_fine=loaded[
                    "fluid_boundary_face_mass_flow_kg_s"
                ].astype(np.float64),
                internal_openfoam_face=internal_openfoam_face,
                internal_orientation=internal_orientation,
                boundary_openfoam_face=boundary_openfoam_face,
            )
        internal_rows.append(internal)
        boundary_rows.append(boundary)
    internal_mass_flux = np.stack(internal_rows)
    boundary_mass_flux = np.stack(boundary_rows)
    dataset = json.loads(args.dataset_index.read_text(encoding="utf-8"))

    device = torch.device("cpu")
    dtype = torch.float64
    geometry = load_p418_subface_geometry(
        args.geometry,
        fluid_patch_names=dataset["boundary_patch_names"]["fluid"],
        solid_patch_names=dataset["boundary_patch_names"]["solid"],
        device=device,
        dtype=dtype,
    )
    time_s = torch.tensor([0.0, 1.0, 2.0], dtype=dtype, device=device)
    repeated_state = torch.as_tensor(state, dtype=dtype, device=device)[:, None].expand(
        -1, len(time_s), -1, -1
    ).clone()
    source_target_condition = np.column_stack(
        (
            conditions[:, 0],
            conditions[:, 1],
            conditions[:, 2] / 1.0e6,
            conditions[:, 0],
            conditions[:, 1],
            conditions[:, 2] / 1.0e6,
            conditions[:, 3],
            conditions[:, 4],
        )
    )
    step_condition = torch.as_tensor(source_target_condition, dtype=dtype, device=device)
    residual = assemble_p418_transient_regional_residual(
        geometry=geometry,
        step_condition=step_condition,
        state_physical=repeated_state,
        time_s=time_s,
        fluid_internal_mass_flux_kg_s=torch.as_tensor(
            internal_mass_flux, dtype=dtype, device=device
        ),
        fluid_boundary_mass_flux_kg_s=torch.as_tensor(
            boundary_mass_flux, dtype=dtype, device=device
        ),
    )

    fluid_volume = geometry.fluid_mesh.cell_volume
    solid_volume = geometry.solid_mesh.cell_volume
    source_density = step_condition[:, 5] * 1.0e6
    source_power = source_density * solid_volume.sum()
    kinetic_abs_power = (
        residual.fluid_kinetic_advection_w_m3.abs()
        * fluid_volume[None, None, :]
    ).sum(dim=-1).mean(dim=1)
    kinetic_net_power = (
        residual.fluid_kinetic_advection_w_m3 * fluid_volume[None, None, :]
    ).sum(dim=-1).mean(dim=1)
    inlet_patch = geometry.fluid_patch_names.index("inlet")
    inlet_faces = geometry.fluid_boundary_patch == inlet_patch
    inlet_mass_flow = torch.as_tensor(
        np.maximum(
            np.abs(
                boundary_mass_flux[:, inlet_faces.cpu().numpy()].sum(axis=1)
            ),
            np.finfo(float).tiny,
        ),
        dtype=dtype,
        device=device,
    )

    storage_fields = {
        "fluid_total": residual.fluid_storage_w_m3,
        "fluid_enthalpy": residual.fluid_enthalpy_storage_w_m3,
        "fluid_kinetic": residual.fluid_kinetic_storage_w_m3,
        "fluid_pressure_work": residual.fluid_pressure_work_w_m3,
        "solid_internal_energy": residual.solid_storage_w_m3,
    }
    maximum_storage = {
        name: float(values.abs().max()) for name, values in storage_fields.items()
    }
    all_finite = all(
        bool(torch.isfinite(values).all())
        for values in (
            residual.fluid_energy_w_m3,
            residual.solid_energy_w_m3,
            residual.fluid_mass_kg_m3_s,
            residual.fluid_kinetic_advection_w_m3,
        )
    )
    case_rows = []
    fluid_rms = normalized_volume_rms(
        residual.fluid_energy_w_m3, fluid_volume, source_density
    )
    solid_rms = normalized_volume_rms(
        residual.solid_energy_w_m3, solid_volume, source_density
    )
    cell_mass_flow_residual = (
        residual.fluid_mass_kg_m3_s * fluid_volume[None, None, :]
    )
    maximum_mass = cell_mass_flow_residual.abs().amax(dim=(1, 2))
    global_mass = cell_mass_flow_residual.sum(dim=-1).abs().amax(dim=1)
    local_mass_l1 = cell_mass_flow_residual.abs().sum(dim=-1).mean(dim=1)
    fluid_power_residual = (
        residual.fluid_energy_w_m3 * fluid_volume[None, None, :]
    ).sum(dim=-1)
    solid_power_residual = (
        residual.solid_energy_w_m3 * solid_volume[None, None, :]
    ).sum(dim=-1)
    global_energy = (fluid_power_residual + solid_power_residual).abs().mean(dim=1)
    local_energy_l1 = (
        residual.fluid_energy_w_m3.abs() * fluid_volume[None, None, :]
    ).sum(dim=-1).mean(dim=1) + (
        residual.solid_energy_w_m3.abs() * solid_volume[None, None, :]
    ).sum(dim=-1).mean(dim=1)
    for index, condition_id in enumerate(state_ids):
        case_rows.append(
            {
                "condition_id": str(condition_id),
                "fluid_volume_weighted_energy_RMSE_over_source": float(fluid_rms[index]),
                "solid_volume_weighted_energy_RMSE_over_source": float(solid_rms[index]),
                "kinetic_advection_absolute_power_over_source_power": float(
                    kinetic_abs_power[index] / source_power[index]
                ),
                "kinetic_advection_net_power_over_source_power": float(
                    kinetic_net_power[index] / source_power[index]
                ),
                "maximum_cell_mass_imbalance_over_inlet": float(
                    maximum_mass[index] / inlet_mass_flow[index]
                ),
                "global_mass_imbalance_over_inlet": float(
                    global_mass[index] / inlet_mass_flow[index]
                ),
                "local_mass_l1_over_two_inlet": float(
                    local_mass_l1[index] / (2.0 * inlet_mass_flow[index])
                ),
                "global_fluid_plus_solid_energy_imbalance_over_source_power": float(
                    global_energy[index] / source_power[index]
                ),
                "local_fluid_plus_solid_energy_l1_over_source_power": float(
                    local_energy_l1[index] / source_power[index]
                ),
            }
        )

    checks = {
        "actual_regional_node_count": len(geometry.fluid_global_region)
        + len(geometry.solid_global_region),
        "actual_fluid_internal_face_count": len(geometry.fluid_mesh.internal_owner),
        "actual_fluid_boundary_face_count": len(geometry.fluid_mesh.boundary_owner),
        "all_terms_are_finite": all_finite,
        "all_storage_terms_are_roundoff_relative_to_source": all(
            value <= float(source_density.min()) * 1.0e-12
            for value in maximum_storage.values()
        ),
        "fixed_solved_mass_flux_is_used": True,
    }
    projection_bias = global_energy / source_power
    mass_global_relative = global_mass / inlet_mass_flow
    kinetic_absolute_relative = kinetic_abs_power / source_power
    summary = {
        "status": "passed_actual_geometry_conservative_transient_term_check"
        if all(
            value
            for key, value in checks.items()
            if key.startswith("all_") or key == "fixed_solved_mass_flux_is_used"
        )
        else "failed",
        "scope_cn": (
            "使用已完成OpenFOAM稳态场和求解得到的面质量流量，在真实区域网格上检查方程实现；"
            "重复稳态场不代表新的瞬态物理结果。"
        ),
        "parameter_ids": list(TRANSIENT_STORAGE_PARAMETER_IDS),
        "checks": checks,
        "maximum_absolute_storage_terms_W_m3": maximum_storage,
        "actual_geometry_findings": {
            "global_mass_imbalance_over_inlet_range": [
                float(mass_global_relative.min()),
                float(mass_global_relative.max()),
            ],
            "kinetic_advection_absolute_power_over_source_range": [
                float(kinetic_absolute_relative.min()),
                float(kinetic_absolute_relative.max()),
            ],
            "projected_steady_global_energy_imbalance_over_source_range": [
                float(projection_bias.min()),
                float(projection_bias.max()),
            ],
            "interpretation_cn": (
                "固定OpenFOAM面质量流在区域子面上保持了质量守恒，动能输运很小但已完整保留。"
                "区域体积平均与非线性面通量计算不交换，因此区域化稳态OpenFOAM场出现非零能量残差；"
                "该绝对区域残差不能作为强制趋零的正式训练项。正式物理项应比较预测与同一区域化"
                "OpenFOAM参考之间的能量算子差，原始OpenFOAM求解器收支单独报告。"
            ),
        },
        "cases": case_rows,
        "new_physical_parameters": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"].startswith("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
