#!/usr/bin/env python3
"""Evaluate a trained conservative P418 model on an independent packing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hccb_p418_comparison_contract import integrated_heat_transfer_metrics
from hccb_p418_conservative_mixed_operator import (
    HCCBP418ConservativeMixedOperator,
    load_regional_energy_flux_geometry,
    load_regional_mass_flux_geometry,
    regional_energy_balance,
    regional_mass_balance,
)
from hccb_p418_parametric_regional_operator import (
    collapse_mesh_to_level,
    load_p418_regional_mesh,
)
from train_hccb_p418_conservative_mixed_operator import (
    engineering_metrics,
    normalized_conditions,
    normalized_state,
    physical_state,
    regional_state_loss,
)
from train_hccb_p418_regional_operator import build_model, load_scales, volume_weighted_rmse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def training_flux_scales(summary: dict[str, object]) -> dict[str, float]:
    normalization = summary.get("normalization", {})
    required = (
        "internal_mass_scale_kg_s",
        "boundary_mass_scale_kg_s",
        "regional_incident_mass_scale_kg_s",
        "internal_energy_scale_W",
        "boundary_energy_scale_W",
        "regional_incident_energy_scale_W",
    )
    scales = {name: float(normalization[name]) for name in required}
    if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise ValueError("training mass and energy scales must be finite and positive")
    return scales


def validate_protocol(payload: dict[str, object]) -> None:
    if payload.get("status") != "cross_packing_model_protocol_ready_before_new_fields":
        raise ValueError("cross-packing protocol has an unexpected status")
    normalization = payload.get("normalization", {})
    if int(normalization.get("packing_seed", -1)) != 101:
        raise ValueError("cross-packing normalization must come from seed101")
    packings = list(payload.get("evaluation_packings", []))
    by_seed = {int(item["seed"]): item for item in packings}
    if set(by_seed) != {202, 303} or len(packings) != 2:
        raise ValueError("protocol must contain exactly seed202 and seed303")
    if by_seed[202].get("role") != "development_packing":
        raise ValueError("seed202 must be the development packing")
    if by_seed[303].get("role") != "final_zero_shot_packing":
        raise ValueError("seed303 must be the final zero-shot packing")
    for seed, item in by_seed.items():
        conditions = [str(value) for value in item.get("condition_ids", [])]
        if len(conditions) != 9 or len(set(conditions)) != 9:
            raise ValueError(f"seed{seed} must contain nine unique conditions")
    if payload.get("new_physical_parameter_values_added") != []:
        raise ValueError("protocol unexpectedly introduces physical parameter values")


def validate_seed101_training_record(
    *,
    summary: dict[str, object],
    architecture: str,
    split_name: str,
    training_statistics: Path,
) -> dict[str, Path]:
    if summary.get("status") != "conservative_mixed_operator_training_complete":
        raise ValueError("training summary is not a completed conservative seed101 model")
    if summary.get("architecture") != architecture:
        raise ValueError("training summary architecture differs from the requested model")
    if summary.get("split_name") != split_name:
        raise ValueError("training summary split differs from the fixed normalization split")
    if summary.get("normalization", {}).get("scales_use_training_cases_only") is not True:
        raise ValueError("training summary does not confirm train-only scaling")
    if summary.get("new_physical_parameters") != []:
        raise ValueError("training summary unexpectedly introduces physical parameters")

    provenance = summary.get("run_provenance", {})
    if provenance.get("architecture") != architecture:
        raise ValueError("training provenance architecture differs")
    if provenance.get("split_name") != split_name:
        raise ValueError("training provenance split differs")
    common = provenance.get("common_inputs", {})
    required = ("training_statistics", "mass_targets", "energy_targets")
    if any(name not in common for name in required):
        raise ValueError("training provenance lacks seed101 statistics or flux targets")
    if common["training_statistics"].get("sha256") != sha256(training_statistics):
        raise ValueError("training statistics differ from the file used for seed101 training")

    paths: dict[str, Path] = {}
    for name in required:
        path = Path(str(common[name].get("path", ""))).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"seed101 training input is missing: {path}")
        if common[name].get("sha256") != sha256(path):
            raise ValueError(f"seed101 {name} changed after model training")
        if "seed202" in str(path) or "seed303" in str(path):
            raise ValueError(f"seed101 training provenance points to an independent packing: {path}")
        paths[name] = path
    return paths


def write_or_verify_fixed_record(
    *,
    path: Path,
    architecture: str,
    checkpoint: Path,
    training_summary: Path,
    training_statistics: Path,
    protocol: Path,
) -> dict[str, object]:
    payload = {
        "status": "conservative_model_fixed_before_final_packing_evaluation",
        "architecture": architecture,
        "checkpoint_sha256": sha256(checkpoint),
        "training_summary_sha256": sha256(training_summary),
        "training_statistics_sha256": sha256(training_statistics),
        "protocol_sha256": sha256(protocol),
        "normalization_packing_seed": 101,
        "model_loaded_before_final_packing_fields": True,
        "statement": (
            "Architecture, seed101 training statistics, checkpoint settings and weights "
            "were checked and loaded before the final seed303 target fields were opened."
        ),
    }
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise ValueError("the existing conservative fixed-model record differs")
        return previous
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--packing-seed", type=int, choices=(202, 303), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("pinn_data_only", "pinn", "graph", "transolver"),
        required=True,
    )
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.microbatch_size <= 0:
        raise ValueError("microbatch size must be positive")

    root = args.project_root.resolve()
    protocol_path = args.protocol.resolve()
    checkpoint_path = args.checkpoint.resolve()
    training_summary_path = args.training_summary.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    normalization = protocol["normalization"]
    statistics_path = resolve(root, str(normalization["training_statistics"]))
    reference_topology = resolve(root, str(normalization["regional_topology"]))
    reference_geometry = resolve(root, str(normalization["model_geometry"]))
    required_common = (
        checkpoint_path,
        training_summary_path,
        statistics_path,
        reference_topology,
        reference_geometry,
    )
    missing_common = [str(path) for path in required_common if not path.is_file()]
    if missing_common:
        raise FileNotFoundError("missing trained-model files: " + ", ".join(missing_common))

    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    seed101_inputs = validate_seed101_training_record(
        summary=training_summary,
        architecture=args.architecture,
        split_name=str(normalization["split_name"]),
        training_statistics=statistics_path,
    )
    flux_scales = training_flux_scales(training_summary)
    scales = load_scales(statistics_path, str(normalization["split_name"]))

    # Build and load the complete model from seed101 files before any seed303
    # state, mass-flow or heat-flow target is opened.
    reference_mesh = collapse_mesh_to_level(
        load_p418_regional_mesh(reference_topology, reference_geometry),
        int(normalization["regional_level"]),
    )
    boundary_role_count = int(reference_mesh.fine_boundary_role.shape[1])
    with np.load(seed101_inputs["mass_targets"], allow_pickle=False) as loaded:
        seed101_boundary_patch = loaded["boundary_patch"].astype(np.int64)
    with np.load(seed101_inputs["energy_targets"], allow_pickle=False) as loaded:
        seed101_internal_energy_kind = loaded["internal_kind"].astype(np.int64)
        seed101_boundary_energy_kind = loaded["boundary_kind"].astype(np.int64)
    patch_count = int(np.max(seed101_boundary_patch)) + 1
    internal_energy_kind_count = int(np.max(seed101_internal_energy_kind)) + 1
    boundary_energy_kind_count = int(np.max(seed101_boundary_energy_kind)) + 1
    field_architecture = "pinn" if args.architecture == "pinn_data_only" else args.architecture
    field_model, settings = build_model(field_architecture, boundary_role_count)
    model = HCCBP418ConservativeMixedOperator(
        field_operator=field_model,
        patch_count=patch_count,
        internal_mass_scale_kg_s=flux_scales["internal_mass_scale_kg_s"],
        boundary_mass_scale_kg_s=flux_scales["boundary_mass_scale_kg_s"],
        internal_energy_scale_W=flux_scales["internal_energy_scale_W"],
        boundary_energy_scale_W=flux_scales["boundary_energy_scale_W"],
        internal_energy_kind_count=internal_energy_kind_count,
        boundary_energy_kind_count=boundary_energy_kind_count,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("settings") != settings:
        raise ValueError("checkpoint architecture settings differ from the fixed seed101 model")
    model.load_state_dict(checkpoint["model"])

    packing = next(
        item
        for item in protocol["evaluation_packings"]
        if int(item["seed"]) == args.packing_seed
    )
    output = args.output.resolve()
    if packing["role"] == "final_zero_shot_packing":
        if output.exists():
            raise FileExistsError("the first conservative seed303 result already exists")
        write_or_verify_fixed_record(
            path=output.parent / "conservative_model_fixed_before_seed303.json",
            architecture=args.architecture,
            checkpoint=checkpoint_path,
            training_summary=training_summary_path,
            training_statistics=statistics_path,
            protocol=protocol_path,
        )

    topology_path = resolve(root, str(packing["regional_topology"]))
    geometry_path = resolve(root, str(packing["model_geometry"]))
    state_path = resolve(root, str(packing["state_targets"]))
    mass_path = resolve(root, str(packing["mass_targets"]))
    energy_path = resolve(root, str(packing["energy_targets"]))
    required_packing = (topology_path, geometry_path, state_path, mass_path, energy_path)
    missing_packing = [str(path) for path in required_packing if not path.is_file()]
    if missing_packing:
        raise FileNotFoundError(
            f"seed{args.packing_seed} model inputs are incomplete: "
            + ", ".join(missing_packing)
        )

    with np.load(state_path, allow_pickle=False) as loaded:
        condition_ids = loaded["condition_id"].astype(str)
        condition_physical = loaded["condition_physical"].astype(np.float64)
        state_physical = loaded["state_physical"].astype(np.float64)
        node_type_np = loaded["node_type"].astype(np.int64)
        node_volume_np = loaded["node_volume_m3"].astype(np.float64)
    with np.load(mass_path, allow_pickle=False) as loaded:
        mass_ids = loaded["condition_id"].astype(str)
        internal_mass_target = loaded["internal_mass_flow_kg_s"].astype(np.float64)
        boundary_mass_target = loaded["boundary_mass_flow_kg_s"].astype(np.float64)
        boundary_owner = loaded["boundary_owner"].astype(np.int64)
        boundary_patch = loaded["boundary_patch"].astype(np.int64)
        boundary_area = loaded["boundary_face_area_m2"].astype(np.float64)
    with np.load(energy_path, allow_pickle=False) as loaded:
        energy_ids = loaded["condition_id"].astype(str)
        internal_energy_target = loaded["internal_energy_flow_W"].astype(np.float64)
        boundary_energy_target = loaded["boundary_energy_flow_W"].astype(np.float64)
        energy_source = loaded["node_source_power_W"].astype(np.float64)
        internal_energy_kind = loaded["internal_kind"].astype(np.int64)
        internal_energy_kind_name = loaded["internal_kind_name"].astype(str)
        boundary_energy_kind = loaded["boundary_kind"].astype(np.int64)
        boundary_energy_kind_name = loaded["boundary_kind_name"].astype(str)
    if not np.array_equal(condition_ids, mass_ids) or not np.array_equal(
        condition_ids, energy_ids
    ):
        raise ValueError("state, mass and energy condition orders differ")
    expected_ids = [str(value) for value in packing["condition_ids"]]
    if set(condition_ids) != set(expected_ids) or len(condition_ids) != 9:
        raise ValueError("independent packing targets differ from the nine-case plan")
    order = np.asarray(
        [{value: index for index, value in enumerate(condition_ids)}[value] for value in expected_ids],
        dtype=np.int64,
    )

    condition_normalized = normalized_conditions(condition_physical, scales)
    state_normalized = normalized_state(
        state_physical, condition_physical, node_type_np, scales
    )
    device = torch.device(args.device)
    mesh = collapse_mesh_to_level(
        load_p418_regional_mesh(topology_path, geometry_path),
        int(packing["regional_level"]),
    ).to(device)
    if not torch.equal(mesh.levels[0].node_type.cpu(), torch.as_tensor(node_type_np)):
        raise ValueError("independent-packing regional target and mesh nodes differ")

    if int(mesh.fine_boundary_role.shape[1]) != boundary_role_count:
        raise ValueError("independent packing has a different boundary-role definition")
    if int(np.max(boundary_patch)) + 1 != patch_count:
        raise ValueError("independent packing has a different mass-flow patch definition")
    if int(np.max(internal_energy_kind)) + 1 != internal_energy_kind_count:
        raise ValueError("independent packing has different internal heat-flow kinds")
    if int(np.max(boundary_energy_kind)) + 1 != boundary_energy_kind_count:
        raise ValueError("independent packing has different boundary heat-flow kinds")
    model = model.to(device)
    model.eval()
    mass_geometry = load_regional_mass_flux_geometry(
        mass_path, patch_count=patch_count, device=device, dtype=torch.float32
    )
    energy_geometry = load_regional_energy_flux_geometry(
        energy_path, device=device, dtype=torch.float32
    )
    condition_t = torch.as_tensor(condition_normalized, device=device)
    node_type = torch.as_tensor(node_type_np, device=device)
    node_volume = torch.as_tensor(node_volume_np, device=device, dtype=torch.float32)

    cases = []
    with torch.no_grad():
        for start in range(0, len(order), args.microbatch_size):
            selected_np = order[start : start + args.microbatch_size]
            selected = torch.as_tensor(selected_np, device=device, dtype=torch.long)
            prediction = model(
                condition_t[selected], mesh, mass_geometry, energy_geometry
            )
            mass_balance = regional_mass_balance(
                prediction, mass_geometry, int(np.count_nonzero(node_type_np == 0))
            ).cpu().numpy()
            energy_balance = regional_energy_balance(
                prediction,
                energy_geometry,
                torch.as_tensor(energy_source[selected_np], device=device, dtype=torch.float32),
            ).cpu().numpy()
            for local, case_index in enumerate(selected_np):
                predicted_normalized = prediction.regional_state[local]
                target_normalized = torch.as_tensor(
                    state_normalized[case_index], device=device, dtype=torch.float32
                ).unsqueeze(0)
                state_loss, state_channels = regional_state_loss(
                    predicted_normalized.unsqueeze(0), target_normalized, node_type, node_volume
                )
                predicted_state = physical_state(
                    predicted_normalized.cpu().numpy(),
                    condition_physical[case_index],
                    node_type_np,
                    scales,
                )
                reference_state = state_physical[case_index]
                predicted_engineering = engineering_metrics(
                    predicted_state,
                    boundary_owner=boundary_owner,
                    boundary_patch=boundary_patch,
                    boundary_area=boundary_area,
                    inlet_patch=0,
                    outlet_patch=1,
                    node_type=node_type_np,
                )
                reference_engineering = engineering_metrics(
                    reference_state,
                    boundary_owner=boundary_owner,
                    boundary_patch=boundary_patch,
                    boundary_area=boundary_area,
                    inlet_patch=0,
                    outlet_patch=1,
                    node_type=node_type_np,
                )
                predicted_heat = integrated_heat_transfer_metrics(
                    internal_energy_flow_w=prediction.internal_energy_flow_W[local].cpu().numpy(),
                    boundary_energy_flow_w=prediction.boundary_energy_flow_W[local].cpu().numpy(),
                    internal_kind=internal_energy_kind,
                    internal_kind_name=internal_energy_kind_name,
                    boundary_kind=boundary_energy_kind,
                    boundary_kind_name=boundary_energy_kind_name,
                )
                reference_heat = integrated_heat_transfer_metrics(
                    internal_energy_flow_w=internal_energy_target[case_index],
                    boundary_energy_flow_w=boundary_energy_target[case_index],
                    internal_kind=internal_energy_kind,
                    internal_kind_name=internal_energy_kind_name,
                    boundary_kind=boundary_energy_kind,
                    boundary_kind_name=boundary_energy_kind_name,
                )
                fluid = node_type_np == 0
                solid = node_type_np == 1
                predicted_hotspot = int(np.argmax(predicted_state[solid, 4]))
                reference_hotspot = int(np.argmax(reference_state[solid, 4]))
                solid_centroid = mesh.levels[0].centroid_m.cpu().numpy()[solid]
                generated_power = float(np.sum(energy_source[case_index]))
                inlet_mass = abs(
                    float(
                        np.sum(
                            boundary_mass_target[case_index][boundary_patch == 0]
                        )
                    )
                )
                cases.append(
                    {
                        "condition_id": str(condition_ids[case_index]),
                        "state_normalized_rmse": float(torch.sqrt(state_loss).cpu()),
                        "state_channel_rmse": torch.sqrt(state_channels).cpu().tolist(),
                        "fluid_temperature_volume_weighted_rmse_K": volume_weighted_rmse(
                            predicted_state[fluid, 4],
                            reference_state[fluid, 4],
                            node_volume_np[fluid],
                        ),
                        "solid_temperature_volume_weighted_rmse_K": volume_weighted_rmse(
                            predicted_state[solid, 4],
                            reference_state[solid, 4],
                            node_volume_np[solid],
                        ),
                        "solid_hotspot_location_error_m": float(
                            np.linalg.norm(
                                solid_centroid[predicted_hotspot]
                                - solid_centroid[reference_hotspot]
                            )
                        ),
                        "engineering_absolute_errors": {
                            **{
                                name: abs(predicted_engineering[name] - reference_engineering[name])
                                for name in predicted_engineering
                            },
                            **{
                                name: abs(predicted_heat[name] - reference_heat[name])
                                for name in predicted_heat
                            },
                        },
                        "local_mass_l1_over_two_inlet": float(
                            np.sum(np.abs(mass_balance[local])) / (2.0 * inlet_mass)
                        ),
                        "global_mass_imbalance_over_inlet": float(
                            abs(np.sum(mass_balance[local])) / inlet_mass
                        ),
                        "local_energy_l1_over_two_generated_power": float(
                            np.sum(np.abs(energy_balance[local]))
                            / (2.0 * generated_power)
                        ),
                        "global_energy_imbalance_over_generated_power": float(
                            abs(np.sum(energy_balance[local])) / generated_power
                        ),
                    }
                )

    result = {
        "status": "cross_packing_conservative_evaluation_complete",
        "packing_seed": args.packing_seed,
        "packing_role": packing["role"],
        "architecture": args.architecture,
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_summary_sha256": sha256(training_summary_path),
        "training_statistics_sha256": sha256(statistics_path),
        "normalization_packing_seed": 101,
        "model_loaded_before_this_packing_fields": True,
        "model_settings": settings,
        "case_count": len(cases),
        "cases": cases,
        "new_physical_parameter_values_added": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
