#!/usr/bin/env python3
"""Verify that the recorded P418 model settings match the current code.

The table contains neural-network and numerical choices, not pebble-bed
physical parameters.  This script checks every table row against either a
literal code constant or the exact selection/formula text used by the current
implementation.  It deliberately does not import PyTorch, so it can run on the
local manuscript workstation as well as the remote GPU/CFD machine.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def literal_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return values


def normalized_value(value: str) -> Any:
    text = value.strip()
    if text == "validation minimum":
        return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def require_text(path: Path, *parts: str) -> str:
    text = path.read_text(encoding="utf-8")
    missing = [part for part in parts if part not in text]
    if missing:
        raise ValueError(f"{path} lacks code fragments {missing}")
    return "；".join(parts)


def direct_setting_map(root: Path) -> dict[tuple[str, str], tuple[Any, str, str]]:
    observable = root / "code/train_hccb_p418_transient_observable_transformer.py"
    graph_model = root / "code/hccb_p418_spatiotemporal_regional_operator.py"
    graph_train = root / "code/train_hccb_p418_spatiotemporal_regional_operator.py"
    fully_coupled_model = root / "code/hccb_p418_fully_coupled_spatiotemporal_operator.py"
    diffusion_model = root / "code/hccb_p418_temporal_temperature_diffusion.py"
    diffusion_train = root / "code/train_hccb_p418_temporal_temperature_diffusion.py"
    dmdc = root / (
        "reproducibility/formal_training_sources/"
        "train_hccb_p418_regional_dmdc_20260729.py"
    )
    steady_train = root / "code/train_hccb_p418_conservative_mixed_operator.py"
    thermophysical = root / "code/hccb_source_backed_thermophysical.py"
    loss_balancing = root / "code/hccb_p418_loss_balancing.py"
    loss_sources_path = root / "parameters/hccb_p418_loss_balancing_sources.json"

    obs = literal_assignments(observable)
    graph = literal_assignments(graph_model)
    graph_training = literal_assignments(graph_train)
    fully_coupled = literal_assignments(fully_coupled_model)
    diffusion = literal_assignments(diffusion_model)
    diffusion_training = literal_assignments(diffusion_train)
    dmd = literal_assignments(dmdc)
    steady = literal_assignments(steady_train)
    thermo = literal_assignments(thermophysical)
    loss_sources = json.loads(loss_sources_path.read_text(encoding="utf-8"))

    oa = obs["FORMAL_ARCHITECTURE"]
    ot = obs["FORMAL_TRAINING"]
    ga = graph["FORMAL_ARCHITECTURE"]
    gt = graph_training["FORMAL_TRAINING"]
    da = diffusion["FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE"]
    dt = diffusion_training["FORMAL_TRAINING"]
    steady_weights = steady["STEADY_CONSERVATIVE_LOSS_WEIGHTS"]
    loss_candidates = {
        row["candidate_id"]: row for row in loss_sources["formal_candidates"]
    }
    fixed_loss = loss_candidates["fixed_equal_dimensionless"]
    burgers_loss = loss_candidates["relobralo_burgers_table_viii"]
    kirchhoff_loss = loss_candidates["relobralo_kirchhoff_table_viii"]
    helmholtz_loss = loss_candidates["relobralo_helmholtz_table_viii"]

    result: dict[tuple[str, str], tuple[Any, str, str]] = {
        ("工程量时间Transformer", "hidden_width"): (oa["d_model"], "代码固定值一致", str(observable.relative_to(root))),
        ("工程量时间Transformer", "layers"): (oa["layers"], "代码固定值一致", str(observable.relative_to(root))),
        ("工程量时间Transformer", "attention_heads"): (oa["heads"], "代码固定值一致", str(observable.relative_to(root))),
        ("工程量时间Transformer", "feedforward_ratio"): (1, "代码结构一致", str(observable.relative_to(root))),
        ("工程量时间Transformer", "dropout"): (0.0, "代码固定值一致", str(observable.relative_to(root))),
        ("工程量时间Transformer", "epochs"): (ot["epochs"], "代码固定值一致", str(observable.relative_to(root))),
        ("工程量时间Transformer", "effective_batch_size"): (ot["batch_size"], "代码固定值一致", str(observable.relative_to(root))),
        ("工程量时间Transformer", "learning_rate"): (ot["learning_rate"], "代码固定值一致", str(observable.relative_to(root))),
        ("工程量时间Transformer", "weight_decay"): (ot["weight_decay"], "代码固定值一致", str(observable.relative_to(root))),
        ("工程量时间Transformer", "selected_epoch"): ("validation minimum", "验证工况选择规则一致", str(observable.relative_to(root))),

        ("图-Transformer", "hidden_width"): (ga["hidden_dim"], "代码固定值一致", str(graph_model.relative_to(root))),
        ("图-Transformer", "preprocessor_mpnn_iterations"): (ga["local_pre_iterations"], "代码固定值一致", str(graph_model.relative_to(root))),
        ("图-Transformer", "physics_attention_blocks"): (ga["physics_attention_blocks"], "代码固定值一致", str(graph_model.relative_to(root))),
        ("图-Transformer", "refinement_mpnn_iterations"): (ga["local_post_iterations"], "代码固定值一致", str(graph_model.relative_to(root))),
        ("图-Transformer", "physics_attention_heads"): (ga["physics_attention_heads"], "代码固定值一致", str(graph_model.relative_to(root))),
        ("图-Transformer", "physics_slices"): (ga["physics_slices"], "代码固定值一致", str(graph_model.relative_to(root))),
        ("图-Transformer", "temporal_layers"): (ga["temporal_layers"], "代码固定值一致", str(graph_model.relative_to(root))),
        ("图-Transformer", "temporal_heads"): (ga["temporal_heads"], "代码固定值一致", str(graph_model.relative_to(root))),
        ("图-Transformer", "leaky_relu_negative_slope"): (ga["leaky_relu_negative_slope"], "代码固定值一致", str(graph_model.relative_to(root))),
        ("图-Transformer", "epochs"): (gt["epochs"], "代码固定值一致", str(graph_train.relative_to(root))),
        ("图-Transformer", "learning_rate"): (gt["learning_rate"], "代码固定值一致", str(graph_train.relative_to(root))),
        ("图-Transformer", "weight_decay"): (gt["weight_decay"], "代码固定值一致", str(graph_train.relative_to(root))),
        ("图-Transformer", "temperature_data_weight"): (gt["data_weight"], "代码固定值一致", str(graph_train.relative_to(root))),
        ("图-Transformer", "edge_flux_weight"): (gt["edge_flux_weight"], "代码固定值一致", str(graph_train.relative_to(root))),
        ("图-Transformer", "energy_balance_weight"): (gt["energy_weight"], "代码固定值一致", str(graph_train.relative_to(root))),
        ("图-Transformer", "selected_epoch"): ("validation minimum", "验证工况选择规则一致", str(graph_train.relative_to(root))),
        ("图-Transformer", "fluid_temperature_output_range_K"): (
            "300--1000",
            "文献给定氦温度范围与有界输出代码一致",
            str(graph_train.relative_to(root)),
        ),
        ("图-Transformer", "solid_temperature_output_range_K"): (
            "298--1300",
            "文献给定固体热容适用范围与有界输出代码一致",
            str(graph_train.relative_to(root)),
        ),
        ("图-Transformer", "structural_feature_order"): (
            "x,y,z,log_volume,fluid,solid,inlet,outlet,cooling_wall,symmetry,fluid_solid_interface",
            "区域结构量和五类边界面积占比与代码一致",
            str(graph_model.relative_to(root)),
        ),
        ("全耦合图-Transformer", "structural_feature_order"): (
            "x,y,z,log_volume,fluid,solid,inlet,outlet,cooling_wall,symmetry,fluid_solid_interface",
            "区域结构量和五类边界面积占比与代码一致",
            str(fully_coupled_model.relative_to(root)),
        ),
        ("全耦合图-Transformer", "initial_internal_face_flux_message"): (
            fully_coupled["FULLY_COUPLED_FACE_FLUX_CONTEXT"][
                "initial_internal_face_flux_message"
            ],
            "内部面有方向质量流向owner和neighbour传递的代码规则一致",
            str(fully_coupled_model.relative_to(root)),
        ),
        ("全耦合图-Transformer", "initial_active_boundary_face_flux_message"): (
            fully_coupled["FULLY_COUPLED_FACE_FLUX_CONTEXT"][
                "initial_active_boundary_face_flux_message"
            ],
            "入口和出口面质量流向相邻流体区域传递的代码规则一致",
            str(fully_coupled_model.relative_to(root)),
        ),
        ("全耦合图-Transformer", "initial_face_flux_aggregation"): (
            fully_coupled["FULLY_COUPLED_FACE_FLUX_CONTEXT"][
                "initial_face_flux_aggregation"
            ],
            "相邻有效面信息按面数平均的代码规则一致",
            str(fully_coupled_model.relative_to(root)),
        ),

        ("DMDc", "candidate_pod_ranks"): (";".join(map(str, dmd["DEFAULT_RANKS"])), "代码固定值一致", str(dmdc.relative_to(root))),

        ("稳态多输出模型", "state_data_weight"): (steady_weights["state_data"], "代码固定值一致", str(steady_train.relative_to(root))),
        ("稳态多输出模型", "face_flux_weight"): (steady_weights["face_flux"], "代码固定值一致", str(steady_train.relative_to(root))),
        ("稳态多输出模型", "physics_balance_weight"): (steady_weights["physics_balance"], "代码固定值一致", str(steady_train.relative_to(root))),

        ("扩散剩余误差修正", "hidden_width"): (da["hidden_dim"], "代码固定值一致", str(diffusion_model.relative_to(root))),
        ("扩散剩余误差修正", "spatial_layers"): (da["spatial_layers"], "代码固定值一致", str(diffusion_model.relative_to(root))),
        ("扩散剩余误差修正", "spatial_attention_heads"): (da["spatial_attention_heads"], "代码固定值一致", str(diffusion_model.relative_to(root))),
        ("扩散剩余误差修正", "physics_slices"): (da["physics_slices"], "代码固定值一致", str(diffusion_model.relative_to(root))),
        ("扩散剩余误差修正", "temporal_layers"): (da["temporal_layers"], "代码固定值一致", str(diffusion_model.relative_to(root))),
        ("扩散剩余误差修正", "temporal_heads"): (da["temporal_heads"], "代码固定值一致", str(diffusion_model.relative_to(root))),
        ("扩散剩余误差修正", "num_refinement_steps"): (da["num_refinement_steps"], "代码固定值一致", str(diffusion_model.relative_to(root))),
        ("扩散剩余误差修正", "minimum_noise_standard_deviation"): (da["minimum_noise_standard_deviation"], "代码固定值一致", str(diffusion_model.relative_to(root))),
        ("扩散剩余误差修正", "prediction_type"): ("v_prediction", "扩散预测公式一致", str(diffusion_model.relative_to(root))),
        ("扩散剩余误差修正", "ema_decay"): (dt["ema_decay"], "代码固定值一致", str(diffusion_train.relative_to(root))),
        ("扩散剩余误差修正", "epochs"): (dt["epochs"], "代码固定值一致", str(diffusion_train.relative_to(root))),
        ("扩散剩余误差修正", "effective_batch_size"): (dt["batch_size"], "代码固定值一致", str(diffusion_train.relative_to(root))),
        ("扩散剩余误差修正", "microbatch_size"): (dt["microbatch_size"], "代码固定值一致", str(diffusion_train.relative_to(root))),
        ("扩散剩余误差修正", "activation_precision"): (dt["activation_precision"], "代码固定值一致", str(diffusion_train.relative_to(root))),
        ("扩散剩余误差修正", "temporal_node_chunk_size"): (dt["temporal_node_chunk_size"], "56时刻真实网格显存实测后的代码固定值一致", str(diffusion_train.relative_to(root))),
        ("扩散剩余误差修正", "ensemble_samples"): (32, "代码固定值一致", str(diffusion_train.relative_to(root))),
        ("扩散剩余误差修正", "selected_epoch"): ("validation minimum", "验证工况选择规则一致", str(diffusion_train.relative_to(root))),

        ("稳态Transolver", "microbatch_size"): (1, "显存实测后的正式运行设置一致", "code/run_hccb_p418_60_model_comparison.sh"),
        ("稳态图网络", "microbatch_size"): (1, "显存实测后的正式运行设置一致", "code/run_hccb_p418_60_model_comparison.sh"),
        ("共同能量算子", "enthalpy_reference_temperature_K"): (
            thermo["OPENFOAM_TSTD_K"],
            "OpenFOAM焓参考温度代码常量一致",
            str(thermophysical.relative_to(root)),
        ),
        ("全耦合损失权重", "fixed_state_weight"): (
            fixed_loss["state_weight"],
            "候选来源表与代码入口一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "fixed_face_flux_weight"): (
            fixed_loss["face_flux_weight"],
            "候选来源表与代码入口一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "fixed_physics_weight"): (
            fixed_loss["physics_weight"],
            "候选来源表与代码入口一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "burgers_temperature"): (
            burgers_loss["temperature"],
            "Bischof和Kraus表VIII候选值一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "burgers_alpha"): (
            burgers_loss["alpha"],
            "Bischof和Kraus表VIII候选值一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "burgers_expected_rho"): (
            burgers_loss["expected_rho"],
            "Bischof和Kraus表VIII候选值一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "kirchhoff_temperature"): (
            kirchhoff_loss["temperature"],
            "Bischof和Kraus表VIII候选值一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "kirchhoff_alpha"): (
            kirchhoff_loss["alpha"],
            "Bischof和Kraus表VIII候选值一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "kirchhoff_expected_rho"): (
            kirchhoff_loss["expected_rho"],
            "Bischof和Kraus表VIII候选值一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "helmholtz_temperature"): (
            helmholtz_loss["temperature"],
            "Bischof和Kraus表VIII候选值一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "helmholtz_alpha"): (
            helmholtz_loss["alpha"],
            "Bischof和Kraus表VIII候选值一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "helmholtz_expected_rho"): (
            helmholtz_loss["expected_rho"],
            "Bischof和Kraus表VIII候选值一致",
            str(loss_sources_path.relative_to(root)),
        ),
        ("全耦合损失权重", "relobralo_epsilon"): (
            1.0e-12,
            "Bischof和Kraus作者公开程序固定值一致",
            str(loss_balancing.relative_to(root)),
        ),
        ("全耦合损失权重", "selection_score"): (
            "equal mean of three dimensionless validation groups",
            "共同检查分数与实现一致",
            str(loss_balancing.relative_to(root)),
        ),
        ("全耦合损失权重", "independent_test_rule"): (
            "after validation-only method selection",
            "两阶段读取顺序与实现一致",
            "code/run_hccb_p418_loss_balancing_protocol.py",
        ),
    }

    # These snippets distinguish equal-width feed-forward layers, validation-only
    # model selection, diffusion velocity prediction and steady microbatching.
    require_text(observable, "dim_feedforward=args.d_model", "dropout=0.0", "validation_mse < best_validation")
    require_text(graph_train, "validation_score < best_validation")
    require_text(
        graph_train,
        "FLUID_TEMPERATURE_RANGE_K = (300.0, 1000.0)",
        "load_hccb_thermophysical_parameters().solid_cp_temperature_range_k",
    )
    require_text(
        thermophysical,
        "solid_cp_temperature_range_k=(298.0, 1300.0)",
    )
    require_text(
        graph_model,
        "return 6 + self.boundary_role_count",
        "(coordinate, log_volume, material, self.boundary_fraction)",
        "if graph.boundary_role_count != self.boundary_role_count",
    )
    require_text(
        fully_coupled_model,
        "6 + boundary_role_count",
        "if graph.boundary_role_count != self.boundary_role_count",
        "graph.structural_features()",
        "self.internal_initial_flux_encoder",
        "-initial_internal_mass_flux.unsqueeze(-1)",
        "boundary_active[None, :, None]",
        "context / count.clamp_min(1.0)",
    )
    require_text(diffusion_model, "clean = signal * current - noise_scale * velocity", "noise = noise_scale * current + signal * velocity")
    require_text(root / "code/hccb_p418_regional_diffusion_refiner.py", "velocity_target", "signal * noise - noise_scale * residual")
    require_text(diffusion_train, 'default=32', "validation_loss < best_validation")
    require_text(
        root / "code/run_hccb_p418_60_model_comparison.sh",
        "STATE_DATA_WEIGHT=${STATE_DATA_WEIGHT:-5.0}",
        "FACE_FLUX_WEIGHT=${FACE_FLUX_WEIGHT:-1.0}",
        "PHYSICS_BALANCE_WEIGHT=${PHYSICS_BALANCE_WEIGHT:-1.0}",
        '--state-data-weight "${STATE_DATA_WEIGHT}"',
        '--face-flux-weight "${FACE_FLUX_WEIGHT}"',
        '--physics-balance-weight "${PHYSICS_BALANCE_WEIGHT}"',
        "GRAPH_MICROBATCH_SIZE=${GRAPH_MICROBATCH_SIZE:-1}",
        "TRANSOLVER_MICROBATCH_SIZE=${TRANSOLVER_MICROBATCH_SIZE:-1}",
        '--microbatch-size "${GRAPH_MICROBATCH_SIZE}"',
        '--microbatch-size "${TRANSOLVER_MICROBATCH_SIZE}"',
    )
    require_text(
        loss_balancing,
        "epsilon: float = 1.0e-12",
        "return _ordered_losses(groups).mean()",
        "rho * alpha * previous_weights",
        "(1.0 - rho) * alpha * relative_initial",
        "(1.0 - alpha) * relative_previous",
    )
    require_text(
        root / "code/run_hccb_p418_loss_balancing_protocol.py",
        'choices=("plan", "selection", "final")',
        '"--evaluation-stage",',
        '"selection",',
        '"final",',
        '"--selected-method-record",',
    )
    return result


def rule_setting_map(root: Path) -> dict[tuple[str, str], tuple[Any, str, str]]:
    dmdc = root / (
        "reproducibility/formal_training_sources/"
        "train_hccb_p418_regional_dmdc_20260729.py"
    )
    pod = root / "code/train_hccb_p418_low_rank_temperature_residual.py"
    require_text(
        dmdc,
        "time_derivatives.append((reduced[:, 1:] - reduced[:, :-1]) / delta_t[None, :])",
        "midpoint_states.append(0.5 * (reduced[:, 1:] + reduced[:, :-1]))",
        "expm(augmented * step_size) @ augmented_state",
        "np.sqrt(volume / volume.mean())",
        "np.linalg.lstsq",
    )
    require_text(
        dmdc,
        "selected = min(",
        'stable_candidates, key=lambda row: row["solid_temperature_RMSE_K"]',
        '"selection_split": "validation"',
        '"selection_metric": "regional-volume-weighted solid-temperature RMSE in K"',
    )
    require_text(pod, "np.linalg.svd", "np.linalg.lstsq", "range(int(fitted[\"available_rank\"]) + 1)", 'key=lambda row: (row["solid_temperature_RMSE_K"], row["rank"])')
    return {
        ("DMDc", "state_equation"): ("dz/dt=A z+B u", "连续时间推进公式一致", str(dmdc.relative_to(root))),
        ("DMDc", "time_integration"): (
            "matrix exponential over each recorded interval",
            "逐个原始时间间隔推进规则一致",
            str(dmdc.relative_to(root)),
        ),
        ("DMDc", "spatial_weighting"): ("sqrt(volume/mean(volume))", "体积加权公式一致", str(dmdc.relative_to(root))),
        ("DMDc", "selected_rank"): ("validation minimum", "验证工况选择规则一致", str(dmdc.relative_to(root))),
        ("POD低秩修正", "candidate_ranks"): ("0 through training-supported rank", "训练数据决定候选阶数，代码规则一致", str(pod.relative_to(root))),
        ("POD低秩修正", "selected_rank"): ("validation minimum", "验证工况选择规则一致", str(pod.relative_to(root))),
    }


def compare_value(recorded: Any, actual: Any) -> bool:
    if isinstance(recorded, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(recorded) - float(actual)) <= 1.0e-12 * max(abs(float(actual)), 1.0)
    return str(recorded) == str(actual)


def verify_registry_components(root: Path, registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Check the fused architecture settings and stored implementation hashes."""
    entries = {str(item["name"]): item for item in registry["architectures"]}
    expected = {
        "Temporal Transformer trajectory operator": {
            "layers": 5,
            "hidden_size": 256,
            "attention_heads": 8,
            "dropout": 0.0,
            "feedforward_ratio": 1,
            "epochs": 500,
            "effective_batch_size": 8,
            "learning_rate_peak": 1.0e-3,
            "weight_decay": 1.0e-5,
        },
        "Published-component spatiotemporal regional operator": {
            "preprocessor_mpnn_iterations": 2,
            "physics_attention_blocks": 2,
            "refinement_mpnn_iterations": 2,
            "attention_heads": 4,
            "physical_tokens": 128,
            "hidden_path": [64, 32, 64],
            "temporal_layers": 3,
            "temporal_heads": 1,
            "leaky_relu_negative_slope": 0.01,
        },
        "Volume-weighted DMDc baseline": {
            "published_state_equation": "z(k+1)=A z(k)+B u(k)",
            "project_state_equation": "dz/dt=A z+B u",
            "project_fit": "midpoint reduced states and adjacent finite-difference time derivatives using the actual recorded time intervals",
            "project_time_integration": "exact augmented linear-system matrix exponential over each recorded interval",
            "candidate_pod_ranks": [1, 2, 3, 4, 8, 12, 16, 24, 32],
            "rank_selection": "lowest validation solid-temperature RMSE",
            "spatial_weighting": "square root of regional volume normalized by mean regional volume",
        },
        "PDE-Refiner-style diffusion refinement": {
            "num_refinement_steps": 3,
            "minimum_noise_standard_deviation": 4.0e-7,
            "prediction_type": "v_prediction",
            "ema_decay": 0.995,
            "effective_batch_size": 8,
            "curve_microbatch_size": 1,
            "activation_precision_on_cuda": "bfloat16",
            "formal_stochastic_samples_per_curve": 32,
            "sample_count_convergence": [8, 16, 32],
        },
    }
    results: list[dict[str, Any]] = []
    for name, values in expected.items():
        actual = entries[name]["source_settings"]
        passed = all(actual.get(key) == value for key, value in values.items())
        results.append(
            {
                "architecture": name,
                "check": "source_settings",
                "expected": values,
                "actual": actual,
                "passed": passed,
            }
        )

    hash_contracts = [
        ("Temporal Transformer trajectory operator", "implementation_sha256", "code/train_hccb_p418_transient_observable_transformer.py"),
        ("Published-component spatiotemporal regional operator", "implementation_model_sha256", "code/hccb_p418_spatiotemporal_regional_operator.py"),
        ("Published-component spatiotemporal regional operator", "implementation_trainer_sha256", "code/train_hccb_p418_spatiotemporal_regional_operator.py"),
        ("Published-component spatiotemporal regional operator", "source_contract_sha256", "parameters/hccb_p418_mgnt_temporal_pino_contract.yaml"),
        (
            "Volume-weighted DMDc baseline",
            "implementation_sha256",
            "reproducibility/formal_training_sources/"
            "train_hccb_p418_regional_dmdc_20260729.py",
        ),
        ("Snapshot-POD low-rank temperature-residual correction", "implementation_sha256", "code/train_hccb_p418_low_rank_temperature_residual.py"),
        ("PDE-Refiner-style diffusion refinement", "regional_model_sha256", "code/hccb_p418_regional_diffusion_refiner.py"),
        ("PDE-Refiner-style diffusion refinement", "regional_trainer_sha256", "code/train_hccb_p418_regional_diffusion_refiner.py"),
        ("PDE-Refiner-style diffusion refinement", "temporal_model_sha256", "code/hccb_p418_temporal_temperature_diffusion.py"),
        ("PDE-Refiner-style diffusion refinement", "temporal_trainer_sha256", "code/train_hccb_p418_temporal_temperature_diffusion.py"),
        ("PDE-Refiner-style diffusion refinement", "physical_state_verification_script_sha256", "code/verify_hccb_p418_diffusion_physical_state.py"),
        ("PDE-Refiner-style diffusion refinement", "physical_state_verification_result_sha256", "results/hccb_p418_diffusion_physical_state/summary.json"),
        ("PDE-Refiner-style diffusion refinement", "observation_scope_verification_script_sha256", "code/verify_hccb_p418_diffusion_observation_scope.py"),
        ("PDE-Refiner-style diffusion refinement", "observation_scope_verification_result_sha256", "results/hccb_p418_diffusion_observation_scope/summary.json"),
        ("Published-component spatiotemporal regional operator", "fully_coupled_extension_implementation_sha256", "code/hccb_p418_fully_coupled_spatiotemporal_operator.py"),
    ]
    for architecture, field, relative_path in hash_contracts:
        path = root / relative_path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_hash = str(entries[architecture][field])
        results.append(
            {
                "architecture": architecture,
                "check": field,
                "path": relative_path,
                "expected": expected_hash,
                "actual": actual,
                "passed": actual == expected_hash,
            }
        )
    return results


def verify(root: Path, settings_path: Path, registry_path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(settings_path.open(encoding="utf-8")))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    architecture_names = {str(item["name"]) for item in registry["architectures"]}
    mapping = direct_setting_map(root)
    mapping.update(rule_setting_map(root))
    results = []
    failures = []
    for row in rows:
        key = (row["model"], row["setting"])
        if key not in mapping:
            failures.append(f"参数表条目没有对应代码检查：{key}")
            continue
        actual, check_kind, code_path = mapping[key]
        recorded = normalized_value(row["value"])
        same = compare_value(recorded, actual)
        source_exists = (root / row["source_path"]).exists()
        implementation_exists = all(
            (root / item.strip()).exists()
            for item in row["implementation_path"].split(";")
            if item.strip()
        )
        nonphysical = row["is_physical_parameter"].strip().lower() == "no"
        passed = same and source_exists and implementation_exists and nonphysical
        result = {
            "model": row["model"],
            "setting": row["setting"],
            "recorded_value": row["value"],
            "actual_code_value_or_rule": actual,
            "check_kind_cn": check_kind,
            "code_path": code_path,
            "source_path_exists": source_exists,
            "implementation_path_exists": implementation_exists,
            "is_not_physical_parameter": nonphysical,
            "passed": passed,
        }
        results.append(result)
        if not passed:
            failures.append(json.dumps(result, ensure_ascii=False))
    unused = sorted(set(mapping).difference((row["model"], row["setting"]) for row in rows))
    if unused:
        failures.append(f"代码检查项不在参数表中：{unused}")
    registry_checks = verify_registry_components(root, registry)
    failures.extend(
        f"架构来源记录与当前代码不一致：{item['architecture']} {item['check']}"
        for item in registry_checks
        if not item["passed"]
    )
    return {
        "status": "current_p418_model_settings_match_code_and_sources" if not failures else "model_setting_mismatch",
        "setting_count": len(rows),
        "verified_setting_count": sum(bool(row["passed"]) for row in results),
        "all_settings_are_nonphysical_numerical_choices": all(bool(row["is_not_physical_parameter"]) for row in results),
        "architecture_registry_entry_count": len(architecture_names),
        "architecture_registry_names": sorted(architecture_names),
        "architecture_registry_check_count": len(registry_checks),
        "architecture_registry_checks": registry_checks,
        "results": results,
        "failures": failures,
    }


def write_chinese_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# P418模型设置与当前代码对应结果",
        "",
        f"本次逐项检查了 **{payload['setting_count']}** 个模型数值设置，"
        f"其中 **{payload['verified_setting_count']}** 个与当前实际代码及来源文件一致。",
        "",
        "这些数值包括网络宽度、层数、注意力头数、训练轮数、学习率、扩散修正步数等。"
        "它们是模型结构或计算设置，不是球床材料参数，也没有改变P418的温度、流速、热源、物性或边界条件。",
        "",
        "## 分模型结果",
        "",
        "| 模型 | 已核对条目数 | 结果 |",
        "|---|---:|---|",
    ]
    models = sorted({row["model"] for row in payload["results"]})
    for model in models:
        selected = [row for row in payload["results"] if row["model"] == model]
        passed = sum(bool(row["passed"]) for row in selected)
        lines.append(f"| {model} | {len(selected)} | {'全部一致' if passed == len(selected) else f'{passed}/{len(selected)}一致'} |")
    lines.extend(
        [
            "",
            "## 科研含义",
            "",
            "1. 图-Transformer、时间Transformer和扩散修正使用的正式结构，与参数表记录一致。",
            "2. DMDc和POD的阶数由训练与验证工况决定，测试工况不参与选择。",
            "3. 扩散模型只修正确定性温度预测的剩余误差，不改变OpenFOAM方程和文献物理参数。",
            "4. 稳态多输出模型的状态、面流量和物理收支采用公开PINO配置中的5:1:1分组权重；纯数据坐标网络与物理PINN的共同数据项权重完全相同。",
            "5. 稳态图网络和Transolver逐工况累积梯度，是20 GB显存下的计算实现方式；等效训练批量不因此改变。",
            "",
            f"程序结果：`{payload['status']}`。",
        ]
    )
    if payload["failures"]:
        lines.extend(["", "## 需要修正", ""] + [f"- {item}" for item in payload["failures"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--settings", type=Path, default=Path("parameters/hccb_p418_model_numerical_settings.csv"))
    parser.add_argument("--registry", type=Path, default=Path("parameters/hccb_p418_ai_architecture_sources.json"))
    parser.add_argument("--output", type=Path, default=Path("results/hccb_p418_model_setting_verification.json"))
    parser.add_argument("--chinese-summary", type=Path, default=Path("parameters/HCCB_P418_MODEL_SETTINGS_CN.md"))
    args = parser.parse_args()
    root = args.root.resolve()
    settings = args.settings if args.settings.is_absolute() else root / args.settings
    registry = args.registry if args.registry.is_absolute() else root / args.registry
    output = args.output if args.output.is_absolute() else root / args.output
    chinese = args.chinese_summary if args.chinese_summary.is_absolute() else root / args.chinese_summary
    payload = verify(root, settings, registry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    chinese.parent.mkdir(parents=True, exist_ok=True)
    write_chinese_summary(chinese, payload)
    print(json.dumps({key: payload[key] for key in ("status", "setting_count", "verified_setting_count", "failures")}, ensure_ascii=False, indent=2))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
