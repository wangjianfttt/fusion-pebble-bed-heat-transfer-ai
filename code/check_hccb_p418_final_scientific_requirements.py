#!/usr/bin/env python3
"""Report whether every scientific result needed by the final P418 paper exists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from hccb_p418_selected_fixed_flow_chain import (
    STRICT_SPLIT,
    selected_model_directories,
)


def count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


def file_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_status(path: Path, expected: str | None = None) -> bool:
    if not file_ok(path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return expected is None or payload.get("status") == expected


def json_matches(path: Path, **expected: object) -> bool:
    payload = json_payload(path)
    return payload is not None and all(payload.get(key) == value for key, value in expected.items())


def json_true(path: Path, key: str) -> bool:
    if not file_ok(path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return payload.get(key) is True


def json_payload(path: Path) -> dict | None:
    if not file_ok(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _valid_seed_split(payload: dict, key: str) -> bool:
    split = payload.get(key)
    if not isinstance(split, dict):
        return False
    for role in ("train", "validation", "test"):
        identifiers = split.get(role)
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or len(identifiers) != len(set(str(value) for value in identifiers))
        ):
            return False
    return True


def steady_seed_robustness_complete(path: Path) -> bool:
    payload = json_payload(path)
    expected_architectures = {"pinn_data_only", "pinn", "graph", "transolver"}
    seeds = payload.get("seeds") if payload is not None else None
    metrics = payload.get("metrics") if payload is not None else None
    if not (
        payload is not None
        and payload.get("status")
        == "completed_p418_main_steady_split_seed_robustness"
        and isinstance(seeds, list)
        and len(seeds) == 3
        and len(set(seeds)) == 3
        and set(payload.get("architectures", [])) == expected_architectures
        and isinstance(metrics, list)
        and len(metrics) == 20
        and payload.get("new_physical_parameters") == []
        and bool(payload.get("common_comparison_fingerprint"))
        and _valid_seed_split(payload, "split_case_ids")
    ):
        return False
    return all(
        row.get("architecture") in expected_architectures
        and int(row.get("seed_count", -1)) == 3
        and math.isfinite(float(row.get("mean", math.nan)))
        and math.isfinite(float(row.get("sample_std", math.nan)))
        and float(row.get("sample_std", -1.0)) >= 0.0
        for row in metrics
    )


def transient_seed_robustness_complete(path: Path) -> bool:
    payload = json_payload(path)
    expected_models = {
        "observable_transformer",
        "graph_transformer_data_only",
        "graph_transformer_energy_flux",
        "low_rank_residual_correction",
        "diffusion_residual_correction",
    }
    seeds = payload.get("seeds") if payload is not None else None
    metrics = payload.get("metrics") if payload is not None else None
    if not (
        payload is not None
        and payload.get("status") == "completed_p418_strict_split_seed_robustness"
        and payload.get("split_name") == "pair_disjoint_stress_test"
        and isinstance(seeds, list)
        and len(seeds) >= 3
        and len(set(seeds)) == len(seeds)
        and set(payload.get("models", [])) == expected_models
        and isinstance(metrics, list)
        and len(metrics) == len(expected_models)
        and payload.get("new_physical_parameters") == []
        and _valid_seed_split(payload, "complete_curve_split_ids")
    ):
        return False
    return all(
        row.get("model") in expected_models
        and int(row.get("seed_count", -1)) == len(seeds)
        and math.isfinite(float(row.get("mean_K", math.nan)))
        and math.isfinite(float(row.get("sample_std_K", math.nan)))
        and float(row.get("sample_std_K", -1.0)) >= 0.0
        for row in metrics
    )


def steady_learning_curve_complete(path: Path) -> bool:
    payload = json_payload(path)
    expected_architectures = {
        "response_surface",
        "pinn_data_only",
        "pinn",
        "graph",
        "transolver",
    }
    expected_counts = [9, 18, 27, 36]
    if not (
        payload is not None
        and payload.get("status") == "p418_steady_learning_curve_complete"
        and payload.get("training_condition_counts") == expected_counts
        and payload.get("fixed_validation_condition_count") == [12]
        and payload.get("fixed_test_condition_count") == [12]
        and set(payload.get("architectures", [])) == expected_architectures
        and payload.get("new_physical_parameters") == []
    ):
        return False
    table_name = str(payload.get("table", ""))
    table_path = path.parent / table_name
    if not table_name or not file_ok(table_path):
        return False
    try:
        with table_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        combinations = {
            (str(row["architecture"]), int(row["train_case_count"])) for row in rows
        }
        expected_combinations = {
            (architecture, count)
            for architecture in expected_architectures
            for count in expected_counts
        }
        return (
            len(rows) == len(expected_combinations)
            and combinations == expected_combinations
            and all(
                int(row["validation_case_count"]) == 12
                and int(row["test_case_count"]) == 12
                and math.isfinite(float(row["test_solid_temperature_normalized_rmse"]))
                and float(row["test_solid_temperature_normalized_rmse"]) >= 0.0
                for row in rows
            )
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def transient_learning_curve_complete(path: Path) -> bool:
    payload = json_payload(path)
    runs = payload.get("runs") if payload is not None else None
    if not (
        payload is not None
        and payload.get("status") == "completed_p418_transient_learning_curve"
        and payload.get("training_trajectory_counts") == [3, 6]
        and payload.get("fixed_validation_trajectory_count") == 2
        and payload.get("fixed_test_trajectory_count") == 4
        and payload.get("model")
        == "physics_constrained_factorized_graph_transformer"
        and isinstance(runs, list)
        and len(runs) == 3
        and payload.get("new_physical_parameters") == []
    ):
        return False
    expected = {(3, "up"), (3, "down"), (6, "both")}
    try:
        actual = {
            (int(row["training_trajectory_count"]), str(row["training_direction"]))
            for row in runs
        }
        return actual == expected and all(
            int(row["validation_trajectory_count"]) == 2
            and int(row["test_trajectory_count"]) == 4
            and int(row["selected_epoch"]) > 0
            and math.isfinite(float(row["test_solid_temperature_RMSE_K"]))
            and float(row["test_solid_temperature_RMSE_K"]) >= 0.0
            and len(str(row.get("source_summary_sha256", ""))) == 64
            for row in runs
        )
    except (KeyError, TypeError, ValueError):
        return False


def corrected_steady_comparison_complete(root: Path) -> bool:
    comparison = (
        root
        / "results/hccb_p418_60_corrected_20260731_model_comparison_100epoch"
    )
    assembly = json_payload(comparison / "corrected_result_assembly.json")
    return bool(
        assembly is not None
        and assembly.get("status") == "corrected_steady_result_assembly_complete"
        and assembly.get("result_count") == 25
        and assembly.get("method_count") == 5
        and assembly.get("split_count") == 5
        and assembly.get("corrected_split") == "heat_source_extrapolation"
        and assembly.get("new_physical_parameters") == []
        and file_ok(comparison / "model_comparison.csv")
        and file_ok(
            root / "manuscript/generated_steady_model_comparison_validated.tex"
        )
    )


def steady_physics_evidence(root: Path) -> tuple[bool, Path]:
    """Return the current 60-case conservation summary, with legacy fallback."""
    candidates = (
        root / "results/hccb_p418_sourceflow_complete_physics_60/summary.json",
        root / "results/hccb_p418_60_sourceflow_r3_completed_physics/summary.json",
    )
    for path in candidates:
        payload = json_payload(path)
        if payload is None:
            continue
        if (
            payload.get("completed_case_count") == 60
            and payload.get("new_physical_parameters") == []
            and isinstance(payload.get("maximum_relative_mass_difference"), (int, float))
            and isinstance(payload.get("maximum_relative_energy_difference"), (int, float))
        ):
            return True, path
    return False, candidates[0]


def steady_extrapolation_evidence(root: Path) -> tuple[bool, Path]:
    """Require formal results for all three steady extrapolation directions."""
    path = (
        root
        / "results/hccb_p418_60_corrected_20260731_model_comparison_100epoch"
        / "corrected_result_assembly.json"
    )
    payload = json_payload(path)
    if payload is None or payload.get("new_physical_parameters") != []:
        return False, path
    records = payload.get("records")
    if not isinstance(records, list):
        return False, path
    required_splits = {
        "temperature_extrapolation",
        "velocity_extrapolation",
        "heat_source_extrapolation",
    }
    methods = {row.get("method") for row in records if isinstance(row, dict)}
    split_methods = {
        split: {
            row.get("method")
            for row in records
            if isinstance(row, dict) and row.get("split") == split
        }
        for split in required_splits
    }
    complete = bool(
        payload.get("status") == "corrected_steady_result_assembly_complete"
        and len(methods) == 5
        and all(split_methods[split] == methods for split in required_splits)
    )
    return complete, path


def three_mesh_sensitivity_evidence(root: Path) -> tuple[bool, Path]:
    """Require the formal three-mesh result and its two source tables."""
    path = root / "results/hccb_p418_three_mesh_cht_sensitivity/summary.json"
    engineering_path = path.parent / "engineering_observables.csv"
    gci_path = path.parent / "mesh_gci.csv"
    verification_path = path.parent / "formal_recovery_verification.json"
    payload = json_payload(path)
    verification = json_payload(verification_path)
    if payload is None or verification is None:
        return False, path
    levels = payload.get("mesh_levels")
    convergence = payload.get("grid_convergence")
    if not isinstance(levels, list) or not isinstance(convergence, list):
        return False, path
    expected_levels = {"coarse", "medium", "fine"}
    level_names = {
        row.get("mesh_level") for row in levels if isinstance(row, dict)
    }
    expected_metrics = {
        "pressure_drop_Pa",
        "outlet_temperature_change_K",
        "solid_maximum_temperature_change_K",
        "cooling_wall_heat_fraction",
    }
    metric_names = {
        row.get("metric") for row in convergence if isinstance(row, dict)
    }
    meshes_pass = all(
        isinstance(row, dict)
        and row.get("fluid_basic_check_passes") is True
        and row.get("solid_basic_check_passes") is True
        for row in levels
    )
    verification_hashes = verification.get("file_sha256")
    verified_files = bool(
        isinstance(verification_hashes, dict)
        and file_ok(engineering_path)
        and file_ok(gci_path)
        and verification_hashes.get("summary.json") == file_sha256(path)
        and verification_hashes.get("engineering_observables.csv")
        == file_sha256(engineering_path)
        and verification_hashes.get("mesh_gci.csv") == file_sha256(gci_path)
    )
    verification_checks = verification.get("checks")
    verified_formal_recovery = bool(
        verification.get("status")
        == "verified_formal_p418_three_mesh_recovery"
        and verification.get("new_physical_parameters") == []
        and isinstance(verification_checks, dict)
        and verification_checks
        and all(value is True for value in verification_checks.values())
        and verified_files
    )
    return (
        bool(
            payload.get("status")
            == "completed_three_mesh_p418_cht_comparison"
            and payload.get("new_physical_parameters") == []
            and len(levels) == 3
            and level_names == expected_levels
            and meshes_pass
            and expected_metrics.issubset(metric_names)
            and verified_formal_recovery
        ),
        path,
    )


def fixed_flow_loss_balancing_complete(selection_path: Path) -> bool:
    """Require validation-only weight selection followed by one test read."""
    selection = json_payload(selection_path)
    if selection is None:
        return False
    records = selection.get("candidate_records")
    selected_id = selection.get("selected_candidate_id")
    if not (
        selection.get("status")
        == "p418_loss_balancing_selected_on_validation_only"
        and isinstance(records, list)
        and len(records) == 4
        and selection.get("independent_test_read") is False
        and isinstance(selected_id, str)
        and selected_id
        and selection.get("new_physical_parameters") == []
    ):
        return False
    candidate_ids = {
        row.get("candidate_id") for row in records if isinstance(row, dict)
    }
    if selected_id not in candidate_ids or len(candidate_ids) != 4:
        return False
    final_summary = json_payload(
        selection_path.parent / selected_id / "final_summary.json"
    )
    final_ok = bool(
        final_summary is not None
        and final_summary.get("status")
        == "completed_p418_spatiotemporal_regional_operator"
        and final_summary.get("evaluation_stage") == "final"
        and final_summary.get("test_evaluated") is True
        and isinstance(final_summary.get("loss_balancing"), dict)
        and final_summary["loss_balancing"].get("candidate_id") == selected_id
        and final_summary.get("new_physical_parameters") == []
    )
    if not final_ok:
        return False
    try:
        selected_model_directories(selection_path.parent.parent, STRICT_SPLIT)
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return False
    return True


def fixed_vs_fully_coupled_complete(path: Path) -> bool:
    payload = json_payload(path)
    required_signals = {
        "pressure_drop_Pa",
        "outlet_temperature_K",
        "maximum_solid_temperature_K",
        "volume_average_fluid_temperature_K",
        "volume_average_solid_temperature_K",
        "cooling_wall_power_W",
        "signed_mass_residual_kg_s",
        "net_outward_enthalpy_flow_W",
    }
    return bool(
        payload is not None
        and payload.get("status")
        == "completed_p418_fixed_vs_fully_coupled_step_comparison"
        and payload.get("sequence_count") == 12
        and set(payload.get("signals", [])) == required_signals
        and payload.get("new_physical_parameters") == []
        and file_ok(path.parent / "fixed_vs_fully_coupled_steps.csv")
    )


def fully_coupled_scope_limit_complete(path: Path) -> bool:
    """Accept a bounded negative result when fully coupled startup is out of scope."""
    payload = json_payload(path)
    if (
        payload is None
        or payload.get("status") != "P418_SCOPE_LIMITS_EVIDENCE_SYNCED"
    ):
        return False
    records = payload.get("records")
    if not isinstance(records, list):
        return False
    failed_jobs = {
        str(row.get("job_id", ""))
        for row in records
        if isinstance(row, dict)
        and row.get("status") == "failed_solver_exit_propagated"
        and row.get("slurm_state") == "FAILED"
    }
    return len(failed_jobs - {""}) >= 3


def verified_fixed_step_progress(root: Path) -> tuple[int, int, str]:
    """Count the current regional dataset, with the legacy raw-case layout as fallback."""
    index = (
        root
        / "results/hccb_p418_physical_steps_12/regional_sequences/dataset_index.json"
    )
    payload = json_payload(index)
    if payload is not None:
        sequences = payload.get("sequences")
        declared_count = payload.get("sequence_count")
        if (
            payload.get("status") == "p418_regional_thermal_step_sequences_ready"
            and isinstance(sequences, list)
            and declared_count == len(sequences)
        ):
            return len(sequences), 12, str(index.resolve())

    collection = (
        root / "results/hccb_p418_formal_fixed_flow_collection_20260729.json"
    )
    payload = json_payload(collection)
    if payload is not None:
        if (
            payload.get("status") == "completed_p418_formal_fixed_flow_collection"
            and payload.get("sequence_count") == 12
            and payload.get("completed_sequence_count") == 12
            and payload.get("points_per_sequence") == 16401
            and payload.get("all_time_axes_strictly_increasing") is True
            and payload.get("all_valid_values_finite") is True
        ):
            return 12, 12, str(collection.resolve())

    legacy = root / "hccb_p418_physical_steps_12"
    return (
        count_files(legacy, "*/step_response_complete.json"),
        12,
        str(legacy.resolve()),
    )


def verified_high_re_fixed_progress(root: Path) -> tuple[int, int, str]:
    """Count only recovered 300 s curves with finite 16,401-point summaries."""
    recovered = (
        root / "results/hccb_p418_high_re_independent_fixed_steps_6/by_sequence"
    )
    completed = 0
    if recovered.exists():
        for case in recovered.iterdir():
            marker = json_payload(case / "cloud_sequence_complete.json")
            summary = json_payload(case / "results/summary.json")
            if marker is None or summary is None:
                continue
            if (
                marker.get("status")
                not in {
                    "completed_fixed_hydrodynamics_sequence",
                    "completed_fixed_hydrodynamics_step_sequence",
                    "completed_fixed_flow_sequence",
                    "completed_p418_high_re_independent_fixed_flow_sequence",
                }
                and marker.get("solver_finished") is not True
            ):
                continue
            if (
                summary.get("completed_case_count") == 1
                and summary.get("maximum_time_points") == 16401
            ):
                completed += 1
        return completed, 6, str(recovered.resolve())

    legacy = root / "hccb_p418_high_re_independent_fixed_steps_6"
    return (
        count_files(legacy, "*/step_response_complete.json"),
        6,
        str(legacy.resolve()),
    )


def verified_steady_progress(root: Path) -> tuple[int, int, str]:
    coverage = (
        root / "results/hccb_p418_training_data_coverage_partial/summary.json"
    )
    if file_ok(coverage):
        payload = json.loads(coverage.read_text(encoding="utf-8"))
        return (
            int(payload["completed_case_count"]),
            int(payload["expected_case_count"]),
            str(coverage.resolve()),
        )

    matrix = root / "hccb_dense_cht_p418_60_sourceflow_r3"
    return (
        count_files(matrix, "*/formal_sample_complete.json"),
        60,
        str(matrix.resolve()),
    )


def build_requirements(root: Path) -> list[dict]:
    results = root / "results"
    manuscript = root / "manuscript"
    figures = root / "figures"
    transient_root = results / "hccb_p418_physical_steps_12"
    steady_completed, steady_required, steady_source = verified_steady_progress(root)
    fixed_completed, fixed_required, fixed_source = verified_fixed_step_progress(root)
    high_re_fixed_completed, high_re_fixed_required, high_re_fixed_source = (
        verified_high_re_fixed_progress(root)
    )
    high_re_comparison = (
        results
        / "hccb_p418_high_re_three_bounded_model_evaluation/comparison/summary.json"
    )
    packing_summary = (
        results / "hccb_p418_cross_packing_seed202_integral_9/summary.json"
    )
    scope_summary = (
        results / "hccb_p418_scope_limits_20260730/scope_limits_summary.json"
    )
    transient_summary = transient_root / "model_comparison/summary.json"
    fixed_coupled_summary = (
        results
        / "hccb_p418_fully_coupled_steps_12/fixed_vs_fully_coupled/summary.json"
    )
    final_narrative = results / "hccb_p418_final_manuscript_narrative.json"
    loss_balancing_selection = (
        transient_root
        / "fixed_flow_loss_balancing_pair_disjoint_stress_test"
        / "selected_loss_balancing_method.json"
    )
    loss_balancing_complete = fixed_flow_loss_balancing_complete(
        loss_balancing_selection
    )
    fully_coupled_comparison_complete = fixed_vs_fully_coupled_complete(
        fixed_coupled_summary
    )
    fully_coupled_limit_complete = fully_coupled_scope_limit_complete(
        scope_summary
    )
    fully_coupled_evidence_complete = (
        fully_coupled_comparison_complete or fully_coupled_limit_complete
    )
    corrected_steady_complete = corrected_steady_comparison_complete(root)
    steady_physics_complete, steady_physics_path = steady_physics_evidence(root)
    steady_extrapolation_complete, steady_extrapolation_path = (
        steady_extrapolation_evidence(root)
    )
    three_mesh_complete, three_mesh_path = three_mesh_sensitivity_evidence(root)
    transient_comparison_complete = json_matches(
        transient_summary,
        status="completed_p418_physical_step_model_comparison",
        strict_split_loss_balancing_stage="validation_selected",
    )
    transient_figure_record = figures / "hccb_p418_transient_model_comparison.json"
    transient_figure_complete = json_matches(
        transient_figure_record,
        status="complete_formal_p418_transient_model_comparison_figure",
        strict_split_loss_balancing_stage="validation_selected",
    )
    field_figure_record = figures / "hccb_p418_openfoam_model_field_comparison.json"
    field_figure_complete = json_matches(
        field_figure_record,
        status="complete_same_scale_openfoam_model_field_comparison",
        strict_split_loss_balancing_stage="validation_selected",
    )
    formal_results_complete = all(
        (
            steady_completed == steady_required,
            fixed_completed == fixed_required,
            high_re_fixed_completed == high_re_fixed_required,
            fully_coupled_evidence_complete,
            transient_comparison_complete,
            json_status(
                high_re_comparison,
                "completed_p418_high_re_three_fixed_model_comparison",
            ),
            loss_balancing_complete,
            json_status(
                packing_summary,
                "completed_seed101_seed202_integral_response_comparison",
            ),
            json_status(
                scope_summary,
                "P418_SCOPE_LIMITS_EVIDENCE_SYNCED",
            ),
            three_mesh_complete,
        )
    )
    requirements = [
        {
            "group": "三维物理计算",
            "name": "60个稳态工况",
            "complete": steady_completed == steady_required,
            "current": steady_completed,
            "required": steady_required,
            "progress_source": steady_source,
        },
        {
            "group": "三维物理计算",
            "name": "12条固定流场瞬态轨迹",
            "complete": fixed_completed == fixed_required,
            "current": fixed_completed,
            "required": fixed_required,
            "progress_source": fixed_source,
        },
        {
            "group": "三维物理计算",
            "name": "6条高雷诺数固定流场轨迹",
            "complete": high_re_fixed_completed == high_re_fixed_required,
            "current": high_re_fixed_completed,
            "required": high_re_fixed_required,
            "progress_source": high_re_fixed_source,
        },
        {
            "group": "三维物理计算",
            "name": "全耦合瞬态的适用范围已判定",
            "complete": fully_coupled_evidence_complete,
            "path": str(
                (
                    fixed_coupled_summary
                    if fully_coupled_comparison_complete
                    else scope_summary
                ).resolve()
            ),
            "note": (
                "若同一12组起终点对照完成，则使用定量对照；"
                "否则只在至少三个独立启动计算明确越出已登记"
                "物性范围时，将其作为本文的适用范围结论，"
                "不报告全耦合精度。"
            ),
        },
        {
            "group": "三维物理计算",
            "name": "第二套球床装填的9个匹配工况",
            "complete": json_status(
                packing_summary,
                "completed_seed101_seed202_integral_response_comparison",
            ),
            "path": str(packing_summary.resolve()),
        },
    ]
    requirements.extend(
        [
            {
                "group": "模型比较",
                "name": "稳态经典模型、PINN、图网络和Transformer比较",
                "complete": corrected_steady_complete,
                "path": str(
                    (
                        results
                        / "hccb_p418_60_corrected_20260731_model_comparison_100epoch"
                        / "corrected_result_assembly.json"
                    ).resolve()
                ),
            },
            {
                "group": "模型比较",
                "name": "固定流场物理损失权重候选比较",
                "complete": loss_balancing_complete,
                "path": str(loss_balancing_selection.resolve()),
                "note": (
                    "四种已登记方案只用验证集选择，选定后才读取一次独立测试集。"
                ),
            },
            {
                "group": "论文成品",
                "name": "稳态模型图",
                "complete": corrected_steady_complete
                and file_ok(figures / "hccb_p418_steady_model_comparison.pdf"),
                "path": str(
                    (figures / "hccb_p418_steady_model_comparison.pdf").resolve()
                ),
            },
            {
                "group": "论文成品",
                "name": "瞬态模型图",
                "complete": transient_figure_complete
                and file_ok(
                    manuscript
                    / "generated_transient_model_comparison_validated.tex"
                )
                and file_ok(figures / "hccb_p418_transient_model_comparison.pdf"),
                "path": str(
                    (figures / "hccb_p418_transient_model_comparison.pdf").resolve()
                ),
            },
            {
                "group": "论文成品",
                "name": "OpenFOAM与模型温度云图",
                "complete": field_figure_complete
                and file_ok(
                    manuscript
                    / "generated_openfoam_model_field_comparison_validated.tex"
                )
                and file_ok(
                    figures / "hccb_p418_openfoam_model_field_comparison.pdf"
                ),
                "path": str(
                    (
                        figures / "hccb_p418_openfoam_model_field_comparison.pdf"
                    ).resolve()
                ),
            },
        ]
    )

    result_files = [
        (
            "数值可靠性",
            "未通过的全域网格和全耦合启动如实记录",
            scope_summary,
        ),
        (
            "模型比较",
            "瞬态DMDc、三种图模型和扩散修正比较",
            transient_summary,
        ),
        (
            "模型比较",
            "六条高流速曲线的三种冻结模型比较",
            high_re_comparison,
        ),
        (
            "重复训练",
            "稳态三随机种子",
            results
            / "hccb_p418_60_steady_seed_robustness_100epoch/summary.json",
        ),
        (
            "重复训练",
            "瞬态三随机种子",
            transient_root
            / "seed_robustness_pair_disjoint_stress_test/summary.json",
        ),
        (
            "训练数据量",
            "稳态训练工况数量曲线",
            results
            / "hccb_p418_learning_curve_model_comparison_100epoch/learning_curve_summary.json",
        ),
        (
            "训练数据量",
            "瞬态完整轨迹数量曲线",
            results / "hccb_p418_transient_learning_curve/summary.json",
        ),
        (
            "外推与独立预测",
            "第二套颗粒装填的直接物理比较",
            packing_summary,
        ),
        (
            "实验与关联式",
            "公开温度、换热和压降数据比较",
            results / "hccb_heat_ai_external_evidence/summary.json",
        ),
        (
            "论文成品",
            "物理模型与计算域图",
            figures / "hccb_p418_physical_model_domain.pdf",
        ),
        (
            "论文成品",
            "主要物理响应图",
            figures / "hccb_p418_physical_response.pdf",
        ),
        (
            "论文成品",
            "独立装填物理对比图",
            figures / "hccb_p418_seed202_integral_9.pdf",
        ),
    ]
    requirements.extend(
        [
            {
                "group": "数值可靠性",
                "name": "已采用网格与固定流场时间步结果",
                "complete": three_mesh_complete,
                "path": str(three_mesh_path.resolve()),
                "note": (
                    "必须保留正式三网格summary、工程量表和GCI表；"
                    "正文中的限制说明不能替代源结果。"
                ),
            },
            {
                "group": "数值可靠性",
                "name": "60组质量和能量守恒结果",
                "complete": steady_physics_complete,
                "path": str(steady_physics_path.resolve()),
            },
            {
                "group": "外推与独立预测",
                "name": "温度、速度和热源范围外预测",
                "complete": steady_extrapolation_complete,
                "path": str(steady_extrapolation_path.resolve()),
            },
        ]
    )
    steady_seed_summary = (
        results / "hccb_p418_60_steady_seed_robustness_100epoch/summary.json"
    )
    transient_seed_summary = (
        transient_root / "seed_robustness_pair_disjoint_stress_test/summary.json"
    )
    steady_learning_summary = (
        results
        / "hccb_p418_learning_curve_model_comparison_100epoch/learning_curve_summary.json"
    )
    transient_learning_summary = (
        results / "hccb_p418_transient_learning_curve/summary.json"
    )
    result_completion_overrides = {
        transient_summary: transient_comparison_complete,
        steady_seed_summary: steady_seed_robustness_complete(steady_seed_summary),
        transient_seed_summary: transient_seed_robustness_complete(
            transient_seed_summary
        ),
        steady_learning_summary: steady_learning_curve_complete(
            steady_learning_summary
        ),
        transient_learning_summary: transient_learning_curve_complete(
            transient_learning_summary
        ),
    }
    for group, name, path in result_files:
        requirements.append(
            {
                "group": group,
                "name": name,
                "complete": result_completion_overrides.get(path, file_ok(path)),
                "path": str(path.resolve()),
            }
        )
    for name, path in (
        ("英文论文PDF", manuscript / "main.pdf"),
        ("论文中文便读版", manuscript / "P418_论文中文便读版.md"),
    ):
        requirements.append(
            {
                "group": "论文成品",
                "name": name,
                "complete": file_ok(path) and formal_results_complete,
                "path": str(path.resolve()),
                "note": (
                    "当前文件已存在，但正式三模型结果尚未全部写入。"
                    if file_ok(path) and not formal_results_complete
                    else ""
                ),
            }
        )
    requirements.append(
        {
            "group": "论文成品",
            "name": "正式摘要、讨论和结论均由已完成结果生成",
            "complete": json_status(
                final_narrative,
                "complete_p418_final_manuscript_narrative",
            ),
            "path": str(final_narrative.resolve()),
        }
    )
    submission_check = (
        results / "hccb_p418_ijhmt_submission_check/summary.json"
    )
    requirements.append(
        {
            "group": "论文成品",
            "name": "IJHMT投稿格式与草稿残留检查",
            "complete": json_status(
                submission_check, "completed_p418_ijhmt_submission_check"
            ),
            "path": str(submission_check.resolve()),
        }
    )
    reproducibility_manifest = (
        results / "hccb_p418_reproducibility_manifest/manifest.json"
    )
    requirements.append(
        {
            "group": "论文成品",
            "name": "P418复现清单与环境说明",
            "complete": json_true(reproducibility_manifest, "source_package_ready"),
            "path": str(reproducibility_manifest.resolve()),
        }
    )
    return requirements


MODEL_CHAIN_REQUIREMENTS = {
    "固定流场物理损失权重候选比较",
    "瞬态DMDc、三种图模型和扩散修正比较",
    "稳态三随机种子",
    "瞬态三随机种子",
    "瞬态完整轨迹数量曲线",
}


def summarize_remaining_dependencies(requirements: list[dict]) -> dict:
    """Separate unfinished model calculations from automatic paper outputs."""
    incomplete = [row for row in requirements if not row["complete"]]
    physics = [row["name"] for row in incomplete if row["group"] == "三维物理计算"]
    model_chain = [
        row["name"] for row in incomplete if row["name"] in MODEL_CHAIN_REQUIREMENTS
    ]
    paper_outputs = [
        row["name"]
        for row in incomplete
        if row["name"] not in MODEL_CHAIN_REQUIREMENTS
        and row["group"] == "论文成品"
    ]
    other = [
        row["name"]
        for row in incomplete
        if row["name"] not in MODEL_CHAIN_REQUIREMENTS
        and row["group"] not in {"三维物理计算", "论文成品"}
    ]
    return {
        "unfinished_count": len(incomplete),
        "unfinished_physics_calculations": physics,
        "waiting_for_model_chain": model_chain,
        "generated_after_model_chain": paper_outputs,
        "other_unfinished_items": other,
    }


def write_chinese(
    path: Path, requirements: list[dict], remaining: dict | None = None
) -> None:
    completed = sum(bool(row["complete"]) for row in requirements)
    remaining = remaining or summarize_remaining_dependencies(requirements)
    lines = [
        "# P418最终论文还缺什么",
        "",
        f"目前完成 {completed}/{len(requirements)} 项。这里的“完成”只表示正式结果文件已经存在，不用预期结果代替。",
        "",
        "## 剩余工作的关系",
        "",
        f"- 尚缺项目：{remaining['unfinished_count']} 项。",
        f"- 尚缺三维物理计算：{len(remaining['unfinished_physics_calculations'])} 项。",
        f"- 等待当前模型训练与比较：{len(remaining['waiting_for_model_chain'])} 项。",
        f"- 模型结果齐全后自动生成：{len(remaining['generated_after_model_chain'])} 项。",
        f"- 其他未完成项目：{len(remaining['other_unfinished_items'])} 项。",
        "",
    ]
    for group in dict.fromkeys(row["group"] for row in requirements):
        lines.extend([f"## {group}", ""])
        for row in (item for item in requirements if item["group"] == group):
            mark = "已完成" if row["complete"] else "未完成"
            amount = (
                f"（{row['current']}/{row['required']}）"
                if "current" in row
                else ""
            )
            lines.append(f"- {mark}：{row['name']}{amount}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    requirements = build_requirements(root)
    completed = sum(bool(row["complete"]) for row in requirements)
    remaining = summarize_remaining_dependencies(requirements)
    payload = {
        "status": (
            "completed_p418_final_scientific_requirements"
            if completed == len(requirements)
            else "p418_final_scientific_requirements_incomplete"
        ),
        "completed_count": completed,
        "required_count": len(requirements),
        "requirements": requirements,
        "remaining_dependencies": remaining,
        "new_physical_parameters": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_chinese(
        args.output_dir / "P418_最终论文还缺什么_CN.md",
        requirements,
        remaining,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require_complete and completed != len(requirements):
        raise SystemExit(
            f"final paper inputs are incomplete: {completed}/{len(requirements)}"
        )


if __name__ == "__main__":
    main()
