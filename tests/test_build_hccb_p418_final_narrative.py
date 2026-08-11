import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_final_narrative.py"
ENERGY = "projection_aware_volume_weighted_energy_equation_normalized_RMSE"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def run_builder(
    tmp_path: Path,
    *,
    joint: bool,
    learned_best: bool = False,
    completed_count: int = 60,
    coupled_scope_count: int = 3,
    high_re_status: str = (
        "completed_p418_high_re_three_fixed_model_comparison"
    ),
) -> dict:
    steady = tmp_path / "steady.json"
    completed = tmp_path / "completed.json"
    hotspots = tmp_path / "hotspots.json"
    steady_seed = tmp_path / "steady_seed.json"
    transient_seed = tmp_path / "transient_seed.json"
    steady_learning = tmp_path / "steady_learning.json"
    steady_learning_csv = tmp_path / "learning_curve_efficiency.csv"
    transient_learning = tmp_path / "transient_learning.json"
    loss_balancing = tmp_path / "loss_balancing"
    loss_balancing_selection = loss_balancing / "selected_loss_balancing_method.json"
    transient = tmp_path / "transient.json"
    metrics = tmp_path / "metrics.csv"
    cost = tmp_path / "cost.json"
    high_re = tmp_path / "high_re.json"
    high_re_csv = tmp_path / "high_re.csv"
    packing = tmp_path / "packing.json"
    external = tmp_path / "external.json"
    scope = tmp_path / "scope.json"
    abstract = tmp_path / "abstract.tex"
    discussion = tmp_path / "discussion.tex"
    conclusion = tmp_path / "conclusion.tex"
    summary = tmp_path / "summary.json"

    selected_loss_id = "relobralo_kirchhoff_table_viii"
    best_transient_model = (
        "graph_transformer_energy_flux"
        if learned_best
        else "initial_temperature_persistence"
    )
    best_transient_temperature = 3.0 if learned_best else 4.0
    loss_balancing.mkdir()
    write_json(
        loss_balancing_selection,
        {
            "status": "p418_loss_balancing_selected_on_validation_only",
            "candidate_records": [
                {"candidate_id": candidate_id}
                for candidate_id in (
                    "fixed_registered_5_1_1",
                    "relobralo_burgers_table_viii",
                    selected_loss_id,
                    "relobralo_helmholtz_table_viii",
                )
            ],
            "selected_candidate_id": selected_loss_id,
            "selected_validation_score": 0.45,
            "independent_test_read": False,
            "new_physical_parameters": [],
        },
    )
    selected_loss_dir = loss_balancing / selected_loss_id
    selected_loss_dir.mkdir()
    write_json(
        selected_loss_dir / "final_summary.json",
        {
            "status": "completed_p418_spatiotemporal_regional_operator",
            "evaluation_stage": "final",
            "test_evaluated": True,
            "loss_balancing": {"candidate_id": selected_loss_id},
            "metrics": {
                "test": {
                    "solid_temperature_RMSE_K": 3.25,
                    "projection_aware_energy_equation_normalized_RMSE": 0.018,
                }
            },
            "new_physical_parameters": [],
        },
    )

    write_json(
        steady,
        {
            "status": "complete_p418_steady_manuscript_text",
            "leaders_by_metric": {
                "test_solid_temperature_normalized_rmse": {
                    "method": "pinn",
                    "value": 0.032,
                }
            },
        },
    )
    write_json(
        completed,
        {
            "status": "completed_p418_case_physics_summarized",
            "completed_case_count": completed_count,
            "thermal_regime_summary": {
                "cooling_wall_heat_direction_counts": {
                    "wall_to_fluid": 24,
                    "fluid_to_wall": 36,
                    "zero": 0,
                },
                "solid_maximum_temperature_range_K": [634.9, 921.4],
            },
            "wall_heat_zero_crossings": [
                {
                    "interpolated_zero_wall_heat_inlet_temperature_K": 608.5
                },
                {
                    "interpolated_zero_wall_heat_inlet_temperature_K": 626.2
                },
            ],
            "complete_factorial_decomposition_available": True,
            "factorial_variance_decomposition": [
                {
                    "observable": observable,
                    "effect": effect,
                    "variance_fraction_percent": value,
                }
                for observable in (
                    "pressure_drop_Pa",
                    "outlet_temperature_K",
                    "solid_maximum_temperature_K",
                    "cooling_wall_heat_into_fluid_W",
                )
                for effect, value in (
                    ("inlet_velocity", 55.0),
                    ("inlet_temperature", 30.0),
                    ("solid_heat_source", 15.0),
                )
            ],
        },
    )
    write_json(
        hotspots,
        {
            "status": "p418_steady_hotspots_ready",
            "completed_case_count": completed_count,
            "factor_summary": [
                {"maximum_adjacent_hotspot_distance_m": 0.0048},
                {"maximum_adjacent_hotspot_distance_m": 0.0026},
                {"maximum_adjacent_hotspot_distance_m": 0.0031},
            ],
        },
    )
    write_json(
        steady_seed,
        {
            "status": "completed_p418_main_steady_split_seed_robustness",
            "seeds": [20260717, 20260718, 20260719],
            "metrics": [
                {
                    "architecture": architecture,
                    "metric": "solid_temperature_normalized_rmse",
                    "mean": mean,
                    "sample_std": sample_std,
                }
                for architecture, mean, sample_std in (
                    ("pinn_data_only", 0.040, 0.004),
                    ("pinn", 0.030, 0.002),
                    ("graph", 0.025, 0.001),
                    ("transolver", 0.035, 0.002),
                )
            ],
            "new_physical_parameters": [],
        },
    )
    write_json(
        transient_seed,
        {
            "status": "completed_p418_strict_split_seed_robustness",
            "seeds": [20260717, 20260718, 20260719],
            "metrics": [
                {
                    "model": model,
                    "sample_std_K": sample_std,
                }
                for model, sample_std in (
                    ("observable_transformer", 0.8),
                    ("graph_transformer_data_only", 0.5),
                    ("graph_transformer_energy_flux", 0.4),
                    ("low_rank_residual_correction", 0.6),
                    ("diffusion_residual_correction", 0.7),
                )
            ],
            "new_physical_parameters": [],
        },
    )
    write_json(
        steady_learning,
        {
            "status": "p418_steady_learning_curve_complete",
            "training_condition_counts": [9, 18, 27, 36],
            "table": steady_learning_csv.name,
            "new_physical_parameters": [],
        },
    )
    steady_learning_rows = [
        {
            "architecture": "pinn",
            "train_case_count": count,
            "test_solid_temperature_normalized_rmse": value,
        }
        for count, value in (
            (9, 0.080),
            (18, 0.055),
            (27, 0.041),
            (36, 0.032),
        )
    ]
    with steady_learning_csv.open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(steady_learning_rows[0])
        )
        writer.writeheader()
        writer.writerows(steady_learning_rows)
    write_json(
        transient_learning,
        {
            "status": "completed_p418_transient_learning_curve",
            "training_trajectory_counts": [3, 6],
            "runs": [
                {
                    "training_trajectory_count": 3,
                    "training_direction": "up",
                    "test_solid_temperature_RMSE_K": 12.0,
                },
                {
                    "training_trajectory_count": 3,
                    "training_direction": "down",
                    "test_solid_temperature_RMSE_K": 10.0,
                },
                {
                    "training_trajectory_count": 6,
                    "training_direction": "both",
                    "test_solid_temperature_RMSE_K": 6.5,
                },
            ],
            "new_physical_parameters": [],
        },
    )
    write_json(
        transient,
        {
            "status": "completed_p418_physical_step_model_comparison",
            "lowest_test_temperature_RMSE_model_by_split": {
                "pair_disjoint_stress_test": {
                    "lowest_test_solid_temperature_RMSE_model": (
                        best_transient_model
                    ),
                    "lowest_test_solid_temperature_RMSE_K": (
                        best_transient_temperature
                    ),
                }
            },
            "diffusion_temperature_and_energy_effect_by_split": {
                "pair_disjoint_stress_test": {
                    "deterministic_solid_temperature_RMSE_K": 6.0,
                    "diffusion_refined_solid_temperature_RMSE_K": 5.0,
                    "deterministic_projection_aware_energy_normalized_RMSE": (
                        0.025
                    ),
                    "diffusion_refined_projection_aware_energy_normalized_RMSE": (
                        0.020 if joint else 0.060
                    ),
                    "diffusion_90pct_interval_coverage_fraction": 0.47,
                    "diffusion_90pct_interval_mean_width_K": 15.8,
                    "diffusion_is_joint_temperature_energy_improvement": joint,
                    "model_selection_uses_this_outcome": False,
                }
            },
        },
    )
    metric_rows = []
    for model, temperature, energy, temperature_metric in (
        (
            "initial_temperature_persistence",
            10.0 if learned_best else 4.0,
            0.02,
            "solid_temperature_RMSE_K",
        ),
        ("dmdc", 8.0, 0.05, "solid_temperature_RMSE_K"),
        (
            "graph_transformer_data_only",
            9.0,
            0.07,
            "solid_temperature_RMSE_K",
        ),
        (
            "graph_transformer_energy_flux",
            3.0 if learned_best else 7.0,
            0.03,
            "solid_temperature_RMSE_K",
        ),
        (
            "graph_transformer_factorized_energy_flux",
            6.0,
            0.025,
            "solid_temperature_RMSE_K",
        ),
        (
            "low_rank_residual_correction",
            5.5,
            0.04,
            "solid_temperature_RMSE_K",
        ),
        (
            "diffusion_residual_correction",
            5.0,
            0.020 if joint else 0.060,
            "diffusion_refined_solid_temperature_RMSE_K",
        ),
    ):
        for metric, value in (
            (temperature_metric, temperature),
            (ENERGY, energy),
        ):
            metric_rows.append(
                {
                    "split_name": "pair_disjoint_stress_test",
                    "model": model,
                    "data_role": "test",
                    "metric": metric,
                    "value": value,
                }
            )
    with metrics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    write_json(
        cost,
        {
            "status": "complete_p418_transient_cost_table",
            "split_name": "pair_disjoint_stress_test",
            "records": [
                {
                    "model": model,
                    "speedup_vs_32_rank_openfoam": speedup,
                    "full_workflow_break_even_curve_count": count,
                }
                for model, speedup, count in (
                    ("initial_temperature_persistence", 120.0, 0),
                    ("dmdc", 110.0, 2),
                    ("graph_transformer_data_only", 80.0, 30),
                    ("graph_transformer_energy_flux", 75.0, 35),
                    (
                        "graph_transformer_factorized_energy_flux",
                        95.0,
                        28,
                    ),
                    ("low_rank_residual_correction", 100.0, 4),
                    ("diffusion_residual_correction", 70.0, 30),
                )
            ],
        },
    )
    write_json(
        high_re,
        {
            "status": high_re_status,
            "curve_count": 6,
            "training_or_model_selection_performed": False,
            "fully_coupled_model_used_for_accuracy_ranking": False,
        },
    )
    high_re_rows = [
        {
            "metric": metric,
            "data_only": data,
            "physics_constrained": physics,
            "factorized": factorized,
        }
        for metric, data, physics, factorized in (
            ("fluid_temperature_volume_weighted_RMSE_K", 4.0, 3.0, 2.5),
            ("solid_temperature_volume_weighted_RMSE_K", 5.0, 3.5, 3.0),
            ("solid_maximum_temperature_history_RMSE_K", 6.0, 4.0, 3.5),
            (
                "solid_regional_hotspot_location_mean_error_m",
                0.003,
                0.002,
                0.0015,
            ),
            (
                "solid_regional_hotspot_location_p95_error_m",
                0.006,
                0.004,
                0.003,
            ),
        )
    ]
    with high_re_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(high_re_rows[0]))
        writer.writeheader()
        writer.writerows(high_re_rows)
    write_json(
        packing,
        {
            "status": "completed_seed101_seed202_integral_response_comparison",
            "complete_nine_case_comparison": True,
            "accepted_common_case_count": 9,
            "failed_seed202_case_count": 0,
            "metric_summary": {
                "outlet_temperature_K": {
                    "maximum_absolute_relative_change_percent": 0.67
                },
                "maximum_solid_temperature_K": {
                    "maximum_absolute_relative_change_percent": 0.31
                },
                "pressure_drop_Pa": {
                    "maximum_absolute_relative_change_percent": 18.0
                },
            },
        },
    )
    write_json(
        scope,
        {
            "status": "P418_SCOPE_LIMITS_EVIDENCE_SYNCED",
            "records": [
                {
                    "status": (
                        "hccb_pore_resolved_openfoam_mesh_preflight_failed"
                    ),
                    "job_id": "14630",
                    "slurm_state": "FAILED",
                },
                *[
                    {
                        "status": "failed_solver_exit_propagated",
                        "job_id": str(14721 + index),
                        "slurm_state": "FAILED",
                    }
                    for index in range(coupled_scope_count)
                ],
                *[
                    {
                        "status": "supporting_scope_record",
                        "job_id": str(14800 + index),
                        "slurm_state": "FAILED",
                    }
                    for index in range(5 - coupled_scope_count)
                ],
            ],
        },
    )
    write_json(
        external,
        {
            "status": "external_thermal_hydraulic_comparison_complete",
            "new_physical_parameters": [],
            "use_in_p418_training": False,
            "hcpb_annulus": {
                "n": 4,
                "points_inside_published_uncertainty": 4,
                "mean_absolute_relative_error_percent": 3.8665394701,
            },
            "fixed_bed_pressure": {
                "n": 6,
                "median_absolute_relative_error_percent": 3.7080552744,
            },
            "premux": {"n": 65, "rmse": 33.4118166118},
            "tesomex": {
                "snapshot_A": {"n": 7, "rmse": 56.0286150434},
                "snapshot_B": {"n": 7, "rmse": 34.7521203786},
            },
        },
    )
    command = [
        "python3",
        str(SCRIPT),
        "--steady-summary",
        str(steady),
        "--completed-physics-summary",
        str(completed),
        "--steady-hotspot-summary",
        str(hotspots),
        "--steady-seed-robustness",
        str(steady_seed),
        "--transient-seed-robustness",
        str(transient_seed),
        "--steady-learning-curve",
        str(steady_learning),
        "--transient-learning-curve",
        str(transient_learning),
        "--loss-balancing-selection",
        str(loss_balancing_selection),
        "--transient-summary",
        str(transient),
        "--transient-metrics",
        str(metrics),
        "--transient-cost",
        str(cost),
        "--high-re-comparison",
        str(high_re),
        "--high-re-aggregate",
        str(high_re_csv),
        "--cross-packing-summary",
        str(packing),
        "--external-evidence",
        str(external),
        "--scope-limits",
        str(scope),
        "--abstract-output",
        str(abstract),
        "--discussion-output",
        str(discussion),
        "--conclusion-output",
        str(conclusion),
        "--summary-output",
        str(summary),
    ]
    completed_process = subprocess.run(
        command, capture_output=True, text=True
    )
    return {
        "completed": completed_process,
        "abstract": abstract,
        "discussion": discussion,
        "conclusion": conclusion,
        "summary": summary,
    }


def test_builds_current_scope_final_sections(tmp_path: Path) -> None:
    result = run_builder(tmp_path, joint=True)
    assert result["completed"].returncode == 0, result["completed"].stderr
    abstract = result["abstract"].read_text(encoding="utf-8")
    discussion = result["discussion"].read_text(encoding="utf-8")
    conclusion = result["conclusion"].read_text(encoding="utf-8")
    assert "60 three-dimensional steady" in abstract
    assert "Disjoint condition and endpoint-pair splits" in abstract
    assert "separate validation from testing" in abstract
    assert "9.00 / 7.00 / 6.00 K" in abstract
    assert "initial-temperature persistence" in abstract
    assert "4.00 K" in abstract
    assert "2.50 and 3.00 K" in abstract
    assert "0.670\\% and 0.310\\%" in abstract
    assert "18.0\\%" in abstract
    assert "3.87\\% and 3.71\\%" in abstract
    assert "full-domain accuracy" not in discussion
    assert "no full-domain solver was run" in discussion
    assert "specified helium-property range" in discussion
    assert "registered helium-property range" not in discussion
    assert "Maximum seed variation" in discussion
    assert "10.0\\%" in discussion
    assert "from 0.080 to 0.032" in discussion
    assert "from 10.0--12.0 to 6.50 K" in discussion
    assert "selected the Kirchhoff ReLoBRaLo setting" in discussion
    assert "solid-temperature RMSE of 3.25 K" in discussion
    assert "No learned field model outperforms" in discussion
    assert "projection-aware energy RMSEs" in discussion
    assert "covered 47.0\\%" in discussion
    assert "mean width 15.8 K" in discussion
    assert "indicating under-dispersion" in discussion
    assert "energy-equation difference" not in abstract + discussion + conclusion
    assert "bulk porosity alone did not capture" in discussion
    assert "not cellwise validation" in discussion
    assert "finite-volume-consistent comparison" in conclusion
    assert (
        "Validation selects models and loss weights before independent testing"
        in conclusion
    )
    assert "sampled operating conditions and two packing realizations" in conclusion
    assert "seed303" not in abstract + discussion + conclusion
    assert "fully coupled predictor" not in abstract + discussion + conclusion
    word_count = lambda text: len(re.findall(r"\b[\w'-]+\b", text))
    assert word_count(abstract) <= 230
    assert word_count(discussion) <= 315
    assert word_count(conclusion) <= 120
    assert len([part for part in discussion.split("\n\n") if part.strip()]) <= 4
    payload = json.loads(result["summary"].read_text(encoding="utf-8"))
    assert payload["best_strict_transient_model"] == (
        "initial_temperature_persistence"
    )
    assert payload["best_strict_transient_model_is_persistence"] is True
    assert payload["best_high_re_model"] == "factorized"
    assert (
        "strict_transient_projection_aware_energy_normalized_RMSE"
        in payload
    )
    assert "strict_transient_energy_difference" not in payload
    assert payload["full_domain_solver_started"] is False
    assert payload["fully_coupled_accuracy_claimed"] is False
    assert payload["diffusion_90pct_interval_coverage_fraction"] == 0.47
    assert payload["diffusion_90pct_interval_mean_width_K"] == 15.8
    assert payload["diffusion_90pct_interval_is_underdispersed"] is True
    assert payload["new_physical_parameters"] == []
    assert payload["section_word_counts"] == {
        "abstract": word_count(abstract),
        "discussion": word_count(discussion),
        "conclusion": word_count(conclusion),
    }
    assert payload["section_word_limits"] == {
        "abstract": 230,
        "discussion": 315,
        "conclusion": 120,
    }
    transient_source = (
        ROOT / "code/build_hccb_p418_transient_result_text.py"
    ).read_text(encoding="utf-8")
    transient_limit = re.search(
        r"^RESULT_WORD_LIMIT = (\d+)$", transient_source, re.MULTILINE
    )
    assert transient_limit is not None
    assert (
        sum(payload["section_word_limits"].values())
        + int(transient_limit.group(1))
        <= 1055
    )
    assert payload["external_consistency"] == {
        "hcpb_annulus_mean_absolute_relative_error_percent": 3.8665394701,
        "fixed_bed_pressure_median_absolute_relative_error_percent": (
            3.7080552744
        ),
        "premux_temperature_RMSE_K": 33.4118166118,
        "tesomex_temperature_RMSE_range_K": [
            34.7521203786,
            56.0286150434,
        ],
        "used_in_p418_training": False,
        "cellwise_validation_claimed": False,
    }
    assert payload["robustness_and_learning_curve"] == {
        "maximum_steady_seed_cv_percent": 10.0,
        "maximum_transient_seed_std_K": 0.8,
        "steady_learning_low_count_nrmse": 0.08,
        "steady_learning_high_count_nrmse": 0.032,
        "transient_three_curve_RMSE_range_K": [10.0, 12.0],
        "transient_six_curve_RMSE_K": 6.5,
    }
    assert payload["fixed_flow_loss_balancing"] == {
        "candidate_id": "relobralo_kirchhoff_table_viii",
        "label": "the Kirchhoff ReLoBRaLo setting",
        "validation_selection_score": 0.45,
        "test_solid_temperature_RMSE_K": 3.25,
        "test_projection_aware_energy_normalized_RMSE": 0.018,
    }


def test_learned_model_cost_language_is_inference_specific() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Its complete-chain inference is" in source
    assert "training and reference-data cost recovered" in source
    assert "complete-chain acceleration" not in source


def test_learned_leader_reports_speedup_without_baseline_claim(
    tmp_path: Path,
) -> None:
    result = run_builder(tmp_path, joint=True, learned_best=True)
    assert result["completed"].returncode == 0, result["completed"].stderr
    discussion = result["discussion"].read_text(encoding="utf-8")
    conclusion = result["conclusion"].read_text(encoding="utf-8")
    assert "physics-constrained graph--Transformer" in discussion
    assert "75.0 times faster" in discussion
    assert "recovered after 35 predicted trajectories" in discussion
    assert "No learned field model outperforms" not in discussion + conclusion
    payload = json.loads(result["summary"].read_text(encoding="utf-8"))
    assert payload["best_strict_transient_model"] == (
        "graph_transformer_energy_flux"
    )
    assert payload["best_strict_transient_model_is_persistence"] is False


def test_diffusion_tradeoff_is_not_promoted(tmp_path: Path) -> None:
    result = run_builder(tmp_path, joint=False)
    assert result["completed"].returncode == 0, result["completed"].stderr
    combined = "".join(
        result[name].read_text(encoding="utf-8")
        for name in ("abstract", "discussion", "conclusion")
    )
    assert "retained as a trade-off" in combined
    assert "reduced both" not in combined


def test_partial_steady_matrix_is_rejected(tmp_path: Path) -> None:
    result = run_builder(tmp_path, joint=True, completed_count=59)
    assert result["completed"].returncode != 0
    assert "requires all 60 steady fields" in result["completed"].stderr


def test_old_high_re_comparison_is_rejected(tmp_path: Path) -> None:
    result = run_builder(
        tmp_path,
        joint=True,
        high_re_status=(
            "completed_p418_high_re_fixed_vs_fully_coupled_comparison"
        ),
    )
    assert result["completed"].returncode != 0
    assert "unexpected status" in result["completed"].stderr


def test_incomplete_fully_coupled_scope_evidence_is_rejected(
    tmp_path: Path,
) -> None:
    result = run_builder(tmp_path, joint=True, coupled_scope_count=2)
    assert result["completed"].returncode != 0
    assert (
        "three independent fully coupled startup failures are required"
        in result["completed"].stderr
    )
