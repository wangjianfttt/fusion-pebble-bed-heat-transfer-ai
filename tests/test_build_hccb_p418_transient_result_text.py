import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_transient_result_text.py"
SPLITS = (
    "direction_down_test",
    "direction_up_test",
    "pair_disjoint_stress_test",
)
ENERGY = "projection_aware_volume_weighted_energy_equation_normalized_RMSE"


def write_inputs(tmp_path: Path, *, joint: bool) -> tuple[Path, Path, Path]:
    summary = tmp_path / "summary.json"
    metrics = tmp_path / "metrics.csv"
    cost = tmp_path / "cost.json"
    summary.write_text(
        json.dumps(
            {
                "status": "completed_p418_physical_step_model_comparison",
                "splits": list(SPLITS),
                "lowest_test_temperature_RMSE_model_by_split": {
                    "direction_down_test": {
                        "lowest_test_solid_temperature_RMSE_model": "dmdc",
                        "lowest_test_solid_temperature_RMSE_K": 4.2,
                    },
                    "direction_up_test": {
                        "lowest_test_solid_temperature_RMSE_model": "low_rank_residual_correction",
                        "lowest_test_solid_temperature_RMSE_K": 3.1,
                    },
                    "pair_disjoint_stress_test": {
                        "lowest_test_solid_temperature_RMSE_model": "diffusion_residual_correction",
                        "lowest_test_solid_temperature_RMSE_K": 5.3,
                    },
                },
                "diffusion_temperature_and_energy_effect_by_split": {
                    "pair_disjoint_stress_test": {
                        "deterministic_solid_temperature_RMSE_K": 7.0,
                        "diffusion_refined_solid_temperature_RMSE_K": 5.3,
                        "deterministic_projection_aware_energy_normalized_RMSE": 0.030,
                        "diffusion_refined_projection_aware_energy_normalized_RMSE": (
                            0.020 if joint else 0.050
                        ),
                        "diffusion_member_projection_aware_energy_equation_normalized_RMSE_p95": 0.080,
                        "diffusion_member_joint_temperature_energy_improvement_fraction": 0.625,
                        "diffusion_member_sample_count": 32,
                        "diffusion_is_joint_temperature_energy_improvement": joint,
                        "held_out_outcome": (
                            "joint_improvement" if joint else "not_joint_improvement"
                        ),
                        "outcome_reason": (
                            "joint_temperature_and_energy_improvement"
                            if joint
                            else "energy_residual_increased_despite_lower_temperature_error"
                        ),
                        "model_selection_uses_this_outcome": False,
                    }
                },
                "strict_split_seed_robustness": {
                    "split_name": "pair_disjoint_stress_test",
                    "seeds": [101, 202, 303],
                    "lowest_three_seed_mean_field_model": "graph_transformer_energy_flux",
                    "lowest_three_seed_mean_field_RMSE_K": 6.2,
                    "corresponding_sample_std_K": 0.4,
                },
                "fair_comparison_checks": {
                    "same_complete_curve_splits": True,
                    "same_graph_transformer_architecture": True,
                    "same_graph_transformer_training_normalization": True,
                    "all_models_selected_using_validation_curves": True,
                    "test_curves_used_only_for_final_metrics": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    for model, temperature, energy in (
        ("graph_transformer_data_only", 8.0, 0.060),
        ("graph_transformer_energy_flux", 7.0, 0.030),
    ):
        rows.extend(
            [
                {
                    "split_name": "pair_disjoint_stress_test",
                    "model": model,
                    "result_scope": "regional_temperature_field",
                    "data_role": "test",
                    "metric": "solid_temperature_RMSE_K",
                    "value": temperature,
                    "unit": "K",
                    "training_seconds": 10,
                    "source_summary": "fixture",
                },
                {
                    "split_name": "pair_disjoint_stress_test",
                    "model": model,
                    "result_scope": "transient_energy_balance",
                    "data_role": "test",
                    "metric": ENERGY,
                    "value": energy,
                    "unit": "dimensionless",
                    "training_seconds": 10,
                    "source_summary": "fixture",
                },
            ]
        )
        rows.append(
            {
                "split_name": "pair_disjoint_stress_test",
                "model": model,
                "result_scope": "regional_temperature_field",
                "data_role": "validation",
                "metric": "solid_temperature_RMSE_K",
                "value": temperature + 10.0,
                "unit": "K",
                "training_seconds": 10,
                "source_summary": "fixture",
            }
        )
    rows.extend(
        {
            "split_name": "pair_disjoint_stress_test",
            "model": "diffusion_residual_correction",
            "result_scope": "regional_temperature_field",
            "data_role": "test",
            "metric": metric,
            "value": value,
            "unit": unit,
            "training_seconds": 30,
            "source_summary": "fixture",
        }
        for metric, value, unit in (
            (
                "diffusion_unobserved_dynamic_solid_90pct_interval_coverage_fraction",
                0.74,
                "fraction",
            ),
            (
                "diffusion_unobserved_dynamic_solid_90pct_interval_mean_width_K",
                12.5,
                "K",
            ),
            ("diffusion_unobserved_dynamic_solid_CRPS_K", 3.2, "K"),
        )
    )
    rows.extend(
        {
            "split_name": "pair_disjoint_stress_test",
            "model": model,
            "result_scope": "regional_temperature_field",
            "data_role": "test",
            "metric": metric,
            "value": value,
            "unit": unit,
            "training_seconds": 20,
            "source_summary": "fixture",
        }
        for model, metric, value, unit in (
            (
                "graph_transformer_energy_flux",
                "solid_maximum_temperature_history_RMSE_K",
                4.5,
                "K",
            ),
            (
                "graph_transformer_energy_flux",
                "solid_regional_hotspot_location_p95_error_m",
                0.024,
                "m",
            ),
            (
                "graph_transformer_energy_flux",
                "solid_hotspot_target_temperature_deficit_p95_K",
                1.2,
                "K",
            ),
            (
                "diffusion_residual_correction",
                "diffusion_refined_solid_maximum_temperature_history_RMSE_K",
                3.8,
                "K",
            ),
            (
                "diffusion_residual_correction",
                "diffusion_refined_solid_regional_hotspot_location_p95_error_m",
                0.015,
                "m",
            ),
            (
                "diffusion_residual_correction",
                "diffusion_refined_solid_hotspot_target_temperature_deficit_p95_K",
                0.8,
                "K",
            ),
        )
    )
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cost.write_text(
        json.dumps(
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
                        ("graph_transformer_energy_flux", 120.0, 24),
                        ("low_rank_residual_correction", 100.0, 28),
                        ("diffusion_residual_correction", 80.0, 35),
                    )
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return summary, metrics, cost


def run_builder(tmp_path: Path, *, joint: bool) -> str:
    summary, metrics, cost = write_inputs(tmp_path, joint=joint)
    output = tmp_path / "result.tex"
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--summary",
            str(summary),
            "--metrics",
            str(metrics),
            "--cost-summary",
            str(cost),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output.read_text(encoding="utf-8")


def test_reports_strict_split_physics_and_complete_chain_cost(tmp_path: Path) -> None:
    text = run_builder(tmp_path, joint=True)
    assert "endpoint-pair-disjoint holdout" in text
    assert "data-only and physics-constrained" in text
    assert "8.00" in text and "7.00" in text
    assert "18.0" in text and "17.0" in text
    assert "different step families" in text
    assert "reported separately rather than pooled" in text
    assert "0.060" in text and "0.030" in text
    assert "120" in text and "24 curves" in text
    assert "80.0" in text and "35 curves" in text
    assert "\\times" in text
    assert "\t" not in text
    assert "not for checkpoint or architecture selection" in text
    assert "nominal 90\\% ensemble interval covered" in text
    assert "74.0\\%" in text
    assert "12.5" in text and "3.20" in text
    assert "do not assume that it is calibrated" in text
    assert "32 stochastic prediction sets" in text
    assert "0.080" in text and "62.5\\%" in text
    assert "not only for their ensemble mean" in text
    assert "dynamic solid hotspot" in text
    assert "4.50" in text and "3.80" in text
    assert "0.024" in text and "0.015" in text
    assert "1.20" in text and "0.800" in text
    assert "rank exchange between nearly equal neighbouring regions" in text
    assert "not pebble-internal maxima" in text
    assert len(re.findall(r"\b[\w'-]+\b", text)) <= 390


def test_reports_diffusion_tradeoff_instead_of_hiding_it(tmp_path: Path) -> None:
    text = run_builder(tmp_path, joint=False)
    assert "temperature--energy trade-off" in text
    assert "larger projection-aware energy RMSE" in text
    assert "energy-equation difference" not in text
    assert "joint improvement on this split" not in text


def test_rejects_held_out_outcome_used_for_selection(tmp_path: Path) -> None:
    summary, metrics, cost = write_inputs(tmp_path, joint=True)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["diffusion_temperature_and_energy_effect_by_split"][
        "pair_disjoint_stress_test"
    ]["model_selection_uses_this_outcome"] = True
    summary.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--summary",
            str(summary),
            "--metrics",
            str(metrics),
            "--cost-summary",
            str(cost),
            "--output",
            str(tmp_path / "result.tex"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "used for model selection" in completed.stderr
