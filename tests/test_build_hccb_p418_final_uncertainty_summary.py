from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_final_uncertainty_summary import (  # noqa: E402
    build_summary,
    load_diffusion_metrics,
    write_chinese,
    write_tex,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_args(tmp_path: Path) -> argparse.Namespace:
    mesh = tmp_path / "mesh.json"
    fixed = tmp_path / "fixed.json"
    scope = tmp_path / "scope.json"
    steady = tmp_path / "steady.json"
    transient = tmp_path / "transient.json"
    metrics = tmp_path / "metrics.csv"
    packing = tmp_path / "packing.json"
    external = tmp_path / "external.json"
    external_metrics = tmp_path / "external_metrics.csv"
    write_json(
        mesh,
        {
            "status": "completed_three_mesh_p418_cht_comparison",
            "grid_convergence": [
                {
                    "metric": "pressure_drop_Pa",
                    "convergence_status": "monotonic_gci_reported",
                    "fine_gci_fraction": 0.012,
                },
                {
                    "metric": "solid_maximum_temperature_K",
                    "convergence_status": "monotonic_gci_reported",
                    "coarse_value": -0.14,
                    "medium_value": 0.05,
                    "fine_value": 0.16,
                    "fine_gci_fraction": 3.85,
                },
            ],
        },
    )
    write_json(
        fixed,
        {
            "status": "completed_p418_thermal_timestep_sensitivity",
            "gci_results": [
                {
                    "signal": "solid_maximum_temperature_K",
                    "quantity": "endpoint",
                    "convergence_status": "monotonic_gci_reported",
                    "fine_gci_fraction": 0.008,
                }
            ],
        },
    )
    write_json(
        scope,
        {
            "status": "P418_SCOPE_LIMITS_EVIDENCE_SYNCED",
            "records": [
                {
                    "filename": f"maxCo_{label}_job_failure.json",
                    "status": "failed_solver_exit_propagated",
                    "scientific_meaning": (
                        f"Formal maxCo={max_co} fully coupled run stopped at "
                        f"{stop_time} s because a thermophysical-property query "
                        "left the available range."
                    ),
                }
                for label, max_co, stop_time in (
                    ("0p8", 0.8, 0.0011),
                    ("0p4", 0.4, 0.0008),
                    ("0p2", 0.2, 0.0005),
                )
            ],
        },
    )
    write_json(
        steady,
        {
            "status": "completed_p418_main_steady_split_seed_robustness",
            "metrics": [
                {
                    "architecture": "pinn",
                    "metric": "solid_maximum_temperature_p95_K",
                    "unit": "K",
                    "mean": 3.0,
                    "sample_std": 0.2,
                },
                {
                    "architecture": "graph",
                    "metric": "solid_temperature_normalized_rmse",
                    "unit": "",
                    "mean": 0.03,
                    "sample_std": 0.005,
                },
            ],
        },
    )
    write_json(
        transient,
        {
            "status": "completed_p418_strict_split_seed_robustness",
            "metrics": [
                {
                    "model": "graph_transformer_energy_flux",
                    "metric": "solid_temperature_RMSE_K",
                    "mean_K": 1.2,
                    "sample_std_K": 0.1,
                },
                {
                    "model": "diffusion_residual_correction",
                    "metric": "diffusion_refined_solid_temperature_RMSE_K",
                    "mean_K": 0.9,
                    "sample_std_K": 0.15,
                },
            ],
        },
    )
    base = {
        "split_name": "pair_disjoint_stress_test",
        "model": "diffusion_residual_correction",
        "result_scope": "regional_temperature_field",
        "data_role": "test",
        "unit": "",
        "training_seconds": "1",
        "source_summary": "fixture",
    }
    write_csv(
        metrics,
        [
            {
                **base,
                "metric": (
                    "diffusion_unobserved_dynamic_solid_90pct_interval_coverage_fraction"
                ),
                "value": 0.88,
            },
            {
                **base,
                "metric": (
                    "diffusion_unobserved_dynamic_solid_90pct_interval_mean_width_K"
                ),
                "value": 2.4,
            },
            {
                **base,
                "metric": "diffusion_unobserved_dynamic_solid_CRPS_K",
                "value": 0.5,
            },
        ],
    )
    write_json(
        packing,
        {
            "status": "completed_seed101_seed202_integral_response_comparison",
            "complete_nine_case_comparison": True,
            "accepted_common_case_count": 9,
            "failed_seed202_case_count": 0,
            "metric_summary": {
                metric: {
                    "maximum_absolute_relative_change_percent": maximum,
                    "mean_absolute_relative_change_percent": mean,
                    "median_absolute_relative_change_percent": median,
                }
                for metric, maximum, mean, median in (
                    ("outlet_temperature_K", 0.66, 0.53, 0.56),
                    ("maximum_solid_temperature_K", 0.31, 0.13, 0.006),
                    ("pressure_drop_Pa", 17.99, 16.10, 15.48),
                )
            },
        },
    )
    write_json(
        external,
        {
            "status": "external_thermal_hydraulic_comparison_complete",
            "use_in_p418_training": False,
        },
    )
    write_csv(
        external_metrics,
        [
            {
                "experiment": name,
                "quantity": "q",
                "comparison": "c",
                "metric": "m",
                "value": index,
            }
            for index, name in enumerate(
                ("PREMUX", "TESOMEX A", "TESOMEX B", "HELOKA", "fixed bed")
            )
        ],
    )
    return argparse.Namespace(
        mesh_summary=mesh,
        fixed_timestep_summary=fixed,
        scope_limit_summary=scope,
        steady_seed_summary=steady,
        transient_seed_summary=transient,
        transient_metrics=metrics,
        cross_packing_summary=packing,
        external_summary=external,
        external_metrics=external_metrics,
        output_dir=tmp_path / "output",
        tex_output=tmp_path / "generated.tex",
    )


def test_builds_separate_uncertainty_components(tmp_path: Path) -> None:
    summary, rows = build_summary(fixture_args(tmp_path))
    assert summary["status"] == "completed_p418_final_uncertainty_summary"
    assert summary["headline_results"]["mesh"]["largest_finite_fraction"] == 0.012
    assert summary["headline_results"]["mesh"]["unavailable_count"] == 1
    assert summary["headline_results"]["fully_coupled_scope"]["gci_reported"] is False
    assert (
        summary["headline_results"]["packing_realization"]
        ["largest_change_percent"]
        == 17.99
    )
    assert summary["headline_results"]["diffusion_ensemble"]["coverage"] == 0.88
    assert summary["material_parameter_probability_propagation"]["performed"] is False
    assert {row["source_kind"] for row in rows} >= {
        "spatial_mesh",
        "fixed_flow_timestep",
        "fully_coupled_applicability",
        "steady_training_seed",
        "transient_training_seed",
        "packing_realization",
        "diffusion_predictive_interval",
    }
    tex = tmp_path / "generated.tex"
    chinese = tmp_path / "result_cn.md"
    write_tex(tex, summary["headline_results"])
    write_chinese(
        chinese,
        summary["headline_results"],
        summary["headline_results"]["external_comparisons"],
    )
    assert "\\paragraph{Numerical and model sensitivity.}" in tex.read_text(
        encoding="utf-8"
    )
    assert "为什么没有随意做“所有参数±5%”" in chinese.read_text(encoding="utf-8")


def test_rejects_external_data_used_for_training(tmp_path: Path) -> None:
    args = fixture_args(tmp_path)
    payload = json.loads(args.external_summary.read_text(encoding="utf-8"))
    payload["use_in_p418_training"] = True
    write_json(args.external_summary, payload)
    with pytest.raises(ValueError, match="outside P418 training"):
        build_summary(args)


def test_rejects_missing_diffusion_interval_metric(tmp_path: Path) -> None:
    args = fixture_args(tmp_path)
    with args.transient_metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    write_csv(args.transient_metrics, rows[:2])
    with pytest.raises(ValueError, match="expected one strict-split diffusion metric"):
        load_diffusion_metrics(args.transient_metrics)
