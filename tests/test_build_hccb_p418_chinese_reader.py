import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_chinese_reader.py"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def final_payload() -> dict:
    return {
        "status": "complete_p418_final_manuscript_narrative",
        "steady_case_count": 60,
        "fixed_flow_trajectory_count": 12,
        "high_re_curve_count": 6,
        "independent_packing_case_count": 9,
        "steady_solid_temperature_leader": "pinn",
        "steady_solid_temperature_worst_case_nrmse": 0.032,
        "solid_maximum_temperature_range_K": [634.9, 921.4],
        "wall_heat_direction_case_counts": {
            "wall_to_fluid": 24,
            "fluid_to_wall": 36,
            "zero": 0,
        },
        "wall_heat_reversal_temperature_range_K": [608.5, 626.2],
        "maximum_adjacent_reference_hotspot_distance_m": 0.0048,
        "strict_transient_temperature_RMSE_K": {
            "graph_transformer_data_only": 9.0,
            "graph_transformer_energy_flux": 7.0,
            "graph_transformer_factorized_energy_flux": 6.0,
        },
        "strict_transient_projection_aware_energy_normalized_RMSE": {
            "graph_transformer_data_only": 0.07,
            "graph_transformer_energy_flux": 0.03,
            "graph_transformer_factorized_energy_flux": 0.025,
        },
        "best_strict_transient_model": (
            "graph_transformer_factorized_energy_flux"
        ),
        "best_strict_transient_speedup": 95.0,
        "best_strict_transient_break_even_curves": 28,
        "diffusion_joint_temperature_energy_improvement": False,
        "diffusion_temperature_RMSE_K": {
            "deterministic": 6.0,
            "refined": 5.0,
        },
        "diffusion_projection_aware_energy_normalized_RMSE": {
            "deterministic": 0.025,
            "refined": 0.060,
        },
        "diffusion_90pct_interval_coverage_fraction": 0.47,
        "diffusion_90pct_interval_mean_width_K": 15.8,
        "best_high_re_model": "factorized",
        "best_high_re_fluid_temperature_RMSE_K": 2.5,
        "best_high_re_solid_temperature_RMSE_K": 3.0,
        "independent_packing_maximum_absolute_change_percent": {
            "outlet_temperature": 0.67,
            "maximum_solid_temperature": 0.31,
            "pressure_drop": 18.0,
        },
        "external_consistency": {
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
        },
        "robustness_and_learning_curve": {
            "maximum_steady_seed_cv_percent": 10.0,
            "maximum_transient_seed_std_K": 0.8,
            "steady_learning_low_count_nrmse": 0.08,
            "steady_learning_high_count_nrmse": 0.032,
            "transient_three_curve_RMSE_range_K": [10.0, 12.0],
            "transient_six_curve_RMSE_K": 6.5,
        },
        "full_domain_solver_started": False,
        "fully_coupled_accuracy_claimed": False,
        "new_physical_parameters": [],
    }


def mesh_payload(*, solid_passes: bool = True) -> dict:
    return {
        "status": "completed_three_mesh_p418_cht_comparison",
        "new_physical_parameters": [],
        "mesh_levels": [
            {
                "mesh_level": "coarse",
                "fluid_cells": 160989,
                "solid_cells": 200162,
                "fluid_basic_check_passes": True,
                "solid_basic_check_passes": solid_passes,
            },
            {
                "mesh_level": "medium",
                "fluid_cells": 432384,
                "solid_cells": 515540,
                "fluid_basic_check_passes": True,
                "solid_basic_check_passes": True,
            },
            {
                "mesh_level": "fine",
                "fluid_cells": 858419,
                "solid_cells": 1011645,
                "fluid_basic_check_passes": True,
                "solid_basic_check_passes": True,
            },
        ],
        "grid_convergence": [
            {
                "metric": "pressure_drop_Pa",
                "fine_gci_fraction": 0.3715,
            },
            {
                "metric": "outlet_temperature_change_K",
                "fine_gci_fraction": 0.0196,
            },
            {
                "metric": "solid_maximum_temperature_change_K",
                "fine_gci_absolute": 0.620,
            },
            {
                "metric": "cooling_wall_heat_fraction",
                "fine_gci_fraction": 0.8666,
            },
        ],
    }


def run_builder(
    tmp_path: Path, *, final: dict, mesh: dict
) -> subprocess.CompletedProcess[str]:
    final_path = tmp_path / "final.json"
    mesh_path = tmp_path / "mesh.json"
    output = tmp_path / "reader.md"
    summary = tmp_path / "summary.json"
    write_json(final_path, final)
    write_json(mesh_path, mesh)
    return subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--final-narrative-summary",
            str(final_path),
            "--mesh-sensitivity-summary",
            str(mesh_path),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ],
        capture_output=True,
        text=True,
    )


def test_builds_concise_complete_chinese_reader(tmp_path: Path) -> None:
    completed = run_builder(
        tmp_path, final=final_payload(), mesh=mesh_payload()
    )
    assert completed.returncode == 0, completed.stderr
    reader = (tmp_path / "reader.md").read_text(encoding="utf-8")
    assert reader.count("\n## ") == 7
    assert "60 个三维稳态" in reader
    assert "12 条 0--300 s" in reader
    assert "压降 37.1%" in reader
    assert "出口温度变化 1.96%" in reader
    assert "固体最高温度变化绝对值 0.620 K" in reader
    assert "纯数据、能量与热流约束" in reader
    assert "扩散修正没有同时改善" in reader
    assert "实际覆盖率为47.0%" in reader
    assert "该区间明显过窄" in reader
    assert "3.87%" in reader
    assert "3.71%" in reader
    assert "不作为当前三维网格逐单元精度证明" in reader
    assert "全文域稳态求解" in reader
    assert all(
        token not in reader
        for token in ("尚待", "待完成", "完成后", "预计结果")
    )
    assert len(reader) < 9000
    payload = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "completed_p418_concise_chinese_reader"
    assert payload["section_count"] == 7
    assert payload["steady_case_count"] == 60
    assert payload["new_physical_parameters"] == []
    assert len(payload["source_files"]) == 2


def test_accepts_parameter_free_transient_leader(tmp_path: Path) -> None:
    final = final_payload()
    final["best_strict_transient_model"] = "initial_temperature_persistence"
    completed = run_builder(tmp_path, final=final, mesh=mesh_payload())
    assert completed.returncode == 0, completed.stderr
    reader = (tmp_path / "reader.md").read_text(encoding="utf-8")
    assert "初始温度场持续性基线" in reader
    assert "持续性基线、DMDc、三种图 Transformer" in reader
    assert "它不需要训练，因此不报告训练成本回收曲线数" in reader
    assert "约预测 28 条曲线" not in reader


def test_rejects_incomplete_final_result(tmp_path: Path) -> None:
    final = final_payload()
    final["steady_case_count"] = 59
    completed = run_builder(
        tmp_path, final=final, mesh=mesh_payload()
    )
    assert completed.returncode != 0
    assert "steady_case_count 必须为 60" in completed.stderr


def test_rejects_mesh_that_failed_basic_check(tmp_path: Path) -> None:
    completed = run_builder(
        tmp_path,
        final=final_payload(),
        mesh=mesh_payload(solid_passes=False),
    )
    assert completed.returncode != 0
    assert "网格未通过流体区和固体区基础检查" in completed.stderr
