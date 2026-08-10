from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_hccb_p418_cross_packing_conservative_operator",
    ROOT / "code" / "evaluate_hccb_p418_cross_packing_conservative_operator.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def valid_summary():
    return {
        "normalization": {
            "internal_mass_scale_kg_s": 1.0,
            "boundary_mass_scale_kg_s": 2.0,
            "regional_incident_mass_scale_kg_s": 3.0,
            "internal_energy_scale_W": 4.0,
            "boundary_energy_scale_W": 5.0,
            "regional_incident_energy_scale_W": 6.0,
        }
    }


def test_training_flux_scales_come_from_seed101_summary():
    assert MODULE.training_flux_scales(valid_summary()) == {
        "internal_mass_scale_kg_s": 1.0,
        "boundary_mass_scale_kg_s": 2.0,
        "regional_incident_mass_scale_kg_s": 3.0,
        "internal_energy_scale_W": 4.0,
        "boundary_energy_scale_W": 5.0,
        "regional_incident_energy_scale_W": 6.0,
    }


def test_training_flux_scales_reject_zero():
    summary = valid_summary()
    summary["normalization"]["internal_mass_scale_kg_s"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        MODULE.training_flux_scales(summary)


def test_protocol_rejects_seed303_normalization():
    payload = {
        "status": "cross_packing_model_protocol_ready_before_new_fields",
        "normalization": {"packing_seed": 303},
        "evaluation_packings": [
            {
                "seed": 202,
                "role": "development_packing",
                "condition_ids": [f"case_{index}" for index in range(9)],
            },
            {
                "seed": 303,
                "role": "final_zero_shot_packing",
                "condition_ids": [f"case_{index}" for index in range(9)],
            },
        ],
        "new_physical_parameter_values_added": [],
    }
    with pytest.raises(ValueError, match="normalization must come from seed101"):
        MODULE.validate_protocol(payload)


def test_conservative_cross_packing_reports_required_physics():
    text = (
        ROOT / "code" / "evaluate_hccb_p418_cross_packing_conservative_operator.py"
    ).read_text(encoding="utf-8")
    for name in (
        "integrated_heat_transfer_metrics",
        "fluid_temperature_volume_weighted_rmse_K",
        "solid_temperature_volume_weighted_rmse_K",
        "solid_hotspot_location_error_m",
        "local_mass_l1_over_two_inlet",
        "local_energy_l1_over_two_generated_power",
    ):
        assert name in text
    assert '"normalization_packing_seed": 101' in text


def test_tiny_nine_case_conservative_evaluation_runs_end_to_end(tmp_path: Path):
    # Software fixture only. These numbers are not research physical inputs.
    code = ROOT / "code"
    project = tmp_path / "project"
    parameters = project / "parameters"
    results = project / "results"
    parameters.mkdir(parents=True)
    results.mkdir(parents=True)

    topology = results / "topology.npz"
    np.savez_compressed(
        topology,
        fine_node_centroid_m=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        fine_node_volume_m3=np.ones(2),
        fine_node_type=np.array([0, 1]),
        level_0_centroid_m=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        level_0_volume_m3=np.ones(2),
        level_0_node_type=np.array([0, 1]),
        level_0_parent_from_finer=np.array([0, 1]),
        level_0_edge_source=np.array([0, 1]),
        level_0_edge_target=np.array([1, 0]),
        level_0_edge_kind=np.array([2, 2]),
        level_0_edge_area_m2=np.ones(2),
        level_0_edge_area_vector_m2=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        level_0_edge_centroid_m=np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    geometry = results / "geometry.npz"
    np.savez_compressed(
        geometry,
        fine_boundary_role=np.zeros((2, 2)),
        level_0_boundary_volume_fraction=np.zeros((2, 2)),
        coordinate_center_m=np.array([0.5, 0.0, 0.0]),
        coordinate_scale_m=np.ones(3),
        volume_scale_m3=np.array(1.0),
    )
    condition_ids = np.array([f"case_{index}" for index in range(9)])
    condition = np.tile(np.array([1.0, 300.0, 1.0, 1.0e5, 635.0]), (9, 1))
    state = np.zeros((9, 2, 5))
    state[:, 0, 0] = 1.0
    state[:, 0, 3] = 1.0e5
    state[:, 0, 4] = 300.0
    state[:, 1, 4] = 400.0
    state_targets = results / "state.npz"
    np.savez_compressed(
        state_targets,
        condition_id=condition_ids,
        condition_physical=condition,
        state_physical=state,
        node_type=np.array([0, 1]),
        node_volume_m3=np.ones(2),
    )
    mass_targets = results / "mass.npz"
    np.savez_compressed(
        mass_targets,
        condition_id=condition_ids,
        internal_mass_flow_kg_s=np.zeros((9, 1)),
        boundary_mass_flow_kg_s=np.tile(np.array([-1.0, 1.0]), (9, 1)),
        fluid_global_region=np.array([0]),
        internal_owner=np.array([0]),
        internal_neighbour=np.array([0]),
        internal_face_centroid_m=np.array([[0.5, 0.0, 0.0]]),
        internal_face_area_vector_m2=np.array([[1.0, 0.0, 0.0]]),
        internal_face_area_m2=np.ones(1),
        boundary_owner=np.array([0, 0]),
        boundary_patch=np.array([0, 1]),
        boundary_face_centroid_m=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        boundary_face_area_vector_m2=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        boundary_face_area_m2=np.ones(2),
    )
    energy_targets = results / "energy.npz"
    np.savez_compressed(
        energy_targets,
        condition_id=condition_ids,
        node_type=np.array([0, 1]),
        internal_energy_flow_W=np.ones((9, 1)),
        boundary_energy_flow_W=np.tile(np.array([-1.0, 0.0]), (9, 1)),
        node_source_power_W=np.tile(np.array([0.0, 1.0]), (9, 1)),
        internal_owner=np.array([0]),
        internal_neighbour=np.array([1]),
        internal_kind=np.array([0]),
        internal_kind_name=np.array(["fluid_to_solid"]),
        internal_face_centroid_m=np.array([[0.5, 0.0, 0.0]]),
        internal_face_area_vector_m2=np.array([[1.0, 0.0, 0.0]]),
        internal_face_area_m2=np.ones(1),
        boundary_owner=np.array([0, 1]),
        boundary_kind=np.array([0, 1]),
        boundary_kind_name=np.array(["fluid:coolingWall", "solid:other"]),
        boundary_face_centroid_m=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        boundary_face_area_vector_m2=np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
        boundary_face_area_m2=np.ones(2),
    )

    statistics = results / "statistics.json"
    statistics.write_text(
        json.dumps(
            {
                "splits": {
                    "base": {
                        "condition_input": {
                            "mean": [0.0] * 5,
                            "standard_deviation": [1.0] * 5,
                        },
                        "targets": {
                            "fluid_velocity_m_s": {"mean": [0.0] * 3, "standard_deviation": [1.0] * 3},
                            "fluid_gauge_pressure_Pa": {"mean": [0.0], "standard_deviation": [1.0]},
                            "fluid_temperature_K": {"mean": [300.0], "standard_deviation": [1.0]},
                            "solid_temperature_K": {"mean": [400.0], "standard_deviation": [1.0]},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    from hccb_p418_conservative_mixed_operator import HCCBP418ConservativeMixedOperator
    from train_hccb_p418_regional_operator import build_model

    field_model, settings = build_model("pinn", 2)
    model = HCCBP418ConservativeMixedOperator(
        field_operator=field_model,
        patch_count=2,
        internal_mass_scale_kg_s=1.0,
        boundary_mass_scale_kg_s=2.0,
        internal_energy_scale_W=4.0,
        boundary_energy_scale_W=5.0,
        internal_energy_kind_count=1,
        boundary_energy_kind_count=2,
    )
    checkpoint = results / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "settings": settings}, checkpoint)

    training_summary = results / "training_summary.json"
    summary = valid_summary()
    summary.update(
        {
            "status": "conservative_mixed_operator_training_complete",
            "architecture": "pinn",
            "split_name": "base",
            "new_physical_parameters": [],
            "run_provenance": {
                "architecture": "pinn",
                "split_name": "base",
                "common_inputs": {
                    "training_statistics": {
                        "path": str(statistics.resolve()),
                        "sha256": MODULE.sha256(statistics),
                    },
                    "mass_targets": {
                        "path": str(mass_targets.resolve()),
                        "sha256": MODULE.sha256(mass_targets),
                    },
                    "energy_targets": {
                        "path": str(energy_targets.resolve()),
                        "sha256": MODULE.sha256(energy_targets),
                    },
                },
            },
        }
    )
    summary["normalization"]["scales_use_training_cases_only"] = True
    training_summary.write_text(json.dumps(summary), encoding="utf-8")

    protocol = parameters / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "status": "cross_packing_model_protocol_ready_before_new_fields",
                "normalization": {
                    "packing_seed": 101,
                    "training_statistics": str(statistics.relative_to(project)),
                    "regional_topology": str(topology.relative_to(project)),
                    "model_geometry": str(geometry.relative_to(project)),
                    "regional_level": 0,
                    "split_name": "base",
                },
                "evaluation_packings": [
                    {
                        "seed": 202,
                        "role": "development_packing",
                        "regional_topology": str(topology.relative_to(project)),
                        "model_geometry": str(geometry.relative_to(project)),
                        "state_targets": str(state_targets.relative_to(project)),
                        "mass_targets": str(mass_targets.relative_to(project)),
                        "energy_targets": str(energy_targets.relative_to(project)),
                        "regional_level": 0,
                        "condition_ids": condition_ids.tolist(),
                    },
                    {
                        "seed": 303,
                        "role": "final_zero_shot_packing",
                        "regional_topology": str(topology.relative_to(project)),
                        "model_geometry": str(geometry.relative_to(project)),
                        "state_targets": str(state_targets.relative_to(project)),
                        "mass_targets": str(mass_targets.relative_to(project)),
                        "energy_targets": str(energy_targets.relative_to(project)),
                        "regional_level": 0,
                        "condition_ids": condition_ids.tolist(),
                    }
                ],
                "new_physical_parameter_values_added": [],
            }
        ),
        encoding="utf-8",
    )
    output = results / "evaluation.json"
    subprocess.run(
        [
            sys.executable,
            str(code / "evaluate_hccb_p418_cross_packing_conservative_operator.py"),
            "--project-root",
            str(project),
            "--protocol",
            str(protocol),
            "--packing-seed",
            "202",
            "--checkpoint",
            str(checkpoint),
            "--training-summary",
            str(training_summary),
            "--architecture",
            "pinn",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["case_count"] == 9
    assert len(result["cases"]) == 9
    assert result["normalization_packing_seed"] == 101
    assert result["model_loaded_before_this_packing_fields"] is True
    assert all(np.isfinite(case["state_normalized_rmse"]) for case in result["cases"])
