from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "plot_hccb_p418_field_cloud_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("field_cloud_comparison", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def common_arrays() -> dict[str, np.ndarray]:
    return {
        "sequence_id": np.array(["case_a"]),
        "time_s": np.array([[0.0, 1.0]]),
        "node_type": np.array([0, 1], dtype=np.int8),
        "node_volume_m3": np.array([1.0, 2.0]),
        "node_centroid_m": np.array(
            [[0.0, 0.0, 0.0], [1.0e-3, 2.0e-3, 3.0e-3]]
        ),
    }


def test_reads_physical_temperature_fields(tmp_path: Path) -> None:
    path = tmp_path / "physical.npz"
    truth = np.array([[[300.0, 600.0], [301.0, 602.0]]])
    prediction = truth + 2.0
    np.savez(
        path,
        **common_arrays(),
        temperature_target_K=truth,
        temperature_prediction_K=prediction,
    )
    with np.load(path, allow_pickle=False) as data:
        _, restored_truth, restored_prediction, representation = (
            MODULE.read_temperature_fields(data)
        )
    np.testing.assert_allclose(restored_truth, truth)
    np.testing.assert_allclose(restored_prediction, prediction)
    assert representation == "physical temperature in K"


def test_restores_node_type_normalized_fields(tmp_path: Path) -> None:
    path = tmp_path / "normalized.npz"
    target = np.array([[[[0.0], [1.0]], [[1.0], [0.0]]]])
    prediction = target + 0.5
    np.savez(
        path,
        **common_arrays(),
        target_temperature_normalized=target,
        baseline_temperature_normalized=prediction,
        temperature_mean_K_by_node_type=np.array([300.0, 600.0]),
        temperature_std_K_by_node_type=np.array([10.0, 20.0]),
    )
    with np.load(path, allow_pickle=False) as data:
        _, restored_truth, restored_prediction, representation = (
            MODULE.read_temperature_fields(data)
        )
    np.testing.assert_allclose(
        restored_truth, np.array([[[300.0, 620.0], [310.0, 600.0]]])
    )
    np.testing.assert_allclose(
        restored_prediction, np.array([[[305.0, 630.0], [315.0, 610.0]]])
    )
    assert representation == "node-type-normalized temperature restored to K"


def test_prefers_low_rank_corrected_field_over_embedded_baseline(tmp_path: Path) -> None:
    path = tmp_path / "low_rank.npz"
    target = np.array([[[[0.0], [1.0]], [[1.0], [0.0]]]])
    np.savez(
        path,
        **common_arrays(),
        target_temperature_normalized=target,
        baseline_temperature_normalized=target + 0.9,
        corrected_temperature_normalized=target + 0.2,
        temperature_mean_K_by_node_type=np.array([300.0, 600.0]),
        temperature_std_K_by_node_type=np.array([10.0, 20.0]),
    )
    with np.load(path, allow_pickle=False) as data:
        _, _, prediction, representation = MODULE.read_temperature_fields(data)
    np.testing.assert_allclose(
        prediction, np.array([[[302.0, 624.0], [312.0, 604.0]]])
    )
    assert representation.startswith("low-rank-corrected")


def test_reads_diffusion_refined_field(tmp_path: Path) -> None:
    path = tmp_path / "diffusion.npz"
    target = np.array([[[[0.0], [1.0]], [[1.0], [0.0]]]])
    np.savez(
        path,
        **common_arrays(),
        target_temperature_normalized=target,
        refined_temperature_normalized=target + 0.1,
        temperature_mean_K_by_node_type=np.array([300.0, 600.0]),
        temperature_std_K_by_node_type=np.array([10.0, 20.0]),
    )
    with np.load(path, allow_pickle=False) as data:
        _, _, prediction, representation = MODULE.read_temperature_fields(data)
    np.testing.assert_allclose(
        prediction, np.array([[[301.0, 622.0], [311.0, 602.0]]])
    )
    assert representation.startswith("diffusion-refined")


def test_particle_section_uses_circular_sphere_intersection() -> None:
    horizontal, vertical = np.meshgrid(
        np.linspace(-1.0, 1.0, 101), np.linspace(-1.0, 1.0, 101)
    )
    centers = np.array([[0.0, 0.0, 0.0], [0.0, 2.0e-3, 0.0]])
    mask, count = MODULE.particle_section_mask(
        horizontal,
        vertical,
        centers,
        radius_m=1.0e-3,
        section_y_m=0.0,
    )
    assert count == 1
    assert mask[50, 50]
    assert mask[50, 100]
    assert not mask[0, 0]


def test_geometry_check_accepts_float32_rounding() -> None:
    arrays = common_arrays()
    arrays["node_volume_m3"] = arrays["node_volume_m3"].astype(np.float32)
    arrays["node_centroid_m"] = arrays["node_centroid_m"].astype(np.float32)
    MODULE.validate_prediction_geometry(
        arrays,
        common_arrays()["node_type"],
        common_arrays()["node_volume_m3"],
        common_arrays()["node_centroid_m"],
    )


def test_geometry_check_rejects_reordered_nodes() -> None:
    arrays = common_arrays()
    arrays["node_type"] = arrays["node_type"][::-1]
    arrays["node_volume_m3"] = arrays["node_volume_m3"][::-1]
    arrays["node_centroid_m"] = arrays["node_centroid_m"][::-1]
    try:
        MODULE.validate_prediction_geometry(
            arrays,
            common_arrays()["node_type"],
            common_arrays()["node_volume_m3"],
            common_arrays()["node_centroid_m"],
        )
    except ValueError as error:
        assert "coordinate misalignment" in str(error)
    else:
        raise AssertionError("reordered prediction nodes must be rejected")


def test_robust_error_limit_retains_extreme_values_for_metrics() -> None:
    error = np.concatenate([np.linspace(-5.0, 5.0, 1000), np.array([400.0])])
    display_limit = MODULE.robust_symmetric_error_limit(error, 99.0)
    assert 4.9 < display_limit < 6.0
    assert np.max(np.abs(error)) == 400.0


def test_robust_error_limit_rejects_ambiguous_percentile() -> None:
    try:
        MODULE.robust_symmetric_error_limit(np.array([1.0, 2.0]), 80.0)
    except ValueError as error:
        assert "between 90 and 100" in str(error)
    else:
        raise AssertionError("an overly aggressive display percentile must be rejected")


def test_error_localization_records_boundary_role_and_volume_fraction() -> None:
    result = MODULE.error_localization_metrics(
        error=np.array([2.0, -400.0, 3.0]),
        volume=np.array([4.0, 1.0, 5.0]),
        coordinates_m=np.array(
            [[0.0, 0.0, 0.0], [1.0e-3, 2.0e-3, 3.0e-3], [0.0, 0.0, 1.0e-3]]
        ),
        boundary_fraction=np.array(
            [[0.0, 0.0], [1.0, 0.25], [0.0, 0.0]]
        ),
        boundary_role_names=np.array(["inlet", "interface"]),
        selected=np.array([True, True, False]),
    )
    maximum = result["maximum_error_node"]
    assert maximum["index"] == 1
    assert maximum["signed_error_K"] == -400.0
    assert maximum["centroid_mm"] == [1.0, 2.0, 3.0]
    assert maximum["regional_volume_fraction"] == 0.2
    assert maximum["boundary_fraction"] == {"inlet": 1.0, "interface": 0.25}


def test_figure_uses_two_columns_and_three_rows() -> None:
    assert MODULE.FIGURE_WIDTH_INCH == 5.40
    assert MODULE.FIGURE_HEIGHT_INCH == 6.70
    assert MODULE.FIGURE_WIDTH_MM == 5.40 * 25.4
    assert MODULE.FIGURE_HEIGHT_MM == 6.70 * 25.4
    assert MODULE.PANEL_ROWS == 3
    assert MODULE.PANEL_COLUMNS == 2
    np.testing.assert_allclose(
        MODULE.FIGURE_SIZE_INCH,
        np.array([5.40, 6.70]),
    )


def test_field_figure_preserves_declared_canvas_size() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "canvas_bounds = figure.bbox_inches" in source
    assert "figure.savefig(pdf, dpi=600, bbox_inches=canvas_bounds)" in source
    assert "figure.savefig(svg, dpi=600, bbox_inches=canvas_bounds)" in source
    assert "figure.savefig(png, dpi=600, bbox_inches=canvas_bounds)" in source


def test_subplots_do_not_restore_titles_above_axes() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert ".set_title(" not in source
    assert "Column headings are shown once" not in source


def test_field_panels_are_checked_for_equal_axis_sizes() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"panel_axis_bounds": panel_axis_bounds' in source
    assert '"panel_width_to_height_ratio": panel_widths[0] / panel_heights[0]' in source
    assert 'raise RuntimeError("field-comparison panels do not have equal widths")' in source
    assert 'raise RuntimeError("field-comparison panels do not have equal heights")' in source


def test_default_packing_is_the_registered_local_openfoam_geometry() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "runs/hccb_dense_snappy_g2_nativezone_r2/geometry/packing_crop.npz" in source
    assert "expected-particle-count" in source
    assert "default=125" in source


def test_validation_marker_exports_model_identity_and_field_metrics() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "if args.selection is not None or args.validation_marker is not None" in source
    assert "a formal validation marker requires --selection" in source
    assert r"\PFieldModelLabel" in source
    assert r"\PFieldFluidRMSE" in source
    assert r"\PFieldSolidRMSE" in source
    assert r"\PFieldFluidMaxError" in source
    assert r"\PFieldSolidMaxError" in source
    assert 'selection_record.get("selection_data_role") != "validation"' in source
    assert 'selection_record.get("display_data_role") != "test"' in source


def test_formal_field_output_requires_validation_selection() -> None:
    args = SimpleNamespace(
        selection=None,
        allow_unselected_diagnostic=False,
        validation_marker=None,
        output_stem=MODULE.FORMAL_OUTPUT_STEM,
    )
    try:
        MODULE.resolve_output_mode(args)
    except ValueError as error:
        assert "requires validation-only --selection" in str(error)
    else:
        raise AssertionError("an unselected prediction must not create a formal field figure")


def test_unselected_field_output_is_forced_to_diagnostic_name() -> None:
    args = SimpleNamespace(
        selection=None,
        allow_unselected_diagnostic=True,
        validation_marker=None,
        output_stem=MODULE.FORMAL_OUTPUT_STEM,
    )
    stem, formal = MODULE.resolve_output_mode(args)
    assert stem == f"{MODULE.FORMAL_OUTPUT_STEM}_unselected_diagnostic"
    assert formal is False


def test_selected_field_output_remains_formal() -> None:
    args = SimpleNamespace(
        selection=Path("selection.json"),
        allow_unselected_diagnostic=False,
        validation_marker=None,
        output_stem=MODULE.FORMAL_OUTPUT_STEM,
    )
    stem, formal = MODULE.resolve_output_mode(args)
    assert stem == MODULE.FORMAL_OUTPUT_STEM
    assert formal is True


def test_selected_file_hash_is_rechecked_before_plotting(tmp_path: Path) -> None:
    selected = tmp_path / "prediction.npz"
    selected.write_bytes(b"selected prediction")
    record = {
        "prediction_file": str(selected),
        "prediction_file_sha256": MODULE.sha256(selected),
    }
    assert (
        MODULE.verify_selected_file(
            record, "prediction_file", "prediction_file_sha256"
        )
        == selected.resolve()
    )
    selected.write_bytes(b"changed after selection")
    try:
        MODULE.verify_selected_file(
            record, "prediction_file", "prediction_file_sha256"
        )
    except ValueError as error:
        assert "changed after selection" in str(error)
    else:
        raise AssertionError("a modified selected prediction must be rejected")
