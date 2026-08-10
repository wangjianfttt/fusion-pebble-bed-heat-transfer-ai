#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${ROOT}"
OPENFOAM_BASHRC=${OPENFOAM_BASHRC:-/opt/openfoam13/etc/bashrc}
if [[ ! -f ${OPENFOAM_BASHRC} ]]; then
  echo "OpenFOAM environment file is missing: ${OPENFOAM_BASHRC}" >&2
  exit 1
fi
# OpenFOAM's bashrc contains harmless probes that may return non-zero.  Do not
# let `set -e` stop the post-processing runner in the middle of that file.
set +u
set +e
source "${OPENFOAM_BASHRC}"
openfoam_source_status=$?
set -e
set -u
if [[ ${openfoam_source_status} -ne 0 ]]; then
  echo "OpenFOAM environment could not be loaded: ${OPENFOAM_BASHRC}" >&2
  exit "${openfoam_source_status}"
fi
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
DATASET_ROOT=${DATASET_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3_dataset}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_60_sourceflow_r3}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${ROOT}/results/hccb_p418_regional_topology_r2/regional_topology.npz}
SUBFACE_GEOMETRY=${SUBFACE_GEOMETRY:-${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}
NATIVE_GRAPH=${NATIVE_GRAPH:-${ROOT}/hccb_dense_cht_native_r2/native_multiregion_graph/native_multiregion_graph.npz}
BOUNDARY_ROLES=${BOUNDARY_ROLES:-${ROOT}/parameters/hccb_dense_cht_boundary_roles.json}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_model_splits.json}
EXPECTED_CASES=${EXPECTED_CASES:-60}
# Sixteen cases retain paired full fields over the final steady-iteration
# window.  Cloud-recovered cases retain the complete iteration-200 field and
# engineering histories, but not fabricated 175--200 partition fields.
MINIMUM_FULL_FIELD_COUNT=${MINIMUM_FULL_FIELD_COUNT:-16}

completed=$(find "${MATRIX_ROOT}" -mindepth 2 -maxdepth 2 -name formal_sample_complete.json | wc -l | tr -d ' ')
if [[ ${completed} -ne ${EXPECTED_CASES} ]]; then
  echo "P418 matrix is incomplete: ${completed}/${EXPECTED_CASES} formal samples" >&2
  exit 1
fi

INPUT_CHECK_DIR=${ROOT}/results/hccb_p418_60_actual_case_input_check
TRAINING_COVERAGE_DIR=${ROOT}/results/hccb_p418_training_data_coverage
STEADY_TAIL_DIR=${ROOT}/results/hccb_p418_60_sourceflow_r3_steady_final_windows
STEADY_HOTSPOT_DIR=${ROOT}/results/hccb_p418_60_sourceflow_r3_steady_hotspots
python3 "${ROOT}/code/update_hccb_p418_matrix_parameter_sources.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --parameter-manifest "${ROOT}/parameters/literature_parameter_manifest.csv" \
  --output "${ROOT}/results/hccb_p418_matrix_parameter_sources/summary.json"
python3 "${ROOT}/code/verify_hccb_p418_actual_case_inputs.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --parameter-source "${ROOT}/parameters/hccb_p418_physical_parameter_sources.csv" \
  --canonical-helium "${ROOT}/results/apd006_hccb_openfoam_helium_property_table/physicalProperties" \
  --output "${INPUT_CHECK_DIR}/summary.json" \
  --markdown-output "${INPUT_CHECK_DIR}/P418_正式算例参数对应_CN.md"
python3 "${ROOT}/code/build_hccb_p418_source_summary.py" \
  --actual-cases "${INPUT_CHECK_DIR}/summary.json" \
  --json-output "${ROOT}/results/hccb_p418_source_summary.json" \
  --markdown-output "${ROOT}/parameters/HCCB_P418_PARAMETER_AND_MODEL_SOURCES_CN.md"
python3 "${ROOT}/code/summarize_hccb_p418_training_data_coverage.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --expected-case-count "${EXPECTED_CASES}" \
  --output-dir "${TRAINING_COVERAGE_DIR}" \
  > "${RESULT_PREFIX}_training_data_coverage.log"
python3 "${ROOT}/code/summarize_hccb_p418_formal_steady_tails.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --expected-case-count "${EXPECTED_CASES}" \
  --minimum-full-field-count "${MINIMUM_FULL_FIELD_COUNT}" \
  --output-dir "${STEADY_TAIL_DIR}" \
  --latex-output "${ROOT}/manuscript/generated_steady_final_windows.tex" \
  > "${RESULT_PREFIX}_steady_final_windows.log"

rm -rf "${DATASET_ROOT}"
python3 "${ROOT}/code/build_hccb_p418_shared_mesh_dataset.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --sample-paths-from-completion-markers \
  --expected-case-count "${EXPECTED_CASES}" \
  --require-completion-markers \
  --require-sourceflow-mapping \
  --require-steady-final-window \
  --output-dir "${DATASET_ROOT}" \
  > "${RESULT_PREFIX}_dataset.log"

DATASET_INDEX=${DATASET_ROOT}/dataset_index.json
TIME_SCALE_DIR=${ROOT}/results/hccb_p418_velocity_step_time_scales
MODEL_GEOMETRY_DIR=${RESULT_PREFIX}_model_geometry
BOUNDARY_HEAT_DIR=${RESULT_PREFIX}_boundary_heat_flux_targets
STATE_DIR=${RESULT_PREFIX}_regional_state_targets
REPRESENTATION_FIDELITY_DIR=${RESULT_PREFIX}_regional_representation_fidelity
NATIVE_RECONSTRUCTION_DIR=${RESULT_PREFIX}_native_reconstruction
MASS_DIR=${RESULT_PREFIX}_regional_mass_flux_targets
ENERGY_DIR=${RESULT_PREFIX}_regional_energy_flux_targets
EXPERIMENT_TARGET_DIR=${RESULT_PREFIX}_experimental_comparison_targets
PRESSURE_DENSITY_DIR=${RESULT_PREFIX}_pressure_density_consistency
PRESSURE_CORRELATION_DIR=${RESULT_PREFIX}_pressure_correlation
TRAINING_STATISTICS=${RESULT_PREFIX}_training_statistics.json

# Recompute flow/thermal time scales from the corrected highest-velocity field.
# This replaces the temporary estimate made from the corrected low-velocity
# preflight and prevents any pre-sourceflow field from entering the manuscript.
python3 "${ROOT}/code/analyze_hccb_p418_velocity_step_time_scales.py" \
  --topology "${DATASET_ROOT}/shared_mesh_topology.npz" \
  --field "${DATASET_ROOT}/fields/u0p25_T300_q4p85.npz" \
  --parameter-source "${ROOT}/parameters/hccb_p418_physical_parameter_sources.csv" \
  --particle-scale-summary "${ROOT}/results/hccb_p418_transient_time_resolution/summary.json" \
  --sourceflow-input-summary "${INPUT_CHECK_DIR}/summary.json" \
  --output-dir "${TIME_SCALE_DIR}" \
  > "${RESULT_PREFIX}_velocity_step_time_scales.log"

python3 "${ROOT}/code/check_hccb_p418_pressure_density_consistency.py" \
  --dataset-index "${DATASET_INDEX}" \
  --output-dir "${PRESSURE_DENSITY_DIR}" \
  > "${RESULT_PREFIX}_pressure_density_consistency.log"

python3 "${ROOT}/code/build_hccb_p418_model_geometry.py" \
  --dataset-index "${DATASET_INDEX}" \
  --regional-topology "${REGIONAL_TOPOLOGY}" \
  --boundary-roles "${BOUNDARY_ROLES}" \
  --output-dir "${MODEL_GEOMETRY_DIR}" \
  > "${RESULT_PREFIX}_model_geometry.log"

python3 "${ROOT}/code/export_hccb_p418_boundary_heat_flux_targets.py" \
  --dataset-index "${DATASET_INDEX}" \
  --case-root "${MATRIX_ROOT}" \
  --time-from-completion-marker \
  --output-dir "${BOUNDARY_HEAT_DIR}" \
  > "${RESULT_PREFIX}_boundary_heat_flux_targets.log"

python3 "${ROOT}/code/summarize_hccb_p418_completed_matrix_physics.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --time-from-completion-marker \
  --output-dir "${RESULT_PREFIX}_completed_physics" \
  > "${RESULT_PREFIX}_completed_physics.log"

python3 "${ROOT}/code/build_hccb_p418_factorial_table.py" \
  --physics-summary "${RESULT_PREFIX}_completed_physics/summary.json" \
  --output "${ROOT}/manuscript/generated_steady_factorial_effects.tex" \
  --summary-output "${RESULT_PREFIX}_completed_physics/factorial_manuscript_table.json" \
  > "${RESULT_PREFIX}_factorial_manuscript_table.log"

python3 "${ROOT}/code/summarize_hccb_p418_steady_hotspots.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --expected-case-count "${EXPECTED_CASES}" \
  --output-dir "${STEADY_HOTSPOT_DIR}" \
  > "${RESULT_PREFIX}_steady_hotspots.log"

python3 "${ROOT}/code/plot_hccb_p418_physical_response.py" \
  --physical-csv "${RESULT_PREFIX}_completed_physics/completed_case_physics.csv" \
  --output-dir "${ROOT}/figures" \
  > "${RESULT_PREFIX}_physical_response_figure.log"

python3 "${ROOT}/code/analyze_hccb_p418_dimensionless_heat_transfer.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --parameter-manifest "${ROOT}/parameters/literature_parameter_manifest.csv" \
  --boundary-heat-summary "${BOUNDARY_HEAT_DIR}/summary.json" \
  --output-dir "${RESULT_PREFIX}_dimensionless_heat_transfer" \
  > "${RESULT_PREFIX}_dimensionless_heat_transfer.log"

python3 "${ROOT}/code/analyze_hccb_p418_pressure_correlation.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --parameter-manifest "${ROOT}/parameters/literature_parameter_manifest.csv" \
  --physical-csv "${RESULT_PREFIX}_completed_physics/completed_case_physics.csv" \
  --output-dir "${PRESSURE_CORRELATION_DIR}" \
  --expected-case-count "${EXPECTED_CASES}" \
  > "${RESULT_PREFIX}_pressure_correlation.log"

python3 "${ROOT}/code/build_hccb_p418_same_source_correlation_text.py" \
  --input-summary "${RESULT_PREFIX}_dimensionless_heat_transfer/summary.json" \
  --pressure-summary "${PRESSURE_CORRELATION_DIR}/summary.json" \
  --output "${ROOT}/manuscript/generated_same_source_correlation.tex" \
  --summary "${RESULT_PREFIX}_dimensionless_heat_transfer/manuscript_text.json" \
  > "${RESULT_PREFIX}_dimensionless_heat_transfer_text.log"

python3 "${ROOT}/code/export_hccb_p418_experimental_comparison_targets.py" \
  --completed-physics-csv "${RESULT_PREFIX}_completed_physics/completed_case_physics.csv" \
  --dimensionless-heat-csv "${RESULT_PREFIX}_dimensionless_heat_transfer/dimensionless_heat_transfer.csv" \
  --observable-matrix "${ROOT}/parameters/hccb_p418_experimental_observable_matrix.csv" \
  --output-dir "${EXPERIMENT_TARGET_DIR}" \
  > "${RESULT_PREFIX}_experimental_comparison_targets.log"

python3 "${ROOT}/code/build_hccb_p418_regional_state_targets.py" \
  --dataset-index "${DATASET_INDEX}" \
  --subface-geometry "${SUBFACE_GEOMETRY}" \
  --output-dir "${STATE_DIR}" \
  > "${RESULT_PREFIX}_regional_state_targets.log"

python3 "${ROOT}/code/quantify_hccb_p418_regional_representation_fidelity.py" \
  --dataset-index "${DATASET_INDEX}" \
  --subface-geometry "${SUBFACE_GEOMETRY}" \
  --regional-state-targets "${STATE_DIR}/regional_state_targets.npz" \
  --parameter-source "${ROOT}/parameters/hccb_p418_physical_parameter_sources.csv" \
  --output-dir "${REPRESENTATION_FIDELITY_DIR}" \
  > "${RESULT_PREFIX}_regional_representation_fidelity.log"

python3 "${ROOT}/code/compare_hccb_p418_native_reconstruction.py" \
  --dataset-index "${DATASET_INDEX}" \
  --subface-geometry "${SUBFACE_GEOMETRY}" \
  --regional-state-targets "${STATE_DIR}/regional_state_targets.npz" \
  --parameter-source "${ROOT}/parameters/hccb_p418_physical_parameter_sources.csv" \
  --output-dir "${NATIVE_RECONSTRUCTION_DIR}" \
  > "${RESULT_PREFIX}_native_reconstruction.log"

python3 "${ROOT}/code/build_hccb_p418_regional_fidelity_text.py" \
  --representation-summary "${REPRESENTATION_FIDELITY_DIR}/summary.json" \
  --reconstruction-summary "${NATIVE_RECONSTRUCTION_DIR}/summary.json" \
  --output "${ROOT}/manuscript/generated_regional_fidelity_formal.tex" \
  --summary "${RESULT_PREFIX}_regional_fidelity_manuscript_text.json" \
  > "${RESULT_PREFIX}_regional_fidelity_manuscript_text.log"

python3 "${ROOT}/code/build_hccb_p418_regional_mass_flux_targets.py" \
  --dataset-index "${DATASET_INDEX}" \
  --regional-topology "${REGIONAL_TOPOLOGY}" \
  --level 5 \
  --output-dir "${MASS_DIR}" \
  > "${RESULT_PREFIX}_regional_mass_flux_targets.log"

python3 "${ROOT}/code/build_hccb_p418_regional_energy_flux_targets.py" \
  --dataset-index "${DATASET_INDEX}" \
  --regional-topology "${REGIONAL_TOPOLOGY}" \
  --native-graph "${NATIVE_GRAPH}" \
  --boundary-heat-targets "${BOUNDARY_HEAT_DIR}/boundary_heat_flux_targets.npz" \
  --level 5 \
  --output-dir "${ENERGY_DIR}" \
  > "${RESULT_PREFIX}_regional_energy_flux_targets.log"

python3 "${ROOT}/code/build_hccb_p418_training_statistics.py" \
  --dataset-index "${DATASET_INDEX}" \
  --split-file "${SPLITS}" \
  --output "${TRAINING_STATISTICS}" \
  > "${RESULT_PREFIX}_training_statistics.log"

python3 - \
  "${DATASET_INDEX}" \
  "${MODEL_GEOMETRY_DIR}/summary.json" \
  "${BOUNDARY_HEAT_DIR}/summary.json" \
  "${STATE_DIR}/summary.json" \
  "${REPRESENTATION_FIDELITY_DIR}/summary.json" \
  "${NATIVE_RECONSTRUCTION_DIR}/summary.json" \
  "${MASS_DIR}/summary.json" \
  "${ENERGY_DIR}/summary.json" \
  "${TRAINING_STATISTICS}" \
  "${EXPERIMENT_TARGET_DIR}/summary.json" \
  "${PRESSURE_DENSITY_DIR}/summary.json" \
  "${TRAINING_COVERAGE_DIR}/summary.json" \
  "${STEADY_TAIL_DIR}/summary.json" \
  "${STEADY_HOTSPOT_DIR}/summary.json" \
  "${EXPECTED_CASES}" \
  "${RESULT_PREFIX}_postprocess_summary.json" <<'PY'
import hashlib
import json
import pathlib
import sys

paths = [pathlib.Path(value).resolve() for value in sys.argv[1:-2]]
expected = int(sys.argv[-2])
output = pathlib.Path(sys.argv[-1]).resolve()
documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
dataset = documents[0]
case_counts = [
    int(document.get("counts", {}).get("cases", expected))
    for document in documents[1:8]
]
checks = {
    "dataset_has_expected_conditions": int(dataset["case_count"]) == expected,
    "all_target_builders_cover_expected_conditions": all(count == expected for count in case_counts),
    "all_summary_statuses_are_ready": all(
        "ready" in str(document.get("status", ""))
        for document in documents[1:]
    ),
}
payload = {
    "status": "p418_60_training_data_ready" if all(checks.values()) else "failed",
    "expected_case_count": expected,
    "checks": checks,
    "case_counts": case_counts,
    "files": [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ],
    "new_physical_parameters": [],
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if payload["status"] != "p418_60_training_data_ready":
    raise SystemExit(1)
PY
