#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results}
STEADY_RESULT_NAMESPACE=${STEADY_RESULT_NAMESPACE:-hccb_p418_60_corrected_20260731}
STEADY_COMPARISON_DIR=${STEADY_COMPARISON_DIR:-${RESULT_ROOT}/${STEADY_RESULT_NAMESPACE}_model_comparison_100epoch}
SELECTED_PLAN=${SELECTED_PLAN:-${RESULT_ROOT}/hccb_p418_thermal_timestep_sensitivity/formal_step_plan.json}
PREFLIGHT_CASE=${PREFLIGHT_CASE:-${ROOT}/hccb_dense_cht_p418_sourceflow_preflight/u0p05_T300_q4p85}
FIRST_FORMAL_CASE=${FIRST_FORMAL_CASE:-${MATRIX_ROOT}/u0p05_T300_q4p85}
FIRST_FORMAL_COMPARISON=${RESULT_ROOT}/hccb_p418_first_formal_consistency/summary.json
NP_PER_CASE=${NP_PER_CASE:-32}
STEP_CONCURRENT_CASES=${STEP_CONCURRENT_CASES:-${CONCURRENT_CASES:-1}}

if ! [[ ${NP_PER_CASE} =~ ^[1-9][0-9]*$ ]] \
  || ! [[ ${STEP_CONCURRENT_CASES} =~ ^[1-9][0-9]*$ ]]; then
  echo "NP_PER_CASE and STEP_CONCURRENT_CASES must be positive integers" >&2
  exit 2
fi

completed=$(find "${MATRIX_ROOT}" -mindepth 2 -maxdepth 2 \
  -name formal_sample_complete.json | wc -l | tr -d ' ')
if [[ ${completed} -ne 60 ]]; then
  echo "post-steady P418 pipeline requires 60 completed cases; found ${completed}" >&2
  exit 1
fi

python3 "${ROOT}/code/compare_hccb_p418_preflight_formal_case.py" \
  --preflight-case "${PREFLIGHT_CASE}" \
  --formal-case "${FIRST_FORMAL_CASE}" \
  --output "${FIRST_FORMAL_COMPARISON}"

# Compare pressure drop, outlet temperature, solid maximum temperature and wall heat
# on the same P418 condition before using the fine-mesh fields for model training.
bash "${ROOT}/code/run_hccb_p418_mesh_sensitivity.sh" \
  > "${RESULT_ROOT}/hccb_p418_three_mesh_cht_sensitivity.log" 2>&1

# The steady data must exist before either transient exporter is allowed to run.
bash "${ROOT}/code/run_hccb_p418_60_postprocess.sh"

# Compare the completed OpenFOAM fields with any filled experimental records.
# Empty templates remain empty and produce no artificial comparison values.
bash "${ROOT}/code/run_hccb_p418_experimental_comparison.sh"

# The steady model comparison can use the GPU while the time-step calculations
# use CPU cores.  The older solver-iteration observable Transformer is not part
# of the paper's physical thermal-step question and is not run here.
DEVICE=cuda GRAPH_MICROBATCH_SIZE=1 TRANSOLVER_MICROBATCH_SIZE=1 \
SPLIT_NAMES="interleaved_all_ranges temperature_extrapolation velocity_extrapolation heat_source_interpolation heat_source_extrapolation" \
EPOCHS=100 THREADS=4 RESULT_NAMESPACE="${STEADY_RESULT_NAMESPACE}" \
COMPARISON_OUTPUT_DIR="${STEADY_COMPARISON_DIR}" \
  bash "${ROOT}/code/run_hccb_p418_60_model_comparison.sh" \
  > "${RESULT_ROOT}/hccb_p418_60_models_after_steady.log" 2>&1 &
steady_model_pid=$!

# The 12 physical histories wait for the three predeclared staged-step
# calculations.  Every stage is refined by the same factor of two.
NP_PER_CASE="${NP_PER_CASE}" bash "${ROOT}/code/run_hccb_p418_thermal_timestep_sensitivity.sh" \
  > "${RESULT_ROOT}/hccb_p418_thermal_timestep_sensitivity.log" 2>&1

python3 "${ROOT}/code/build_hccb_p418_selected_timestep_plan.py" \
  --base-plan "${ROOT}/parameters/hccb_p418_transient_step_plan.json" \
  --sensitivity-summary "${RESULT_ROOT}/hccb_p418_thermal_timestep_sensitivity/thermal_timestep_sensitivity.json" \
  --output "${SELECTED_PLAN}"

python3 "${ROOT}/code/build_hccb_p418_timestep_table.py" \
  --summary "${RESULT_ROOT}/hccb_p418_thermal_timestep_sensitivity/thermal_timestep_sensitivity.json" \
  --output "${ROOT}/manuscript/generated_timestep_sensitivity.tex"

NP_PER_CASE="${NP_PER_CASE}" CONCURRENT_CASES="${STEP_CONCURRENT_CASES}" \
STEP_SPLIT_NAMES="direction_down_test direction_up_test pair_disjoint_stress_test" \
PLAN="${SELECTED_PLAN}" RUN_MODEL_TRAINING=0 \
  bash "${ROOT}/code/run_hccb_p418_step_responses.sh" \
  > "${RESULT_ROOT}/hccb_p418_step_responses_openfoam.log" 2>&1

wait "${steady_model_pid}"

# Repeat only the main steady split with three independent neural-network
# initializations. The deterministic response surface is not repeated.
DEVICE=cuda GRAPH_MICROBATCH_SIZE=1 TRANSOLVER_MICROBATCH_SIZE=1 THREADS=4 \
RESULT_NAMESPACE="${STEADY_RESULT_NAMESPACE}" \
  bash "${ROOT}/code/run_hccb_p418_steady_seed_robustness.sh" \
  > "${RESULT_ROOT}/hccb_p418_steady_seed_robustness.log" 2>&1

# If the common 100-epoch comparison still improves at its right boundary,
# rerun only those models from scratch with the epoch count archived from the
# corresponding source. Both runs are retained and compared by physical output.
DEVICE=cuda GRAPH_MICROBATCH_SIZE=1 TRANSOLVER_MICROBATCH_SIZE=1 THREADS=4 \
RESULT_NAMESPACE="${STEADY_RESULT_NAMESPACE}" \
BASE_COMPARISON_DIR="${STEADY_COMPARISON_DIR}" \
  bash "${ROOT}/code/run_hccb_p418_steady_epoch_followup.sh" \
  > "${RESULT_ROOT}/hccb_p418_steady_epoch_followup.log" 2>&1

# For each architecture, choose between the common 100-epoch result and any
# source-length rerun using seed101 validation conditions only.  The resulting
# checkpoint map is fixed before either independent packing is evaluated.
CROSS_PACKING_MODEL_SOURCES=${RESULT_ROOT}/hccb_p418_cross_packing_seed101_model_sources.json
python3 "${ROOT}/code/build_hccb_p418_cross_packing_model_sources.py" \
  --project-root "${ROOT}" \
  --result-namespace "${STEADY_RESULT_NAMESPACE}" \
  --initial-epochs 100 \
  --split-name interleaved_all_ranges \
  --followup-plan "${RESULT_ROOT}/${STEADY_RESULT_NAMESPACE}_source_epoch_followup/epoch_followup_plan.json" \
  --output "${CROSS_PACKING_MODEL_SOURCES}"

# Select the steady PINN by validation loss only. The independent steady test
# conditions are not used to choose the initial-field model for the chain.
STEADY_CHAIN_SELECTION=${RESULT_ROOT}/hccb_p418_steady_chain_source.json
python3 "${ROOT}/code/select_hccb_p418_steady_chain_source.py" \
  --base-summary "${RESULT_ROOT}/${STEADY_RESULT_NAMESPACE}_pinn_interleaved_all_ranges_100epoch/summary.json" \
  --followup-plan "${RESULT_ROOT}/${STEADY_RESULT_NAMESPACE}_source_epoch_followup/epoch_followup_plan.json" \
  --project-root "${ROOT}" \
  --architecture pinn \
  --split-name interleaved_all_ranges \
  --output "${STEADY_CHAIN_SELECTION}"
STEADY_CHAIN_SUMMARY=$(python3 - "${STEADY_CHAIN_SELECTION}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_summary"])
PY
)

# Repeat the selected coordinate-PINN schedule with three loss ratios taken
# directly from archived official PINO configurations.  All three runs use the
# same seed101 split, initialization, optimizer and epoch count.
DEVICE=cuda THREADS=4 SELECTION="${STEADY_CHAIN_SELECTION}" \
  bash "${ROOT}/code/run_hccb_p418_steady_loss_weight_sensitivity.sh" \
  > "${RESULT_ROOT}/hccb_p418_steady_loss_weight_sensitivity.log" 2>&1
LOSS_WEIGHT_SUMMARY=${RESULT_ROOT}/hccb_p418_steady_loss_weight_sensitivity/summary.json
if [[ ! -f ${LOSS_WEIGHT_SUMMARY} ]]; then
  echo "loss-weight sensitivity summary is missing" >&2
  exit 1
fi

# The remote machine has one GPU.  The first call above completes and exports
# the 12 OpenFOAM histories without starting a neural model.  Only after every
# steady-model GPU run has finished do we revisit the already completed step
# directories and train the transient Transformer and diffusion models.  The
# step runner skips completed OpenFOAM cases, so this call does not recompute
# any physical trajectory.
NP_PER_CASE="${NP_PER_CASE}" CONCURRENT_CASES="${STEP_CONCURRENT_CASES}" \
STEP_SPLIT_NAMES="direction_down_test direction_up_test pair_disjoint_stress_test" \
PLAN="${SELECTED_PLAN}" RUN_MODEL_TRAINING=1 \
  bash "${ROOT}/code/run_hccb_p418_step_responses.sh" \
  > "${RESULT_ROOT}/hccb_p418_step_response_model_training.log" 2>&1

# Evaluate the deployable chain: steady physics PINN initial field followed by
# the transient energy-constrained graph--Transformer on held-out steps.
DEVICE=cuda DIFFUSION_DEVICE=cuda ENERGY_DEVICE=cpu STEADY_SUMMARY="${STEADY_CHAIN_SUMMARY}" \
  bash "${ROOT}/code/run_hccb_p418_chained_initial_state_evaluation.sh" \
  > "${RESULT_ROOT}/hccb_p418_chained_initial_state.log" 2>&1
FUSED_CHAIN_TABLE=${ROOT}/manuscript/generated_fused_chain_results.tex
FUSED_CHAIN_TABLE_SUMMARY=${RESULT_ROOT}/hccb_p418_physical_steps_12/chained_initial_state/manuscript_table_summary.json
TRANSIENT_COST_TABLE=${ROOT}/manuscript/generated_transient_cost.tex
TRANSIENT_COST_SUMMARY=${RESULT_ROOT}/hccb_p418_physical_steps_12/model_comparison/transient_cost_table.json
TRANSIENT_PERFORMANCE_TABLE=${ROOT}/manuscript/generated_transient_performance.tex
TRANSIENT_PERFORMANCE_SUMMARY=${RESULT_ROOT}/hccb_p418_physical_steps_12/model_comparison/transient_performance_table.json
TRANSIENT_RESULT_TEXT=${ROOT}/manuscript/generated_transient_result_text.tex
TRANSITION_COVERAGE_SUMMARY=${RESULT_ROOT}/hccb_p418_physical_steps_12/transition_temperature_coverage/summary.json
TRANSITION_COVERAGE_TEXT=${ROOT}/manuscript/generated_transition_temperature_coverage.tex
for required in "${FUSED_CHAIN_TABLE}" "${FUSED_CHAIN_TABLE_SUMMARY}" \
  "${TRANSIENT_PERFORMANCE_TABLE}" "${TRANSIENT_PERFORMANCE_SUMMARY}" \
  "${TRANSIENT_COST_TABLE}" "${TRANSIENT_COST_SUMMARY}" "${TRANSIENT_RESULT_TEXT}" \
  "${TRANSITION_COVERAGE_SUMMARY}" "${TRANSITION_COVERAGE_TEXT}"; do
  if [[ ! -f ${required} ]]; then
    echo "completed fused-chain manuscript result is missing: ${required}" >&2
    exit 1
  fi
done

# Quantify how many expensive OpenFOAM conditions each steady model needs.
# Validation and test cases remain fixed; only the training-condition count changes.
DEVICE=cuda GRAPH_MICROBATCH_SIZE=1 TRANSOLVER_MICROBATCH_SIZE=1 THREADS=4 \
  bash "${ROOT}/code/run_hccb_p418_steady_learning_curve.sh" \
  > "${RESULT_ROOT}/hccb_p418_steady_learning_curve.log" 2>&1

# Repeat the selected transient architecture with three and six complete
# OpenFOAM trajectories. Validation and independent-prediction curves remain
# fixed; saved times from one trajectory are never counted as separate cases.
ROOT="${ROOT}" EXECUTE=1 \
DATASET_INDEX="${RESULT_ROOT}/hccb_p418_physical_steps_12/regional_sequences/dataset_index.json" \
RESIDUAL_GEOMETRY="${SUBFACE_GEOMETRY:-${RESULT_ROOT}/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}" \
  bash "${ROOT}/code/run_hccb_p418_transient_learning_curve.sh" \
  > "${RESULT_ROOT}/hccb_p418_transient_learning_curve.log" 2>&1

# Keep the principal physical and model results together with the completion
# record. A resumed run may reuse them only if every file still exists with the
# same content; this prevents an older figure or partial table entering the PDF.
formal_result_specs=(
  "preflight_formal_consistency=${FIRST_FORMAL_COMPARISON}"
  "physical_and_model_source_summary=${RESULT_ROOT}/hccb_p418_source_summary.json"
  "physical_and_model_source_text=${ROOT}/parameters/HCCB_P418_PARAMETER_AND_MODEL_SOURCES_CN.md"
  "completed_physics_csv=${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_completed_physics/completed_case_physics.csv"
  "steady_factorial_effects=${ROOT}/manuscript/generated_steady_factorial_effects.tex"
  "steady_factorial_effects_summary=${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_completed_physics/factorial_manuscript_table.json"
  "steady_hotspot_summary=${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_steady_hotspots/summary.json"
  "steady_hotspot_csv=${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_steady_hotspots/steady_hotspots.csv"
  "steady_hotspot_movements_csv=${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_steady_hotspots/steady_hotspot_movements.csv"
  "steady_final_window_summary=${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_steady_final_windows/summary.json"
  "steady_final_window_text=${ROOT}/manuscript/generated_steady_final_windows.tex"
  "mesh_sensitivity_summary=${RESULT_ROOT}/hccb_p418_three_mesh_cht_sensitivity/summary.json"
  "mesh_sensitivity_gci=${RESULT_ROOT}/hccb_p418_three_mesh_cht_sensitivity/mesh_gci.csv"
  "mesh_sensitivity_table=${ROOT}/manuscript/generated_mesh_sensitivity.tex"
  "dimensionless_heat_summary=${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_dimensionless_heat_transfer/summary.json"
  "pressure_correlation_summary=${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_pressure_correlation/summary.json"
  "same_source_correlation_text=${ROOT}/manuscript/generated_same_source_correlation.tex"
  "physical_response_figure=${ROOT}/figures/hccb_p418_physical_response.pdf"
  "regional_fidelity_text=${ROOT}/manuscript/generated_regional_fidelity_formal.tex"
  "steady_model_comparison_csv=${STEADY_COMPARISON_DIR}/model_comparison.csv"
  "steady_model_comparison_figure=${ROOT}/figures/hccb_p418_steady_model_comparison.pdf"
  "steady_performance_table=${ROOT}/manuscript/generated_steady_performance.tex"
  "thermal_regime_split_coverage=${STEADY_COMPARISON_DIR}/thermal_regime_split_coverage.json"
  "steady_result_text=${ROOT}/manuscript/generated_steady_result_text.tex"
  "native_cell_model_comparison=${STEADY_COMPARISON_DIR}/native_cell_model_comparison.json"
  "native_cell_performance_table=${ROOT}/manuscript/generated_native_cell_performance.tex"
  "steady_seed_robustness_table=${ROOT}/manuscript/generated_steady_seed_robustness.tex"
  "steady_seed_robustness_summary=${RESULT_ROOT}/${STEADY_RESULT_NAMESPACE}_steady_seed_robustness_100epoch/summary.json"
  "transient_model_metrics=${RESULT_ROOT}/hccb_p418_physical_steps_12/model_comparison/physical_step_model_metrics.csv"
  "transient_model_comparison_summary=${RESULT_ROOT}/hccb_p418_physical_steps_12/model_comparison/summary.json"
  "transient_seed_robustness_summary=${RESULT_ROOT}/hccb_p418_physical_steps_12/seed_robustness_pair_disjoint_stress_test/summary.json"
  "transient_performance_table=${TRANSIENT_PERFORMANCE_TABLE}"
  "transient_performance_summary=${TRANSIENT_PERFORMANCE_SUMMARY}"
  "transient_model_figure=${ROOT}/figures/hccb_p418_transient_model_comparison.pdf"
  "transient_result_text=${TRANSIENT_RESULT_TEXT}"
  "transition_temperature_coverage=${TRANSITION_COVERAGE_SUMMARY}"
  "transition_temperature_coverage_text=${TRANSITION_COVERAGE_TEXT}"
  "steady_learning_curve=${RESULT_ROOT}/hccb_p418_learning_curve_model_comparison_100epoch/learning_curve_summary.json"
  "transient_learning_curve=${RESULT_ROOT}/hccb_p418_transient_learning_curve/summary.json"
  "transient_learning_curve_csv=${RESULT_ROOT}/hccb_p418_transient_learning_curve/transient_learning_curve.csv"
  "transient_learning_curve_table=${ROOT}/manuscript/generated_transient_learning_curve.tex"
)
for spec in "${formal_result_specs[@]}"; do
  path=${spec#*=}
  if [[ ! -f ${path} ]]; then
    echo "principal formal result is missing: ${path}" >&2
    exit 1
  fi
done

TIMESTEP_SUMMARY=${RESULT_ROOT}/hccb_p418_thermal_timestep_sensitivity/thermal_timestep_sensitivity.json
TIMESTEP_GCI=${RESULT_ROOT}/hccb_p418_thermal_timestep_sensitivity/thermal_timestep_gci.csv
TIMESTEP_TABLE=${ROOT}/manuscript/generated_timestep_sensitivity.tex
for required in "${SELECTED_PLAN}" "${TIMESTEP_SUMMARY}" "${TIMESTEP_GCI}" "${TIMESTEP_TABLE}"; do
  if [[ ! -f ${required} ]]; then
    echo "completed time-step result is missing: ${required}" >&2
    exit 1
  fi
done

python3 - "${RESULT_ROOT}/hccb_p418_poststeady_pipeline_complete.json" \
  "${SELECTED_PLAN}" "${TIMESTEP_SUMMARY}" "${TIMESTEP_GCI}" "${TIMESTEP_TABLE}" \
  "${CROSS_PACKING_MODEL_SOURCES}" "${LOSS_WEIGHT_SUMMARY}" \
  "${FUSED_CHAIN_TABLE}" "${FUSED_CHAIN_TABLE_SUMMARY}" \
  "${TRANSIENT_COST_TABLE}" "${TRANSIENT_COST_SUMMARY}" \
  "${formal_result_specs[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
selected_plan = pathlib.Path(sys.argv[2])
timestep_summary = pathlib.Path(sys.argv[3])
timestep_gci = pathlib.Path(sys.argv[4])
timestep_table = pathlib.Path(sys.argv[5])
model_sources = pathlib.Path(sys.argv[6])
loss_weight_summary = pathlib.Path(sys.argv[7])
fused_chain_table = pathlib.Path(sys.argv[8])
fused_chain_table_summary = pathlib.Path(sys.argv[9])
transient_cost_table = pathlib.Path(sys.argv[10])
transient_cost_summary = pathlib.Path(sys.argv[11])
formal_result_specs = sys.argv[12:]
formal_results = []
for spec in formal_result_specs:
    label, raw_path = spec.split("=", 1)
    path = pathlib.Path(raw_path)
    if not path.is_file():
        raise SystemExit(f"principal formal result is missing: {path}")
    formal_results.append(
        {
            "label": label,
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
plan = json.loads(selected_plan.read_text(encoding="utf-8"))
output.write_text(
    json.dumps(
        {
            "status": "completed_p418_poststeady_heat_transfer_pipeline",
            "selected_delta_t_s": plan["numerical_time_design"]["delta_t_s"],
            "selected_time_step_schedule": plan["numerical_time_design"]["time_step_schedule"],
            "selected_field_write_schedule": plan["numerical_time_design"]["field_write_schedule"],
            "selected_timestep_plan": str(selected_plan.resolve()),
            "selected_timestep_plan_sha256": hashlib.sha256(selected_plan.read_bytes()).hexdigest(),
            "thermal_timestep_sensitivity_summary": str(timestep_summary.resolve()),
            "thermal_timestep_sensitivity_summary_sha256": hashlib.sha256(timestep_summary.read_bytes()).hexdigest(),
            "thermal_timestep_gci": str(timestep_gci.resolve()),
            "thermal_timestep_gci_sha256": hashlib.sha256(timestep_gci.read_bytes()).hexdigest(),
            "thermal_timestep_manuscript_table": str(timestep_table.resolve()),
            "thermal_timestep_manuscript_table_sha256": hashlib.sha256(timestep_table.read_bytes()).hexdigest(),
            "cross_packing_seed101_model_sources": str(model_sources.resolve()),
            "cross_packing_seed101_model_sources_sha256": hashlib.sha256(model_sources.read_bytes()).hexdigest(),
            "steady_loss_weight_sensitivity": str(loss_weight_summary.resolve()),
            "steady_loss_weight_sensitivity_sha256": hashlib.sha256(loss_weight_summary.read_bytes()).hexdigest(),
            "fused_chain_manuscript_table": str(fused_chain_table.resolve()),
            "fused_chain_manuscript_table_sha256": hashlib.sha256(fused_chain_table.read_bytes()).hexdigest(),
            "fused_chain_manuscript_table_summary": str(fused_chain_table_summary.resolve()),
            "fused_chain_manuscript_table_summary_sha256": hashlib.sha256(fused_chain_table_summary.read_bytes()).hexdigest(),
            "transient_cost_manuscript_table": str(transient_cost_table.resolve()),
            "transient_cost_manuscript_table_sha256": hashlib.sha256(transient_cost_table.read_bytes()).hexdigest(),
            "transient_cost_summary": str(transient_cost_summary.resolve()),
            "transient_cost_summary_sha256": hashlib.sha256(transient_cost_summary.read_bytes()).hexdigest(),
            "formal_result_files": formal_results,
            "new_physical_parameters": [],
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
