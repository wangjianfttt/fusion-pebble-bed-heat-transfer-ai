#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_60_sourceflow_r3}
RESULT_NAMESPACE=${RESULT_NAMESPACE:-hccb_p418_60}
COMPARISON_OUTPUT_DIR=${COMPARISON_OUTPUT_DIR:-${ROOT}/results/${RESULT_NAMESPACE}_model_comparison_${EPOCHS:-100}epoch}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${ROOT}/results/hccb_p418_regional_topology_r2/regional_topology.npz}
DATASET_INDEX=${DATASET_INDEX:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3_dataset/dataset_index.json}
SUBFACE_GEOMETRY=${SUBFACE_GEOMETRY:-${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}
MODEL_GEOMETRY=${MODEL_GEOMETRY:-${RESULT_PREFIX}_model_geometry/model_geometry.npz}
STATE_TARGETS=${STATE_TARGETS:-${RESULT_PREFIX}_regional_state_targets/regional_state_targets.npz}
MASS_TARGETS=${MASS_TARGETS:-${RESULT_PREFIX}_regional_mass_flux_targets/regional_mass_flux_targets.npz}
ENERGY_TARGETS=${ENERGY_TARGETS:-${RESULT_PREFIX}_regional_energy_flux_targets/regional_energy_flux_targets.npz}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_model_splits.json}
TRAINING_STATISTICS=${TRAINING_STATISTICS:-${RESULT_PREFIX}_training_statistics.json}
POSTPROCESS_SUMMARY=${POSTPROCESS_SUMMARY:-${RESULT_PREFIX}_postprocess_summary.json}
EPOCHS=${EPOCHS:-100}
EXPECTED_CASES=${EXPECTED_CASES:-60}
THREADS=${THREADS:-16}
DEVICE=${DEVICE:-cpu}
MODEL_SEED=${MODEL_SEED:-20260717}
STATE_DATA_WEIGHT=${STATE_DATA_WEIGHT:-5.0}
FACE_FLUX_WEIGHT=${FACE_FLUX_WEIGHT:-1.0}
PHYSICS_BALANCE_WEIGHT=${PHYSICS_BALANCE_WEIGHT:-1.0}
GRAPH_MICROBATCH_SIZE=${GRAPH_MICROBATCH_SIZE:-1}
TRANSOLVER_MICROBATCH_SIZE=${TRANSOLVER_MICROBATCH_SIZE:-1}
ARCHITECTURES=${ARCHITECTURES:-pinn_data_only pinn graph transolver}
SPLIT_NAMES=${SPLIT_NAMES:-interleaved_all_ranges temperature_extrapolation velocity_extrapolation heat_source_interpolation heat_source_extrapolation}
FORMAL_PAPER_OUTPUTS=${FORMAL_PAPER_OUTPUTS:-auto}
PARALLEL_RESULT_POLL_SECONDS=${PARALLEL_RESULT_POLL_SECONDS:-30}

paper_outputs=0
if [[ ${FORMAL_PAPER_OUTPUTS} == 1 ]]; then
  paper_outputs=1
elif [[ ${FORMAL_PAPER_OUTPUTS} == auto \
    && ${ARCHITECTURES} == "pinn_data_only pinn graph transolver" \
    && ${SPLIT_NAMES} == "interleaved_all_ranges temperature_extrapolation velocity_extrapolation heat_source_interpolation heat_source_extrapolation" ]]; then
  paper_outputs=1
elif [[ ${FORMAL_PAPER_OUTPUTS} != 0 && ${FORMAL_PAPER_OUTPUTS} != auto ]]; then
  echo "FORMAL_PAPER_OUTPUTS must be auto, 0 or 1" >&2
  exit 1
fi

result_is_current() {
  local architecture=$1
  local split_name=$2
  local output=$3
  seed_args=()
  if [[ ${architecture} != response_surface ]]; then
    seed_args=(--training-seed "${MODEL_SEED}")
  fi
  python3 "${ROOT}/code/check_hccb_p418_steady_result_current.py" \
    --summary "${output}/summary.json" \
    --architecture "${architecture}" \
    --epochs "${EPOCHS}" \
    --split-name "${split_name}" \
    --state-targets "${STATE_TARGETS}" \
    --mass-targets "${MASS_TARGETS}" \
    --energy-targets "${ENERGY_TARGETS}" \
    --split-file "${SPLITS}" \
    --training-statistics "${TRAINING_STATISTICS}" \
    "${seed_args[@]}"
}

active_training_pids_for_output() {
  local output=$1
  python3 - "${output}" <<'PY'
import os
import pathlib
import sys

expected = os.path.abspath(sys.argv[1])
matches = []
for proc_dir in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        state = proc_dir.joinpath("stat").read_text().split()[2]
        args = [
            os.fsdecode(arg)
            for arg in proc_dir.joinpath("cmdline").read_bytes().split(b"\0")
            if arg
        ]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if state == "Z" or not any("train_hccb_p418" in arg for arg in args):
        continue
    outputs = []
    for index, arg in enumerate(args):
        if arg == "--output-dir" and index + 1 < len(args):
            outputs.append(args[index + 1])
        elif arg.startswith("--output-dir="):
            outputs.append(arg.split("=", 1)[1])
    if any(os.path.abspath(value) == expected for value in outputs):
        matches.append(int(proc_dir.name))
print(" ".join(str(pid) for pid in sorted(matches)))
PY
}

wait_for_parallel_training() {
  local output=$1
  local pid_file=${output}/parallel_chain_pid.txt
  local marker_pid=
  if [[ -f ${pid_file} ]]; then
    read -r marker_pid < "${pid_file}" || marker_pid=
  fi

  local active_pids
  active_pids=$(active_training_pids_for_output "${output}")
  if [[ -z ${active_pids} ]]; then
    if [[ -n ${marker_pid} ]]; then
      echo "ignore inactive parallel pid ${marker_pid} for ${output}"
    fi
    return 0
  fi

  echo "wait for active training pid(s) ${active_pids}: ${output}"
  while [[ -n ${active_pids} ]]; do
    sleep "${PARALLEL_RESULT_POLL_SECONDS}"
    active_pids=$(active_training_pids_for_output "${output}")
  done
}

prepare_output() {
  local architecture=$1
  local split_name=$2
  local output=$3
  wait_for_parallel_training "${output}"
  if [[ -f ${output}/summary.json ]]; then
    if result_is_current "${architecture}" "${split_name}" "${output}"; then
      echo "reuse current ${architecture} ${split_name}"
      return 1
    fi
    local archived="${output}.older.$(date +%Y%m%dT%H%M%S).$$"
    mv "${output}" "${archived}"
    mkdir -p "${output}"
    echo "kept completed result from different data, settings or code at ${archived}"
    return 0
  fi
  if [[ ${architecture} != response_surface && -f ${output}/training_checkpoint.pt ]]; then
    echo "resume interrupted ${architecture} ${split_name}"
    return 0
  fi
  if [[ -e ${output} ]]; then
    local archived="${output}.older.$(date +%Y%m%dT%H%M%S).$$"
    mv "${output}" "${archived}"
    echo "kept older result at ${archived}"
  fi
  mkdir -p "${output}"
  return 0
}

python3 - "${POSTPROCESS_SUMMARY}" <<'PY'
import json
import pathlib
import sys

summary = pathlib.Path(sys.argv[1])
if not summary.is_file():
    raise SystemExit(f"missing completed P418 data summary: {summary}")
payload = json.loads(summary.read_text(encoding="utf-8"))
if payload.get("status") != "p418_60_training_data_ready":
    raise SystemExit("P418 60-condition training data are not ready")
PY

python3 "${ROOT}/code/validate_hccb_p418_steady_comparison_inputs.py" \
  --state-targets "${STATE_TARGETS}" \
  --mass-targets "${MASS_TARGETS}" \
  --energy-targets "${ENERGY_TARGETS}" \
  --split-file "${SPLITS}" \
  --training-statistics "${TRAINING_STATISTICS}" \
  --expected-cases "${EXPECTED_CASES}" \
  --output "${COMPARISON_OUTPUT_DIR}/common_input_check.json"

for split_name in ${SPLIT_NAMES}; do
  python3 "${ROOT}/code/summarize_hccb_p418_fv_loss_scales.py" \
    --state-targets "${STATE_TARGETS}" \
    --mass-targets "${MASS_TARGETS}" \
    --energy-targets "${ENERGY_TARGETS}" \
    --split-file "${SPLITS}" \
    --training-statistics "${TRAINING_STATISTICS}" \
    --split-name "${split_name}" \
    --output "${COMPARISON_OUTPUT_DIR}/fv_loss_scales_${split_name}.json" \
    --chinese-summary "${COMPARISON_OUTPUT_DIR}/P418_有限体积损失量级_${split_name}_CN.md" \
    > "${COMPARISON_OUTPUT_DIR}/fv_loss_scales_${split_name}.log"
done

if [[ ${paper_outputs} -eq 1 ]]; then
  python3 "${ROOT}/code/summarize_hccb_p418_thermal_regime_split_coverage.py" \
    --physical-csv "${RESULT_PREFIX}_completed_physics/completed_case_physics.csv" \
    --split-file "${SPLITS}" \
    --output-dir "${COMPARISON_OUTPUT_DIR}"
fi

for split_name in ${SPLIT_NAMES}; do
  baseline_output=${ROOT}/results/${RESULT_NAMESPACE}_response_surface_${split_name}_${EPOCHS}epoch
  if prepare_output response_surface "${split_name}" "${baseline_output}"; then
    python3 "${ROOT}/code/train_hccb_p418_regional_response_surface.py" \
      --state-targets "${STATE_TARGETS}" \
      --mass-targets "${MASS_TARGETS}" \
      --energy-targets "${ENERGY_TARGETS}" \
      --split-file "${SPLITS}" \
      --training-statistics "${TRAINING_STATISTICS}" \
      --split-name "${split_name}" \
      --comparison-epochs "${EPOCHS}" \
      --output-dir "${baseline_output}" \
      > "${baseline_output}/run.log" 2>&1
  fi
  for architecture in ${ARCHITECTURES}; do
    output=${ROOT}/results/${RESULT_NAMESPACE}_${architecture}_${split_name}_${EPOCHS}epoch
    if ! prepare_output "${architecture}" "${split_name}" "${output}"; then
      continue
    fi
    microbatch_args=()
    if [[ ${architecture} == graph && ${DEVICE} == cuda* ]]; then
      microbatch_args=(--microbatch-size "${GRAPH_MICROBATCH_SIZE}")
    elif [[ ${architecture} == transolver && ${DEVICE} == cuda* ]]; then
      microbatch_args=(--microbatch-size "${TRANSOLVER_MICROBATCH_SIZE}")
    fi
    python3 "${ROOT}/code/train_hccb_p418_conservative_mixed_operator.py" \
      --regional-topology "${REGIONAL_TOPOLOGY}" \
      --model-geometry "${MODEL_GEOMETRY}" \
      --state-targets "${STATE_TARGETS}" \
      --mass-targets "${MASS_TARGETS}" \
      --energy-targets "${ENERGY_TARGETS}" \
      --split-file "${SPLITS}" \
      --training-statistics "${TRAINING_STATISTICS}" \
      --split-name "${split_name}" \
      --regional-level 5 \
      --architecture "${architecture}" \
      --epochs "${EPOCHS}" \
      --resume \
      --threads "${THREADS}" \
      --device "${DEVICE}" \
      --seed "${MODEL_SEED}" \
      --state-data-weight "${STATE_DATA_WEIGHT}" \
      --face-flux-weight "${FACE_FLUX_WEIGHT}" \
      --physics-balance-weight "${PHYSICS_BALANCE_WEIGHT}" \
      "${microbatch_args[@]}" \
      --output-dir "${output}" \
      > "${output}/run.log" 2>&1
    echo "completed ${architecture} ${split_name}"
  done
done

python3 "${ROOT}/code/summarize_hccb_p418_60_model_comparison.py" \
  --results-root "${ROOT}/results" \
  --result-prefix "${RESULT_NAMESPACE}" \
  --epochs "${EPOCHS}" \
  --architectures response_surface ${ARCHITECTURES} \
  --splits ${SPLIT_NAMES} \
  --split-file "${SPLITS}" \
  --output-dir "${COMPARISON_OUTPUT_DIR}"

python3 "${ROOT}/code/assess_hccb_p418_training_convergence.py" \
  --results-root "${ROOT}/results" \
  --result-prefix "${RESULT_NAMESPACE}" \
  --epochs "${EPOCHS}" \
  --architectures ${ARCHITECTURES} \
  --splits ${SPLIT_NAMES} \
  --architecture-registry "${ROOT}/parameters/hccb_p418_ai_architecture_sources.json" \
  --output-dir "${COMPARISON_OUTPUT_DIR}"

python3 "${ROOT}/code/summarize_hccb_p418_engineering_metric_leaders.py" \
  --comparison-csv "${COMPARISON_OUTPUT_DIR}/model_comparison.csv" \
  --output-dir "${COMPARISON_OUTPUT_DIR}"

python3 "${ROOT}/code/plot_hccb_p418_steady_engineering_comparison.py" \
  --comparison-csv "${COMPARISON_OUTPUT_DIR}/model_comparison.csv" \
  --output-dir "${COMPARISON_OUTPUT_DIR}"

if [[ ${paper_outputs} -eq 1 ]]; then
  native_result_args=()
  for architecture in response_surface ${ARCHITECTURES}; do
    model_output=${ROOT}/results/${RESULT_NAMESPACE}_${architecture}_interleaved_all_ranges_${EPOCHS}epoch
    native_output=${COMPARISON_OUTPUT_DIR}/native_cell_${architecture}_interleaved_all_ranges
    python3 "${ROOT}/code/evaluate_hccb_p418_native_cell_prediction.py" \
      --dataset-index "${DATASET_INDEX}" \
      --subface-geometry "${SUBFACE_GEOMETRY}" \
      --regional-state-targets "${STATE_TARGETS}" \
      --regional-predictions "${model_output}/test_regional_predictions.npz" \
      --training-statistics "${TRAINING_STATISTICS}" \
      --split-name interleaved_all_ranges \
      --output-dir "${native_output}"
    native_result_args+=(--result "${architecture}=${native_output}/summary.json")
  done

  python3 "${ROOT}/code/summarize_hccb_p418_native_cell_predictions.py" \
    "${native_result_args[@]}" \
    --output-dir "${COMPARISON_OUTPUT_DIR}"

  python3 "${ROOT}/code/build_hccb_p418_native_cell_model_table.py" \
    --comparison-summary "${COMPARISON_OUTPUT_DIR}/native_cell_model_comparison.json" \
    --output "${ROOT}/manuscript/generated_native_cell_performance.tex" \
    --summary "${COMPARISON_OUTPUT_DIR}/native_cell_model_table.json"

  python3 "${ROOT}/code/plot_hccb_p418_steady_model_comparison.py" \
    --comparison-csv "${COMPARISON_OUTPUT_DIR}/model_comparison.csv" \
    --output-dir "${ROOT}/figures"

  python3 "${ROOT}/code/build_hccb_p418_steady_performance_table.py" \
    --comparison-csv "${COMPARISON_OUTPUT_DIR}/model_comparison.csv" \
    --output "${ROOT}/manuscript/generated_steady_performance.tex" \
    --summary "${COMPARISON_OUTPUT_DIR}/steady_performance_table.json"

  python3 "${ROOT}/code/build_hccb_p418_steady_result_text.py" \
    --comparison-csv "${COMPARISON_OUTPUT_DIR}/model_comparison.csv" \
    --thermal-regime-coverage "${COMPARISON_OUTPUT_DIR}/thermal_regime_split_coverage.json" \
    --output "${ROOT}/manuscript/generated_steady_result_text.tex" \
    --summary "${COMPARISON_OUTPUT_DIR}/steady_result_text.json"
  printf '%% Generated only after all five corrected steady-model results were assembled.\n' \
    > "${ROOT}/manuscript/generated_steady_model_comparison_validated.tex"
else
  echo "skip formal 5x5 paper figure and table for this partial/smoke comparison"
fi

if [[ ${paper_outputs} -eq 1 ]]; then
  python3 "${ROOT}/code/summarize_hccb_p418_thermal_regime_model_errors.py" \
    --physical-csv "${RESULT_PREFIX}_completed_physics/completed_case_physics.csv" \
    --results-root "${ROOT}/results" \
    --result-prefix "${RESULT_NAMESPACE}" \
    --epochs "${EPOCHS}" \
    --architectures response_surface ${ARCHITECTURES} \
    --splits ${SPLIT_NAMES} \
    --output-dir "${COMPARISON_OUTPUT_DIR}"

  # Use one fixed representative split for like-for-like experimental comparison.
  # Empty experimental tables create no artificial comparison values.
  for architecture in response_surface ${ARCHITECTURES}; do
    model_output=${ROOT}/results/${RESULT_NAMESPACE}_${architecture}_interleaved_all_ranges_${EPOCHS}epoch
    MODEL_OUTPUT="${model_output}" \
    MODEL_NAME="${architecture} P418 interleaved-all-ranges" \
    SPLIT_NAME=interleaved_all_ranges \
    RESULT_PREFIX="${RESULT_PREFIX}" \
    REGIONAL_TOPOLOGY="${REGIONAL_TOPOLOGY}" \
    REFERENCE_STATE_TARGETS="${STATE_TARGETS}" \
    MASS_TARGETS="${MASS_TARGETS}" \
    TRAINING_STATISTICS="${TRAINING_STATISTICS}" \
    OUTPUT_DIR="${COMPARISON_OUTPUT_DIR}/experimental_${architecture}" \
      bash "${ROOT}/code/run_hccb_p418_learned_model_experimental_comparison.sh"
  done
else
  echo "skip complete-matrix thermal-regime and experimental outputs for this partial/smoke comparison"
fi
