#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_60_sourceflow_r3}
RESULT_NAMESPACE=${RESULT_NAMESPACE:-hccb_p418_60}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${ROOT}/results/hccb_p418_regional_topology_r2/regional_topology.npz}
MODEL_GEOMETRY=${MODEL_GEOMETRY:-${RESULT_PREFIX}_model_geometry/model_geometry.npz}
STATE_TARGETS=${STATE_TARGETS:-${RESULT_PREFIX}_regional_state_targets/regional_state_targets.npz}
MASS_TARGETS=${MASS_TARGETS:-${RESULT_PREFIX}_regional_mass_flux_targets/regional_mass_flux_targets.npz}
ENERGY_TARGETS=${ENERGY_TARGETS:-${RESULT_PREFIX}_regional_energy_flux_targets/regional_energy_flux_targets.npz}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_model_splits.json}
TRAINING_STATISTICS=${TRAINING_STATISTICS:-${RESULT_PREFIX}_training_statistics.json}
SPLIT_NAME=${SPLIT_NAME:-interleaved_all_ranges}
EPOCHS=${EPOCHS:-100}
PRIMARY_SEED=${PRIMARY_SEED:-20260717}
MODEL_SEEDS=${MODEL_SEEDS:-20260717 20260718 20260719}
ARCHITECTURES=${ARCHITECTURES:-pinn_data_only pinn graph transolver}
THREADS=${THREADS:-4}
DEVICE=${DEVICE:-cuda}
GRAPH_MICROBATCH_SIZE=${GRAPH_MICROBATCH_SIZE:-1}
TRANSOLVER_MICROBATCH_SIZE=${TRANSOLVER_MICROBATCH_SIZE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/${RESULT_NAMESPACE}_steady_seed_robustness_${EPOCHS}epoch}
TEX_OUTPUT=${TEX_OUTPUT:-${ROOT}/manuscript/generated_steady_seed_robustness.tex}

current_result() {
  local architecture=$1
  local seed=$2
  local output=$3
  python3 "${ROOT}/code/check_hccb_p418_steady_result_current.py" \
    --summary "${output}/summary.json" \
    --architecture "${architecture}" \
    --epochs "${EPOCHS}" \
    --split-name "${SPLIT_NAME}" \
    --state-targets "${STATE_TARGETS}" \
    --mass-targets "${MASS_TARGETS}" \
    --energy-targets "${ENERGY_TARGETS}" \
    --split-file "${SPLITS}" \
    --training-statistics "${TRAINING_STATISTICS}" \
    --training-seed "${seed}"
}

for architecture in ${ARCHITECTURES}; do
  for seed in ${MODEL_SEEDS}; do
    if [[ ${seed} == "${PRIMARY_SEED}" ]]; then
      output=${ROOT}/results/${RESULT_NAMESPACE}_${architecture}_${SPLIT_NAME}_${EPOCHS}epoch
      if ! current_result "${architecture}" "${seed}" "${output}"; then
        echo "primary steady result is missing or does not use seed ${PRIMARY_SEED}: ${output}" >&2
        exit 1
      fi
      continue
    fi
    output=${ROOT}/results/${RESULT_NAMESPACE}_${architecture}_${SPLIT_NAME}_${EPOCHS}epoch_seed${seed}
    if [[ -f ${output}/summary.json ]] && current_result "${architecture}" "${seed}" "${output}"; then
      echo "reuse current ${architecture} ${SPLIT_NAME} seed ${seed}"
      continue
    fi
    if [[ -e ${output} ]]; then
      mv "${output}" "${output}.older.$(date +%Y%m%dT%H%M%S).$$"
    fi
    mkdir -p "${output}"
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
      --split-name "${SPLIT_NAME}" \
      --regional-level 5 \
      --architecture "${architecture}" \
      --epochs "${EPOCHS}" \
      --threads "${THREADS}" \
      --device "${DEVICE}" \
      --seed "${seed}" \
      "${microbatch_args[@]}" \
      --output-dir "${output}" \
      > "${output}/run.log" 2>&1
    echo "completed ${architecture} ${SPLIT_NAME} seed ${seed}"
  done
done

python3 "${ROOT}/code/summarize_hccb_p418_steady_seed_robustness.py" \
  --results-root "${ROOT}/results" \
  --result-prefix "${RESULT_NAMESPACE}" \
  --split-file "${SPLITS}" \
  --split-name "${SPLIT_NAME}" \
  --epochs "${EPOCHS}" \
  --primary-seed "${PRIMARY_SEED}" \
  --seeds ${MODEL_SEEDS} \
  --output-dir "${OUTPUT_DIR}" \
  --tex-output "${TEX_OUTPUT}"
