#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_60_sourceflow_r3}
RESULT_NAMESPACE=${RESULT_NAMESPACE:-hccb_p418_60}
BASE_COMPARISON_DIR=${BASE_COMPARISON_DIR:-${ROOT}/results/${RESULT_NAMESPACE}_model_comparison_100epoch}
CONVERGENCE_SUMMARY=${CONVERGENCE_SUMMARY:-${BASE_COMPARISON_DIR}/training_convergence.json}
FOLLOWUP_DIR=${FOLLOWUP_DIR:-${ROOT}/results/${RESULT_NAMESPACE}_source_epoch_followup}
PLAN=${PLAN:-${FOLLOWUP_DIR}/epoch_followup_plan.json}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${ROOT}/results/hccb_p418_regional_topology_r2/regional_topology.npz}
MODEL_GEOMETRY=${MODEL_GEOMETRY:-${RESULT_PREFIX}_model_geometry/model_geometry.npz}
STATE_TARGETS=${STATE_TARGETS:-${RESULT_PREFIX}_regional_state_targets/regional_state_targets.npz}
MASS_TARGETS=${MASS_TARGETS:-${RESULT_PREFIX}_regional_mass_flux_targets/regional_mass_flux_targets.npz}
ENERGY_TARGETS=${ENERGY_TARGETS:-${RESULT_PREFIX}_regional_energy_flux_targets/regional_energy_flux_targets.npz}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_model_splits.json}
TRAINING_STATISTICS=${TRAINING_STATISTICS:-${RESULT_PREFIX}_training_statistics.json}
THREADS=${THREADS:-4}
DEVICE=${DEVICE:-cuda}
MODEL_SEED=${MODEL_SEED:-20260717}
GRAPH_MICROBATCH_SIZE=${GRAPH_MICROBATCH_SIZE:-1}
TRANSOLVER_MICROBATCH_SIZE=${TRANSOLVER_MICROBATCH_SIZE:-1}

mkdir -p "${FOLLOWUP_DIR}"
python3 "${ROOT}/code/build_hccb_p418_epoch_followup_plan.py" \
  --convergence-summary "${CONVERGENCE_SUMMARY}" \
  --output "${PLAN}" > "${FOLLOWUP_DIR}/build_plan.log"

while IFS=$'\t' read -r architecture split_name followup_epochs; do
  [[ -n ${architecture} ]] || continue
  output=${ROOT}/results/${RESULT_NAMESPACE}_${architecture}_${split_name}_${followup_epochs}epoch
  if [[ -f ${output}/summary.json ]] && python3 "${ROOT}/code/check_hccb_p418_steady_result_current.py" \
      --summary "${output}/summary.json" \
      --architecture "${architecture}" \
      --epochs "${followup_epochs}" \
      --split-name "${split_name}" \
      --state-targets "${STATE_TARGETS}" \
      --mass-targets "${MASS_TARGETS}" \
      --energy-targets "${ENERGY_TARGETS}" \
      --split-file "${SPLITS}" \
      --training-statistics "${TRAINING_STATISTICS}" \
      --training-seed "${MODEL_SEED}"; then
    echo "reuse current source-epoch run ${architecture} ${split_name} ${followup_epochs}"
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
    --split-name "${split_name}" \
    --regional-level 5 \
    --architecture "${architecture}" \
    --epochs "${followup_epochs}" \
    --threads "${THREADS}" \
    --device "${DEVICE}" \
    --seed "${MODEL_SEED}" \
    "${microbatch_args[@]}" \
    --output-dir "${output}" > "${output}/run.log" 2>&1
done < <(
  python3 - "${PLAN}" <<'PY'
import json
import sys

for run in json.load(open(sys.argv[1], encoding="utf-8"))["runs"]:
    print(run["architecture"], run["split"], run["followup_epochs"], sep="\t")
PY
)

python3 "${ROOT}/code/compare_hccb_p418_epoch_followup.py" \
  --plan "${PLAN}" \
  --project-root "${ROOT}" \
  --output-dir "${FOLLOWUP_DIR}"
