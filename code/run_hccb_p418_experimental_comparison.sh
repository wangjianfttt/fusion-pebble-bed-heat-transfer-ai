#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/experimental_data_templates}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_60_sourceflow_r3}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/hccb_p418_openfoam_experimental_comparison}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${ROOT}/results/hccb_p418_regional_topology_r2/regional_topology.npz}
STATE_TARGETS=${STATE_TARGETS:-${RESULT_PREFIX}_regional_state_targets/regional_state_targets.npz}
MASS_TARGETS=${MASS_TARGETS:-${RESULT_PREFIX}_regional_mass_flux_targets/regional_mass_flux_targets.npz}
ENERGY_TARGETS=${ENERGY_TARGETS:-${RESULT_PREFIX}_regional_energy_flux_targets/regional_energy_flux_targets.npz}
TEMPORAL_STATE_FILE=${TEMPORAL_STATE_FILE:-}
TEMPORAL_SOURCE=${TEMPORAL_SOURCE:-prediction}

python3 "${ROOT}/code/validate_hccb_p418_experimental_data.py" \
  --schema "${ROOT}/parameters/hccb_p418_experimental_data_schema.json" \
  --data-root "${DATA_ROOT}" \
  --output "${OUTPUT_DIR}/experimental_data_validation.json"

temporal_args=()
if [[ -n ${TEMPORAL_STATE_FILE} ]]; then
  temporal_args=(--temporal-state-file "${TEMPORAL_STATE_FILE}" --temporal-source "${TEMPORAL_SOURCE}")
fi

python3 "${ROOT}/code/compare_hccb_p418_model_to_experiment.py" \
  --data-root "${DATA_ROOT}" \
  --regional-topology "${REGIONAL_TOPOLOGY}" \
  --regional-level 5 \
  --state-file "${STATE_TARGETS}" \
  --mass-targets "${MASS_TARGETS}" \
  --energy-targets "${ENERGY_TARGETS}" \
  "${temporal_args[@]}" \
  --model-name "OpenFOAM P418 regional reference" \
  --output-dir "${OUTPUT_DIR}"
