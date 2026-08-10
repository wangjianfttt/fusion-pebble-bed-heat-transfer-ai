#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-${SCRIPT_ROOT}}
EXECUTE=${EXECUTE:-0}
MODE=${MODE:-fixed}
PAUSE_MARKER=${PAUSE_MARKER:-${ROOT}/control/PAUSE_NEW_P418_CASES_FOR_CLOUD_MIGRATION}
ALLOW_PAUSED_WORKSTATION_RUN=${ALLOW_PAUSED_WORKSTATION_RUN:-0}
RUN_FROZEN_EVALUATION=${RUN_FROZEN_EVALUATION:-1}

if [[ ${MODE} != fixed && ${MODE} != fully_coupled ]]; then
  echo "MODE must be fixed or fully_coupled" >&2
  exit 2
fi

python3 "${ROOT}/code/verify_hccb_p418_high_re_independent_plan.py"

if [[ ${EXECUTE} != 1 ]]; then
  cat <<EOF
P418 high-Re independent test plan only; no OpenFOAM or model training was started.
mode=${MODE}
curves=6
data_role=frozen_model_independent_test_only
Set EXECUTE=1 only after the 60 steady endpoints and the main transient model selection are complete.
EOF
  exit 0
fi

if [[ -f ${PAUSE_MARKER} && ${ALLOW_PAUSED_WORKSTATION_RUN} != 1 ]]; then
  echo "new P418 calculations are paused for cloud migration: ${PAUSE_MARKER}" >&2
  exit 3
fi

if [[ ${MODE} == fixed ]]; then
  ROOT="${ROOT}" \
  PLAN="${ROOT}/parameters/hccb_p418_high_re_independent_step_plan.json" \
  STEP_ROOT="${ROOT}/hccb_p418_high_re_independent_fixed_steps_6" \
  RESULT_DIR="${ROOT}/results/hccb_p418_high_re_independent_fixed_steps_6" \
  RUN_MODEL_TRAINING=0 \
    bash "${ROOT}/code/run_hccb_p418_step_responses.sh"
else
  ROOT="${ROOT}" \
  PLAN="${ROOT}/parameters/hccb_p418_high_re_independent_fully_coupled_step_plan.json" \
  STEP_ROOT="${ROOT}/hccb_p418_high_re_independent_fully_coupled_steps_6" \
  RESULT_DIR="${ROOT}/results/hccb_p418_high_re_independent_fully_coupled_steps_6" \
  FIXED_RESULT_DIR="${ROOT}/results/hccb_p418_high_re_independent_fixed_steps_6" \
  EXECUTE=1 \
  REQUIRE_TIMESTEP_SENSITIVITY=0 \
  COMPARE_FIXED=1 \
  ALLOW_PAUSED_WORKSTATION_RUN="${ALLOW_PAUSED_WORKSTATION_RUN}" \
    bash "${ROOT}/code/run_hccb_p418_fully_coupled_step_responses.sh"
fi

if [[ ${RUN_FROZEN_EVALUATION} == 1 ]]; then
  ROOT="${ROOT}" \
  MODE="${MODE}" \
  EXECUTE=1 \
  ALLOW_PAUSED_WORKSTATION_RUN="${ALLOW_PAUSED_WORKSTATION_RUN}" \
  PAUSE_MARKER="${PAUSE_MARKER}" \
    bash "${ROOT}/code/run_hccb_p418_high_re_independent_evaluation.sh"
fi
