#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results/hccb_p418_physical_steps_12}
DATASET_INDEX=${DATASET_INDEX:-${RESULT_ROOT}/regional_sequences/dataset_index.json}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_step_response_splits.json}
RESIDUAL_GEOMETRY=${RESIDUAL_GEOMETRY:-${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}
WAIT_PID=${WAIT_PID:-}
CPU_LIST=${CPU_LIST:-54,55}
GPU_ID=${GPU_ID:-0}
PHYSICS_DEVICE=${PHYSICS_DEVICE:-cuda}
PHYSICS_TIME_CHUNK_SIZE=${PHYSICS_TIME_CHUNK_SIZE:-8}
SEED=${SEED:-20260717}
SPLIT_NAME=${SPLIT_NAME:-pair_disjoint_stress_test}
LOCK_FILE=${LOCK_FILE:-${RESULT_ROOT}/.strict_regional_graph_chain.lock}
OUTPUT_REVISION=${OUTPUT_REVISION:-bounded}
TEMPERATURE_OUTPUT_MODE=${TEMPERATURE_OUTPUT_MODE:-literature_bounded_residual}

exec 9>"${LOCK_FILE}"
flock -n 9 || {
  echo "another strict regional graph chain is active" >&2
  exit 1
}

check_summary() {
  local summary=$1
  local expected_role=$2
  local expected_mode=$3
  python3 - "${summary}" "${expected_role}" "${expected_mode}" "${SPLIT_NAME}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_role, expected_mode, expected_split = sys.argv[2:5]
if not path.is_file():
    raise SystemExit(f"missing summary: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
checks = {
    "status": (
        payload.get("status"),
        "completed_p418_spatiotemporal_regional_operator",
    ),
    "run_role": (payload.get("run_role"), expected_role),
    "physics_mode": (payload.get("physics_mode"), expected_mode),
    "split_name": (payload.get("split_name"), expected_split),
    "temperature_output_mode": (
        payload.get("architecture", {}).get("temperature_output_mode"),
        "literature_bounded_residual",
    ),
}
bad = {name: pair for name, pair in checks.items() if pair[0] != pair[1]}
if bad:
    raise SystemExit(f"unexpected summary values in {path}: {bad}")
PY
}

wait_for_pid() {
  local pid=$1
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
  done
}

run_model() {
  local label=$1
  local output_dir=$2
  local run_role=$3
  local physics_mode=$4
  shift 4

  local summary="${output_dir}/summary.json"
  if [[ -f "${summary}" ]]; then
    check_summary "${summary}" "${run_role}" "${physics_mode}"
    echo "${label}: existing completed result retained"
    return
  fi

  mkdir -p "${output_dir}"
  printf '%s\n' "$$" > "${output_dir}/chain_pid.txt"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" taskset -c "${CPU_LIST}" \
    python3 -u "${ROOT}/code/train_hccb_p418_spatiotemporal_regional_operator.py" \
      --dataset-index "${DATASET_INDEX}" \
      --splits "${SPLITS}" \
      --split-name "${SPLIT_NAME}" \
      --residual-geometry "${RESIDUAL_GEOMETRY}" \
      --output-dir "${output_dir}" \
      --run-role "${run_role}" \
      --physics-mode "${physics_mode}" \
      --temperature-output-mode "${TEMPERATURE_OUTPUT_MODE}" \
      --seed "${SEED}" \
      --resume \
      "$@" \
      > "${output_dir}/run.log" 2>&1
  check_summary "${summary}" "${run_role}" "${physics_mode}"
  echo "${label}: completed"
}

if [[ -n "${WAIT_PID}" ]]; then
  wait_for_pid "${WAIT_PID}"
fi

run_model \
  graph_data_only \
  "${RESULT_ROOT}/regional_graph_transformer_${OUTPUT_REVISION}_data_only_${SPLIT_NAME}" \
  formal_data_only \
  data_only

run_model \
  graph_physics \
  "${RESULT_ROOT}/regional_graph_transformer_${OUTPUT_REVISION}_physics_${SPLIT_NAME}" \
  formal \
  energy_and_flux \
  --physics-device "${PHYSICS_DEVICE}" \
  --physics-time-chunk-size "${PHYSICS_TIME_CHUNK_SIZE}"

run_model \
  graph_factorized \
  "${RESULT_ROOT}/regional_graph_transformer_${OUTPUT_REVISION}_factorized_${SPLIT_NAME}" \
  formal_factorized \
  energy_and_flux \
  --physics-device "${PHYSICS_DEVICE}" \
  --physics-time-chunk-size "${PHYSICS_TIME_CHUNK_SIZE}" \
  --spatial-temporal-mode factorized_static_spatial
