#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results/hccb_p418_physical_steps_12}
DATA=${DATA:-${RESULT_ROOT}/hccb_p418_transient_observables.npz}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_step_response_splits.json}
CURRENT_PID=${CURRENT_PID:-}
CPU_LIST=${CPU_LIST:-54,55}
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-20260717}
LOCK_FILE=${LOCK_FILE:-${RESULT_ROOT}/.observable_transformer_chain.lock}

exec 9>"${LOCK_FILE}"
flock -n 9 || {
  echo "another observable Transformer chain is active" >&2
  exit 1
}

check_summary() {
  local summary=$1
  python3 - "${summary}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing summary: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
status = str(payload.get("status", ""))
if not status.startswith("completed_p418_physical_step_response_transformer_"):
    raise SystemExit(f"unexpected status in {path}: {status}")
PY
}

if [[ -n "${CURRENT_PID}" ]]; then
  while kill -0 "${CURRENT_PID}" 2>/dev/null; do
    sleep 20
  done
  check_summary "${RESULT_ROOT}/transformer_pair_disjoint_stress_test/summary.json"
fi

for split_name in direction_down_test direction_up_test; do
  output_dir="${RESULT_ROOT}/transformer_${split_name}"
  summary="${output_dir}/summary.json"
  if [[ -f "${summary}" ]]; then
    check_summary "${summary}"
    echo "${split_name}: existing completed result retained"
    continue
  fi

  mkdir -p "${output_dir}"
  printf '%s\n' "$$" > "${output_dir}/chain_pid.txt"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" taskset -c "${CPU_LIST}" \
    python3 -u "${ROOT}/code/train_hccb_p418_transient_observable_transformer.py" \
      --data "${DATA}" \
      --splits "${SPLITS}" \
      --split-name "${split_name}" \
      --output-dir "${output_dir}" \
      --run-role formal \
      --history-kind physical_step_response \
      --seed "${SEED}" \
      > "${output_dir}/run.log" 2>&1
  check_summary "${summary}"
  echo "${split_name}: completed"
done
