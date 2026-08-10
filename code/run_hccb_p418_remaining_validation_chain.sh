#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WAIT_PID=${WAIT_PID:-}
POLL_SECONDS=${POLL_SECONDS:-60}
GPU_ID=${GPU_ID:-0}
CPU_LIST=${CPU_LIST:-110,111}
THREADS=${THREADS:-2}
RESULT_ROOT=${ROOT}/results
TRANSIENT_ROOT=${RESULT_ROOT}/hccb_p418_physical_steps_12
FORMAL_MANIFEST=${FORMAL_MANIFEST:-${TRANSIENT_ROOT}/formal_training_jobs_workstation.json}
LOCK_FILE=${LOCK_FILE:-${RESULT_ROOT}/.p418_remaining_validation_chain.lock}
COMPLETE_RECORD=${COMPLETE_RECORD:-${RESULT_ROOT}/hccb_p418_remaining_validation_chain_complete.json}

exec 9>"${LOCK_FILE}"
flock -n 9 || {
  echo "another P418 remaining-validation chain is active" >&2
  exit 1
}

if [[ -n ${WAIT_PID} ]]; then
  if [[ ! ${WAIT_PID} =~ ^[0-9]+$ ]]; then
    echo "WAIT_PID must be an integer process id" >&2
    exit 2
  fi
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
fi

# A stopped or failed main chain must not release the follow-up training.
python3 - "${FORMAL_MANIFEST}" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
jobs = payload.get("jobs")
if not isinstance(jobs, list) or len(jobs) != 75:
    raise SystemExit("formal training manifest must contain exactly 75 jobs")

incomplete_tokens = (
    "failed", "failure", "incomplete", "not_started", "in_progress",
    "running", "blocked",
)
missing = []
for job in jobs:
    path = pathlib.Path(str(job["completion_file"]))
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(str(job["job_id"]))
        continue
    if path.suffix == ".json":
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            missing.append(str(job["job_id"]))
            continue
        status = str(result.get("status", "")).lower()
        if not status or any(token in status for token in incomplete_tokens):
            missing.append(str(job["job_id"]))
if missing:
    raise SystemExit(
        "formal 75-job chain did not complete; follow-up training was not started: "
        + ", ".join(missing)
    )
print("formal 75-job chain complete; starting remaining validation runs")
PY

json_has_status() {
  python3 - "$1" "$2" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
if not path.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == expected else 1)
PY
}

STEADY_SEEDS=${RESULT_ROOT}/hccb_p418_60_steady_seed_robustness_100epoch/summary.json
if ! json_has_status "${STEADY_SEEDS}" completed_p418_main_steady_split_seed_robustness; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" taskset -c "${CPU_LIST}" \
    env DEVICE=cuda THREADS="${THREADS}" \
      GRAPH_MICROBATCH_SIZE=1 TRANSOLVER_MICROBATCH_SIZE=1 \
      bash "${ROOT}/code/run_hccb_p418_steady_seed_robustness.sh"
fi

STEADY_CURVE=${RESULT_ROOT}/hccb_p418_learning_curve_model_comparison_100epoch/learning_curve_summary.json
if ! json_has_status "${STEADY_CURVE}" p418_steady_learning_curve_complete; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" taskset -c "${CPU_LIST}" \
    env DEVICE=cuda THREADS="${THREADS}" \
      GRAPH_MICROBATCH_SIZE=1 TRANSOLVER_MICROBATCH_SIZE=1 \
      bash "${ROOT}/code/run_hccb_p418_steady_learning_curve.sh"
fi

TRANSIENT_CURVE=${RESULT_ROOT}/hccb_p418_transient_learning_curve/summary.json
if ! json_has_status "${TRANSIENT_CURVE}" completed_p418_transient_learning_curve; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" taskset -c "${CPU_LIST}" \
    env EXECUTE=1 \
      bash "${ROOT}/code/run_hccb_p418_transient_learning_curve.sh"
fi

LOSS_ROOT=${RESULT_ROOT}/hccb_p418_physical_steps_12/fixed_flow_loss_balancing_pair_disjoint_stress_test
LOSS_SELECTION=${LOSS_ROOT}/selected_loss_balancing_method.json
LOSS_INTEGRATION=${LOSS_ROOT}/selected_downstream_integration.json
STEP_DATASET=${RESULT_ROOT}/hccb_p418_physical_steps_12/regional_sequences/dataset_index.json
STEP_SPLITS=${ROOT}/parameters/hccb_p418_step_response_splits.json
STEP_GEOMETRY=${RESULT_ROOT}/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz
LOSS_PROTOCOL=${ROOT}/code/run_hccb_p418_fixed_flow_loss_balancing_protocol.py
LOSS_DOWNSTREAM=${ROOT}/code/run_hccb_p418_selected_loss_downstream.py
POST_SELECTION_OUTPUTS=${ROOT}/code/rerun_hccb_p418_post_selection_outputs.py

if ! json_has_status "${LOSS_SELECTION}" p418_loss_balancing_selected_on_validation_only; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" taskset -c "${CPU_LIST}" \
    python3 "${LOSS_PROTOCOL}" selection \
      --dataset-index "${STEP_DATASET}" \
      --splits "${STEP_SPLITS}" \
      --split-name pair_disjoint_stress_test \
      --residual-geometry "${STEP_GEOMETRY}" \
      --output-root "${LOSS_ROOT}" \
      --physics-device cuda \
      --torch-threads "${THREADS}"
fi

SELECTED_CANDIDATE=$(python3 - "${LOSS_SELECTION}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["selected_candidate_id"])
PY
)
LOSS_FINAL=${LOSS_ROOT}/${SELECTED_CANDIDATE}/final_summary.json
if ! json_has_status "${LOSS_FINAL}" completed_p418_spatiotemporal_regional_operator; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" taskset -c "${CPU_LIST}" \
    python3 "${LOSS_PROTOCOL}" final \
      --dataset-index "${STEP_DATASET}" \
      --splits "${STEP_SPLITS}" \
      --split-name pair_disjoint_stress_test \
      --residual-geometry "${STEP_GEOMETRY}" \
      --output-root "${LOSS_ROOT}" \
      --physics-device cuda \
      --torch-threads "${THREADS}"
fi

if ! json_has_status "${LOSS_INTEGRATION}" completed_p418_selected_loss_balancing_downstream; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" taskset -c "${CPU_LIST}" \
    python3 "${LOSS_DOWNSTREAM}" \
      --dataset-index "${STEP_DATASET}" \
      --splits "${STEP_SPLITS}" \
      --residual-geometry "${STEP_GEOMETRY}" \
      --result-dir "${RESULT_ROOT}/hccb_p418_physical_steps_12" \
      --physics-device cuda \
      --torch-threads "${THREADS}" \
      --execute
fi

python3 "${POST_SELECTION_OUTPUTS}" \
  --root "${ROOT}" \
  --manifest "${FORMAL_MANIFEST}" \
  --result-dir "${TRANSIENT_ROOT}" \
  --execute

python3 - "${COMPLETE_RECORD}" "${STEADY_SEEDS}" "${STEADY_CURVE}" "${TRANSIENT_CURVE}" "${LOSS_INTEGRATION}" \
  "${TRANSIENT_ROOT}/model_comparison/summary.json" \
  "${ROOT}/figures/hccb_p418_transient_model_comparison.json" \
  "${ROOT}/figures/hccb_p418_openfoam_model_field_comparison.json" <<'PY'
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1]).resolve()
expected = {
    pathlib.Path(sys.argv[2]).resolve():
        "completed_p418_main_steady_split_seed_robustness",
    pathlib.Path(sys.argv[3]).resolve():
        "p418_steady_learning_curve_complete",
    pathlib.Path(sys.argv[4]).resolve():
        "completed_p418_transient_learning_curve",
    pathlib.Path(sys.argv[5]).resolve():
        "completed_p418_selected_loss_balancing_downstream",
    pathlib.Path(sys.argv[6]).resolve():
        "completed_p418_physical_step_model_comparison",
    pathlib.Path(sys.argv[7]).resolve():
        "complete_formal_p418_transient_model_comparison_figure",
    pathlib.Path(sys.argv[8]).resolve():
        "complete_same_scale_openfoam_model_field_comparison",
}
files = []
for path, status in expected.items():
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise SystemExit(f"unexpected validation status in {path}")
    files.append(
        {
            "path": str(path),
            "status": status,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
record = {
    "status": "completed_p418_remaining_validation_chain",
    "results": files,
    "openfoam_solver_started": False,
    "new_physical_parameters": [],
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(output)
PY
