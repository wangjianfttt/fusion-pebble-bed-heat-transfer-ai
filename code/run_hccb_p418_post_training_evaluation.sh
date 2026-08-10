#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CHAIN_PID=${CHAIN_PID:-}
DEVICE=${DEVICE:-cuda}
CHECK_INTERVAL_S=${CHECK_INTERVAL_S:-60}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results/hccb_p418_physical_steps_12}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/results/hccb_p418_high_re_three_bounded_model_evaluation}
LOCK_FILE=${LOCK_FILE:-${RESULT_ROOT}/.post_training_evaluation.lock}
LOG_RECORD=${LOG_RECORD:-${OUTPUT_ROOT}/post_training_evaluation_complete.json}

exec 9>"${LOCK_FILE}"
flock -n 9 || {
  echo "another P418 post-training evaluation is active" >&2
  exit 1
}

if [[ -n "${CHAIN_PID}" ]]; then
  while kill -0 "${CHAIN_PID}" 2>/dev/null; do
    sleep "${CHECK_INTERVAL_S}"
  done
fi

summaries=(
  "${RESULT_ROOT}/regional_graph_transformer_bounded_data_only_pair_disjoint_stress_test/summary.json"
  "${RESULT_ROOT}/regional_graph_transformer_bounded_physics_pair_disjoint_stress_test/summary.json"
  "${RESULT_ROOT}/regional_graph_transformer_bounded_factorized_pair_disjoint_stress_test/summary.json"
)

python3 "${ROOT}/code/check_hccb_p418_bounded_training_summaries.py" \
  --data-only "${summaries[0]}" \
  --physics "${summaries[1]}" \
  --factorized "${summaries[2]}" \
  --output "${OUTPUT_ROOT}/bounded_training_summaries_checked.json"

mkdir -p "${OUTPUT_ROOT}"
EXECUTE=1 \
DEVICE="${DEVICE}" \
ALLOW_PAUSED_WORKSTATION_RUN=1 \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
bash "${ROOT}/code/run_hccb_p418_high_re_independent_evaluation.sh"

python3 - "${OUTPUT_ROOT}" "${LOG_RECORD}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
record_path = pathlib.Path(sys.argv[2]).resolve()
required = {
    "data_only": root / "data_only" / "summary.json",
    "physics_constrained": root / "physics_constrained" / "summary.json",
    "factorized": root / "factorized" / "summary.json",
    "comparison": root / "comparison" / "summary.json",
}
files = {}
for label, path in required.items():
    if not path.is_file():
        raise SystemExit(f"post-training result is missing: {path}")
    payload = path.read_bytes()
    files[label] = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
record = {
    "status": "p418_three_fixed_models_high_re_evaluation_complete",
    "models": ["data_only", "physics_constrained", "factorized"],
    "independent_test_family": "six_high_velocity_fixed_hydrodynamics_sequences",
    "training_or_solver_started_by_this_step": False,
    "files": files,
    "new_physical_parameters": [],
}
record_path.parent.mkdir(parents=True, exist_ok=True)
record_path.write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(record_path)
PY
