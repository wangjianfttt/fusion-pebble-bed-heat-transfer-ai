#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
POLL_SECONDS=${POLL_SECONDS:-120}
CORRECTED_NAMESPACE=${CORRECTED_NAMESPACE:-hccb_p418_60_normfix_20260731}
OUTPUT_NAMESPACE=${OUTPUT_NAMESPACE:-hccb_p418_60_corrected_20260731}
RESULT_ROOT=${ROOT}/results
COMPARISON_DIR=${RESULT_ROOT}/${OUTPUT_NAMESPACE}_model_comparison_100epoch
LOCK=${RESULT_ROOT}/.corrected_steady_postprocess.lock

required_methods=(graph transolver)
required_files=(
  summary.json
  train_regional_predictions.npz
  validation_regional_predictions.npz
  test_regional_predictions.npz
)

if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "corrected steady postprocessing is already registered: ${LOCK}" >&2
  exit 3
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

if [[ -s ${COMPARISON_DIR}/corrected_result_assembly.json ]]; then
  python3 - "${COMPARISON_DIR}/corrected_result_assembly.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "corrected_steady_result_assembly_complete":
    raise SystemExit("existing corrected steady assembly has an unexpected status")
if payload.get("result_count") != 25:
    raise SystemExit("existing corrected steady assembly is incomplete")
PY
  exit 0
fi

while :; do
  waiting=0
  for method in "${required_methods[@]}"; do
    result=${RESULT_ROOT}/${CORRECTED_NAMESPACE}_${method}_heat_source_extrapolation_100epoch
    for relative in "${required_files[@]}"; do
      if [[ ! -s ${result}/${relative} ]]; then
        waiting=1
      fi
    done
  done
  if [[ ${waiting} -eq 0 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

for method in "${required_methods[@]}"; do
  result=${RESULT_ROOT}/${CORRECTED_NAMESPACE}_${method}_heat_source_extrapolation_100epoch
  python3 - "${result}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if "complete" not in str(payload.get("status", "")):
    raise SystemExit("corrected steady model summary is not complete")
PY
done

ROOT="${ROOT}" \
CORRECTED_NAMESPACE="${CORRECTED_NAMESPACE}" \
OUTPUT_NAMESPACE="${OUTPUT_NAMESPACE}" \
  bash "${ROOT}/code/run_hccb_p418_corrected_steady_comparison.sh"

python3 - "${COMPARISON_DIR}/corrected_result_assembly.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "corrected_steady_result_assembly_complete":
    raise SystemExit("corrected steady postprocessing did not finish")
if payload.get("result_count") != 25:
    raise SystemExit("corrected steady postprocessing assembled fewer than 25 results")
PY

echo "Corrected steady comparison and manuscript inputs are ready."
