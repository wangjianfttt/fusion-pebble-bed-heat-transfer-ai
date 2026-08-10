#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MATRIX_SOURCE=${MATRIX_SOURCE:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
EXPECTED_CASES=${EXPECTED_CASES:-8}
SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-${ROOT}/hccb_dense_cht_p418_mixed_endpoint_smoke}
DATASET_ROOT=${DATASET_ROOT:-${ROOT}/hccb_dense_cht_p418_mixed_endpoint_smoke_dataset}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_mixed_endpoint_smoke}
SPLITS=${SPLITS:-${ROOT}/results/hccb_p418_mixed_endpoint_smoke_splits.json}
SOURCE_SPLITS=${SOURCE_SPLITS:-${ROOT}/parameters/hccb_p418_model_splits.json}
WAIT_SECONDS=${WAIT_SECONDS:-60}

while true; do
  completed=$(find "${MATRIX_SOURCE}" -mindepth 2 -maxdepth 2 \
    -name formal_sample_complete.json | wc -l | tr -d ' ')
  if (( completed >= EXPECTED_CASES )); then
    break
  fi
  echo "waiting for completed P418 cases: ${completed}/${EXPECTED_CASES}"
  sleep "${WAIT_SECONDS}"
done

rm -rf "${SNAPSHOT_ROOT}"
mkdir -p "${SNAPSHOT_ROOT}"
mapfile -t selected_cases < <(
  find "${MATRIX_SOURCE}" -mindepth 2 -maxdepth 2 \
    -name formal_sample_complete.json -printf '%T@ %h\n' \
    | sort -n | head -n "${EXPECTED_CASES}" | cut -d' ' -f2-
)
if [[ ${#selected_cases[@]} -ne ${EXPECTED_CASES} ]]; then
  echo "could not select ${EXPECTED_CASES} completed cases" >&2
  exit 1
fi
for source_case in "${selected_cases[@]}"; do
  cp -al "${source_case}" "${SNAPSHOT_ROOT}/$(basename "${source_case}")"
done

python3 "${ROOT}/code/build_hccb_p418_completed_smoke_splits.py" \
  --matrix-root "${SNAPSHOT_ROOT}" \
  --source-splits "${SOURCE_SPLITS}" \
  --expected-case-count "${EXPECTED_CASES}" \
  --output "${SPLITS}"

ROOT="${ROOT}" \
MATRIX_ROOT="${SNAPSHOT_ROOT}" \
DATASET_ROOT="${DATASET_ROOT}" \
RESULT_PREFIX="${RESULT_PREFIX}" \
SPLITS="${SPLITS}" \
EXPECTED_CASES="${EXPECTED_CASES}" \
  bash "${ROOT}/code/run_hccb_p418_60_postprocess.sh"

python3 "${ROOT}/code/summarize_hccb_p418_completed_matrix_physics.py" \
  --matrix-root "${SNAPSHOT_ROOT}" \
  --time-from-completion-marker \
  --output-dir "${RESULT_PREFIX}_completed_physics"

python3 "${ROOT}/code/analyze_hccb_p418_dimensionless_heat_transfer.py" \
  --matrix-root "${SNAPSHOT_ROOT}" \
  --parameter-manifest "${ROOT}/parameters/literature_parameter_manifest.csv" \
  --boundary-heat-summary "${RESULT_PREFIX}_boundary_heat_flux_targets/summary.json" \
  --output-dir "${RESULT_PREFIX}_dimensionless_heat_transfer"

python3 - "${SNAPSHOT_ROOT}" "${RESULT_PREFIX}_postprocess_summary.json" <<'PY'
import json
import pathlib
import sys

matrix = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
times = sorted(
    {
        str(json.loads(path.read_text(encoding="utf-8"))["time"])
        for path in matrix.glob("*/formal_sample_complete.json")
    }
)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("status") != "p418_60_training_data_ready":
    raise SystemExit("mixed-endpoint postprocess summary is not ready")
if len(times) < 2:
    raise SystemExit(f"mixed-endpoint check requires at least two completion times: {times}")
payload = {
    "status": "mixed_endpoint_postprocess_ready",
    "completion_times_s": times,
    "case_count": int(summary["expected_case_count"]),
    "postprocess_summary": str(summary_path.resolve()),
    "scientific_scope": "software-path check only; no model accuracy is reported",
}
output = pathlib.Path(str(summary_path).replace("_postprocess_summary.json", "_mixed_endpoint_check.json"))
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
