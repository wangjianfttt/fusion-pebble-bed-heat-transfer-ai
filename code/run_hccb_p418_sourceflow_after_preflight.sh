#!/usr/bin/env bash
# Wait for the corrected source-flow preflight, check its physical result, and
# then start the formal 60-condition matrix on the same remote machine.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PREFLIGHT_ROOT=${PREFLIGHT_ROOT:-${ROOT}/hccb_dense_cht_p418_sourceflow_preflight}
PREFLIGHT_CASE=${PREFLIGHT_CASE:-${PREFLIGHT_ROOT}/u0p05_T300_q4p85}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
RESULT_DIR=${RESULT_DIR:-${ROOT}/results/hccb_p418_sourceflow_preflight}
STATUS_FILE=${STATUS_FILE:-${ROOT}/results/hccb_p418_sourceflow_watch_status.txt}
LOG_FILE=${LOG_FILE:-${ROOT}/results/hccb_p418_60_sourceflow_r3_run.log}
FORMAL_ROUTE_LOG=${FORMAL_ROUTE_LOG:-${ROOT}/results/hccb_p418_formal_calculations.log}
LOCK_FILE=${LOCK_FILE:-${ROOT}/results/hccb_p418_sourceflow_watch.lock}
POLL_SECONDS=${POLL_SECONDS:-300}
NP_PER_CASE=${NP_PER_CASE:-32}
CONCURRENT_CASES=${CONCURRENT_CASES:-3}
PYTHON=${PYTHON:-python3}
FORMAL_INPUT_SUMMARY=${FORMAL_INPUT_SUMMARY:-${RESULT_DIR}/formal_60_input_summary.json}

if ! [[ ${POLL_SECONDS} =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

mkdir -p "${ROOT}/results" "${RESULT_DIR}"

record_exit_status() {
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    printf '%s automatic continuation stopped with return code %d; inspect %s\n' \
      "$(date --iso-8601=seconds)" "${rc}" "${ROOT}/results/hccb_p418_sourceflow_watch.log" \
      > "${STATUS_FILE}"
  fi
}
trap record_exit_status EXIT

formal_input_check_is_current() {
  [[ -f ${FORMAL_INPUT_SUMMARY} ]] || return 1
  "${PYTHON}" - "${FORMAL_INPUT_SUMMARY}" <<'PY' || return 1
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("status") != "hccb_p418_60_actual_case_inputs_verified":
    raise SystemExit(1)
if result.get("case_count") != 60:
    raise SystemExit(1)
if result.get("all_openfoam_dictionary_values_match_registered_sources") is not True:
    raise SystemExit(1)
if result.get("all_operating_points_are_exact_P418_values") is not True:
    raise SystemExit(1)
PY

  # Only files read by verify_hccb_p418_actual_case_inputs.py are considered.
  # Solver logs and written time directories do not invalidate the input check.
  if find "${MATRIX_ROOT}" -type f \
      \( -name cht_smoke_metadata.json \
         -o -path '*/0/fluid/U' -o -path '*/0/fluid/T' -o -path '*/0/fluid/p' \
         -o -path '*/0/solid/T' \
         -o -path '*/constant/fluid/physicalProperties' \
         -o -path '*/constant/solid/physicalProperties' \
         -o -path '*/constant/solid/fvModels' \) \
      -newer "${FORMAL_INPUT_SUMMARY}" -print -quit | grep -q .; then
    return 1
  fi

  local source_file
  for source_file in \
    "${ROOT}/parameters/hccb_p418_physical_parameter_sources.csv" \
    "${ROOT}/results/apd006_hccb_openfoam_helium_property_table/physicalProperties" \
    "${ROOT}/results/apd006_hccb_source_sequence_lammps/sweep/seed101_s80/packing_input_manifest.json" \
    "${ROOT}/data/apd006_hccb_source_sequence_target_packings/seed101_s80_xlo_ycentre/summary.json"; do
    [[ -f ${source_file} ]] || return 1
    [[ ${source_file} -ot ${FORMAL_INPUT_SUMMARY} ]] || return 1
  done
  return 0
}

exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
  echo "the corrected source-flow watcher is already running" >&2
  exit 1
fi

failed_polls=0
while [[ ! -f ${PREFLIGHT_CASE}/formal_sample_complete.json ]]; do
  if [[ -f ${PREFLIGHT_CASE}/formal_sample_failed.json ]]; then
    printf '%s corrected preflight reported failure\n' "$(date --iso-8601=seconds)" \
      > "${STATUS_FILE}"
    exit 1
  fi
  if pgrep -f "${PREFLIGHT_CASE}" >/dev/null; then
    failed_polls=0
    state=$("${PYTHON}" "${ROOT}/code/report_hccb_p418_runtime_progress.py" \
      --matrix-root "${PREFLIGHT_ROOT}" --concurrent-cases 1 --parallel-ranks 32)
    printf '%s corrected preflight running\n%s\n' \
      "$(date --iso-8601=seconds)" "${state}" > "${STATUS_FILE}"
  else
    failed_polls=$((failed_polls + 1))
    printf '%s corrected preflight has no active process (%d/12 checks)\n' \
      "$(date --iso-8601=seconds)" "${failed_polls}" > "${STATUS_FILE}"
    if [[ ${failed_polls} -ge 12 ]]; then
      echo "preflight stopped without a completion record" >&2
      exit 1
    fi
  fi
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" "${ROOT}/code/summarize_hccb_p418_completed_matrix_physics.py" \
  --matrix-root "${PREFLIGHT_ROOT}" \
  --time-from-completion-marker \
  --output-dir "${RESULT_DIR}" \
  > "${RESULT_DIR}/summary_stdout.json"

"${PYTHON}" "${ROOT}/code/analyze_hccb_p418_pressure_correlation.py" \
  --matrix-root "${PREFLIGHT_ROOT}" \
  --parameter-manifest "${ROOT}/parameters/literature_parameter_manifest.csv" \
  --physical-csv "${RESULT_DIR}/completed_case_physics.csv" \
  --output-dir "${RESULT_DIR}/pressure_correlation" \
  --expected-case-count 1 \
  > "${RESULT_DIR}/pressure_correlation_stdout.json"

"${PYTHON}" - "${PREFLIGHT_CASE}" "${RESULT_DIR}/summary.json" \
  "${RESULT_DIR}/pressure_correlation/summary.json" <<'PY'
import json
import math
import pathlib
import sys

case = pathlib.Path(sys.argv[1])
physics = json.load(open(sys.argv[2], encoding="utf-8"))
pressure_check = json.load(open(sys.argv[3], encoding="utf-8"))
metadata = json.load(open(case / "cht_smoke_metadata.json", encoding="utf-8"))
marker = json.load(open(case / "formal_sample_complete.json", encoding="utf-8"))
time_name = str(marker["time"])
result = json.load(open(case / f"cht_result_summary_{time_name}.json", encoding="utf-8"))

if metadata.get("source_channel_volume_flow_preserved") is not True:
    raise SystemExit("preflight does not preserve the source entrance-channel volume flow")
source_velocity = float(metadata["source_inlet_channel_velocity_m_s"])
pore_velocity = float(metadata["pore_opening_boundary_velocity_m_s"])
open_fraction = float(metadata["inlet_open_area_fraction"])
if not math.isclose(pore_velocity * open_fraction, source_velocity, rel_tol=1.0e-12):
    raise SystemExit("preflight pore velocity does not reproduce the source channel flow")
if marker.get("solver_finished") is not True or result.get("solver_finished") is not True:
    raise SystemExit("preflight solver did not finish")
if result.get("all_reported_values_are_finite") is not True:
    raise SystemExit("preflight contains a non-finite reported value")
mass = abs(float(physics["maximum_relative_mass_difference"]))
energy = abs(float(physics["maximum_relative_energy_difference"]))
if mass > 1.0e-6:
    raise SystemExit(f"preflight mass difference is too large: {mass:.6g}")
if energy > 1.0e-3:
    raise SystemExit(f"preflight energy difference is too large: {energy:.6g}")
pressure = float(result["flow"]["pressure_drop_Pa"])
temperatures = (
    float(result["temperature"]["outlet_average_K"]),
    float(result["temperature"]["solid_maximum_K"]),
)
if pressure <= 0.0 or any(value <= 0.0 for value in temperatures):
    raise SystemExit("preflight pressure drop or absolute temperature is non-positive")
if pressure_check.get("case_count") != 1:
    raise SystemExit("preflight pressure/flow reconstruction did not cover exactly one case")
superficial_error = abs(
    float(pressure_check["maximum_superficial_vs_source_channel_velocity_difference_fraction"])
)
boundary_pressure_error = abs(
    float(pressure_check["maximum_boundary_vs_reported_pressure_difference_fraction"])
)
if superficial_error > 1.0e-6:
    raise SystemExit(
        "resolved inlet flow does not reproduce the published source-channel velocity: "
        f"{superficial_error:.6g}"
    )
# The two pressure values are reconstructed from separately written boundary
# and summary fields. This is a numerical consistency tolerance, not a new
# packed-bed parameter or a physical acceptance limit.
if boundary_pressure_error > 1.0e-4:
    raise SystemExit(
        "boundary and reported pressure drops are inconsistent: "
        f"{boundary_pressure_error:.6g}"
    )

check = {
    "status": "corrected_source_flow_preflight_passed",
    "source_channel_velocity_m_s": source_velocity,
    "pore_opening_boundary_velocity_m_s": pore_velocity,
    "inlet_open_area_fraction": open_fraction,
    "relative_mass_difference": mass,
    "relative_energy_difference": energy,
    "pressure_drop_Pa": pressure,
    "resolved_superficial_velocity_relative_difference": superficial_error,
    "boundary_vs_reported_pressure_drop_relative_difference": boundary_pressure_error,
    "published_pressure_relation_absolute_difference_percent": float(
        pressure_check["maximum_absolute_difference_percent"]
    ),
    "published_pressure_relation_scope": (
        "whole-bed literature relation used as a physical comparison, not a fitted "
        "acceptance tolerance for the smaller wall-adjacent crop"
    ),
    "outlet_temperature_K": temperatures[0],
    "solid_maximum_temperature_K": temperatures[1],
}
(pathlib.Path(sys.argv[2]).parent / "preflight_pass.json").write_text(
    json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(check, ensure_ascii=False, indent=2))
PY

if formal_input_check_is_current; then
  printf '%s reusing unchanged verified formal inputs\n' \
    "$(date --iso-8601=seconds)" >> "${RESULT_DIR}/formal_60_input_check_stdout.json"
else
  "${PYTHON}" "${ROOT}/code/verify_hccb_p418_actual_case_inputs.py" \
    --matrix-root "${MATRIX_ROOT}" \
    --output "${FORMAL_INPUT_SUMMARY}" \
    --markdown-output "${RESULT_DIR}/formal_60_input_summary_CN.md" \
    > "${RESULT_DIR}/formal_60_input_check_stdout.json"
fi

printf '%s corrected preflight passed; starting formal 60-condition matrix\n' \
  "$(date --iso-8601=seconds)" > "${STATUS_FILE}"

ROOT="${ROOT}" MATRIX_ROOT="${MATRIX_ROOT}" NP_PER_CASE="${NP_PER_CASE}" \
  CONCURRENT_CASES="${CONCURRENT_CASES}" \
  bash "${ROOT}/code/run_hccb_dense_cht_p418_matrix_parallel.sh" \
  > "${LOG_FILE}" 2>&1

printf '%s formal 60-condition matrix finished\n' "$(date --iso-8601=seconds)" \
  > "${STATUS_FILE}"

printf '%s starting time-step, thermal-step, model and cross-packing calculations\n' \
  "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
ROOT="${ROOT}" P418_PYTHON="${PYTHON}" EXECUTE=1 \
  NP_PER_CASE="${NP_PER_CASE}" CONCURRENT_CASES="${CONCURRENT_CASES}" \
  bash "${ROOT}/code/run_hccb_p418_formal_calculations.sh" \
  > "${FORMAL_ROUTE_LOG}" 2>&1

printf '%s all declared heat-transfer calculations and manuscript refresh finished\n' \
  "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
