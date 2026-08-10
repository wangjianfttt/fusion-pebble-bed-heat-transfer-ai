#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-${SCRIPT_ROOT}}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
STEP_ROOT=${STEP_ROOT:-${ROOT}/hccb_p418_fully_coupled_steps_12}
RESULT_DIR=${RESULT_DIR:-${ROOT}/results/hccb_p418_fully_coupled_steps_12}
FIXED_RESULT_DIR=${FIXED_RESULT_DIR:-${ROOT}/results/hccb_p418_physical_steps_12}
PLAN=${PLAN:-${ROOT}/parameters/hccb_p418_fully_coupled_step_plan.json}
NP_PER_CASE=${NP_PER_CASE:-32}
CONCURRENT_CASES=${CONCURRENT_CASES:-1}
EXECUTE=${EXECUTE:-0}
ALLOW_PAUSED_WORKSTATION_RUN=${ALLOW_PAUSED_WORKSTATION_RUN:-0}
COMPARE_FIXED=${COMPARE_FIXED:-1}
REQUIRE_TIMESTEP_SENSITIVITY=${REQUIRE_TIMESTEP_SENSITIVITY:-1}
TIMESTEP_SUMMARY=${TIMESTEP_SUMMARY:-${ROOT}/results/hccb_p418_fully_coupled_timestep_sensitivity/fully_coupled_timestep_sensitivity.json}
TIMESTEP_CONFIG=${TIMESTEP_CONFIG:-${ROOT}/parameters/hccb_p418_fully_coupled_timestep_sensitivity.json}
REMOVE_PROCESSORS_AFTER_EXPORT=${REMOVE_PROCESSORS_AFTER_EXPORT:-1}
STEADY_DATASET_ROOT=${STEADY_DATASET_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3_dataset}
SHARED_TOPOLOGY=${SHARED_TOPOLOGY:-${STEADY_DATASET_ROOT}/shared_mesh_topology.npz}
STEADY_DATASET_INDEX=${STEADY_DATASET_INDEX:-${STEADY_DATASET_ROOT}/dataset_index.json}
SUBFACE_GEOMETRY=${SUBFACE_GEOMETRY:-${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${ROOT}/results/hccb_p418_regional_topology_r2/regional_topology.npz}
MODEL_GEOMETRY=${MODEL_GEOMETRY:-${ROOT}/results/hccb_p418_60_sourceflow_r3_model_geometry/model_geometry.npz}
PAUSE_MARKER=${PAUSE_MARKER:-${ROOT}/control/PAUSE_NEW_P418_CASES_FOR_CLOUD_MIGRATION}

if [[ ${EXECUTE} != 1 ]]; then
  case_count=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["sequences"]))' "${PLAN}")
  cat <<EOF
P418 fully coupled flow--heat step plan only; no OpenFOAM command was started.
matrix_root=${MATRIX_ROOT}
step_root=${STEP_ROOT}
result_dir=${RESULT_DIR}
plan=${PLAN}
cases=${case_count}
mpi_ranks_per_case=${NP_PER_CASE}
concurrent_cases=${CONCURRENT_CASES}
To run after all 60 steady endpoints are complete, set EXECUTE=1.
EOF
  exit 0
fi

if [[ -f ${PAUSE_MARKER} && ${ALLOW_PAUSED_WORKSTATION_RUN} != 1 ]]; then
  echo "new P418 calculations are paused for cloud migration: ${PAUSE_MARKER}" >&2
  exit 3
fi

if [[ ${REQUIRE_TIMESTEP_SENSITIVITY} == 1 ]]; then
  if [[ ! -f ${TIMESTEP_SUMMARY} ]]; then
    echo "fully coupled formal curves require the completed representative time-step study: ${TIMESTEP_SUMMARY}" >&2
    exit 4
  fi
  python3 "${ROOT}/code/verify_hccb_p418_fully_coupled_timestep_summary.py" \
    --summary "${TIMESTEP_SUMMARY}" \
    --config "${TIMESTEP_CONFIG}" \
    --plan "${PLAN}" \
    --output "${RESULT_DIR}/verified_timestep_selection.json"
fi

set +u
source /opt/openfoam13/etc/bashrc
set -u

python3 "${ROOT}/code/summarize_hccb_p418_step_endpoint_readiness.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --plan "${PLAN}" \
  --output-dir "${RESULT_DIR}/endpoint_readiness"
python3 - "${RESULT_DIR}/endpoint_readiness/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
if summary["ready_sequence_count"] != summary["sequence_count"]:
    raise SystemExit(
        f"P418 fully coupled steps require all {summary['sequence_count']} endpoint "
        f"pairs; only {summary['ready_sequence_count']} are complete"
    )
PY

python3 "${ROOT}/code/build_hccb_p418_fully_coupled_step_cases.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --output-root "${STEP_ROOT}" \
  --plan "${PLAN}" \
  --require-all-ready

latest_complete_parallel_time() {
  local case_dir=$1
  local candidate complete rank field
  [[ -d ${case_dir}/processor0 ]] || return 0
  while IFS= read -r candidate; do
    [[ ${candidate} =~ ^[0-9]+([.][0-9]+)?$ ]] || continue
    complete=1
    for ((rank = 0; rank < NP_PER_CASE; rank++)); do
      for field in fluid/T fluid/U fluid/p fluid/p_rgh fluid/phi solid/T uniform/time; do
        if [[ ! -f ${case_dir}/processor${rank}/${candidate}/${field} ]]; then
          complete=0
          break 2
        fi
      done
    done
    if [[ ${complete} -eq 1 ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(find "${case_dir}/processor0" -mindepth 1 -maxdepth 1 -type d \
    -printf '%f\n' | sort -gr)
}

time_greater_or_equal() {
  awk -v first="$1" -v second="$2" 'BEGIN { exit !((first + 0) >= (second + 0)) }'
}

run_one_step() {
  local case_dir=$1
  local sequence restart_time first_executed_stage
  sequence=$(basename "${case_dir}")
  if [[ -f ${case_dir}/fully_coupled_step_response_complete.json ]]; then
    echo "skip completed ${sequence}"
    return 0
  fi
  if [[ ! -f ${case_dir}/fully_coupled_initial_field_map_complete.json ]]; then
    ROOT="${ROOT}" bash "${ROOT}/code/initialize_hccb_p418_fully_coupled_step_case.sh" \
      "${case_dir}"
  fi
  python3 "${ROOT}/code/add_hccb_pressure_outputs.py" --case "${case_dir}"
  python3 "${ROOT}/code/add_hccb_transient_temperature_outputs.py" --case "${case_dir}"
  restart_time=$(latest_complete_parallel_time "${case_dir}")
  if [[ -n ${restart_time} && ${restart_time} != 0 ]]; then
    printf '\n### resumed from complete parallel time %s at %s ###\n' \
      "${restart_time}" "$(date --iso-8601=seconds)" \
      >> "${case_dir}/log.foamMultiRun.fully_coupled"
    echo "resume ${sequence} from ${restart_time} s"
  else
    restart_time=""
    rm -rf "${case_dir}"/processor*
    decomposePar -case "${case_dir}" -allRegions \
      > "${case_dir}/log.decomposePar.fully_coupled" 2>&1
    : > "${case_dir}/log.foamMultiRun.fully_coupled"
  fi
  local -a execution_stages
  mapfile -t execution_stages < <(python3 - "${case_dir}/fully_coupled_step_metadata.json" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["execution_schedule"]:
    print(row["start_s"], row["end_s"], row["delta_t_s"], row["write_interval_s"])
PY
  )
  local stage_index=0 stage start_time end_time delta_t write_interval
  first_executed_stage=1
  for stage in "${execution_stages[@]}"; do
    read -r start_time end_time delta_t write_interval <<< "${stage}"
    if [[ -n ${restart_time} ]] && time_greater_or_equal "${restart_time}" "${end_time}"; then
      echo "skip completed fully coupled stage ${stage_index}: ${start_time}--${end_time} s"
      stage_index=$((stage_index + 1))
      continue
    fi
    if [[ ${first_executed_stage} -eq 1 && -n ${restart_time} ]]; then
      foamDictionary "${case_dir}/system/controlDict" -entry startFrom -set startTime
      foamDictionary "${case_dir}/system/controlDict" -entry startTime -set "${restart_time}"
    elif [[ ${first_executed_stage} -eq 1 ]]; then
      foamDictionary "${case_dir}/system/controlDict" -entry startFrom -set startTime
      foamDictionary "${case_dir}/system/controlDict" -entry startTime -set "${start_time}"
    else
      foamDictionary "${case_dir}/system/controlDict" -entry startFrom -set latestTime
    fi
    foamDictionary "${case_dir}/system/controlDict" -entry endTime -set "${end_time}"
    foamDictionary "${case_dir}/system/controlDict" -entry deltaT -set "${delta_t}"
    foamDictionary "${case_dir}/system/controlDict" -entry writeControl -set runTime
    foamDictionary "${case_dir}/system/controlDict" -entry writeInterval -set "${write_interval}"
    printf '\n### fully coupled stage %s: %s--%s s, deltaT %s s, fields every %s s ###\n' \
      "${stage_index}" "${start_time}" "${end_time}" "${delta_t}" "${write_interval}" \
      >> "${case_dir}/log.foamMultiRun.fully_coupled"
    mpirun -np "${NP_PER_CASE}" foamMultiRun -case "${case_dir}" -parallel \
      >> "${case_dir}/log.foamMultiRun.fully_coupled" 2>&1
    first_executed_stage=0
    stage_index=$((stage_index + 1))
  done
  ROOT="${ROOT}" REMOVE_PROCESSORS_AFTER_EXPORT="${REMOVE_PROCESSORS_AFTER_EXPORT}" \
    bash "${ROOT}/code/finalize_hccb_p418_fully_coupled_step_response.sh" "${case_dir}" \
    > "${case_dir}/log.finalize.fully_coupled" 2>&1
  echo "completed ${sequence}"
}

export -f latest_complete_parallel_time time_greater_or_equal run_one_step
export ROOT NP_PER_CASE REMOVE_PROCESSORS_AFTER_EXPORT

find "${STEP_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 \
  | sort -z \
  | xargs -0 -n 1 -P "${CONCURRENT_CASES}" bash -c 'run_one_step "$1"' _

python3 "${ROOT}/code/export_hccb_p418_transient_observables.py" \
  --matrix-root "${STEP_ROOT}" \
  --output-dir "${RESULT_DIR}" \
  --history-kind fully_coupled_flow_heat_response

python3 "${ROOT}/code/export_hccb_p418_step_regional_sequences.py" \
  --step-root "${STEP_ROOT}" \
  --shared-topology "${SHARED_TOPOLOGY}" \
  --steady-dataset-index "${STEADY_DATASET_INDEX}" \
  --subface-geometry "${SUBFACE_GEOMETRY}" \
  --regional-topology "${REGIONAL_TOPOLOGY}" \
  --model-geometry "${MODEL_GEOMETRY}" \
  --output-dir "${RESULT_DIR}/regional_sequences" \
  --require-complete \
  --history-mode fully_coupled_flow_heat

if [[ ${COMPARE_FIXED} == 1 && -f ${FIXED_RESULT_DIR}/hccb_p418_transient_observables.npz ]]; then
  python3 "${ROOT}/code/compare_hccb_p418_fixed_and_fully_coupled_steps.py" \
    --fixed-observables "${FIXED_RESULT_DIR}/hccb_p418_transient_observables.npz" \
    --fully-coupled-observables "${RESULT_DIR}/hccb_p418_transient_observables.npz" \
    --output-dir "${RESULT_DIR}/fixed_vs_fully_coupled"
fi
