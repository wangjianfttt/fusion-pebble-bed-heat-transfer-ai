#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
P418_FIXED_STEP_PATCH_ROOT=${P418_FIXED_STEP_PATCH_ROOT:-${ROOT}}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
STEP_ROOT=${STEP_ROOT:-${ROOT}/hccb_p418_physical_steps_12}
RESULT_DIR=${RESULT_DIR:-${ROOT}/results/hccb_p418_physical_steps_12}
NP_PER_CASE=${NP_PER_CASE:-32}
CONCURRENT_CASES=${CONCURRENT_CASES:-3}
PLAN=${PLAN:-${ROOT}/parameters/hccb_p418_transient_step_plan.json}
RUN_MODEL_TRAINING=${RUN_MODEL_TRAINING:-1}
RUN_REGIONAL_PIPELINE=${RUN_REGIONAL_PIPELINE:-1}
ENDPOINT_READINESS_MODE=${ENDPOINT_READINESS_MODE:-full_dataset}
RUN_DATA_ONLY_ABLATION=${RUN_DATA_ONLY_ABLATION:-1}
RUN_DIFFUSION_BENCHMARK=${RUN_DIFFUSION_BENCHMARK:-1}
DIFFUSION_DEVICE=${DIFFUSION_DEVICE:-cuda}
DIFFUSION_MICROBATCH_SIZE=${DIFFUSION_MICROBATCH_SIZE:-1}
DIFFUSION_ACTIVATION_PRECISION=${DIFFUSION_ACTIVATION_PRECISION:-bfloat16}
ENERGY_BALANCE_DEVICE=${ENERGY_BALANCE_DEVICE:-cpu}
TEMPERATURE_OUTPUT_MODE=${TEMPERATURE_OUTPUT_MODE:-literature_bounded_residual}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF
REQUIRE_GRAPH_CUDA=${REQUIRE_GRAPH_CUDA:-1}
GRAPH_GPU_MEASURED_SUMMARY=${GRAPH_GPU_MEASURED_SUMMARY:-${ROOT}/results/hccb_p418_actual_spatiotemporal_operator_56time_gpu_factorized/summary.json}
GRAPH_GPU_READY_RECORD=${GRAPH_GPU_READY_RECORD:-${RESULT_DIR}/graph_gpu_ready.json}
STEP_SPLIT_NAMES=${STEP_SPLIT_NAMES:-"direction_down_test direction_up_test pair_disjoint_stress_test"}
PRIMARY_MODEL_SEED=${PRIMARY_MODEL_SEED:-20260717}
RUN_SEED_ROBUSTNESS=${RUN_SEED_ROBUSTNESS:-1}
ROBUSTNESS_SPLIT=${ROBUSTNESS_SPLIT:-pair_disjoint_stress_test}
ROBUSTNESS_MODEL_SEEDS=${ROBUSTNESS_MODEL_SEEDS:-"20260717 20260718 20260719"}
STEADY_DATASET_ROOT=${STEADY_DATASET_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3_dataset}
SHARED_TOPOLOGY=${SHARED_TOPOLOGY:-${STEADY_DATASET_ROOT}/shared_mesh_topology.npz}
STEADY_DATASET_INDEX=${STEADY_DATASET_INDEX:-${STEADY_DATASET_ROOT}/dataset_index.json}
STEADY_POSTPROCESS_SUMMARY=${STEADY_POSTPROCESS_SUMMARY:-${ROOT}/results/hccb_p418_60_sourceflow_r3_postprocess_summary.json}
SUBFACE_GEOMETRY=${SUBFACE_GEOMETRY:-${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${ROOT}/results/hccb_p418_regional_topology_r2/regional_topology.npz}
MODEL_GEOMETRY=${MODEL_GEOMETRY:-${ROOT}/results/hccb_p418_60_sourceflow_r3_model_geometry/model_geometry.npz}
RESUME_PARALLEL_HISTORY_ROOT=${RESUME_PARALLEL_HISTORY_ROOT:-}
RESUME_THROUGH_TIME_S=${RESUME_THROUGH_TIME_S:-1}

OPENFOAM_BASHRC=${OPENFOAM_BASHRC:-/opt/openfoam13/etc/bashrc}
if [[ -f ${OPENFOAM_BASHRC} ]]; then
  set +u
  source "${OPENFOAM_BASHRC}"
  set -u
fi

if [[ ${ENDPOINT_READINESS_MODE} == full_dataset ]]; then
  python3 "${ROOT}/code/summarize_hccb_p418_step_endpoint_readiness.py" \
    --matrix-root "${MATRIX_ROOT}" \
    --plan "${PLAN}" \
    --output-dir "${RESULT_DIR}/endpoint_readiness"
  python3 - "${RESULT_DIR}/endpoint_readiness/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
if summary["ready_sequence_count"] != summary["sequence_count"]:
    raise SystemExit(
        f"P418 thermal steps require all {summary['sequence_count']} endpoint pairs; "
        f"only {summary['ready_sequence_count']} are complete"
    )
PY
elif [[ ${ENDPOINT_READINESS_MODE} == transient_endpoint_fields ]]; then
  python3 "${ROOT}/code/verify_hccb_p418_transient_endpoint_fields.py" \
    --matrix-root "${MATRIX_ROOT}" \
    --plan "${PLAN}" \
    --output "${RESULT_DIR}/endpoint_readiness/transient_endpoint_fields.json"
else
  echo "未知端点检查方式: ${ENDPOINT_READINESS_MODE}" >&2
  exit 2
fi

python3 "${P418_FIXED_STEP_PATCH_ROOT}/code/build_hccb_p418_step_response_cases.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --output-root "${STEP_ROOT}" \
  --plan "${PLAN}" \
  --require-all-ready

initialise_step_case() {
  local case_dir=$1
  local metadata=${case_dir}/step_case_metadata.json
  local source_case target_case source_time target_time target_u target_T sequence
  sequence=$(basename "${case_dir}")
  source_case=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_case"])' "${metadata}")
  target_case=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_case"])' "${metadata}")
  source_time=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_final_time_s"])' "${metadata}")
  target_time=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_final_time_s"])' "${metadata}")
  target_u=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["target_parameters"].get("pore_opening_boundary_velocity_m_s", d["target_parameters"]["inlet_velocity_m_s"]))' "${metadata}")
  target_T=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_parameters"]["inlet_temperature_K"])' "${metadata}")

  if [[ ! -f ${case_dir}/initial_field_map_complete.json ]]; then
    for field in U p p_rgh phi; do
      cp "${target_case}/${target_time}/fluid/${field}" "${case_dir}/0/fluid/${field}"
    done
    cp "${source_case}/${source_time}/fluid/T" "${case_dir}/0/fluid/T"
    cp "${source_case}/${source_time}/solid/T" "${case_dir}/0/solid/T"
    for field in U T p p_rgh phi; do
      foamDictionary "${case_dir}/0/fluid/${field}" -writePrecision 10 \
        -entry 'FoamFile/location' -set '"0/fluid"'
    done
    foamDictionary "${case_dir}/0/solid/T" -writePrecision 10 \
      -entry 'FoamFile/location' -set '"0/solid"'
    foamDictionary "${case_dir}/0/fluid/U" \
      -writePrecision 10 -entry 'boundaryField/inlet/value' -set "uniform (0 0 ${target_u})"
    foamDictionary "${case_dir}/0/fluid/T" \
      -writePrecision 10 -entry 'boundaryField/inlet/value' -set "uniform ${target_T}"
    foamDictionary "${case_dir}/0/fluid/T" \
      -writePrecision 10 -entry 'boundaryField/outlet/inletValue' -set "uniform ${target_T}"
    python3 "${ROOT}/code/verify_hccb_p418_step_initialization.py" \
      --case "${case_dir}" --write-record
    echo "initialized ${sequence}: target hydrodynamics and source temperatures"
  fi
}

import_parallel_history_if_requested() {
  local case_dir=$1
  local sequence source_case processor_count
  sequence=$(basename "${case_dir}")
  [[ -n ${RESUME_PARALLEL_HISTORY_ROOT} ]] || return 0
  if [[ -f ${case_dir}/parallel_history_import_complete.json ]]; then
    return 0
  fi
  source_case="${RESUME_PARALLEL_HISTORY_ROOT}/by_sequence/${sequence}/steps/${sequence}"
  if [[ ! -d ${source_case} ]]; then
    source_case="${RESUME_PARALLEL_HISTORY_ROOT}/${sequence}"
  fi
  [[ -d ${source_case} ]] || return 0
  if [[ -d ${case_dir}/processor0 ]]; then
    echo "${sequence}: refusing to import history into an already decomposed case" >&2
    return 1
  fi
  foamDictionary "${case_dir}/system/decomposeParDict" \
    -entry numberOfSubdomains -set "${NP_PER_CASE}"
  decomposePar -case "${case_dir}" -allRegions > "${case_dir}/log.decomposePar" 2>&1
  processor_count=$(find "${case_dir}" -maxdepth 1 -type d -name 'processor*' | wc -l)
  if [[ ${processor_count} -ne ${NP_PER_CASE} ]]; then
    echo "${sequence}: decomposePar produced ${processor_count} processor directories; expected ${NP_PER_CASE}" >&2
    return 1
  fi
  python3 "${P418_FIXED_STEP_PATCH_ROOT}/code/import_hccb_p418_parallel_history.py" \
    --source-case "${source_case}" \
    --destination-case "${case_dir}" \
    --through-time "${RESUME_THROUGH_TIME_S}" \
    --mpi-tasks "${NP_PER_CASE}"
  echo "imported ${sequence} history through ${RESUME_THROUGH_TIME_S} s"
}

latest_complete_parallel_time() {
  local case_dir=$1
  local candidate complete rank field require_time_record
  [[ -d ${case_dir}/processor0 ]] || return 0
  while IFS= read -r candidate; do
    [[ ${candidate} =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || continue
    require_time_record=$(awk -v value="${candidate}" \
      'BEGIN { print ((value + 0.0) == 0.0) ? 0 : 1 }')
    complete=1
    for ((rank = 0; rank < NP_PER_CASE; rank++)); do
      for field in fluid/T fluid/U fluid/p fluid/p_rgh fluid/phi solid/T; do
        if [[ ! -f ${case_dir}/processor${rank}/${candidate}/${field} ]]; then
          complete=0
          break 2
        fi
      done
      if [[ ${require_time_record} -eq 1 ]] && \
        [[ ! -f ${case_dir}/processor${rank}/${candidate}/uniform/time ]]; then
        complete=0
        break
      fi
    done
    if [[ ${complete} -eq 1 ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(
    for candidate_path in "${case_dir}"/processor0/*; do
      [[ -d ${candidate_path} ]] || continue
      basename "${candidate_path}"
    done | sort -gr
  )
}

time_greater_or_equal() {
  awk -v first="$1" -v second="$2" 'BEGIN { exit !((first + 0) >= (second + 0)) }'
}

integer_write_steps() {
  awk -v interval="$1" -v delta="$2" '
    BEGIN {
      raw = interval / delta
      rounded = int(raw + 0.5)
      error = raw - rounded
      if (error < 0) error = -error
      if (rounded < 1 || error > 1.0e-8) exit 1
      print rounded
    }
  '
}

parallel_common_time_index() {
  local case_dir=$1
  local time_name=$2
  local rank time_file index common_index="" missing_count=0
  for ((rank = 0; rank < NP_PER_CASE; rank++)); do
    time_file="${case_dir}/processor${rank}/${time_name}/uniform/time"
    if [[ ! -f ${time_file} ]]; then
      missing_count=$((missing_count + 1))
      continue
    fi
    index=$(awk '
      $1 == "index" {
        gsub(/;/, "", $2)
        print $2
        exit
      }
    ' "${time_file}")
    if [[ ! ${index} =~ ^[0-9]+$ ]]; then
      echo "invalid OpenFOAM time index in ${time_file}: ${index}" >&2
      return 1
    fi
    if [[ -z ${common_index} ]]; then
      common_index=${index}
    elif [[ ${index} != "${common_index}" ]]; then
      echo "inconsistent OpenFOAM time indices at ${time_name}: ${common_index} and ${index}" >&2
      return 1
    fi
  done
  if [[ ${missing_count} -eq ${NP_PER_CASE} ]] && \
    awk -v value="${time_name}" 'BEGIN { exit !((value + 0.0) == 0.0) }'; then
    printf '0\n'
    return 0
  fi
  if [[ ${missing_count} -ne 0 ]]; then
    echo "OpenFOAM time records are incomplete at ${time_name}: " \
      "${missing_count}/${NP_PER_CASE} missing" >&2
    return 1
  fi
  printf '%s\n' "${common_index}"
}

recover_incomplete_stage_end() {
  local case_dir=$1
  local sequence=$2
  local stage_index=$3
  local end_time=$4
  local delta_t=$5
  local complete_time recovery_interval recovery_write_steps
  local recovery_start_index recovery_target_index
  complete_time=$(latest_complete_parallel_time "${case_dir}")
  if [[ -n ${complete_time} ]] && time_greater_or_equal "${complete_time}" "${end_time}"; then
    return 0
  fi
  if [[ -z ${complete_time} ]]; then
    echo "${sequence}: no complete parallel time is available after stage ${stage_index}" >&2
    return 1
  fi
  recovery_interval=$(awk -v end="${end_time}" -v start="${complete_time}" \
    'BEGIN { value=end-start; if (value <= 0) exit 1; printf "%.12g\n", value }')
  recovery_write_steps=$(integer_write_steps "${recovery_interval}" "${delta_t}")
  recovery_start_index=$(parallel_common_time_index "${case_dir}" "${complete_time}")
  recovery_target_index=$((recovery_start_index + recovery_write_steps))
  printf '\n### completing stage %s from common time %s to %s s ###\n' \
    "${stage_index}" "${complete_time}" "${end_time}" \
    >> "${case_dir}/log.foamMultiRun.step"
  foamDictionary "${case_dir}/system/controlDict" -entry startFrom -set startTime
  foamDictionary "${case_dir}/system/controlDict" -entry startTime -set "${complete_time}"
  foamDictionary "${case_dir}/system/controlDict" -entry endTime -set "${end_time}"
  foamDictionary "${case_dir}/system/controlDict" -entry deltaT -set "${delta_t}"
  foamDictionary "${case_dir}/system/controlDict" -entry writeControl -set timeStep
  foamDictionary "${case_dir}/system/controlDict" \
    -entry writeInterval -set "${recovery_target_index}"
  if ! mpirun -np "${NP_PER_CASE}" foamMultiRun -case "${case_dir}" -parallel \
    >> "${case_dir}/log.foamMultiRun.step" 2>&1; then
    echo "${sequence}: stage ${stage_index} completion run failed" >&2
    return 1
  fi
  complete_time=$(latest_complete_parallel_time "${case_dir}")
  if [[ -z ${complete_time} ]] || ! time_greater_or_equal "${complete_time}" "${end_time}"; then
    echo "${sequence}: stage ${stage_index} did not produce a complete ${end_time} s state" >&2
    return 1
  fi
}

run_one_step() {
  local case_dir=$1
  local sequence restart_time first_executed_stage processor_count
  local active_write_interval active_write_steps
  local write_source_time write_start_index write_target_index
  sequence=$(basename "${case_dir}")
  if [[ -f ${case_dir}/step_response_complete.json ]]; then
    echo "skip completed ${sequence}"
    return 0
  fi
  initialise_step_case "${case_dir}"
  python3 "${ROOT}/code/add_hccb_pressure_outputs.py" --case "${case_dir}"
  python3 "${P418_FIXED_STEP_PATCH_ROOT}/code/add_hccb_transient_temperature_outputs.py" \
    --case "${case_dir}"
  import_parallel_history_if_requested "${case_dir}"
  restart_time=$(latest_complete_parallel_time "${case_dir}")
  if [[ -n ${restart_time} && ${restart_time} != 0 ]]; then
    printf '\n### resumed from complete parallel time %s at %s ###\n' \
      "${restart_time}" "$(date --iso-8601=seconds)" \
      >> "${case_dir}/log.foamMultiRun.step"
    echo "resume ${sequence} from ${restart_time} s"
  else
    restart_time=""
    rm -rf "${case_dir}"/processor*
    foamDictionary "${case_dir}/system/decomposeParDict" \
      -entry numberOfSubdomains -set "${NP_PER_CASE}"
    decomposePar -case "${case_dir}" -allRegions > "${case_dir}/log.decomposePar" 2>&1
    processor_count=$(find "${case_dir}" -maxdepth 1 -type d -name 'processor*' | wc -l)
    if [[ ${processor_count} -ne ${NP_PER_CASE} ]]; then
      echo "${sequence}: decomposePar produced ${processor_count} processor directories; expected ${NP_PER_CASE}" >&2
      exit 2
    fi
    : > "${case_dir}/log.foamMultiRun.step"
  fi
  local -a execution_stages
  mapfile -t execution_stages < <(python3 - "${case_dir}/step_case_metadata.json" <<'PY'
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
      echo "skip completed transient stage ${stage_index}: ${start_time}--${end_time} s"
      stage_index=$((stage_index + 1))
      continue
    fi
    if [[ ${first_executed_stage} -eq 1 && -n ${restart_time} ]]; then
      foamDictionary "${case_dir}/system/controlDict" -entry startFrom -set startTime
      foamDictionary "${case_dir}/system/controlDict" -entry startTime -set "${restart_time}"
      active_write_interval=$(python3 \
        "${ROOT}/code/compute_hccb_resume_write_interval.py" \
        --stage-start "${start_time}" \
        --restart-time "${restart_time}" \
        --planned-interval "${write_interval}")
    elif [[ ${first_executed_stage} -eq 1 ]]; then
      foamDictionary "${case_dir}/system/controlDict" -entry startFrom -set startTime
      foamDictionary "${case_dir}/system/controlDict" -entry startTime -set "${start_time}"
      active_write_interval=${write_interval}
    else
      foamDictionary "${case_dir}/system/controlDict" -entry startFrom -set latestTime
      active_write_interval=${write_interval}
    fi
    foamDictionary "${case_dir}/system/controlDict" -entry endTime -set "${end_time}"
    foamDictionary "${case_dir}/system/controlDict" -entry deltaT -set "${delta_t}"
    active_write_steps=$(integer_write_steps "${active_write_interval}" "${delta_t}")
    write_source_time=$(latest_complete_parallel_time "${case_dir}")
    if [[ -z ${write_source_time} ]]; then
      echo "${sequence}: no complete parallel source time is available for stage ${stage_index}" >&2
      return 1
    fi
    write_start_index=$(parallel_common_time_index "${case_dir}" "${write_source_time}")
    write_target_index=$((write_start_index + active_write_steps))
    foamDictionary "${case_dir}/system/controlDict" -entry writeControl -set timeStep
    foamDictionary "${case_dir}/system/controlDict" \
      -entry writeInterval -set "${write_target_index}"
    printf '\n### transient stage %s: %s--%s s, deltaT %s s, full field at target index %s ###\n' \
      "${stage_index}" "${start_time}" "${end_time}" "${delta_t}" \
      "${write_target_index}" \
      >> "${case_dir}/log.foamMultiRun.step"
    if ! mpirun -np "${NP_PER_CASE}" foamMultiRun -case "${case_dir}" -parallel \
      >> "${case_dir}/log.foamMultiRun.step" 2>&1; then
      echo "${sequence}: foamMultiRun failed during stage ${stage_index}" >&2
      return 1
    fi
    recover_incomplete_stage_end \
      "${case_dir}" "${sequence}" "${stage_index}" "${end_time}" "${delta_t}" \
      || return 1
    restart_time=$(latest_complete_parallel_time "${case_dir}")
    first_executed_stage=0
    stage_index=$((stage_index + 1))
  done
  ROOT="${ROOT}" OPENFOAM_BASHRC="${OPENFOAM_BASHRC}" \
    bash "${ROOT}/code/finalize_hccb_p418_step_response.sh" "${case_dir}" \
    > "${case_dir}/log.finalize.step" 2>&1
  echo "completed ${sequence}"
}

export -f initialise_step_case latest_complete_parallel_time time_greater_or_equal
export -f import_parallel_history_if_requested
export -f integer_write_steps parallel_common_time_index
export -f recover_incomplete_stage_end run_one_step
export ROOT MATRIX_ROOT STEP_ROOT RESULT_DIR NP_PER_CASE
export P418_FIXED_STEP_PATCH_ROOT
export RESUME_PARALLEL_HISTORY_ROOT RESUME_THROUGH_TIME_S

find "${STEP_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 \
  | sort -z \
  | xargs -0 -n 1 -P "${CONCURRENT_CASES}" \
      bash -euo pipefail -c 'run_one_step "$1"' _

if [[ ${RUN_REGIONAL_PIPELINE} == 0 ]]; then
  python3 "${ROOT}/code/export_hccb_p418_transient_observables.py" \
    --matrix-root "${STEP_ROOT}" \
    --output-dir "${RESULT_DIR}" \
    --history-kind physical_step_response
  echo "completed OpenFOAM step responses and observable export; regional/model pipeline skipped"
  exit 0
fi

python3 "${ROOT}/code/validate_hccb_p418_steady_dataset_ready.py" \
  --postprocess-summary "${STEADY_POSTPROCESS_SUMMARY}" \
  --dataset-index "${STEADY_DATASET_INDEX}" \
  --shared-topology "${SHARED_TOPOLOGY}" \
  --expected-cases 60

python3 "${ROOT}/code/export_hccb_p418_transient_observables.py" \
  --matrix-root "${STEP_ROOT}" \
  --output-dir "${RESULT_DIR}" \
  --history-kind physical_step_response

python3 "${ROOT}/code/export_hccb_p418_step_regional_sequences.py" \
  --step-root "${STEP_ROOT}" \
  --shared-topology "${SHARED_TOPOLOGY}" \
  --steady-dataset-index "${STEADY_DATASET_INDEX}" \
  --subface-geometry "${SUBFACE_GEOMETRY}" \
  --regional-topology "${REGIONAL_TOPOLOGY}" \
  --model-geometry "${MODEL_GEOMETRY}" \
  --output-dir "${RESULT_DIR}/regional_sequences" \
  --require-complete

python3 "${ROOT}/code/summarize_hccb_p418_transition_temperature_coverage.py" \
  --dataset-index "${RESULT_DIR}/regional_sequences/dataset_index.json" \
  --output-dir "${RESULT_DIR}/transition_temperature_coverage" \
  --latex-output "${ROOT}/manuscript/generated_transition_temperature_coverage.tex"

if [[ ${RUN_MODEL_TRAINING} == 1 && ${REQUIRE_GRAPH_CUDA} == 1 ]]; then
  python3 "${ROOT}/code/check_hccb_p418_gpu_training_ready.py" \
    --measured-summary "${GRAPH_GPU_MEASURED_SUMMARY}" \
    --output "${GRAPH_GPU_READY_RECORD}"
fi

if [[ ${RUN_MODEL_TRAINING} == 1 ]]; then
  for split_name in ${STEP_SPLIT_NAMES}; do
    python3 "${ROOT}/code/train_hccb_p418_transient_observable_transformer.py" \
      --data "${RESULT_DIR}/hccb_p418_transient_observables.npz" \
      --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
      --split-name "${split_name}" \
      --output-dir "${RESULT_DIR}/transformer_${split_name}" \
      --run-role formal \
      --history-kind physical_step_response \
      --seed "${PRIMARY_MODEL_SEED}"
    python3 "${ROOT}/code/train_hccb_p418_regional_dmdc.py" \
      --dataset-index "${RESULT_DIR}/regional_sequences/dataset_index.json" \
      --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
      --split-name "${split_name}" \
      --output-dir "${RESULT_DIR}/regional_dmdc_${split_name}"
    if [[ ${RUN_DATA_ONLY_ABLATION} == 1 ]]; then
      python3 "${ROOT}/code/train_hccb_p418_spatiotemporal_regional_operator.py" \
        --dataset-index "${RESULT_DIR}/regional_sequences/dataset_index.json" \
        --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
        --split-name "${split_name}" \
        --residual-geometry "${SUBFACE_GEOMETRY}" \
        --output-dir "${RESULT_DIR}/regional_graph_transformer_bounded_data_only_${split_name}" \
        --run-role formal_data_only \
        --physics-mode data_only \
        --temperature-output-mode "${TEMPERATURE_OUTPUT_MODE}" \
        --seed "${PRIMARY_MODEL_SEED}" \
        --resume
    fi
    python3 "${ROOT}/code/train_hccb_p418_spatiotemporal_regional_operator.py" \
      --dataset-index "${RESULT_DIR}/regional_sequences/dataset_index.json" \
      --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
      --split-name "${split_name}" \
      --residual-geometry "${SUBFACE_GEOMETRY}" \
      --output-dir "${RESULT_DIR}/regional_graph_transformer_bounded_physics_${split_name}" \
      --run-role formal \
      --physics-mode energy_and_flux \
      --temperature-output-mode "${TEMPERATURE_OUTPUT_MODE}" \
      --seed "${PRIMARY_MODEL_SEED}" \
      --resume
    python3 "${ROOT}/code/train_hccb_p418_spatiotemporal_regional_operator.py" \
      --dataset-index "${RESULT_DIR}/regional_sequences/dataset_index.json" \
      --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
      --split-name "${split_name}" \
      --residual-geometry "${SUBFACE_GEOMETRY}" \
      --output-dir "${RESULT_DIR}/regional_graph_transformer_bounded_factorized_${split_name}" \
      --run-role formal_factorized \
      --physics-mode energy_and_flux \
      --temperature-output-mode "${TEMPERATURE_OUTPUT_MODE}" \
      --spatial-temporal-mode factorized_static_spatial \
      --seed "${PRIMARY_MODEL_SEED}" \
      --resume
    python3 "${ROOT}/code/train_hccb_p418_low_rank_temperature_residual.py" \
      --prediction-dir "${RESULT_DIR}/regional_graph_transformer_bounded_physics_${split_name}" \
      --output-dir "${RESULT_DIR}/low_rank_temperature_residual_${split_name}" \
      --split-name "${split_name}" \
      --run-role formal
    if [[ ${RUN_DIFFUSION_BENCHMARK} == 1 ]]; then
      python3 "${ROOT}/code/train_hccb_p418_temporal_temperature_diffusion.py" \
        --prediction-dir "${RESULT_DIR}/regional_graph_transformer_bounded_physics_${split_name}" \
        --residual-geometry "${SUBFACE_GEOMETRY}" \
        --output-dir "${RESULT_DIR}/temporal_diffusion_${split_name}" \
        --run-role computed_residual_benchmark \
        --microbatch-size "${DIFFUSION_MICROBATCH_SIZE}" \
        --activation-precision "${DIFFUSION_ACTIVATION_PRECISION}" \
        --device "${DIFFUSION_DEVICE}" \
        --seed "${PRIMARY_MODEL_SEED}" \
        --resume
    fi
    energy_model_dirs=(
      "regional_dmdc_${split_name}"
      "regional_graph_transformer_bounded_physics_${split_name}"
      "regional_graph_transformer_bounded_factorized_${split_name}"
      "low_rank_temperature_residual_${split_name}"
    )
    if [[ ${RUN_DATA_ONLY_ABLATION} == 1 ]]; then
      energy_model_dirs+=("regional_graph_transformer_bounded_data_only_${split_name}")
    fi
    if [[ ${RUN_DIFFUSION_BENCHMARK} == 1 ]]; then
      energy_model_dirs+=("temporal_diffusion_${split_name}")
    fi
    for model_dir in "${energy_model_dirs[@]}"; do
      python3 "${ROOT}/code/evaluate_hccb_p418_temporal_energy_balance.py" \
        --model-summary "${RESULT_DIR}/${model_dir}/summary.json" \
        --dataset-index "${RESULT_DIR}/regional_sequences/dataset_index.json" \
        --residual-geometry "${SUBFACE_GEOMETRY}" \
        --output "${RESULT_DIR}/${model_dir}/energy_balance_summary.json" \
        --device "${ENERGY_BALANCE_DEVICE}"
    done
  done
  if [[ ${RUN_SEED_ROBUSTNESS} == 1 ]]; then
    for seed in ${ROBUSTNESS_MODEL_SEEDS}; do
      if [[ ${seed} == "${PRIMARY_MODEL_SEED}" ]]; then
        continue
      fi
      suffix=_seed${seed}
      python3 "${ROOT}/code/train_hccb_p418_transient_observable_transformer.py" \
        --data "${RESULT_DIR}/hccb_p418_transient_observables.npz" \
        --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
        --split-name "${ROBUSTNESS_SPLIT}" \
        --output-dir "${RESULT_DIR}/transformer_${ROBUSTNESS_SPLIT}${suffix}" \
        --run-role formal \
        --history-kind physical_step_response \
        --seed "${seed}"
      if [[ ${RUN_DATA_ONLY_ABLATION} == 1 ]]; then
        python3 "${ROOT}/code/train_hccb_p418_spatiotemporal_regional_operator.py" \
          --dataset-index "${RESULT_DIR}/regional_sequences/dataset_index.json" \
          --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
          --split-name "${ROBUSTNESS_SPLIT}" \
          --residual-geometry "${SUBFACE_GEOMETRY}" \
          --output-dir "${RESULT_DIR}/regional_graph_transformer_bounded_data_only_${ROBUSTNESS_SPLIT}${suffix}" \
          --run-role formal_data_only \
          --physics-mode data_only \
          --temperature-output-mode "${TEMPERATURE_OUTPUT_MODE}" \
          --seed "${seed}" \
          --resume
      fi
      python3 "${ROOT}/code/train_hccb_p418_spatiotemporal_regional_operator.py" \
        --dataset-index "${RESULT_DIR}/regional_sequences/dataset_index.json" \
        --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
        --split-name "${ROBUSTNESS_SPLIT}" \
        --residual-geometry "${SUBFACE_GEOMETRY}" \
        --output-dir "${RESULT_DIR}/regional_graph_transformer_bounded_physics_${ROBUSTNESS_SPLIT}${suffix}" \
        --run-role formal \
        --physics-mode energy_and_flux \
        --temperature-output-mode "${TEMPERATURE_OUTPUT_MODE}" \
        --seed "${seed}" \
        --resume
      python3 "${ROOT}/code/train_hccb_p418_low_rank_temperature_residual.py" \
        --prediction-dir "${RESULT_DIR}/regional_graph_transformer_bounded_physics_${ROBUSTNESS_SPLIT}${suffix}" \
        --output-dir "${RESULT_DIR}/low_rank_temperature_residual_${ROBUSTNESS_SPLIT}${suffix}" \
        --split-name "${ROBUSTNESS_SPLIT}" \
        --run-role formal
      if [[ ${RUN_DIFFUSION_BENCHMARK} == 1 ]]; then
        python3 "${ROOT}/code/train_hccb_p418_temporal_temperature_diffusion.py" \
          --prediction-dir "${RESULT_DIR}/regional_graph_transformer_bounded_physics_${ROBUSTNESS_SPLIT}${suffix}" \
          --residual-geometry "${SUBFACE_GEOMETRY}" \
          --output-dir "${RESULT_DIR}/temporal_diffusion_${ROBUSTNESS_SPLIT}${suffix}" \
          --run-role computed_residual_benchmark \
          --microbatch-size "${DIFFUSION_MICROBATCH_SIZE}" \
          --activation-precision "${DIFFUSION_ACTIVATION_PRECISION}" \
          --device "${DIFFUSION_DEVICE}" \
          --seed "${seed}" \
          --resume
      fi
      robustness_energy_dirs=(
        "regional_graph_transformer_bounded_physics_${ROBUSTNESS_SPLIT}${suffix}"
        "low_rank_temperature_residual_${ROBUSTNESS_SPLIT}${suffix}"
      )
      if [[ ${RUN_DATA_ONLY_ABLATION} == 1 ]]; then
        robustness_energy_dirs+=(
          "regional_graph_transformer_bounded_data_only_${ROBUSTNESS_SPLIT}${suffix}"
        )
      fi
      if [[ ${RUN_DIFFUSION_BENCHMARK} == 1 ]]; then
        robustness_energy_dirs+=(
          "temporal_diffusion_${ROBUSTNESS_SPLIT}${suffix}"
        )
      fi
      for model_dir in "${robustness_energy_dirs[@]}"; do
        python3 "${ROOT}/code/evaluate_hccb_p418_temporal_energy_balance.py" \
          --model-summary "${RESULT_DIR}/${model_dir}/summary.json" \
          --dataset-index "${RESULT_DIR}/regional_sequences/dataset_index.json" \
          --residual-geometry "${SUBFACE_GEOMETRY}" \
          --output "${RESULT_DIR}/${model_dir}/energy_balance_summary.json" \
          --device "${ENERGY_BALANCE_DEVICE}"
      done
    done
  fi
  if [[ ${RUN_DATA_ONLY_ABLATION} == 1 && ${RUN_DIFFUSION_BENCHMARK} == 1 ]]; then
    seed_robustness_args=()
    if [[ ${RUN_SEED_ROBUSTNESS} == 1 ]]; then
      python3 "${ROOT}/code/summarize_hccb_p418_step_seed_robustness.py" \
        --result-dir "${RESULT_DIR}" \
        --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
        --split-name "${ROBUSTNESS_SPLIT}" \
        --primary-seed "${PRIMARY_MODEL_SEED}" \
        --seeds ${ROBUSTNESS_MODEL_SEEDS} \
        --output-dir "${RESULT_DIR}/seed_robustness_${ROBUSTNESS_SPLIT}"
      seed_robustness_args=(
        --seed-robustness-summary
        "${RESULT_DIR}/seed_robustness_${ROBUSTNESS_SPLIT}/summary.json"
      )
    fi
    python3 "${ROOT}/code/summarize_hccb_p418_step_model_comparison.py" \
      --result-dir "${RESULT_DIR}" \
      --step-root "${STEP_ROOT}" \
      --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
      --split-names ${STEP_SPLIT_NAMES} \
      "${seed_robustness_args[@]}" \
      --output-dir "${RESULT_DIR}/model_comparison"
    python3 "${ROOT}/code/build_hccb_p418_transient_performance_table.py" \
      --metrics-csv "${RESULT_DIR}/model_comparison/physical_step_model_metrics.csv" \
      --output "${ROOT}/manuscript/generated_transient_performance.tex" \
      --summary "${RESULT_DIR}/model_comparison/transient_performance_table.json"
    python3 "${ROOT}/code/build_hccb_p418_transient_cost_table.py" \
      --speed-csv "${RESULT_DIR}/model_comparison/physical_step_model_speedup.csv" \
      --output "${ROOT}/manuscript/generated_transient_cost.tex" \
      --summary "${RESULT_DIR}/model_comparison/transient_cost_table.json"
    python3 "${ROOT}/code/build_hccb_p418_transient_result_text.py" \
      --summary "${RESULT_DIR}/model_comparison/summary.json" \
      --metrics "${RESULT_DIR}/model_comparison/physical_step_model_metrics.csv" \
      --cost-summary "${RESULT_DIR}/model_comparison/transient_cost_table.json" \
      --output "${ROOT}/manuscript/generated_transient_result_text.tex"
    python3 "${ROOT}/code/plot_hccb_p418_transient_model_comparison.py" \
      --result-dir "${RESULT_DIR}" \
      --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
      --output-dir "${ROOT}/figures"
  fi
fi
