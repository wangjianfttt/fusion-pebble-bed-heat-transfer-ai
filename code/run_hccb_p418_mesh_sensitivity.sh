#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results/hccb_p418_three_mesh_cht_sensitivity}
CONDITION_ID=${CONDITION_ID:-u0p20_T700_q6p85}
END_TIME=${END_TIME:-200}
FINE_CASE=${FINE_CASE:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3/${CONDITION_ID}}

source /opt/openfoam13/etc/bashrc || true

declare -A MESH_CASES=(
  [coarse]="${ROOT}/hccb_dense_snappy_g2_nativezone"
  [medium]="${ROOT}/hccb_dense_snappy_g2_nativezone_m15"
  [fine]="${ROOT}/hccb_dense_snappy_g2_nativezone_r2"
)
declare -A NPROCS=([coarse]=16 [medium]=32)

mkdir -p "${RESULT_ROOT}/mesh_summaries" "${RESULT_ROOT}/cases"

mesh_case_is_current() {
  local case_dir=$1
  [[ -f ${case_dir}/mesh_sensitivity_complete.json ]] || return 1
  [[ -f ${case_dir}/cht_smoke_metadata.json ]] || return 1
  [[ -f ${case_dir}/cht_result_summary_${END_TIME}.json ]] || return 1
  python3 - "${case_dir}/cht_smoke_metadata.json" \
    "${case_dir}/mesh_sensitivity_complete.json" "${END_TIME}" <<'PY'
import json
import math
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
completion = json.load(open(sys.argv[2], encoding="utf-8"))
end_time = float(sys.argv[3])
required = metadata.get("source_channel_volume_flow_preserved") is True
source = float(metadata.get("source_inlet_channel_velocity_m_s", float("nan")))
pore = float(metadata.get("pore_opening_boundary_velocity_m_s", float("nan")))
fraction = float(metadata.get("inlet_open_area_fraction", float("nan")))
required = required and all(math.isfinite(value) for value in (source, pore, fraction))
required = required and math.isclose(pore * fraction, source, rel_tol=1.0e-12)
required = required and math.isclose(float(metadata["end_time"]), end_time, abs_tol=1.0e-12)
required = required and math.isclose(float(completion["time"]), end_time, abs_tol=1.0e-12)
raise SystemExit(0 if required else 1)
PY
}

for level in coarse medium fine; do
  mesh=${MESH_CASES[$level]}
  summary="${RESULT_ROOT}/mesh_summaries/${level}.json"
  python3 "${ROOT}/code/summarize_hccb_dense_mesh_check.py" \
    --case "${mesh}" \
    --fluid-log "${mesh}/log.checkMesh.fluid.basic" \
    --solid-log "${mesh}/log.checkMesh.solid.basic" \
    --output "${summary}" > "${RESULT_ROOT}/mesh_summaries/${level}.log"
  test -s "${summary}"
done

python3 "${ROOT}/code/verify_hccb_p418_mesh_fine_reference.py" \
  --metadata "${FINE_CASE}/cht_smoke_metadata.json" \
  --completion "${FINE_CASE}/formal_sample_complete.json" \
  --result "${FINE_CASE}/cht_result_summary_${END_TIME}.json" \
  --mesh-manifest "${MESH_CASES[fine]}/case_manifest.json" \
  --condition-id "${CONDITION_ID}" \
  --end-time "${END_TIME}" \
  > "${RESULT_ROOT}/fine_reference_check.json"

run_level() {
  local level=$1
  local np=${NPROCS[$level]}
  local mesh=${MESH_CASES[$level]}
  local output_root="${RESULT_ROOT}/cases/${level}"
  local case_dir="${output_root}/${CONDITION_ID}"
  local archived

  if [[ -e ${case_dir} ]] && ! mesh_case_is_current "${case_dir}"; then
    archived="${case_dir}.older.$(date +%Y%m%dT%H%M%S).$$"
    mv "${case_dir}" "${archived}"
    echo "kept earlier ${level} mesh result at ${archived}"
  fi
  if [[ ! -f ${case_dir}/cht_smoke_metadata.json ]]; then
    python3 "${ROOT}/code/build_hccb_dense_cht_p418_matrix.py" \
      --mesh-case "${mesh}" \
      --mesh-manifest "${mesh}/case_manifest.json" \
      --mesh-check-summary "${RESULT_ROOT}/mesh_summaries/${level}.json" \
      --output-root "${output_root}" \
      --mode selected --condition-id "${CONDITION_ID}" \
      --mesh-resolution-label "${level}" \
      --parallel-subdomains "${np}" \
      --end-time "${END_TIME}" --write-interval 25 --energy-correctors 20 \
      > "${RESULT_ROOT}/build_${level}.json"
  fi
  if mesh_case_is_current "${case_dir}"; then
    echo "skip completed ${level} mesh"
    return
  fi

  local property_tmp="${case_dir}/constant/fluid/physicalProperties.tmp.$$"
  cp "${ROOT}/results/apd006_hccb_openfoam_helium_property_table/physicalProperties" \
    "${property_tmp}"
  mv -f "${property_tmp}" "${case_dir}/constant/fluid/physicalProperties"
  python3 "${ROOT}/code/add_hccb_pressure_outputs.py" --case "${case_dir}"
  foamDictionary "${case_dir}/system/controlDict" \
    -entry runTimeModifiable -set false

  local restart_time=""
  if [[ -d ${case_dir}/processor0 ]]; then
    while IFS= read -r candidate; do
      [[ ${candidate} =~ ^[0-9]+([.][0-9]+)?$ ]] || continue
      local complete=1
      local rank field
      for ((rank = 0; rank < np; rank++)); do
        for field in fluid/T fluid/U fluid/p fluid/p_rgh solid/T uniform/time; do
          if [[ ! -f ${case_dir}/processor${rank}/${candidate}/${field} ]]; then
            complete=0
            break 2
          fi
        done
      done
      if [[ ${complete} -eq 1 ]]; then
        restart_time=${candidate}
        break
      fi
    done < <(find "${case_dir}/processor0" -mindepth 1 -maxdepth 1 -type d \
      -printf '%f\n' | sort -gr)
  fi

  if [[ -n ${restart_time} && ${restart_time} != 0 ]]; then
    foamDictionary "${case_dir}/system/controlDict" -entry startFrom -set startTime
    foamDictionary "${case_dir}/system/controlDict" -entry startTime -set "${restart_time}"
    printf '\n===== resumed from complete parallel time %s at %s =====\n' \
      "${restart_time}" "$(date --iso-8601=seconds)" \
      >> "${case_dir}/log.foamMultiRun.mesh_sensitivity"
    echo "resume ${level} mesh from ${restart_time} s"
  else
    foamDictionary "${case_dir}/system/controlDict" -entry startFrom -set startTime
    foamDictionary "${case_dir}/system/controlDict" -entry startTime -set 0
    rm -rf "${case_dir}"/processor*
    decomposePar -case "${case_dir}" -allRegions \
      > "${case_dir}/log.decomposePar.mesh_sensitivity" 2>&1
    : > "${case_dir}/log.foamMultiRun.mesh_sensitivity"
  fi
  mpirun -np "${np}" foamMultiRun -case "${case_dir}" -parallel \
    >> "${case_dir}/log.foamMultiRun.mesh_sensitivity" 2>&1
  ROOT="${ROOT}" bash "${ROOT}/code/finalize_hccb_p418_mesh_sensitivity_case.sh" \
    "${case_dir}" "${CONDITION_ID}"
}

run_level coarse &
coarse_pid=$!
run_level medium &
medium_pid=$!
wait "${coarse_pid}"
wait "${medium_pid}"

python3 "${ROOT}/code/summarize_hccb_p418_mesh_sensitivity.py" \
  --coarse-mesh "${RESULT_ROOT}/mesh_summaries/coarse.json" \
  --medium-mesh "${RESULT_ROOT}/mesh_summaries/medium.json" \
  --fine-mesh "${RESULT_ROOT}/mesh_summaries/fine.json" \
  --coarse-result "${RESULT_ROOT}/cases/coarse/${CONDITION_ID}/cht_result_summary_${END_TIME}.json" \
  --medium-result "${RESULT_ROOT}/cases/medium/${CONDITION_ID}/cht_result_summary_${END_TIME}.json" \
  --fine-result "${FINE_CASE}/cht_result_summary_${END_TIME}.json" \
  --output-dir "${RESULT_ROOT}"

python3 "${ROOT}/code/build_hccb_p418_mesh_sensitivity_table.py" \
  --input-summary "${RESULT_ROOT}/summary.json" \
  --output "${ROOT}/manuscript/generated_mesh_sensitivity.tex"
