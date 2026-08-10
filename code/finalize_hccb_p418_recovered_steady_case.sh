#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "用法: $0 <已恢复到60组主目录的算例路径> <工况编号>" >&2
  exit 2
fi

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CASE_DIR=$1
CONDITION=$2
FINAL_TIME=${FINAL_TIME:-200}
TAIL_START_TIME=${TAIL_START_TIME:-175}
SAMPLE_NAME=${SAMPLE_NAME:-training_sample_${FINAL_TIME}_schema3}
INTERFACE_PAIRS=${INTERFACE_PAIRS:-${ROOT}/hccb_dense_cht_native_r2/interface_pairs/interface_face_pairs.npz}
PARAMETERS=${PARAMETERS:-${ROOT}/parameters/literature_parameter_manifest.csv}
REUSE_BOUNDARY_SAMPLE=${REUSE_BOUNDARY_SAMPLE:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3/u0p05_T300_q4p85/training_sample_200_schema3/fields_and_topology.npz}
REMOVE_PROCESSORS_AFTER_EXPORT=${REMOVE_PROCESSORS_AFTER_EXPORT:-1}
ALLOW_MISSING_TAIL_FIELDS=${ALLOW_MISSING_TAIL_FIELDS:-0}
FINALIZATION_START_EPOCH=$(date +%s.%N)

if [[ $(basename "${CASE_DIR}") != "${CONDITION}" ]]; then
  echo "算例目录名与工况编号不一致: ${CASE_DIR} ${CONDITION}" >&2
  exit 2
fi
for path in \
  "${CASE_DIR}/cht_result_summary_${FINAL_TIME}.json" \
  "${CASE_DIR}/cloud_runtime_resources.json" \
  "${CASE_DIR}/log.foamMultiRun" \
  "${CASE_DIR}/${FINAL_TIME}" \
  "${INTERFACE_PAIRS}" \
  "${PARAMETERS}" \
  "${REUSE_BOUNDARY_SAMPLE}"; do
  if [[ ! -e ${path} ]]; then
    echo "缺少正式后处理输入: ${path}" >&2
    exit 2
  fi
done
for rank in "${CASE_DIR}"/processor*; do
  [[ -d ${rank} ]] || continue
  required_times=("${FINAL_TIME}")
  if [[ ${ALLOW_MISSING_TAIL_FIELDS} != 1 ]]; then
    required_times=("${TAIL_START_TIME}" "${FINAL_TIME}")
  fi
  for time_name in "${required_times[@]}"; do
    for field in fluid/T fluid/U fluid/p fluid/p_rgh solid/T; do
      if [[ ! -f ${rank}/${time_name}/${field} ]]; then
        echo "缺少末段分区场: ${rank}/${time_name}/${field}" >&2
        exit 2
      fi
    done
  done
done
if [[ -e ${CASE_DIR}/formal_sample_complete.json ]]; then
  echo "算例已经有正式训练样本完成标记: ${CASE_DIR}" >&2
  exit 2
fi
if [[ -e ${CASE_DIR}/${SAMPLE_NAME} ]]; then
  echo "训练样本目录已经存在，停止以避免覆盖: ${CASE_DIR}/${SAMPLE_NAME}" >&2
  exit 2
fi

set +u
set +e
set +o pipefail
source /opt/openfoam13/etc/bashrc
set -o pipefail
set -e
set -u
case "${WM_PROJECT_VERSION:-}" in
  13|OpenFOAM-13) ;;
  *)
    echo "需要OpenFOAM Foundation 13，当前为${WM_PROJECT_VERSION:-unset}" >&2
    exit 2
    ;;
esac

cp -a "${CASE_DIR}/log.foamMultiRun" "${CASE_DIR}/log.foamMultiRun.formal"
python3 "${ROOT}/code/export_hccb_cht_training_sample.py" \
  --case "${CASE_DIR}" \
  --time "${FINAL_TIME}" \
  --parameter-manifest "${PARAMETERS}" \
  --interface-pairs "${INTERFACE_PAIRS}" \
  --reuse-boundary-geometry "${REUSE_BOUNDARY_SAMPLE}" \
  --output-dir "${CASE_DIR}/${SAMPLE_NAME}" \
  --run-postprocess > "${CASE_DIR}/log.training_export.${FINAL_TIME}"

tail_args=(
  --case "${CASE_DIR}"
  --start-time "${TAIL_START_TIME}"
  --end-time "${FINAL_TIME}"
  --output "${CASE_DIR}/steady_final_window_${TAIL_START_TIME}_to_${FINAL_TIME}.json"
)
if [[ ${ALLOW_MISSING_TAIL_FIELDS} == 1 ]]; then
  tail_args+=(--allow-missing-fields)
fi
python3 "${ROOT}/code/summarize_hccb_p418_formal_steady_tail.py" \
  "${tail_args[@]}" > "${CASE_DIR}/log.steady_final_window.${FINAL_TIME}"

python3 - \
  "${CASE_DIR}" "${CONDITION}" "${FINAL_TIME}" "${TAIL_START_TIME}" \
  "${SAMPLE_NAME}" "${FINALIZATION_START_EPOCH}" <<'PY'
import hashlib
import json
import math
import pathlib
import sys
import time

case = pathlib.Path(sys.argv[1])
condition = sys.argv[2]
time_name = sys.argv[3]
tail_start = sys.argv[4]
sample_name = sys.argv[5]
start_epoch = float(sys.argv[6])

summary_path = case / f"cht_result_summary_{time_name}.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("solver_finished") is not True:
    raise SystemExit("OpenFOAM结果没有正常完成")
if summary.get("all_reported_values_are_finite") is not True:
    raise SystemExit("OpenFOAM结果含非有限值")
if float(summary.get("reported_iteration", -1)) != float(time_name):
    raise SystemExit("OpenFOAM结果终止时间与正式时间不一致")

sample_dir = case / sample_name
sample = sample_dir / "fields_and_topology.npz"
sample_metadata_path = sample_dir / "metadata.json"
sample_metadata = json.loads(sample_metadata_path.read_text(encoding="utf-8"))
if sample_metadata.get("schema_version") != 3:
    raise SystemExit("训练样本不是schema 3")

resources_path = case / "cloud_runtime_resources.json"
resources = json.loads(resources_path.read_text(encoding="utf-8"))
actual_mpi = int(resources["mpi_process_count"])
sample_metadata["numerical_execution"] = {
    "mpi_process_count": actual_mpi,
    "resource_summary": str(resources_path),
    "resource_summary_sha256": hashlib.sha256(resources_path.read_bytes()).hexdigest(),
    "note": (
        "The actual decomposition used for this run is recorded here. "
        "The physical conditions are unchanged."
    ),
}
sample_metadata_path.write_text(
    json.dumps(sample_metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

tail_path = case / f"steady_final_window_{tail_start}_to_{time_name}.json"
tail = json.loads(tail_path.read_text(encoding="utf-8"))
if tail.get("status") != "formal_steady_final_window_measured":
    raise SystemExit("末段变化结果无效")
full_field_available = bool(tail.get("full_field_available"))

numeric = []
def collect(value):
    if isinstance(value, dict):
        for item in value.values():
            collect(item)
    elif isinstance(value, list):
        for item in value:
            collect(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric.append(float(value))
collect(summary)
collect(tail)
if not numeric or not all(math.isfinite(value) for value in numeric):
    raise SystemExit("正式结果或末段变化含非有限值")

payload = {
    "condition_id": condition,
    "time": time_name,
    "solver_finished": True,
    "actual_mpi_process_count": actual_mpi,
    "relative_mass_difference": summary["flow"]["relative_mass_difference"],
    "relative_energy_difference": summary["heat_balance"]["relative_energy_difference"],
    "result_summary": str(summary_path),
    "result_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "training_sample": str(sample),
    "training_sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
    "training_sample_schema_version": 3,
    "steady_final_window_status": tail["status"],
    "steady_final_window_iterations": tail.get("window_iterations", tail.get("window_s")),
    "solver_time_semantics": "steady_iteration_index",
    "physical_time_s": None,
    "steady_final_window_full_field_available": full_field_available,
    "steady_final_window_summary": str(tail_path),
    "steady_final_window_summary_sha256": hashlib.sha256(tail_path.read_bytes()).hexdigest(),
    "formal_finalization_seconds": max(0.0, time.time() - start_epoch),
}
(case / "formal_sample_complete.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

if [[ ${REMOVE_PROCESSORS_AFTER_EXPORT} == 1 ]]; then
  rm -rf "${CASE_DIR}"/processor*
fi

echo "已生成正式schema-3样本并接入60组主目录: ${CONDITION}"
