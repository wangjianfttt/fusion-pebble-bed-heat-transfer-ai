#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CASE_DIR" >&2
  exit 2
fi

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CASE_DIR=$1
OPENFOAM_BASHRC=${OPENFOAM_BASHRC:-/opt/openfoam13/etc/bashrc}
SKIP_RECONSTRUCT_IF_COMPLETE=${SKIP_RECONSTRUCT_IF_COMPLETE:-0}
if [[ -f ${OPENFOAM_BASHRC} ]]; then
  set +u
  source "${OPENFOAM_BASHRC}"
  set -u
fi

times=$(python3 - "${CASE_DIR}/step_case_metadata.json" <<'PY'
import json, sys

row = json.load(open(sys.argv[1]))
print(",".join(format(float(value), ".12g") for value in row["snapshot_times_s"]))
PY
)
final_time=$(python3 - "${CASE_DIR}/step_case_metadata.json" <<'PY'
import json, sys

value = json.load(open(sys.argv[1]))["duration_s"]
print(format(float(value), ".12g"))
PY
)

if [[ ${SKIP_RECONSTRUCT_IF_COMPLETE} == 1 ]]; then
  python3 - "${CASE_DIR}" "${CASE_DIR}/step_case_metadata.json" <<'PY'
import json
import pathlib
import sys

case = pathlib.Path(sys.argv[1])
metadata = json.load(open(sys.argv[2]))
required = ("fluid/T", "fluid/U", "fluid/p", "fluid/p_rgh", "fluid/phi", "solid/T")
missing = []
for value in metadata["snapshot_times_s"]:
    time_name = format(float(value), ".12g")
    for field in required:
        if not (case / time_name / field).is_file():
            missing.append(f"{time_name}/{field}")
if missing:
    raise SystemExit(
        "cannot skip reconstruction; missing reconstructed fields: "
        + ", ".join(missing[:20])
    )
print(
    f"reusing {len(metadata['snapshot_times_s'])} complete reconstructed "
    "snapshot times"
)
PY
else
  reconstructPar -case "${CASE_DIR}" -allRegions -time "${times}" \
    > "${CASE_DIR}/log.reconstructPar.step" 2>&1
fi
python3 "${ROOT}/code/compute_hccb_gmsh_boundary_heat_flows.py" \
  --case "${CASE_DIR}" --time "${final_time}" \
  --output "${CASE_DIR}/boundary_heat_flows_${final_time}.json" \
  > "${CASE_DIR}/log.boundary_heat_flows.${final_time}"
python3 "${ROOT}/code/summarize_hccb_gmsh_cht_result.py" \
  --case "${CASE_DIR}" \
  --log "${CASE_DIR}/log.foamMultiRun.step" \
  --boundary-heat-flows "${CASE_DIR}/boundary_heat_flows_${final_time}.json" \
  --output "${CASE_DIR}/cht_result_summary_${final_time}.json" \
  > "${CASE_DIR}/log.result_summary.${final_time}"

python3 - "${CASE_DIR}" "${final_time}" <<'PY'
import json, pathlib, sys
case = pathlib.Path(sys.argv[1])
final_time = sys.argv[2]
metadata = json.load(open(case / "step_case_metadata.json"))
summary = json.load(open(case / f"cht_result_summary_{final_time}.json"))
payload = {
    "status": "completed_p418_quasi_steady_flow_thermal_step_response",
    "transient_model": metadata["transient_model"],
    "sequence_id": metadata["sequence_id"],
    "source_condition_id": metadata["source_condition_id"],
    "target_condition_id": metadata["target_condition_id"],
    "changed_physical_input": metadata["changed_physical_input"],
    "duration_s": metadata["duration_s"],
    "solver_finished": summary["solver_finished"],
    "relative_mass_difference": summary["flow"]["relative_mass_difference"],
    "relative_energy_difference": summary["heat_balance"]["relative_energy_difference"],
    "new_physical_parameters": [],
}
(case / "step_response_complete.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

if [[ ${REMOVE_PROCESSORS_AFTER_EXPORT:-1} == 1 ]]; then
  rm -rf "${CASE_DIR}"/processor*
fi
