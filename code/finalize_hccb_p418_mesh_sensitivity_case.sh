#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CASE_DIR CONDITION_ID" >&2
  exit 2
fi

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CASE_DIR=$1
CONDITION_ID=$2

source /opt/openfoam13/etc/bashrc || true

FINAL_TIME=$(python3 - "${CASE_DIR}/cht_smoke_metadata.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["end_time"])
PY
)

reconstructPar -case "${CASE_DIR}" -allRegions -time "${FINAL_TIME}" \
  > "${CASE_DIR}/log.reconstructPar.${FINAL_TIME}" 2>&1
{
  foamPostProcess -case "${CASE_DIR}" -region fluid -time "${FINAL_TIME}" \
    -func 'patchAverage(p,name=inletPressure,patch=inlet)'
  foamPostProcess -case "${CASE_DIR}" -region fluid -time "${FINAL_TIME}" \
    -func 'patchAverage(p,name=outletPressure,patch=outlet)'
} > "${CASE_DIR}/log.pressure.${FINAL_TIME}" 2>&1
cat "${CASE_DIR}/log.pressure.${FINAL_TIME}" >> "${CASE_DIR}/log.foamMultiRun.mesh_sensitivity"

python3 "${ROOT}/code/compute_hccb_gmsh_boundary_heat_flows.py" \
  --case "${CASE_DIR}" --time "${FINAL_TIME}" \
  --output "${CASE_DIR}/boundary_heat_flows_${FINAL_TIME}.json" \
  > "${CASE_DIR}/log.boundary_heat_flows.${FINAL_TIME}"
python3 "${ROOT}/code/summarize_hccb_gmsh_cht_result.py" \
  --case "${CASE_DIR}" \
  --log "${CASE_DIR}/log.foamMultiRun.mesh_sensitivity" \
  --boundary-heat-flows "${CASE_DIR}/boundary_heat_flows_${FINAL_TIME}.json" \
  --output "${CASE_DIR}/cht_result_summary_${FINAL_TIME}.json" \
  > "${CASE_DIR}/log.result_summary.${FINAL_TIME}"

python3 - "${CASE_DIR}" "${CONDITION_ID}" "${FINAL_TIME}" <<'PY'
import json
import pathlib
import sys

case = pathlib.Path(sys.argv[1])
condition_id = sys.argv[2]
time_name = sys.argv[3]
summary_path = case / f"cht_result_summary_{time_name}.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if not summary["solver_finished"]:
    raise SystemExit("mesh-sensitivity OpenFOAM run did not finish")
if not summary["all_reported_values_are_finite"]:
    raise SystemExit("mesh-sensitivity result contains a non-finite value")
payload = {
    "condition_id": condition_id,
    "time": time_name,
    "result_summary": str(summary_path),
    "relative_mass_difference": summary["flow"]["relative_mass_difference"],
    "relative_energy_difference": summary["heat_balance"]["relative_energy_difference"],
}
(case / "mesh_sensitivity_complete.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY

if [[ ${REMOVE_PROCESSORS_AFTER_EXPORT:-1} == 1 ]]; then
  rm -rf "${CASE_DIR}"/processor*
fi
