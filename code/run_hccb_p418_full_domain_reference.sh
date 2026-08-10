#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOCAL_STEADY_ROOT=${LOCAL_STEADY_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
STEP_ROOT=${STEP_ROOT:-${ROOT}/hccb_p418_physical_steps_12}
PACKING=${PACKING:-${ROOT}/data/apd006_hccb_source_sequence_target_packings/seed101_s80_xlo_ycentre/packing.npz}
MESH_CASE=${MESH_CASE:-${ROOT}/hccb_p418_full_domain_g2_seed101_mesh}
RESULT_DIR=${RESULT_DIR:-${ROOT}/results/hccb_p418_full_domain_reference}
LOCAL_REFERENCE=${LOCAL_REFERENCE:-${LOCAL_STEADY_ROOT}/u0p20_T700_q6p85/cht_result_summary_200.json}
LOCAL_MESH_MANIFEST=${LOCAL_MESH_MANIFEST:-${ROOT}/hccb_dense_snappy_g2_nativezone_r2/case_manifest.json}
NP=${NP:-32}
END_TIME=${END_TIME:-5000}
WRITE_INTERVAL=${WRITE_INTERVAL:-500}
RUN_LOCK=${RUN_LOCK:-${ROOT}/results/hccb_p418_full_domain_reference_run.lock}
MANUSCRIPT_TABLE=${MANUSCRIPT_TABLE:-${ROOT}/manuscript/generated_full_domain_reference.tex}

source /opt/openfoam13/etc/bashrc || true
mkdir -p "${RESULT_DIR}"
exec 8>"${RUN_LOCK}"
flock 8

if [[ -f ${RESULT_DIR}/completion.json \
  && -f ${RESULT_DIR}/full_vs_local_comparison.json \
  && -f ${MESH_CASE}/steady_result_audit.json ]]; then
  completion_status=$(python3 - "${RESULT_DIR}/completion.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))
PY
)
  if [[ ${completion_status} == hccb_p418_full_domain_reference_completed ]]; then
    echo "reuse completed full-domain reference: ${RESULT_DIR}"
    exit 0
  fi
fi

steady_completed=$(find "${LOCAL_STEADY_ROOT}" -mindepth 2 -maxdepth 2 \
  -name formal_sample_complete.json | wc -l | tr -d ' ')
if [[ ${steady_completed} -ne 60 ]]; then
  echo "full-domain reference waits for 60 completed local steady cases; found ${steady_completed}" >&2
  exit 1
fi

step_completed=$(find "${STEP_ROOT}" -mindepth 2 -maxdepth 2 \
  -name step_response_complete.json | wc -l | tr -d ' ')
if [[ ${step_completed} -ne 12 ]]; then
  echo "full-domain reference waits for 12 completed OpenFOAM step histories; found ${step_completed}" >&2
  exit 1
fi

if [[ ! -f ${MESH_CASE}/case_manifest.json ]]; then
  python3 "${ROOT}/code/build_hccb_pore_resolved_openfoam_mesh.py" \
    --packing "${PACKING}" \
    --output-dir "${MESH_CASE}" \
    --mesh-level G2 \
    > "${RESULT_DIR}/mesh_build_manifest.json"
fi

if [[ ! -f ${MESH_CASE}/mesh_audit_summary.json ]]; then
  if ! python3 "${ROOT}/code/audit_hccb_pore_resolved_openfoam_mesh.py" \
    --case "${MESH_CASE}" --run-mesh \
    > "${RESULT_DIR}/mesh_check_stdout.json"; then
    cp "${MESH_CASE}/mesh_audit_summary.json" "${RESULT_DIR}/mesh_audit_summary.json"
    echo "full-domain G2 mesh did not pass the declared OpenFOAM mesh checks; no physical result was run" >&2
    exit 3
  fi
fi

mesh_status=$(python3 - "${MESH_CASE}/mesh_audit_summary.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)
if [[ ${mesh_status} != hccb_pore_resolved_openfoam_mesh_preflight_passed ]]; then
  echo "full-domain G2 mesh is not ready for a physical solve: ${mesh_status}" >&2
  exit 3
fi

cp "${MESH_CASE}/mesh_audit_summary.json" "${RESULT_DIR}/mesh_audit_summary.json"

if [[ ! -f ${MESH_CASE}/steady_case_manifest.json ]]; then
  python3 "${ROOT}/code/build_hccb_pore_resolved_openfoam_steady_case.py" \
    --case "${MESH_CASE}" \
    --parallel-subdomains "${NP}" \
    --end-time "${END_TIME}" \
    --write-interval "${WRITE_INTERVAL}" \
    > "${RESULT_DIR}/steady_case_manifest_stdout.json"
fi

if ! grep -q '^ *End *$' "${MESH_CASE}/log.foamMultiRun.steady" 2>/dev/null; then
  (cd "${MESH_CASE}" && nice -n 10 ./Allrun.steady)
fi

python3 "${ROOT}/code/audit_hccb_pore_resolved_openfoam_steady_result.py" \
  --case "${MESH_CASE}" \
  > "${RESULT_DIR}/steady_result_stdout.json"
cp "${MESH_CASE}/steady_result_audit.json" "${RESULT_DIR}/steady_result_audit.json"

python3 "${ROOT}/code/compare_hccb_p418_full_domain_reference.py" \
  --full-result "${MESH_CASE}/steady_result_audit.json" \
  --local-result "${LOCAL_REFERENCE}" \
  --local-mesh-manifest "${LOCAL_MESH_MANIFEST}" \
  --output "${RESULT_DIR}/full_vs_local_comparison.json" \
  > "${RESULT_DIR}/full_vs_local_comparison_stdout.json"

python3 "${ROOT}/code/build_hccb_p418_full_domain_reference_table.py" \
  --comparison "${RESULT_DIR}/full_vs_local_comparison.json" \
  --output "${MANUSCRIPT_TABLE}" \
  > "${RESULT_DIR}/manuscript_table_stdout.json"

python3 - "${RESULT_DIR}/completion.json" "${MESH_CASE}" "${NP}" \
  "${MANUSCRIPT_TABLE}" <<'PY'
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
case = pathlib.Path(sys.argv[2])
result = case / "steady_result_audit.json"
mesh = case / "mesh_audit_summary.json"
comparison = output.parent / "full_vs_local_comparison.json"
manuscript_table = pathlib.Path(sys.argv[4])
payload = {
    "status": "hccb_p418_full_domain_reference_completed",
    "case": str(case.resolve()),
    "parallel_subdomains": int(sys.argv[3]),
    "mesh_audit": str(mesh.resolve()),
    "mesh_audit_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
    "steady_result": str(result.resolve()),
    "steady_result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
    "full_vs_local_comparison": str(comparison.resolve()),
    "full_vs_local_comparison_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
    "manuscript_table": str(manuscript_table.resolve()),
    "manuscript_table_sha256": hashlib.sha256(manuscript_table.read_bytes()).hexdigest(),
    "new_physical_parameters": [],
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
