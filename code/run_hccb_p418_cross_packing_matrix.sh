#!/usr/bin/env bash
# Run and post-process one nine-condition independent packing matrix.
# Dry run by default. Set EXECUTE=1 only after the seed101 matrix releases CPUs.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SEED=${SEED:-202}
EXECUTE=${EXECUTE:-0}
NP_PER_CASE=${NP_PER_CASE:-32}
CONCURRENT_CASES=${CONCURRENT_CASES:-3}

if [[ ${SEED} != 202 && ${SEED} != 303 ]]; then
    echo "SEED must be 202 or 303" >&2
    exit 1
fi
if [[ ${EXECUTE} != 0 && ${EXECUTE} != 1 ]]; then
    echo "EXECUTE must be 0 or 1" >&2
    exit 1
fi

MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_cross_packing_seed${SEED}_screen9}
MESH_CASE=${MESH_CASE:-${ROOT}/hccb_dense_snappy_g2_nativezone_r2_seed${SEED}}
MESH_MANIFEST=${MESH_MANIFEST:-${MESH_CASE}/case_manifest.json}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results}
LOCK_FILE=${LOCK_FILE:-${MATRIX_ROOT}.pipeline.lock}
PLAN_FILE=${PLAN_FILE:-${ROOT}/parameters/hccb_p418_cross_packing_plan.json}

echo "seed${SEED} independent-packing matrix"
echo "  mesh: ${MESH_CASE}"
echo "  cases: ${MATRIX_ROOT}"
echo "  parallelism: ${CONCURRENT_CASES} cases x ${NP_PER_CASE} ranks"

if [[ ${EXECUTE} == 0 ]]; then
    echo "dry run only: no mesh, solver or post-processing command was started"
    exit 0
fi

exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
    echo "seed${SEED} cross-packing pipeline is already running" >&2
    exit 1
fi

MATRIX_MANIFEST=${MATRIX_ROOT}/matrix_manifest.json
if [[ ! -f ${MATRIX_MANIFEST} ]]; then
    echo "seed${SEED} case matrix is missing; run cross-packing setup first" >&2
    exit 1
fi
python3 "${ROOT}/code/verify_hccb_p418_cross_packing_matrix.py" \
    --seed "${SEED}" \
    --plan "${PLAN_FILE}" \
    --matrix-manifest "${MATRIX_MANIFEST}" \
    --matrix-root "${MATRIX_ROOT}" \
    --mesh-manifest "${MESH_MANIFEST}" \
    --output "${RESULT_ROOT}/hccb_p418_cross_packing_seed${SEED}_matrix_check.json"

reference_case=$(find "${MATRIX_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name 'u*_T*_q*' | sort | head -n 1)
if [[ ! -f ${reference_case}/interface_pairs/interface_face_pairs.npz ]]; then
    CASE="${reference_case}" \
        bash "${ROOT}/code/build_hccb_dense_cht_learning_geometry.sh"
fi
INTERFACE_PAIRS=${reference_case}/interface_pairs/interface_face_pairs.npz
if [[ ! -f ${INTERFACE_PAIRS} ]]; then
    echo "seed${SEED} interface-pair file was not generated" >&2
    exit 1
fi

MATRIX_ROOT="${MATRIX_ROOT}" \
MESH_CASE="${MESH_CASE}" \
MESH_MANIFEST="${MESH_MANIFEST}" \
INTERFACE_PAIRS="${INTERFACE_PAIRS}" \
NP_PER_CASE="${NP_PER_CASE}" \
CONCURRENT_CASES="${CONCURRENT_CASES}" \
LOCK_FILE="${MATRIX_ROOT}.solver.lock" \
    bash "${ROOT}/code/run_hccb_dense_cht_p418_matrix_parallel.sh"

completed=$(find "${MATRIX_ROOT}" -mindepth 2 -maxdepth 2 \
    -name formal_sample_complete.json | wc -l | tr -d ' ')
if [[ ${completed} -ne 9 ]]; then
    echo "seed${SEED} matrix incomplete after solver: ${completed}/9" >&2
    exit 1
fi

SEED="${SEED}" EXECUTE=1 \
    bash "${ROOT}/code/run_hccb_p418_cross_packing_postprocess.sh"

python3 - "${SEED}" "${INTERFACE_PAIRS}" \
    "${RESULT_ROOT}/hccb_p418_cross_packing_seed${SEED}_complete.json" <<'PY'
import hashlib
import json
import pathlib
import sys

seed = int(sys.argv[1])
interface = pathlib.Path(sys.argv[2]).resolve()
output = pathlib.Path(sys.argv[3]).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(
        {
            "status": "cross_packing_openfoam_and_postprocess_complete",
            "packing_seed": seed,
            "condition_count": 9,
            "packing_specific_interface_pairs": str(interface),
            "interface_pairs_sha256": hashlib.sha256(interface.read_bytes()).hexdigest(),
            "new_physical_parameter_values_added": [],
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(output)
PY

echo "seed${SEED} nine-condition OpenFOAM matrix and model inputs are complete"
