#!/usr/bin/env bash
# Prepare seed202/303 meshes and their nine exact-P418 CHT case directories.
# The default is a dry run. Set EXECUTE=1 only when the current OpenFOAM matrix
# no longer occupies the workstation.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PLAN=${PLAN:-${ROOT}/parameters/hccb_p418_cross_packing_plan.json}
BASE_MESH_MANIFEST=${BASE_MESH_MANIFEST:-}
EXECUTE=${EXECUTE:-0}

if [[ -z ${BASE_MESH_MANIFEST} ]]; then
    for candidate in \
        "${ROOT}/hccb_dense_snappy_g2_nativezone_r2/case_manifest.json" \
        "${ROOT}/runs/hccb_dense_snappy_g2_nativezone_r2/case_manifest.json"; do
        if [[ -f ${candidate} ]]; then
            BASE_MESH_MANIFEST=${candidate}
            break
        fi
    done
fi

if [[ ! -f ${PLAN} ]]; then
    echo "cross-packing plan is missing: ${PLAN}" >&2
    exit 1
fi
if [[ ! -f ${BASE_MESH_MANIFEST} ]]; then
    echo "seed101 mesh record is missing: ${BASE_MESH_MANIFEST}" >&2
    exit 1
fi
if [[ ${EXECUTE} != 0 && ${EXECUTE} != 1 ]]; then
    echo "EXECUTE must be 0 or 1" >&2
    exit 1
fi

CONDITIONS=()
while IFS= read -r condition; do
    CONDITIONS+=("${condition}")
done < <(
    python3 - "${PLAN}" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
for condition in plan["screening_design"]["conditions"]:
    print(condition["condition_id"])
PY
)
if [[ ${#CONDITIONS[@]} -ne 9 ]]; then
    echo "the plan must contain nine screening conditions" >&2
    exit 1
fi

# Read the local crop and all mesh controls from the completed seed101 mesh.
# These are numerical settings for a like-for-like packing comparison, not new
# material properties or operating conditions.
read -r crop_x0 crop_x1 crop_y0 crop_y1 crop_z0 crop_z1 \
    cells_per_dp sphere_subdivisions surface_refinement cells_between_levels < <(
    python3 - "${BASE_MESH_MANIFEST}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
crop = payload.get("crop_box_dp", [])
controls = payload.get("numerical_controls", {})
values = [
    *crop,
    controls.get("background_cells_per_particle_diameter"),
    controls.get("sphere_icosphere_subdivisions"),
    controls.get("surface_refinement_level"),
    controls.get("cells_between_levels"),
]
if len(crop) != 6 or any(value is None for value in values):
    raise SystemExit("seed101 mesh record does not contain the required crop and mesh controls")
if payload.get("new_physical_parameters") != []:
    raise SystemExit("seed101 mesh record unexpectedly adds a physical parameter")
print(*values)
PY
)

echo "reference seed101 local mesh"
echo "  crop box (dp): ${crop_x0} ${crop_x1} ${crop_y0} ${crop_y1} ${crop_z0} ${crop_z1}"
echo "  cells/dp=${cells_per_dp}, sphere subdivisions=${sphere_subdivisions}, surface refinement=${surface_refinement}, cells between levels=${cells_between_levels}"

if [[ ${EXECUTE} == 1 ]]; then
    source /opt/openfoam13/etc/bashrc || true
fi

while IFS=$'\t' read -r seed relative_packing expected_hash; do
    packing=${ROOT}/${relative_packing}
    mesh_case=${ROOT}/hccb_dense_snappy_g2_nativezone_r2_seed${seed}
    mesh_summary=${ROOT}/results/hccb_dense_mesh_seed${seed}_r2_summary.json
    matrix_root=${ROOT}/hccb_dense_cht_p418_cross_packing_seed${seed}_screen9

    echo "seed ${seed}"
    echo "  packing: ${packing}"
    echo "  mesh: ${mesh_case}"
    echo "  CHT cases: ${matrix_root} (${#CONDITIONS[@]} exact P418 conditions)"

    if [[ ${EXECUTE} == 0 ]]; then
        continue
    fi
    if [[ ! -f ${packing} ]]; then
        echo "packing is missing: ${packing}" >&2
        exit 1
    fi
    actual_hash=$(sha256sum "${packing}" | awk '{print $1}')
    if [[ ${actual_hash} != ${expected_hash} ]]; then
        echo "packing checksum mismatch for seed ${seed}" >&2
        exit 1
    fi
    mesh_complete=1
    for required in \
        "${mesh_case}/case_manifest.json" \
        "${mesh_case}/constant/fluid/polyMesh/boundary" \
        "${mesh_case}/constant/solid/polyMesh/boundary" \
        "${mesh_case}/log.checkMesh.fluid" \
        "${mesh_case}/log.checkMesh.solid"; do
        if [[ ! -f ${required} ]]; then
            mesh_complete=0
        fi
    done
    if [[ -e ${mesh_case} && ${mesh_complete} -ne 1 ]]; then
        echo "existing seed${seed} mesh is incomplete; it is left unchanged: ${mesh_case}" >&2
        exit 1
    fi
    if [[ ${mesh_complete} -eq 1 ]]; then
        echo "reuse completed seed${seed} mesh"
    else
        mesh_stage=${mesh_case}.building.$(date +%Y%m%d%H%M%S).$$
        echo "build seed${seed} mesh in temporary directory ${mesh_stage}"
        python3 "${ROOT}/code/build_hccb_dense_snappy_case.py" \
            --packing "${packing}" \
            --output-dir "${mesh_stage}" \
            --crop-box-dp "${crop_x0}" "${crop_x1}" "${crop_y0}" "${crop_y1}" "${crop_z0}" "${crop_z1}" \
            --cells-per-diameter "${cells_per_dp}" \
            --sphere-subdivisions "${sphere_subdivisions}" \
            --surface-refinement "${surface_refinement}" \
            --cells-between-levels "${cells_between_levels}" \
            --solid-cell-source snappy-zone \
            > "${ROOT}/results/hccb_dense_mesh_seed${seed}_r2_build.json"
        (cd "${mesh_stage}" && ./Allmesh > log.Allmesh 2>&1)
        checkMesh -case "${mesh_stage}" -region fluid -allTopology -allGeometry \
            > "${mesh_stage}/log.checkMesh.fluid" 2>&1
        checkMesh -case "${mesh_stage}" -region solid -allTopology -allGeometry \
            > "${mesh_stage}/log.checkMesh.solid" 2>&1
        if [[ -e ${mesh_case} ]]; then
            echo "refusing to replace seed${seed} mesh created during staging" >&2
            exit 1
        fi
        mv "${mesh_stage}" "${mesh_case}"
    fi

    python3 - "${BASE_MESH_MANIFEST}" "${mesh_case}/case_manifest.json" "${seed}" <<'PY'
import json
import math
import sys

base = json.load(open(sys.argv[1], encoding="utf-8"))
new = json.load(open(sys.argv[2], encoding="utf-8"))
seed = int(sys.argv[3])
if len(base["crop_box_dp"]) != len(new["crop_box_dp"]) or any(
    not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-12)
    for a, b in zip(base["crop_box_dp"], new["crop_box_dp"])
):
    raise SystemExit(f"seed{seed} crop differs from seed101")
keys = (
    "background_cells_per_particle_diameter",
    "sphere_icosphere_subdivisions",
    "surface_refinement_level",
    "cells_between_levels",
)
for key in keys:
    a = base["numerical_controls"][key]
    b = new["numerical_controls"][key]
    if isinstance(a, float):
        same = math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-12)
    else:
        same = a == b
    if not same:
        raise SystemExit(f"seed{seed} mesh setting {key} differs from seed101")
if new.get("new_physical_parameters") != []:
    raise SystemExit(f"seed{seed} mesh unexpectedly adds a physical parameter")
PY
    if [[ ! -f ${mesh_summary} ]]; then
        python3 "${ROOT}/code/summarize_hccb_dense_mesh_check.py" \
            --case "${mesh_case}" \
            --fluid-log "${mesh_case}/log.checkMesh.fluid" \
            --solid-log "${mesh_case}/log.checkMesh.solid" \
            --output "${mesh_summary}"
    fi

    condition_args=()
    for condition in "${CONDITIONS[@]}"; do
        condition_args+=(--condition-id "${condition}")
    done
    if [[ -f ${matrix_root}/matrix_manifest.json ]]; then
        echo "reuse completed seed${seed} nine-condition case directory"
    else
        python3 "${ROOT}/code/build_hccb_dense_cht_p418_matrix.py" \
            --mesh-case "${mesh_case}" \
            --mesh-manifest "${mesh_case}/case_manifest.json" \
            --mesh-check-summary "${mesh_summary}" \
            --output-root "${matrix_root}" \
            --mode selected \
            --mesh-resolution-label fine \
            --resume-existing \
            "${condition_args[@]}"
    fi
done < <(
    python3 - "${PLAN}" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
for packing in plan["packing_realisations"]:
    if packing["seed"] in (202, 303):
        print(
            f'{packing["seed"]}\t{packing["packing_path"]}\t'
            f'{packing["packing_npz_sha256"]}'
        )
PY
)

if [[ ${EXECUTE} == 0 ]]; then
    echo "dry run only: no mesh or CHT case was created"
else
    geometry_args=()
    physical_csv=${ROOT}/results/hccb_p418_60_sourceflow_r3_completed_physics/completed_case_physics.csv
    if [[ -f ${physical_csv} ]]; then
        geometry_args+=(--physical-csv "${physical_csv}")
    fi
    python3 "${ROOT}/code/summarize_hccb_p418_cross_packing_geometry.py" \
        --root "${ROOT}" \
        --plan "${PLAN}" \
        --manifest-seed101 "${BASE_MESH_MANIFEST}" \
        --manifest-seed202 "${ROOT}/hccb_dense_snappy_g2_nativezone_r2_seed202/case_manifest.json" \
        --manifest-seed303 "${ROOT}/hccb_dense_snappy_g2_nativezone_r2_seed303/case_manifest.json" \
        --output-dir "${ROOT}/results/hccb_p418_cross_packing_geometry" \
        --tex-output "${ROOT}/manuscript/generated_cross_packing_geometry.tex" \
        "${geometry_args[@]}"
    echo "seed202 and seed303 meshes and nine-condition case directories are ready"
fi
