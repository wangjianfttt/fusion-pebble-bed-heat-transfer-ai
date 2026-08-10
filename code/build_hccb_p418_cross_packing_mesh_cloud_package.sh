#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/cloud_migration_build}
PACKAGE_NAME=${PACKAGE_NAME:-p418_cross_packing_mesh_preprocess}
PACKAGE_DIR=${OUTPUT_ROOT}/${PACKAGE_NAME}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}

case "${OUTPUT_ROOT}" in
  /home|/home/*)
    echo "cloud package output must not be under /home: ${OUTPUT_ROOT}" >&2
    exit 2
    ;;
esac

for required in \
  "${ROOT}/data/apd006_hccb_source_sequence_target_packings/seed202_s80_xlo_ycentre/packing.npz" \
  "${ROOT}/data/apd006_hccb_source_sequence_target_packings/seed303_s80_xlo_ycentre/packing.npz" \
  "${ROOT}/parameters/hccb_p418_cross_packing_plan.json" \
  "${ROOT}/parameters/hccb_p418_physical_parameter_sources.csv" \
  "${ROOT}/parameters/literature_parameter_manifest.csv" \
  "${ROOT}/cloud_migration/reference_seed101_mesh_manifest.json" \
  "${ROOT}/cloud_migration/CROSS_PACKING_MESH_README_CN.md" \
  "${ROOT}/cloud_migration/run_cross_packing_mesh_preprocess.sh" \
  "${ROOT}/cloud_migration/submit_cross_packing_mesh_array_example.sh" \
  "${ROOT}/code/build_hccb_dense_snappy_case.py" \
  "${ROOT}/code/build_hccb_pore_resolved_openfoam_mesh.py" \
  "${ROOT}/code/check_hccb_source_sequence_lammps_packing.py" \
  "${ROOT}/code/build_clipped_hccb_solid_surface_vtk.py" \
  "${ROOT}/code/summarize_hccb_dense_mesh_check.py" \
  "${ROOT}/code/run_with_resource_monitor.py"; do
  if [[ ! -f ${required} ]]; then
    echo "required mesh-package input is missing: ${required}" >&2
    exit 1
  fi
done

rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/packings" "${PACKAGE_DIR}/parameters" \
  "${PACKAGE_DIR}/reference" "${PACKAGE_DIR}/code" "${PACKAGE_DIR}/scripts"

cp -a \
  "${ROOT}/data/apd006_hccb_source_sequence_target_packings/seed202_s80_xlo_ycentre/packing.npz" \
  "${PACKAGE_DIR}/packings/seed202_packing.npz"
cp -a \
  "${ROOT}/data/apd006_hccb_source_sequence_target_packings/seed303_s80_xlo_ycentre/packing.npz" \
  "${PACKAGE_DIR}/packings/seed303_packing.npz"
cp -a "${ROOT}/parameters/hccb_p418_cross_packing_plan.json" \
  "${PACKAGE_DIR}/parameters/"
cp -a "${ROOT}/parameters/hccb_p418_physical_parameter_sources.csv" \
  "${PACKAGE_DIR}/parameters/"
cp -a "${ROOT}/parameters/literature_parameter_manifest.csv" \
  "${PACKAGE_DIR}/parameters/"
cp -a "${ROOT}/cloud_migration/reference_seed101_mesh_manifest.json" \
  "${PACKAGE_DIR}/reference/"
cp -a "${ROOT}/cloud_migration/CROSS_PACKING_MESH_README_CN.md" \
  "${PACKAGE_DIR}/README_CN.md"
cp -a "${ROOT}/cloud_migration/run_cross_packing_mesh_preprocess.sh" \
  "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/cloud_migration/submit_cross_packing_mesh_array_example.sh" \
  "${PACKAGE_DIR}/scripts/"
for script in \
  build_hccb_dense_snappy_case.py \
  build_hccb_pore_resolved_openfoam_mesh.py \
  check_hccb_source_sequence_lammps_packing.py \
  build_clipped_hccb_solid_surface_vtk.py \
  summarize_hccb_dense_mesh_check.py \
  run_with_resource_monitor.py; do
  cp -a "${ROOT}/code/${script}" "${PACKAGE_DIR}/code/"
done
chmod +x "${PACKAGE_DIR}/scripts/"*.sh "${PACKAGE_DIR}/code/"*.py

python3 - "${PACKAGE_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
plan = json.loads(
    (root / "parameters/hccb_p418_cross_packing_plan.json").read_text(encoding="utf-8")
)
for seed in (202, 303):
    expected = next(
        item["packing_npz_sha256"]
        for item in plan["packing_realisations"]
        if int(item["seed"]) == seed
    )
    path = root / f"packings/seed{seed}_packing.npz"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"seed{seed} packing hash does not match the plan")
reference = json.loads(
    (root / "reference/reference_seed101_mesh_manifest.json").read_text(encoding="utf-8")
)
if reference.get("new_physical_parameters") != []:
    raise SystemExit("reference mesh manifest adds an unexpected physical parameter")
PY

if find "${PACKAGE_DIR}" -mindepth 1 \
    \( -name polyMesh -o -name 'processor*' -o -name dynamicCode \
       -o -name postProcessing -o -name log.Allmesh -o -name 'log.checkMesh*' \) \
    -print -quit | grep -q .; then
  echo "generated mesh or solver output entered the preprocessing package" >&2
  exit 1
fi
if grep -REn \
    '^[[:space:]]*(foamMultiRun|mpirun|decomposePar|reconstructPar)([[:space:]]|$)' \
    "${PACKAGE_DIR}/scripts" >/dev/null; then
  echo "a heat-transfer solver command entered the mesh-only package" >&2
  exit 1
fi

(
  cd "${PACKAGE_DIR}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

if [[ ${CREATE_ARCHIVE} == 1 ]]; then
  if command -v zstd >/dev/null 2>&1; then
    archive=${OUTPUT_ROOT}/${PACKAGE_NAME}.tar.zst
    rm -f "${archive}"
    tar --zstd -cf "${archive}" -C "${OUTPUT_ROOT}" "${PACKAGE_NAME}"
  else
    archive=${OUTPUT_ROOT}/${PACKAGE_NAME}.tar.gz
    rm -f "${archive}"
    tar -czf "${archive}" -C "${OUTPUT_ROOT}" "${PACKAGE_NAME}"
  fi
  sha256sum "${archive}" > "${archive}.sha256"
  echo "archive=${archive}"
fi
echo "package_dir=${PACKAGE_DIR}"
