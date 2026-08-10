#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/cloud_migration_build}
PACKAGE_NAME=${PACKAGE_NAME:-p418_seed303_contact_gap_repair_candidate}
PACKAGE_DIR=${OUTPUT_ROOT}/${PACKAGE_NAME}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}

case "${OUTPUT_ROOT}" in
  /home|/home/*)
    echo "cloud package output must not be under /home: ${OUTPUT_ROOT}" >&2
    exit 2
    ;;
esac

for required in \
  "${ROOT}/data/apd006_hccb_source_sequence_target_packings/seed303_s80_xlo_ycentre/packing.npz" \
  "${ROOT}/parameters/hccb_p418_seed303_contact_gap_repair_candidate.json" \
  "${ROOT}/cloud_migration/seed303_contact_gap_reference/original_case_manifest.json" \
  "${ROOT}/cloud_migration/seed303_contact_gap_reference/original_mesh_check_summary.json" \
  "${ROOT}/cloud_migration/run_seed303_contact_gap_mesh_repair.sh" \
  "${ROOT}/code/prepare_hccb_p418_seed303_contact_gap_repair.py" \
  "${ROOT}/code/build_hccb_dense_snappy_case.py" \
  "${ROOT}/code/build_hccb_pore_resolved_openfoam_mesh.py" \
  "${ROOT}/code/check_hccb_source_sequence_lammps_packing.py" \
  "${ROOT}/code/build_clipped_hccb_solid_surface_vtk.py" \
  "${ROOT}/code/summarize_hccb_dense_mesh_check.py" \
  "${ROOT}/code/run_with_resource_monitor.py"; do
  if [[ ! -f ${required} ]]; then
    echo "required seed303 repair input is missing: ${required}" >&2
    exit 1
  fi
done

rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/packings" "${PACKAGE_DIR}/parameters" \
  "${PACKAGE_DIR}/reference" "${PACKAGE_DIR}/code" "${PACKAGE_DIR}/scripts"
cp -a \
  "${ROOT}/data/apd006_hccb_source_sequence_target_packings/seed303_s80_xlo_ycentre/packing.npz" \
  "${PACKAGE_DIR}/packings/seed303_packing.npz"
cp -a "${ROOT}/parameters/hccb_p418_seed303_contact_gap_repair_candidate.json" \
  "${PACKAGE_DIR}/parameters/"
cp -a "${ROOT}/cloud_migration/seed303_contact_gap_reference/original_case_manifest.json" \
  "${PACKAGE_DIR}/reference/"
cp -a "${ROOT}/cloud_migration/seed303_contact_gap_reference/original_mesh_check_summary.json" \
  "${PACKAGE_DIR}/reference/"
cp -a "${ROOT}/cloud_migration/run_seed303_contact_gap_mesh_repair.sh" \
  "${PACKAGE_DIR}/scripts/"
for script in \
  prepare_hccb_p418_seed303_contact_gap_repair.py \
  build_hccb_dense_snappy_case.py \
  build_hccb_pore_resolved_openfoam_mesh.py \
  check_hccb_source_sequence_lammps_packing.py \
  build_clipped_hccb_solid_surface_vtk.py \
  summarize_hccb_dense_mesh_check.py \
  run_with_resource_monitor.py; do
  cp -a "${ROOT}/code/${script}" "${PACKAGE_DIR}/code/"
done
chmod +x "${PACKAGE_DIR}/scripts/"*.sh "${PACKAGE_DIR}/code/"*.py

cat > "${PACKAGE_DIR}/PACKAGE_STATUS.json" <<'JSON'
{
  "status": "seed303_contact_gap_repair_candidate_preflight_only",
  "execution_default": 0,
  "execution_approved": false,
  "mesh_generator_started_during_packaging": false,
  "heat_transfer_solver_included": false,
  "new_physical_parameters": []
}
JSON

if find "${PACKAGE_DIR}" -mindepth 1 \
  \( -name polyMesh -o -name 'processor*' -o -name postProcessing \
     -o -name log.Allmesh -o -name 'log.checkMesh*' \) \
  -print -quit | grep -q .; then
  echo "generated mesh or solver output entered the repair package" >&2
  exit 1
fi
if grep -REn \
  '^[[:space:]]*(foamMultiRun|mpirun|decomposePar|reconstructPar)([[:space:]]|$)' \
  "${PACKAGE_DIR}" >/dev/null; then
  echo "a heat-transfer solver command entered the repair package" >&2
  exit 1
fi

(
  cd "${PACKAGE_DIR}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

PREFLIGHT_OUTPUT=${PACKAGE_DIR}/seed303_contact_gap_repair_preflight.json \
  EXECUTE=0 \
  "${PACKAGE_DIR}/scripts/run_seed303_contact_gap_mesh_repair.sh" \
  > "${PACKAGE_DIR}/preflight.stdout.log"
if find "${PACKAGE_DIR}" -mindepth 1 \
  \( -name polyMesh -o -name 'processor*' -o -name log.Allmesh \) \
  -print -quit | grep -q .; then
  echo "preflight unexpectedly generated a mesh" >&2
  exit 1
fi

(
  cd "${PACKAGE_DIR}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

if [[ ${CREATE_ARCHIVE} == 1 ]]; then
  archive=${OUTPUT_ROOT}/${PACKAGE_NAME}.tar.gz
  rm -f "${archive}"
  COPYFILE_DISABLE=1 tar -czf "${archive}" -C "${OUTPUT_ROOT}" "${PACKAGE_NAME}"
  sha256sum "${archive}" > "${archive}.sha256"
  echo "archive=${archive}"
fi
echo "package_dir=${PACKAGE_DIR}"
