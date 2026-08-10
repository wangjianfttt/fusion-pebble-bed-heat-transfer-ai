#!/usr/bin/env bash
# Reproduce the P418 paper in explicit stages without starting a solver by default.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-preflight}
PYTHON_BIN=${PYTHON:-python3}

usage() {
  cat <<'EOF'
Usage: bash scripts/reproduce_p418_paper.sh MODE

Modes:
  preflight    Read-only parameter, route, and data-completeness checks.
  manifest     Rebuild the reproducibility file/checksum manifest.
  archive      Build the deterministic small source archive.
  postprocess  Process already completed formal OpenFOAM results.
  paper        Regenerate final values, figures, manuscript, and supplement.
  help         Show this message.

This script never starts a formal OpenFOAM solver or model training.
Formal calculations remain behind: make p418-formal-plan / make p418-formal-run
EOF
}

build_manifest() {
  "${PYTHON_BIN}" "${ROOT}/code/build_hccb_p418_public_figure_data.py" \
    --project-root "${ROOT}" \
    --output-dir "${ROOT}/results/hccb_p418_public_figure_data"
  "${PYTHON_BIN}" "${ROOT}/code/build_hccb_p418_public_data_release.py" \
    --project-root "${ROOT}" \
    --output-dir "${ROOT}/results/hccb_p418_public_data_release_preflight"
  "${PYTHON_BIN}" "${ROOT}/code/build_hccb_p418_reproducibility_manifest.py" \
    --project-root "${ROOT}" \
    --output-dir "${ROOT}/results/hccb_p418_reproducibility_manifest"
}

build_archive() {
  build_manifest
  "${PYTHON_BIN}" "${ROOT}/code/package_hccb_p418_reproducibility_source.py" \
    --project-root "${ROOT}" \
    --manifest "${ROOT}/results/hccb_p418_reproducibility_manifest/manifest.json" \
    --output "${ROOT}/results/hccb_p418_reproducibility_manifest/p418_reproduction_source.tar.gz" \
    --record "${ROOT}/results/hccb_p418_reproducibility_manifest/source_archive_record.json"
}

case "${MODE}" in
  preflight)
    make -C "${ROOT}" p418-fused-preflight
    make -C "${ROOT}" p418-research-route-check
    build_manifest
    ;;
  manifest)
    build_manifest
    ;;
  archive)
    build_archive
    ;;
  postprocess)
    bash "${ROOT}/code/run_hccb_p418_60_postprocess.sh"
    build_manifest
    ;;
  paper)
    bash "${ROOT}/code/run_hccb_p418_manuscript_refresh.sh"
    build_manifest
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
