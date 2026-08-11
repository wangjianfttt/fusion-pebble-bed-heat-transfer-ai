#!/usr/bin/env bash
#SBATCH --job-name=p418_gci_cmp
#SBATCH --partition=cn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=p418_gci_compare_%j.out
#SBATCH --error=p418_gci_compare_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?export the project root on the target cluster}
PACKAGE_ROOT=${PACKAGE_ROOT:-${PROJECT_ROOT}/incoming/p418_gci_solver_runtime_20260729}
RESULT_ROOT=${RESULT_ROOT:-${PROJECT_ROOT}/work/gci_solver_20260729}
SOLVER_ARRAY_JOB_ID=${SOLVER_ARRAY_JOB_ID:?export the completed GCI solver array job id}
OUTPUT_DIR=${RESULT_ROOT}/comparison_${SOLVER_ARRAY_JOB_ID}
coarse=${RESULT_ROOT}/cases/coarse_${SOLVER_ARRAY_JOB_ID}_0
medium=${RESULT_ROOT}/cases/medium_${SOLVER_ARRAY_JOB_ID}_1

test -f "${coarse}/mesh_sensitivity_complete.json"
test -f "${medium}/mesh_sensitivity_complete.json"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

python3 "${PACKAGE_ROOT}/code/summarize_hccb_p418_mesh_sensitivity.py" \
  --coarse-mesh "${coarse}/mesh_summary.json" \
  --medium-mesh "${medium}/mesh_summary.json" \
  --fine-mesh "${PACKAGE_ROOT}/reference/fine_mesh_summary.json" \
  --coarse-result "${coarse}/cht_result_summary_200.json" \
  --medium-result "${medium}/cht_result_summary_200.json" \
  --fine-result "${PACKAGE_ROOT}/reference/fine_cht_result_summary_200.json" \
  --output-dir "${OUTPUT_DIR}"

python3 "${PACKAGE_ROOT}/code/build_hccb_p418_mesh_sensitivity_table.py" \
  --input-summary "${OUTPUT_DIR}/summary.json" \
  --output "${OUTPUT_DIR}/generated_mesh_sensitivity.tex"

python3 - "${OUTPUT_DIR}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if summary.get("status") != "completed_three_mesh_p418_cht_comparison":
    raise SystemExit("three-mesh comparison is incomplete")
payload = {
    "status": "completed_p418_three_mesh_gci_postprocess",
    "summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
    "gci_csv_sha256": hashlib.sha256((root / "mesh_gci.csv").read_bytes()).hexdigest(),
    "new_physical_parameters": [],
}
(root / "GCI_COMPARISON_COMPLETE.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY
