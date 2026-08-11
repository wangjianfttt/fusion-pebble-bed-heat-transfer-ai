#!/usr/bin/env bash
#SBATCH --job-name=p418_gci_cht
#SBATCH --partition=cn
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=p418_gci_cht_%A_%a.out
#SBATCH --error=p418_gci_cht_%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?export the project root on the target cluster}
PACKAGE_ROOT=${PACKAGE_ROOT:-${PROJECT_ROOT}/incoming/p418_gci_solver_runtime_20260729}
PHYSICS_PACKAGE=${PHYSICS_PACKAGE:-${PROJECT_ROOT}/incoming/p418_openfoam13_pending_46}
OPENFOAM_ROOT=${OPENFOAM_ROOT:-${PROJECT_ROOT}/software/of13_20260624_r2/OpenFOAM-13}
RESULT_ROOT=${RESULT_ROOT:-${PROJECT_ROOT}/work/gci_solver_20260729}
CONDITION_ID=${CONDITION_ID:-u0p20_T700_q6p85}

levels=(coarse medium)
mesh_roots=(
  "${PROJECT_ROOT}/work/gci_mesh_preflight_20260729/gci_local20_l2_14573"
  "${PROJECT_ROOT}/work/gci_mesh_preflight_20260729/gci_local30_l2_14571_0"
)
level=${levels[${SLURM_ARRAY_TASK_ID:?}]}
mesh_root=${mesh_roots[${SLURM_ARRAY_TASK_ID}]}
source_template=${PHYSICS_PACKAGE}/case_inputs/${CONDITION_ID}
prepared_template=${RESULT_ROOT}/prepared_inputs/${level}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
run_root=${RESULT_ROOT}/cases/${level}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

test -f "${mesh_root}/mesh_summary.json"
python3 - "${mesh_root}/mesh_summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("all_checks_pass") is not True:
    raise SystemExit("candidate mesh did not pass both basic checkMesh runs")
PY
test -d "${mesh_root}/constant/fluid/polyMesh"
test -d "${mesh_root}/constant/solid/polyMesh"
test -d "${source_template}"
test -x "${PHYSICS_PACKAGE}/scripts/run_openfoam13_case.sh"
test ! -e "${prepared_template}"
test ! -e "${run_root}"
(cd "${source_template}" && sha256sum -c INPUT_SHA256SUMS)

mkdir -p "$(dirname "${prepared_template}")" "$(dirname "${run_root}")"
cp -a "${source_template}" "${prepared_template}"
cp -a "${mesh_root}/mesh_summary.json" "${prepared_template}/mesh_summary.json"
cp -a "${mesh_root}/case_manifest.json" "${prepared_template}/mesh_case_manifest.json"
python3 - "${prepared_template}" "${level}" "${mesh_root}" <<'PY'
import hashlib
import json
import pathlib
import sys

case = pathlib.Path(sys.argv[1])
level = sys.argv[2]
mesh_root = pathlib.Path(sys.argv[3])
metadata_path = case / "cht_smoke_metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata["mesh_resolution_label"] = level
metadata["mesh_case"] = str(mesh_root)
metadata["mesh_case_manifest_sha256"] = hashlib.sha256(
    (case / "mesh_case_manifest.json").read_bytes()
).hexdigest()
metadata["mesh_check_summary_sha256"] = hashlib.sha256(
    (case / "mesh_summary.json").read_bytes()
).hexdigest()
metadata["new_physical_parameters"] = []
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY

mkdir -p "${prepared_template}/mesh_hashes"
(
  cd "${mesh_root}/constant"
  find fluid/polyMesh solid/polyMesh -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "${prepared_template}/mesh_hashes/polyMesh_SHA256SUMS"
)

export OPENFOAM_BASHRC=${OPENFOAM_ROOT}/etc/bashrc
export CASE_TEMPLATE=${prepared_template}
export SHARED_MESH_ROOT=${mesh_root}/constant
export RUN_ROOT=${run_root}
export NP=${SLURM_NTASKS:-32}
export END_TIME_OVERRIDE=200
export REFERENCE_SUMMARY=${PACKAGE_ROOT}/reference/fine_cht_result_summary_200.json
export POSTPROCESS_SCRIPT=${PHYSICS_PACKAGE}/scripts/postprocess_openfoam13_case.sh
export PYTHON_BIN=python3
export ZSH_NAME=${ZSH_NAME:-}

module purge
module load openmpi/4.1.6_gcc11.5
set +u
# shellcheck disable=SC1090
source "${OPENFOAM_BASHRC}"
set -u

bash "${PHYSICS_PACKAGE}/scripts/run_openfoam13_case.sh"
cp -a "${prepared_template}/mesh_summary.json" "${run_root}/mesh_summary.json"
cp -a "${prepared_template}/mesh_case_manifest.json" "${run_root}/mesh_case_manifest.json"
cp -a "${prepared_template}/mesh_hashes" "${run_root}/mesh_hashes"

python3 - "${run_root}" "${CONDITION_ID}" "${level}" <<'PY'
import hashlib
import json
import pathlib
import sys

case = pathlib.Path(sys.argv[1])
condition = sys.argv[2]
level = sys.argv[3]
summary_path = case / "cht_result_summary_200.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("solver_finished") is not True:
    raise SystemExit("OpenFOAM steady iteration did not finish")
if summary.get("all_reported_values_are_finite") is not True:
    raise SystemExit("OpenFOAM result contains non-finite values")
payload = {
    "status": "completed_p418_gci_mesh_level",
    "condition_id": condition,
    "mesh_level": level,
    "reported_iteration": 200,
    "result_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "mesh_summary_sha256": hashlib.sha256(
        (case / "mesh_summary.json").read_bytes()
    ).hexdigest(),
    "new_physical_parameters": [],
}
(case / "mesh_sensitivity_complete.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY
