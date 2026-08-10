#!/usr/bin/env bash
# Convert one completed nine-case packing matrix into model-ready graph data.
# Default is a dry run. Set EXECUTE=1 after all nine OpenFOAM cases complete.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SEED=${SEED:-202}
EXECUTE=${EXECUTE:-0}
EXPECTED_CASES=9
ROLE_MANIFEST_TEMPLATE=${ROLE_MANIFEST:-${ROOT}/parameters/hccb_dense_cht_boundary_roles.json}

if [[ ${SEED} != 202 && ${SEED} != 303 ]]; then
    echo "SEED must be 202 or 303" >&2
    exit 1
fi
if [[ ${EXECUTE} != 0 && ${EXECUTE} != 1 ]]; then
    echo "EXECUTE must be 0 or 1" >&2
    exit 1
fi

MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_cross_packing_seed${SEED}_screen9}
DATASET_ROOT=${DATASET_ROOT:-${ROOT}/hccb_dense_cht_p418_cross_packing_seed${SEED}_screen9_dataset}
TOPOLOGY_DIR=${TOPOLOGY_DIR:-${ROOT}/results/hccb_p418_cross_packing_seed${SEED}_regional_topology}
MODEL_GEOMETRY_DIR=${MODEL_GEOMETRY_DIR:-${ROOT}/results/hccb_p418_cross_packing_seed${SEED}_model_geometry}
SUBFACE_DIR=${SUBFACE_DIR:-${ROOT}/results/hccb_p418_cross_packing_seed${SEED}_subface_geometry}
BOUNDARY_HEAT_DIR=${BOUNDARY_HEAT_DIR:-${ROOT}/results/hccb_p418_cross_packing_seed${SEED}_boundary_heat_flux_targets}
STATE_DIR=${STATE_DIR:-${ROOT}/results/hccb_p418_cross_packing_seed${SEED}_regional_state_targets}
MASS_DIR=${MASS_DIR:-${ROOT}/results/hccb_p418_cross_packing_seed${SEED}_regional_mass_flux_targets}
ENERGY_DIR=${ENERGY_DIR:-${ROOT}/results/hccb_p418_cross_packing_seed${SEED}_regional_energy_flux_targets}
SUMMARY=${SUMMARY:-${ROOT}/results/hccb_p418_cross_packing_seed${SEED}_postprocess_summary.json}
PACKING_ROLE_MANIFEST=${PACKING_ROLE_MANIFEST:-${ROOT}/results/hccb_p418_cross_packing_seed${SEED}_boundary_roles.json}

echo "seed${SEED} cross-packing post-processing"
echo "  OpenFOAM cases: ${MATRIX_ROOT}"
echo "  shared dataset: ${DATASET_ROOT}"
echo "  packing-specific regional graph: ${TOPOLOGY_DIR}"
echo "  packing-specific model geometry: ${MODEL_GEOMETRY_DIR}"
echo "  mass and energy targets: ${MASS_DIR}, ${ENERGY_DIR}"
echo "  normalization: seed101 training statistics (not rebuilt here)"

if [[ ${EXECUTE} == 0 ]]; then
    echo "dry run only: no dataset or graph was created"
    exit 0
fi

completed=$(find "${MATRIX_ROOT}" -mindepth 2 -maxdepth 2 \
    -name formal_sample_complete.json | wc -l | tr -d ' ')
if [[ ${completed} -ne ${EXPECTED_CASES} ]]; then
    echo "seed${SEED} matrix is incomplete: ${completed}/${EXPECTED_CASES}" >&2
    exit 1
fi

readarray -t reference < <(
    python3 - "${MATRIX_ROOT}" "${EXPECTED_CASES}" <<'PY'
import json
import pathlib
import sys

matrix = pathlib.Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
valid = []
problems = []
for marker_path in sorted(matrix.glob("*/formal_sample_complete.json")):
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    sample = pathlib.Path(marker["training_sample"])
    if not sample.is_absolute():
        sample = marker_path.parent / sample
    metadata = sample.parent / "metadata.json"
    if not sample.is_file() or not metadata.is_file():
        problems.append(f"{marker_path.parent.name}: sample or metadata missing")
        continue
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 3:
        problems.append(
            f"{marker_path.parent.name}: schema_version={payload.get('schema_version')}"
        )
        continue
    valid.append((marker_path.parent, sample.resolve()))
if len(valid) != expected:
    problems.append(f"valid schema-v3 samples: {len(valid)}/{expected}")
if problems:
    raise SystemExit("\n".join(problems))
print(valid[0][0])
print(valid[0][1])
PY
)
REFERENCE_CASE=${reference[0]}
REFERENCE_SAMPLE=${reference[1]}

python3 - "${ROLE_MANIFEST_TEMPLATE}" "${PACKING_ROLE_MANIFEST}" "${REFERENCE_CASE}" "${SEED}" <<'PY'
import json
import pathlib
import sys

template = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2]).resolve()
reference_case = pathlib.Path(sys.argv[3]).resolve()
seed = int(sys.argv[4])

payload = json.loads(template.read_text(encoding="utf-8"))
payload["case"] = str(reference_case)
payload["packing_seed"] = seed
payload["provenance"] = (
    "Physical patch roles are copied without modification from the declared "
    "solid-breeder CHT boundary-role template; only the reference-case path "
    "and packing seed are specialized for this independent packing."
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

if [[ ! -f ${REFERENCE_CASE}/native_multiregion_graph/native_multiregion_graph.npz \
    || ! -f ${REFERENCE_CASE}/boundary_faces/summary.json \
    || ! -f ${REFERENCE_CASE}/boundary_conditions/summary.json ]]; then
    CASE="${REFERENCE_CASE}" ROLE_MANIFEST="${PACKING_ROLE_MANIFEST}" \
        bash "${ROOT}/code/build_hccb_dense_cht_learning_geometry.sh"
fi

python3 "${ROOT}/code/build_hccb_p418_shared_mesh_dataset.py" \
    --matrix-root "${MATRIX_ROOT}" \
    --sample-paths-from-completion-markers \
    --expected-case-count "${EXPECTED_CASES}" \
    --require-completion-markers \
    --output-dir "${DATASET_ROOT}"

python3 "${ROOT}/code/build_hccb_p418_regional_topology.py" \
    --shared-topology "${REFERENCE_SAMPLE}" \
    --native-graph "${REFERENCE_CASE}/native_multiregion_graph/native_multiregion_graph.npz" \
    --levels 6 \
    --subsample-factor 4 \
    --output-dir "${TOPOLOGY_DIR}"

python3 "${ROOT}/code/build_hccb_p418_model_geometry.py" \
    --dataset-index "${DATASET_ROOT}/dataset_index.json" \
    --regional-topology "${TOPOLOGY_DIR}/regional_topology.npz" \
    --boundary-roles "${PACKING_ROLE_MANIFEST}" \
    --output-dir "${MODEL_GEOMETRY_DIR}"

python3 "${ROOT}/code/build_hccb_p418_subface_residual_geometry.py" \
    --dataset-index "${DATASET_ROOT}/dataset_index.json" \
    --regional-topology "${TOPOLOGY_DIR}/regional_topology.npz" \
    --native-graph "${REFERENCE_CASE}/native_multiregion_graph/native_multiregion_graph.npz" \
    --level 5 \
    --output-dir "${SUBFACE_DIR}"

python3 "${ROOT}/code/export_hccb_p418_boundary_heat_flux_targets.py" \
    --dataset-index "${DATASET_ROOT}/dataset_index.json" \
    --case-root "${MATRIX_ROOT}" \
    --time-from-completion-marker \
    --output-dir "${BOUNDARY_HEAT_DIR}"

python3 "${ROOT}/code/build_hccb_p418_regional_state_targets.py" \
    --dataset-index "${DATASET_ROOT}/dataset_index.json" \
    --subface-geometry "${SUBFACE_DIR}/subface_residual_geometry.npz" \
    --output-dir "${STATE_DIR}"

python3 "${ROOT}/code/build_hccb_p418_regional_mass_flux_targets.py" \
    --dataset-index "${DATASET_ROOT}/dataset_index.json" \
    --regional-topology "${TOPOLOGY_DIR}/regional_topology.npz" \
    --level 5 \
    --output-dir "${MASS_DIR}"

python3 "${ROOT}/code/build_hccb_p418_regional_energy_flux_targets.py" \
    --dataset-index "${DATASET_ROOT}/dataset_index.json" \
    --regional-topology "${TOPOLOGY_DIR}/regional_topology.npz" \
    --native-graph "${REFERENCE_CASE}/native_multiregion_graph/native_multiregion_graph.npz" \
    --boundary-heat-targets "${BOUNDARY_HEAT_DIR}/boundary_heat_flux_targets.npz" \
    --level 5 \
    --output-dir "${ENERGY_DIR}"

python3 - \
    "${SEED}" \
    "${DATASET_ROOT}/dataset_index.json" \
    "${TOPOLOGY_DIR}/summary.json" \
    "${MODEL_GEOMETRY_DIR}/summary.json" \
    "${SUBFACE_DIR}/summary.json" \
    "${BOUNDARY_HEAT_DIR}/summary.json" \
    "${STATE_DIR}/summary.json" \
    "${MASS_DIR}/summary.json" \
    "${ENERGY_DIR}/summary.json" \
    "${SUMMARY}" <<'PY'
import hashlib
import json
import pathlib
import sys

seed = int(sys.argv[1])
paths = [pathlib.Path(value).resolve() for value in sys.argv[2:-1]]
output = pathlib.Path(sys.argv[-1]).resolve()
documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
dataset, topology, geometry, subface, boundary_heat, state, mass, energy = documents
energy_ready_statuses = {
    "p418_regional_energy_flux_targets_ready",
    "p418_regional_energy_flux_targets_ready_with_reported_interface_mismatch",
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

conditions = dataset.get("conditions", [])
checks = {
    "nine_conditions": len(conditions) == 9,
    "regional_topology_ready": topology.get("status")
    == "p418_multiregion_regional_topology_ready",
    "model_geometry_ready": geometry.get("status")
    == "p418_regional_model_geometry_ready",
    "subface_geometry_ready": subface.get("status")
    == "subface_residual_geometry_ready",
    "boundary_heat_targets_ready": boundary_heat.get("status")
    == "p418_boundary_heat_flux_targets_ready",
    "regional_state_targets_ready": state.get("status")
    == "regional_state_targets_ready",
    "regional_mass_targets_ready": mass.get("status")
    == "regional_mass_flux_targets_ready",
    "regional_energy_targets_ready": energy.get("status") in energy_ready_statuses,
    "packing_specific_topology": str(seed) in str(paths[1].parent),
    "packing_specific_geometry": str(seed) in str(paths[2].parent),
}
strict_interface_reference_met = (
    energy.get("status") == "p418_regional_energy_flux_targets_ready"
)
base_ready = all(checks.values())
status = (
    "cross_packing_model_inputs_ready"
    if base_ready and strict_interface_reference_met
    else "cross_packing_model_inputs_ready_with_reported_interface_mismatch"
    if base_ready
    else "failed"
)
payload = {
    "status": status,
    "packing_seed": seed,
    "condition_count": len(conditions),
    "checks": checks,
    "files": {
        "dataset_index": str(paths[0]),
        "dataset_index_sha256": digest(paths[0]),
        "regional_topology": str(paths[1].parent / "regional_topology.npz"),
        "regional_topology_sha256": digest(paths[1].parent / "regional_topology.npz"),
        "model_geometry": str(paths[2].parent / "model_geometry.npz"),
        "model_geometry_sha256": digest(paths[2].parent / "model_geometry.npz"),
        "subface_geometry": str(paths[3].parent / "subface_residual_geometry.npz"),
        "subface_geometry_sha256": digest(paths[3].parent / "subface_residual_geometry.npz"),
        "boundary_heat_targets": str(paths[4].parent / "boundary_heat_flux_targets.npz"),
        "boundary_heat_targets_sha256": digest(paths[4].parent / "boundary_heat_flux_targets.npz"),
        "regional_state_targets": str(paths[5].parent / "regional_state_targets.npz"),
        "regional_state_targets_sha256": digest(paths[5].parent / "regional_state_targets.npz"),
        "regional_mass_targets": str(paths[6].parent / "regional_mass_flux_targets.npz"),
        "regional_mass_targets_sha256": digest(paths[6].parent / "regional_mass_flux_targets.npz"),
        "regional_energy_targets": str(paths[7].parent / "regional_energy_flux_targets.npz"),
        "regional_energy_targets_sha256": digest(paths[7].parent / "regional_energy_flux_targets.npz"),
    },
    "normalization_statement": (
        "Evaluation uses seed101 training statistics. No statistics are fitted "
        "from this independent packing."
    ),
    "interface_energy_statement": (
        "The fluid-solid target uses the symmetric mean of the two independently "
        "reconstructed interface powers. The strict seed101-side consistency reference "
        f"is {'met' if strict_interface_reference_met else 'not met'}; the measured "
        "pre-averaging mismatch is retained in the energy-target summary."
    ),
    "new_physical_parameter_values_added": [],
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if payload["status"] == "failed":
    raise SystemExit("cross-packing model inputs failed")
PY
