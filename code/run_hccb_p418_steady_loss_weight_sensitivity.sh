#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results}
RESULT_PREFIX=${RESULT_PREFIX:-${RESULT_ROOT}/hccb_p418_60_sourceflow_r3}
SETTINGS=${SETTINGS:-${ROOT}/parameters/hccb_p418_steady_loss_weight_sensitivity.csv}
SELECTION=${SELECTION:-${RESULT_ROOT}/hccb_p418_steady_chain_source.json}
OUTPUT_DIR=${OUTPUT_DIR:-${RESULT_ROOT}/hccb_p418_steady_loss_weight_sensitivity}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${RESULT_ROOT}/hccb_p418_regional_topology_r2/regional_topology.npz}
MODEL_GEOMETRY=${MODEL_GEOMETRY:-${RESULT_PREFIX}_model_geometry/model_geometry.npz}
STATE_TARGETS=${STATE_TARGETS:-${RESULT_PREFIX}_regional_state_targets/regional_state_targets.npz}
MASS_TARGETS=${MASS_TARGETS:-${RESULT_PREFIX}_regional_mass_flux_targets/regional_mass_flux_targets.npz}
ENERGY_TARGETS=${ENERGY_TARGETS:-${RESULT_PREFIX}_regional_energy_flux_targets/regional_energy_flux_targets.npz}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_model_splits.json}
TRAINING_STATISTICS=${TRAINING_STATISTICS:-${RESULT_PREFIX}_training_statistics.json}
THREADS=${THREADS:-4}
DEVICE=${DEVICE:-cuda}
MODEL_SEED=${MODEL_SEED:-20260717}

readarray -t selection_values < <(python3 - "${SELECTION}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
if p.get("status") != "steady_PINN_chain_source_selected":
    raise SystemExit("steady PINN source has not been selected")
summary = json.load(open(p["selected_summary"], encoding="utf-8"))
if summary.get("architecture") != "pinn":
    raise SystemExit("selected steady source is not a PINN")
print(p["selected_epochs"])
print(p["split_name"])
print(summary["effective_batch_size"])
print(summary["microbatch_size"])
print(summary["training_seed"])
print(summary["optimizer_name"])
PY
)
epochs=${selection_values[0]}
split_name=${selection_values[1]}
effective_batch_size=${selection_values[2]}
microbatch_size=${selection_values[3]}
selected_seed=${selection_values[4]}
selected_optimizer=${selection_values[5]}
if [[ ${selected_optimizer} != Adam ]]; then
  echo "selected steady PINN does not use Adam" >&2
  exit 1
fi
if [[ ${MODEL_SEED} != ${selected_seed} ]]; then
  echo "MODEL_SEED differs from the selected steady PINN" >&2
  exit 1
fi

while IFS=$'\t' read -r setting_id state_weight face_weight physics_weight; do
  [[ -n ${setting_id} ]] || continue
  if [[ ${setting_id} == pino_standard_5_1_1 ]]; then
    continue
  fi
  output=${RESULT_ROOT}/hccb_p418_loss_weight_${setting_id}_${epochs}epoch
  current=0
  if [[ -f ${output}/summary.json ]] && python3 "${ROOT}/code/check_hccb_p418_steady_result_current.py" \
      --summary "${output}/summary.json" --architecture pinn --epochs "${epochs}" \
      --split-name "${split_name}" --state-targets "${STATE_TARGETS}" \
      --mass-targets "${MASS_TARGETS}" --energy-targets "${ENERGY_TARGETS}" \
      --split-file "${SPLITS}" --training-statistics "${TRAINING_STATISTICS}" \
      --training-seed "${MODEL_SEED}" >/dev/null; then
    if python3 - "${output}/summary.json" "${state_weight}" "${face_weight}" "${physics_weight}" "${effective_batch_size}" "${microbatch_size}" "${selected_optimizer}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"state_data": float(sys.argv[2]), "face_flux": float(sys.argv[3]), "physics_balance": float(sys.argv[4])}
same = (
    p.get("loss_group_weights") == expected
    and int(p.get("effective_batch_size", -1)) == int(sys.argv[5])
    and int(p.get("microbatch_size", -1)) == int(sys.argv[6])
    and p.get("optimizer_name") == sys.argv[7]
)
raise SystemExit(0 if same else 1)
PY
    then
      current=1
    fi
  fi
  if [[ ${current} -eq 1 ]]; then
    echo "reuse ${setting_id} ${epochs} epochs"
    continue
  fi
  if [[ -e ${output} ]]; then
    mv "${output}" "${output}.older.$(date +%Y%m%dT%H%M%S).$$"
  fi
  mkdir -p "${output}"
  python3 "${ROOT}/code/train_hccb_p418_conservative_mixed_operator.py" \
    --regional-topology "${REGIONAL_TOPOLOGY}" --model-geometry "${MODEL_GEOMETRY}" \
    --state-targets "${STATE_TARGETS}" --mass-targets "${MASS_TARGETS}" \
    --energy-targets "${ENERGY_TARGETS}" --split-file "${SPLITS}" \
    --training-statistics "${TRAINING_STATISTICS}" --split-name "${split_name}" \
    --regional-level 5 --architecture pinn --epochs "${epochs}" --threads "${THREADS}" \
    --device "${DEVICE}" --seed "${selected_seed}" \
    --effective-batch-size "${effective_batch_size}" --microbatch-size "${microbatch_size}" \
    --state-data-weight "${state_weight}" --face-flux-weight "${face_weight}" \
    --physics-balance-weight "${physics_weight}" --output-dir "${output}" \
    > "${output}/run.log" 2>&1
done < <(
  python3 - "${SETTINGS}" "${ROOT}" <<'PY'
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[2]) / "code"))
from summarize_hccb_p418_steady_loss_weight_sensitivity import load_settings
for row in load_settings(Path(sys.argv[1]), Path(sys.argv[2])):
    print(row["setting_id"], row["state_data_weight"], row["face_flux_weight"], row["physics_balance_weight"], sep="\t")
PY
)

python3 "${ROOT}/code/summarize_hccb_p418_steady_loss_weight_sensitivity.py" \
  --project-root "${ROOT}" --settings "${SETTINGS}" --selection "${SELECTION}" \
  --result-root "${RESULT_ROOT}" --output-dir "${OUTPUT_DIR}"
