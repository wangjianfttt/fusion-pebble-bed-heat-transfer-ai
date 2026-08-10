#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
PILOT_ROOT=${PILOT_ROOT:-${ROOT}/hccb_dense_cht_p418_pilot}
TIME_NAME=${TIME_NAME:-300}
SAMPLE_NAME=${SAMPLE_NAME:-training_sample_${TIME_NAME}_schema3}

updated=0
for marker in "${MATRIX_ROOT}"/*/formal_sample_complete.json; do
  [[ -f ${marker} ]] || continue
  case_dir=$(dirname "${marker}")
  condition=$(basename "${case_dir}")
  source_dir=${PILOT_ROOT}/${condition}/${SAMPLE_NAME}
  target_dir=${case_dir}/${SAMPLE_NAME}

  if [[ ! -f ${target_dir}/fields_and_topology.npz ]]; then
    if [[ ! -f ${source_dir}/fields_and_topology.npz ]]; then
      echo "missing schema-v3 source for completed condition: ${condition}" >&2
      exit 1
    fi
    cp -al "${source_dir}" "${target_dir}"
  fi

  python3 - "${marker}" "${target_dir}" <<'PY'
import hashlib
import json
import pathlib
import sys

marker = pathlib.Path(sys.argv[1])
sample_dir = pathlib.Path(sys.argv[2])
sample = sample_dir / "fields_and_topology.npz"
metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
if metadata.get("schema_version") != 3:
    raise SystemExit(f"not a schema-v3 sample: {sample_dir}")
payload = json.loads(marker.read_text(encoding="utf-8"))
payload["training_sample"] = str(sample.resolve())
payload["training_sample_sha256"] = hashlib.sha256(sample.read_bytes()).hexdigest()
payload["training_sample_schema_version"] = 3
marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  updated=$((updated + 1))
done

echo "schema-v3 samples prepared for ${updated} completed matrix cases"
