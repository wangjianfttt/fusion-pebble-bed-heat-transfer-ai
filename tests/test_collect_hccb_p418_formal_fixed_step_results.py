import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/collect_hccb_p418_formal_fixed_step_results.py"


def load_module():
    spec = importlib.util.spec_from_file_location("formal_collector", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_sequence(root: Path, sequence_id: str, value: float) -> None:
    task = root / "by_sequence" / sequence_id
    results = task / "results"
    results.mkdir(parents=True)
    artifact = results / "hccb_p418_transient_observables.npz"
    np.savez_compressed(
        artifact,
        case_id=np.asarray([sequence_id], dtype=object),
        complete=np.asarray([True]),
        conditions=np.asarray([[value]]),
        condition_names=np.asarray(["target"], dtype=object),
        time_s=np.asarray([[0.0, 1.0]]),
        time_mask=np.asarray([[True, True]]),
        values=np.asarray([[[value], [value + 1.0]]]),
        signal_names=np.asarray(["temperature"], dtype=object),
    )
    with (results / "hccb_p418_transient_observables_long.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition_id", "time_s"])
        writer.writeheader()
        writer.writerow({"condition_id": sequence_id, "time_s": 0.0})
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (task / "cloud_sequence_complete.json").write_text(
        json.dumps(
            {
                "status": "completed_p418_formal_fixed_hydrodynamics_sequence",
                "sequence_id": sequence_id,
                "observable_artifact_sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def test_combine_two_sequences(tmp_path: Path) -> None:
    combine = load_module().combine
    work = tmp_path / "work"
    write_sequence(work, "a", 1.0)
    write_sequence(work, "b", 2.0)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"sequences": [{"sequence_id": "a"}, {"sequence_id": "b"}]}),
        encoding="utf-8",
    )
    output = tmp_path / "combined"
    summary = combine(work, plan, output)
    assert summary["sequence_count"] == 2
    assert summary["completed_sequence_count"] == 2
    with np.load(output / "hccb_p418_formal_fixed_step_observables.npz", allow_pickle=True) as data:
        assert data["case_id"].tolist() == ["a", "b"]
        assert data["values"].shape == (2, 2, 1)
