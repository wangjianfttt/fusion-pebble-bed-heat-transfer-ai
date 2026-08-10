from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_fixed_flow_loss_balancing_protocol.py"
SPEC = importlib.util.spec_from_file_location("fixed_flow_loss_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_candidate_arguments_cover_recorded_fixed_flow_candidates() -> None:
    sources = json.loads(MODULE.DEFAULT_SOURCES.read_text(encoding="utf-8"))
    candidates = sources["formal_candidates"]
    assert len(candidates) == 4
    for candidate in candidates:
        arguments = MODULE.candidate_arguments(candidate)
        assert arguments == [
            "--loss-balancing-candidate-id",
            candidate["candidate_id"],
            "--loss-balancing-sources",
            str(MODULE.DEFAULT_SOURCES.resolve()),
        ]


def test_common_arguments_keep_the_formal_physics_model_unchanged(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        dataset_index=tmp_path / "dataset_index.json",
        splits=tmp_path / "splits.json",
        split_name="pair_disjoint_stress_test",
        residual_geometry=tmp_path / "geometry.npz",
        physics_device="cuda",
        seed=20260717,
        torch_threads=None,
    )
    arguments = MODULE.common_arguments(args, tmp_path / "output")
    assert arguments[arguments.index("--run-role") + 1] == "formal"
    assert arguments[arguments.index("--physics-mode") + 1] == "energy_and_flux"
    assert arguments[arguments.index("--physics-device") + 1] == "cuda"
    assert (
        arguments[arguments.index("--temperature-output-mode") + 1]
        == "literature_bounded_residual"
    )
