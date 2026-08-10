from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/assemble_hccb_p418_corrected_steady_results.py"
METHODS = ("response_surface", "pinn_data_only", "pinn", "graph", "transolver")
SPLITS = (
    "interleaved_all_ranges",
    "temperature_extrapolation",
    "velocity_extrapolation",
    "heat_source_interpolation",
    "heat_source_extrapolation",
)


def populate(root: Path, corrected_complete: bool = True) -> None:
    for method in METHODS:
        for split in SPLITS:
            namespace = "corrected" if split == "heat_source_extrapolation" else "formal"
            result = root / f"{namespace}_{method}_{split}_100epoch"
            result.mkdir(parents=True)
            status = "training_complete"
            if not corrected_complete and namespace == "corrected" and method == "graph":
                status = "running"
            (result / "summary.json").write_text(
                json.dumps({"status": status}), encoding="utf-8"
            )
            for role in ("train", "validation", "test"):
                (result / f"{role}_regional_predictions.npz").write_bytes(
                    f"{method}:{split}:{role}".encode()
                )


def command(results: Path, manifest: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--results-root",
        str(results),
        "--source-namespace",
        "formal",
        "--corrected-namespace",
        "corrected",
        "--output-namespace",
        "assembled",
        "--manifest",
        str(manifest),
    ]


def test_assembles_only_after_all_sources_are_complete(tmp_path: Path) -> None:
    results = tmp_path / "results"
    populate(results, corrected_complete=False)
    process = subprocess.run(command(results, tmp_path / "manifest.json"), text=True)
    assert process.returncode != 0
    assert not list(results.glob("assembled_*"))


def test_assembles_twenty_five_traceable_links(tmp_path: Path) -> None:
    results = tmp_path / "results"
    populate(results)
    manifest = tmp_path / "manifest.json"
    subprocess.run(command(results, manifest), check=True, capture_output=True, text=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "corrected_steady_result_assembly_complete"
    assert payload["result_count"] == 25
    assert len(list(results.glob("assembled_*"))) == 25
    for method in METHODS:
        corrected = results / f"assembled_{method}_heat_source_extrapolation_100epoch"
        unchanged = results / f"assembled_{method}_temperature_extrapolation_100epoch"
        assert corrected.is_symlink()
        assert "corrected_" in corrected.resolve().name
        assert unchanged.is_symlink()
        assert "formal_" in unchanged.resolve().name
