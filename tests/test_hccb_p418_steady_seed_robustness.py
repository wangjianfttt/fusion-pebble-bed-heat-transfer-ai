from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/summarize_hccb_p418_steady_seed_robustness.py"
RUNNER = ROOT / "code/run_hccb_p418_steady_seed_robustness.sh"
ARCHITECTURES = ("pinn_data_only", "pinn", "graph", "transolver")
SEEDS = (20260717, 20260718, 20260719)


def fake_summary(architecture: str, seed: int) -> dict:
    cases = []
    for condition, scale in (("d", 1.0), ("e", 2.0)):
        cases.append(
            {
                "condition_id": condition,
                "generated_power_W": 100.0,
                "engineering_absolute_errors": {
                    "solid_maximum_temperature_K": scale * seed / 1.0e7,
                    "pressure_drop_Pa": scale,
                    "cooling_wall_heat_into_fluid_W": scale * 2.0,
                },
                "local_energy_l1_over_two_generated_power": scale * 0.01,
            }
        )
    return {
        "architecture": architecture,
        "split_name": "interleaved_all_ranges",
        "split_case_ids": {
            "train": ["a", "b"],
            "validation": ["c"],
            "test": ["d", "e"],
        },
        "epochs": 100,
        "training_seed": seed,
        "run_provenance": {"common_comparison_fingerprint": "same-fields"},
        "evaluations": {
            "test": {
                "metrics": {"state_channel_rmse": [0, 0, 0, 0, 0.1, 0.2]},
                "cases": cases,
            }
        },
    }


def make_inputs(tmp_path: Path) -> tuple[Path, Path]:
    split = tmp_path / "splits.json"
    split.write_text(
        json.dumps(
            {
                "splits": {
                    "interleaved_all_ranges": {
                        "train": ["a", "b"],
                        "validation": ["c"],
                        "test": ["d", "e"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            suffix = "" if seed == SEEDS[0] else f"_seed{seed}"
            directory = (
                results
                / f"hccb_p418_60_{architecture}_interleaved_all_ranges_100epoch{suffix}"
            )
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(
                json.dumps(fake_summary(architecture, seed)), encoding="utf-8"
            )
    return split, results


def command(tmp_path: Path, split: Path, results: Path) -> list[str]:
    return [
        "python3",
        str(SCRIPT),
        "--results-root",
        str(results),
        "--split-file",
        str(split),
        "--output-dir",
        str(tmp_path / "summary"),
        "--tex-output",
        str(tmp_path / "seed_table.tex"),
        "--text-output",
        str(tmp_path / "seed_text.tex"),
    ]


def test_three_seed_summary_and_tex_are_generated(tmp_path: Path) -> None:
    split, results = make_inputs(tmp_path)
    completed = subprocess.run(command(tmp_path, split, results), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    rows = list(
        csv.DictReader((tmp_path / "summary/steady_seed_metrics.csv").open(encoding="utf-8"))
    )
    assert len(rows) == 12
    assert all(not Path(row["source_summary"]).is_absolute() for row in rows)
    assert all(row["source_summary"].startswith("results/") for row in rows)
    aggregate = list(
        csv.DictReader((tmp_path / "summary/steady_seed_summary.csv").open(encoding="utf-8"))
    )
    assert len(aggregate) == 20
    assert {int(row["seed_count"]) for row in aggregate} == {3}
    tex = (tmp_path / "seed_table.tex").read_text(encoding="utf-8")
    assert "mean $\\pm$ sample standard deviation" in tex
    assert "Physics PINN" in tex
    result_text = (tmp_path / "seed_text.tex").read_text(encoding="utf-8")
    assert "three independent initializations" in result_text
    assert "does not imply local energy closure" in result_text
    assert "wall-heat p95 error" in result_text
    assert (tmp_path / "summary/README_CN.md").is_file()


def test_wrong_recorded_seed_is_rejected(tmp_path: Path) -> None:
    split, results = make_inputs(tmp_path)
    wrong = (
        results
        / "hccb_p418_60_graph_interleaved_all_ranges_100epoch_seed20260718"
        / "summary.json"
    )
    payload = json.loads(wrong.read_text(encoding="utf-8"))
    payload["training_seed"] = 0
    wrong.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(command(tmp_path, split, results), capture_output=True, text=True)
    assert completed.returncode != 0
    assert "does not record seed 20260718" in completed.stderr


def test_runner_uses_three_seeds_without_repeating_response_surface() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    text = RUNNER.read_text(encoding="utf-8")
    assert "MODEL_SEEDS=${MODEL_SEEDS:-20260717 20260718 20260719}" in text
    assert "ARCHITECTURES=${ARCHITECTURES:-pinn_data_only pinn graph transolver}" in text
    assert "response_surface" not in text
    assert '--seed "${seed}"' in text
    assert '--training-seed "${seed}"' in text
    assert 'TEX_OUTPUT=${TEX_OUTPUT:-${ROOT}/manuscript/generated_steady_seed_robustness.tex}' in text
    assert '--tex-output "${TEX_OUTPUT}"' in text
