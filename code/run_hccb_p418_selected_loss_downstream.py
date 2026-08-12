#!/usr/bin/env python3
"""Build all final strict-split models from the validation-selected loss scheme."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from hccb_p418_selected_fixed_flow_chain import (
    INTEGRATION_RECORD_NAME,
    LOSS_ROOT_NAME,
    STRICT_SPLIT,
    load_json,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "code/train_hccb_p418_spatiotemporal_regional_operator.py"
LOW_RANK = ROOT / "code/train_hccb_p418_low_rank_temperature_residual.py"
DIFFUSION = ROOT / "code/train_hccb_p418_temporal_temperature_diffusion.py"
ENERGY = ROOT / "code/evaluate_hccb_p418_temporal_energy_balance.py"
SOURCES = ROOT / "parameters/hccb_p418_fixed_flow_loss_balancing_candidates.json"


def run(command: list[str], *, execute: bool) -> None:
    print("+", shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, check=True)


ENERGY_STATUSES = (
    "completed_p418_common_transient_energy_balance",
    "completed_p418_common_transient_energy_balance_with_rejected_roles",
)


def has_status(path: Path, expected: str | tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        expected_values = (expected,) if isinstance(expected, str) else expected
        return load_json(path).get("status") in expected_values
    except (OSError, json.JSONDecodeError):
        return False


def energy_command(
    *, summary: Path, dataset_index: Path, residual_geometry: Path, output: Path
) -> list[str]:
    return [
        sys.executable,
        str(ENERGY),
        "--model-summary",
        str(summary),
        "--dataset-index",
        str(dataset_index),
        "--residual-geometry",
        str(residual_geometry),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]


def require_selected_final(loss_root: Path) -> tuple[dict, dict, Path]:
    selection_path = loss_root / "selected_loss_balancing_method.json"
    selection = load_json(selection_path)
    if selection.get("status") != "p418_loss_balancing_selected_on_validation_only":
        raise ValueError("loss-weight selection is incomplete")
    if selection.get("independent_test_read") is not False:
        raise ValueError("the loss-weight selection read the independent test too early")
    selected_id = str(selection["selected_candidate_id"])
    selected_dir = loss_root / selected_id
    final_path = selected_dir / "final_summary.json"
    final = load_json(final_path)
    if final.get("status") != "completed_p418_spatiotemporal_regional_operator":
        raise ValueError("selected physics model is incomplete")
    if final.get("evaluation_stage") != "final" or final.get("test_evaluated") is not True:
        raise ValueError("selected physics model lacks the one-time final test evaluation")
    if final.get("split_name") != STRICT_SPLIT:
        raise ValueError("selected physics model uses another split")
    if final.get("loss_balancing", {}).get("candidate_id") != selected_id:
        raise ValueError("final physics model differs from the validation-selected method")
    if final.get("selected_method_record_sha256") != sha256(selection_path):
        raise ValueError("selection record changed before the final evaluation")
    if sha256(selected_dir / "summary.json") != sha256(final_path):
        raise ValueError("summary.json and final_summary.json differ for the selected model")
    return selection, final, selected_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--residual-geometry", type=Path, required=True)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "results/hccb_p418_physical_steps_12",
    )
    parser.add_argument("--physics-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    dataset_index = args.dataset_index.resolve()
    splits = args.splits.resolve()
    residual_geometry = args.residual_geometry.resolve()
    result_dir = args.result_dir.resolve()
    loss_root = result_dir / LOSS_ROOT_NAME
    selection, physics, physics_dir = require_selected_final(loss_root)
    selected_id = str(selection["selected_candidate_id"])
    selection_path = loss_root / "selected_loss_balancing_method.json"
    integration_path = loss_root / INTEGRATION_RECORD_NAME

    factorized_dir = loss_root / f"selected_{selected_id}_factorized"
    low_rank_dir = loss_root / f"selected_{selected_id}_low_rank"
    diffusion_dir = loss_root / f"selected_{selected_id}_diffusion"

    commands: list[tuple[list[str], Path, str | tuple[str, ...]]] = []
    physics_energy = physics_dir / "energy_balance_summary.json"
    commands.append(
        (
            energy_command(
                summary=physics_dir / "summary.json",
                dataset_index=dataset_index,
                residual_geometry=residual_geometry,
                output=physics_energy,
            ),
            physics_energy,
            ENERGY_STATUSES,
        )
    )

    factorized_command = [
        sys.executable,
        str(TRAINER),
        "--dataset-index",
        str(dataset_index),
        "--splits",
        str(splits),
        "--split-name",
        STRICT_SPLIT,
        "--residual-geometry",
        str(residual_geometry),
        "--output-dir",
        str(factorized_dir),
        "--run-role",
        "formal_factorized",
        "--physics-mode",
        "energy_and_flux",
        "--physics-device",
        args.physics_device,
        "--spatial-temporal-mode",
        "factorized_static_spatial",
        "--temperature-output-mode",
        "literature_bounded_residual",
        "--loss-balancing-candidate-id",
        selected_id,
        "--loss-balancing-sources",
        str(SOURCES.resolve()),
        "--evaluation-stage",
        "final",
        "--selected-method-record",
        str(selection_path),
        "--seed",
        str(args.seed),
        "--torch-threads",
        str(args.torch_threads),
    ]
    if (factorized_dir / "training_checkpoint.pt").is_file():
        factorized_command.append("--resume")
    commands.append(
        (
            factorized_command,
            factorized_dir / "summary.json",
            "completed_p418_spatiotemporal_regional_operator",
        )
    )
    factorized_energy = factorized_dir / "energy_balance_summary.json"
    commands.append(
        (
            energy_command(
                summary=factorized_dir / "summary.json",
                dataset_index=dataset_index,
                residual_geometry=residual_geometry,
                output=factorized_energy,
            ),
            factorized_energy,
            ENERGY_STATUSES,
        )
    )

    commands.append(
        (
            [
                sys.executable,
                str(LOW_RANK),
                "--prediction-dir",
                str(physics_dir),
                "--output-dir",
                str(low_rank_dir),
                "--split-name",
                STRICT_SPLIT,
                "--run-role",
                "formal",
            ],
            low_rank_dir / "summary.json",
            "completed_p418_low_rank_temperature_residual",
        )
    )
    low_rank_energy = low_rank_dir / "energy_balance_summary.json"
    commands.append(
        (
            energy_command(
                summary=low_rank_dir / "summary.json",
                dataset_index=dataset_index,
                residual_geometry=residual_geometry,
                output=low_rank_energy,
            ),
            low_rank_energy,
            ENERGY_STATUSES,
        )
    )

    diffusion_command = [
        sys.executable,
        str(DIFFUSION),
        "--prediction-dir",
        str(physics_dir),
        "--residual-geometry",
        str(residual_geometry),
        "--output-dir",
        str(diffusion_dir),
        "--run-role",
        "computed_residual_benchmark",
        "--microbatch-size",
        "1",
        "--activation-precision",
        "bfloat16",
        "--device",
        "cuda",
        "--threads",
        str(args.torch_threads),
        "--seed",
        str(args.seed),
    ]
    if (diffusion_dir / "training_checkpoint.pt").is_file():
        diffusion_command.append("--resume")
    commands.append(
        (
            diffusion_command,
            diffusion_dir / "summary.json",
            "completed_p418_temporal_temperature_diffusion",
        )
    )
    diffusion_energy = diffusion_dir / "energy_balance_summary.json"
    commands.append(
        (
            energy_command(
                summary=diffusion_dir / "summary.json",
                dataset_index=dataset_index,
                residual_geometry=residual_geometry,
                output=diffusion_energy,
            ),
            diffusion_energy,
            ENERGY_STATUSES,
        )
    )

    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "plan_only_no_training_started",
                    "selected_candidate_id": selected_id,
                    "commands": [shlex.join(command) for command, _, _ in commands],
                    "integration_record": str(integration_path),
                    "new_physical_parameters": [],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    for command, completion, status in commands:
        if has_status(completion, status):
            continue
        run(command, execute=True)
        if not has_status(completion, status):
            raise RuntimeError(f"expected completed result was not written: {completion}")

    summaries = {
        "graph_transformer_energy_flux": physics_dir / "summary.json",
        "graph_transformer_factorized_energy_flux": factorized_dir / "summary.json",
        "low_rank_residual_correction": low_rank_dir / "summary.json",
        "diffusion_residual_correction": diffusion_dir / "summary.json",
    }
    factorized = load_json(summaries["graph_transformer_factorized_energy_flux"])
    if factorized.get("loss_weights") != physics.get("loss_weights"):
        raise ValueError("factorized and repeated-query models use different selected weights")
    if factorized.get("loss_balancing", {}).get("candidate_id") != selected_id:
        raise ValueError("factorized model does not use the selected loss method")
    for path in (physics_energy, factorized_energy, low_rank_energy, diffusion_energy):
        if not has_status(path, ENERGY_STATUSES):
            raise ValueError(f"energy result is incomplete: {path}")

    model_paths = {}
    for model_name, summary_path in summaries.items():
        directory = summary_path.parent
        model_paths[model_name] = {
            "directory_relative_to_result_root": str(directory.relative_to(result_dir)),
            "summary_sha256": sha256(summary_path),
        }
    integration = {
        "status": "completed_p418_selected_loss_balancing_downstream",
        "split_name": STRICT_SPLIT,
        "selected_candidate_id": selected_id,
        "selection_record": str(selection_path.relative_to(result_dir)),
        "selection_record_sha256": sha256(selection_path),
        "independent_test_read_after_validation_selection": True,
        "factorized_model_uses_selected_weights_without_reselection": True,
        "low_rank_and_diffusion_upstream": "graph_transformer_energy_flux",
        "model_paths": model_paths,
        "energy_summary_sha256": {
            "graph_transformer_energy_flux": sha256(physics_energy),
            "graph_transformer_factorized_energy_flux": sha256(factorized_energy),
            "low_rank_residual_correction": sha256(low_rank_energy),
            "diffusion_residual_correction": sha256(diffusion_energy),
        },
        "new_physical_parameters": [],
    }
    integration_path.write_text(
        json.dumps(integration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(integration_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
