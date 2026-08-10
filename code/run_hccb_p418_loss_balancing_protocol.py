#!/usr/bin/env python3
"""Run the two-stage P418 loss-balancing comparison without test leakage."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "code/train_hccb_p418_fully_coupled_spatiotemporal_operator.py"
SELECTOR = ROOT / "code/select_hccb_p418_loss_balancing_method.py"
DEFAULT_SOURCES = ROOT / "parameters/hccb_p418_loss_balancing_sources.json"


def candidate_arguments(candidate: dict[str, object]) -> list[str]:
    arguments = [
        "--loss-balance-candidate-id",
        str(candidate["candidate_id"]),
        "--loss-balance-method",
        str(candidate["method"]),
        "--state-weight",
        str(candidate["state_weight"]),
        "--face-flux-weight",
        str(candidate["face_flux_weight"]),
        "--physics-weight",
        str(candidate["physics_weight"]),
    ]
    if candidate["method"] == "relobralo":
        arguments.extend(
            [
                "--relobralo-temperature",
                str(candidate["temperature"]),
                "--relobralo-alpha",
                str(candidate["alpha"]),
                "--relobralo-rho",
                str(candidate["expected_rho"]),
            ]
        )
    return arguments


def common_arguments(args: argparse.Namespace, output_dir: Path) -> list[str]:
    arguments = [
        sys.executable,
        str(TRAINER),
        "--dataset-index",
        str(args.dataset_index.resolve()),
        "--splits",
        str(args.splits.resolve()),
        "--split-name",
        args.split_name,
        "--residual-geometry",
        str(args.residual_geometry.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--run-role",
        "formal",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.torch_threads is not None:
        arguments.extend(["--torch-threads", str(args.torch_threads)])
    return arguments


def run(command: list[str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "selection", "final"))
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--residual-geometry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    sources_path = args.sources.resolve()
    if sources_path != DEFAULT_SOURCES.resolve():
        raise ValueError("formal runs must use the recorded project loss-balancing source file")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    candidates = sources["formal_candidates"]
    output_root = args.output_root.resolve()
    selection_record = output_root / "selected_loss_balancing_method.json"

    selection_commands: list[list[str]] = []
    for candidate in candidates:
        candidate_output = output_root / str(candidate["candidate_id"])
        command = [
            *common_arguments(args, candidate_output),
            *candidate_arguments(candidate),
            "--evaluation-stage",
            "selection",
        ]
        if (candidate_output / "training_checkpoint.pt").is_file():
            command.append("--resume")
        selection_commands.append(command)

    selector_command = [
        sys.executable,
        str(SELECTOR),
        "--candidate-root",
        str(output_root),
        "--sources",
        str(sources_path),
        "--output",
        str(selection_record),
    ]

    if args.stage == "plan":
        plan = {
            "stage": "plan_only_no_training_started",
            "selection_commands": [shlex.join(command) for command in selection_commands],
            "selector_command": shlex.join(selector_command),
            "final_command": (
                "Generated only after selected_loss_balancing_method.json exists."
            ),
            "independent_test_read": False,
            "new_physical_parameters": [],
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if args.stage == "selection":
        output_root.mkdir(parents=True, exist_ok=True)
        if selection_record.exists():
            raise FileExistsError(
                "selection record already exists; keep it unchanged or use a new output root"
            )
        for command in selection_commands:
            run(command)
        run(selector_command)
        return 0

    if not selection_record.is_file():
        raise FileNotFoundError(
            "run the selection stage before final independent-test evaluation"
        )
    selected = json.loads(selection_record.read_text(encoding="utf-8"))
    selected_id = str(selected["selected_candidate_id"])
    matching = [
        candidate
        for candidate in candidates
        if str(candidate["candidate_id"]) == selected_id
    ]
    if len(matching) != 1:
        raise ValueError("selected candidate is not present in the recorded source file")
    candidate_output = output_root / selected_id
    if (candidate_output / "final_summary.json").is_file():
        raise FileExistsError(
            "final independent-test result already exists; it is not evaluated twice"
        )
    final_command = [
        *common_arguments(args, candidate_output),
        *candidate_arguments(matching[0]),
        "--evaluation-stage",
        "final",
        "--selected-method-record",
        str(selection_record),
        "--resume",
    ]
    run(final_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
