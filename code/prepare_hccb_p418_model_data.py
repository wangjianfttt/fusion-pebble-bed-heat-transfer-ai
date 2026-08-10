#!/usr/bin/env python3
"""Plan or run P418 data preparation without starting model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "parameters/hccb_p418_model_data_pipeline.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_condition_ids(matrix_root: Path) -> list[str]:
    manifest = matrix_root / "matrix_manifest.json"
    if not manifest.is_file():
        return []
    rows = read_json(manifest).get("published_conditions", [])
    return [str(row["condition_id"]) for row in rows]


def step_sequences(root: Path, plan_value: str) -> list[dict[str, Any]]:
    plan = read_json(resolve(root, plan_value))
    return list(plan["sequences"])


def completion_state(
    source_root: Path,
    expected_ids: list[str],
    marker_name: str,
    fallback_expected_count: int,
) -> dict[str, Any]:
    existing = [identifier for identifier in expected_ids if (source_root / identifier).is_dir()]
    completed = [
        identifier
        for identifier in expected_ids
        if (source_root / identifier / marker_name).is_file()
    ]
    if expected_ids:
        expected_count = len(expected_ids)
        missing_directories = sorted(set(expected_ids) - set(existing))
        incomplete = sorted(set(existing) - set(completed))
    else:
        expected_count = fallback_expected_count
        completed = (
            sorted(path.parent.name for path in source_root.glob(f"*/{marker_name}"))
            if source_root.is_dir()
            else []
        )
        missing_directories = []
        incomplete = []
    return {
        "source_root": str(source_root),
        "directory_available": source_root.is_dir(),
        "completion_marker": marker_name,
        "expected_count": expected_count,
        "completed_count": len(completed),
        "complete": len(completed) == expected_count and expected_count > 0,
        "completed_ids": completed,
        "missing_directories": missing_directories,
        "incomplete_ids": incomplete,
    }


def command_record(
    *,
    name: str,
    argv: list[str],
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    command_text = " ".join(argv)
    if any(Path(token).name.startswith("train_") for token in argv):
        raise ValueError(f"training command entered data-only pipeline: {command_text}")
    return {
        "name": name,
        "argv": argv,
        "environment": environment or {},
        "starts_model_training": False,
    }


def artifact_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        record: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
        }
        if path.is_file():
            record["bytes"] = path.stat().st_size
            record["sha256"] = sha256(path)
        records.append(record)
    return records


def build_plan(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("new_physical_parameters") != []:
        raise ValueError("data pipeline introduces new physical parameters")

    steady_cfg = config["steady"]
    fixed_cfg = config["fixed_hydrodynamics_thermal_steps"]
    coupled_cfg = config["fully_coupled_flow_heat_steps"]
    shared_cfg = config["shared_inputs"]

    steady_root = resolve(root, steady_cfg["source_root"])
    fixed_root = resolve(root, fixed_cfg["source_root"])
    coupled_root = resolve(root, coupled_cfg["source_root"])
    steady_ids = matrix_condition_ids(steady_root)
    fixed_sequences = step_sequences(root, fixed_cfg["plan"])
    coupled_sequences = step_sequences(root, coupled_cfg["plan"])
    if fixed_sequences != coupled_sequences:
        raise ValueError("fixed and fully coupled data do not use the same endpoint pairs")
    fixed_ids = [str(row["sequence_id"]) for row in fixed_sequences]
    coupled_ids = [str(row["sequence_id"]) for row in coupled_sequences]

    states = {
        "steady": completion_state(
            steady_root,
            steady_ids,
            steady_cfg["completion_marker"],
            int(steady_cfg["expected_case_count"]),
        ),
        "fixed_hydrodynamics_thermal_steps": completion_state(
            fixed_root,
            fixed_ids,
            fixed_cfg["completion_marker"],
            len(fixed_ids),
        ),
        "fully_coupled_flow_heat_steps": completion_state(
            coupled_root,
            coupled_ids,
            coupled_cfg["completion_marker"],
            len(coupled_ids),
        ),
    }

    shared = {key: resolve(root, value) for key, value in shared_cfg.items()}
    steady_postprocess = resolve(root, steady_cfg["postprocess_summary"])
    steady_outputs_ready = steady_postprocess.is_file() and all(
        path.is_file() for path in shared.values()
    )

    python = sys.executable
    integrated_exporter = str(resolve(root, config["integrated_observable_exporter"]))
    regional_exporter = str(resolve(root, config["regional_sequence_exporter"]))
    fixed_result = resolve(root, fixed_cfg["result_root"])
    coupled_result = resolve(root, coupled_cfg["result_root"])

    commands = {
        "steady": [
            command_record(
                name="prepare_steady_shared_mesh_and_regional_targets",
                argv=["bash", str(resolve(root, steady_cfg["postprocess_runner"]))],
                environment={
                    "ROOT": str(root),
                    "MATRIX_ROOT": str(steady_root),
                    "DATASET_ROOT": str(resolve(root, steady_cfg["dataset_root"])),
                },
            )
        ],
        "fixed_hydrodynamics_thermal_steps": [
            command_record(
                name="export_fixed_step_integrated_observables",
                argv=[
                    python,
                    integrated_exporter,
                    "--matrix-root",
                    str(fixed_root),
                    "--output-dir",
                    str(fixed_result),
                    "--history-kind",
                    fixed_cfg["integrated_observable_history_kind"],
                ],
            ),
            command_record(
                name="export_fixed_step_regional_sequences",
                argv=[
                    python,
                    regional_exporter,
                    "--step-root",
                    str(fixed_root),
                    "--shared-topology",
                    str(shared["shared_topology"]),
                    "--steady-dataset-index",
                    str(shared["steady_dataset_index"]),
                    "--subface-geometry",
                    str(shared["subface_geometry"]),
                    "--regional-topology",
                    str(shared["regional_topology"]),
                    "--model-geometry",
                    str(shared["model_geometry"]),
                    "--output-dir",
                    str(fixed_result / "regional_sequences"),
                    "--require-complete",
                    "--history-mode",
                    "fixed_hydrodynamics_thermal",
                ],
            ),
        ],
        "fully_coupled_flow_heat_steps": [
            command_record(
                name="export_fully_coupled_integrated_observables",
                argv=[
                    python,
                    integrated_exporter,
                    "--matrix-root",
                    str(coupled_root),
                    "--output-dir",
                    str(coupled_result),
                    "--history-kind",
                    coupled_cfg["integrated_observable_history_kind"],
                ],
            ),
            command_record(
                name="export_fully_coupled_regional_sequences",
                argv=[
                    python,
                    regional_exporter,
                    "--step-root",
                    str(coupled_root),
                    "--shared-topology",
                    str(shared["shared_topology"]),
                    "--steady-dataset-index",
                    str(shared["steady_dataset_index"]),
                    "--subface-geometry",
                    str(shared["subface_geometry"]),
                    "--regional-topology",
                    str(shared["regional_topology"]),
                    "--model-geometry",
                    str(shared["model_geometry"]),
                    "--output-dir",
                    str(coupled_result / "regional_sequences"),
                    "--require-complete",
                    "--history-mode",
                    "fully_coupled_flow_heat",
                ],
            ),
        ],
    }

    readiness = {
        "steady": states["steady"]["complete"],
        "fixed_hydrodynamics_thermal_steps": (
            states["fixed_hydrodynamics_thermal_steps"]["complete"]
            and steady_outputs_ready
        ),
        "fully_coupled_flow_heat_steps": (
            states["fully_coupled_flow_heat_steps"]["complete"]
            and steady_outputs_ready
        ),
    }
    outputs = artifact_records(
        [
            steady_postprocess,
            shared["steady_dataset_index"],
            shared["shared_topology"],
            fixed_result / "summary.json",
            fixed_result / "regional_sequences/dataset_index.json",
            coupled_result / "summary.json",
            coupled_result / "regional_sequences/dataset_index.json",
        ]
    )
    return {
        "status": (
            "all_p418_model_data_stages_ready"
            if all(readiness.values())
            else "p418_model_data_waiting_for_openfoam_outputs"
        ),
        "config": str(config_path),
        "states": states,
        "steady_outputs_ready": steady_outputs_ready,
        "stage_ready_to_run": readiness,
        "commands": commands,
        "artifacts": outputs,
        "fixed_and_fully_coupled_endpoint_pairs_identical": True,
        "starts_model_training": False,
        "new_physical_parameters": [],
    }


def run_stage(root: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        if record["starts_model_training"]:
            raise ValueError("data-only stage contains a model-training command")
        environment = os.environ.copy()
        environment.update(record["environment"])
        subprocess.run(record["argv"], cwd=root, env=environment, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/hccb_p418_model_data_preparation/summary.json",
    )
    parser.add_argument(
        "--execute-stage",
        action="append",
        choices=(
            "steady",
            "fixed_hydrodynamics_thermal_steps",
            "fully_coupled_flow_heat_steps",
        ),
        default=[],
        help="Omit this option to inspect readiness without running data preparation.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config.resolve()
    plan = build_plan(root, config)
    for stage in args.execute_stage:
        refreshed = build_plan(root, config)
        if not refreshed["stage_ready_to_run"][stage]:
            state = refreshed["states"][stage]
            raise RuntimeError(
                f"{stage} is not ready: "
                f"{state['completed_count']}/{state['expected_count']} complete"
            )
        run_stage(root, refreshed["commands"][stage])
    final = build_plan(root, config)
    final["executed_stages"] = list(args.execute_stage)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
