#!/usr/bin/env python3
"""Build a dependency-aware manifest for formal P418 transient model training."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SEED = 20260717
ROBUSTNESS_SEEDS = (20260718, 20260719)
ROBUSTNESS_SPLIT = "pair_disjoint_stress_test"
TEMPERATURE_OUTPUT_MODE = "literature_bounded_residual"


def valid_completion(path: Path) -> bool:
    """Return true only for a non-empty, non-failure completion artifact."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix != ".json":
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status", "")).lower()
    incomplete_tokens = (
        "failed",
        "failure",
        "incomplete",
        "not_started",
        "in_progress",
        "running",
        "blocked",
    )
    return not any(token in status for token in incomplete_tokens)


def command_text(parts: list[str | Path]) -> str:
    return shlex.join([str(part) for part in parts])


def add_job(
    jobs: list[dict],
    *,
    job_id: str,
    stage: str,
    model_family: str,
    split_name: str | None,
    seed: int | None,
    device: str,
    depends_on: list[str],
    output_dir: Path,
    command: list[str | Path],
    measured_peak_gpu_gb: float | None = None,
    completion_file: Path | None = None,
) -> None:
    jobs.append(
        {
            "job_id": job_id,
            "stage": stage,
            "model_family": model_family,
            "split_name": split_name,
            "seed": seed,
            "device": device,
            "depends_on": depends_on,
            "output_dir": str(output_dir),
            "completion_file": str(completion_file or (output_dir / "summary.json")),
            "command": command_text(command),
            "measured_peak_gpu_GB": measured_peak_gpu_gb,
            "preferred_gpu_vram_GB": (
                24 if measured_peak_gpu_gb is not None else None
            ),
            "cpu_cores": 8 if device == "cuda" else 4,
            "host_memory_GB": 64 if device == "cuda" else 32,
            "new_physical_parameters": [],
        }
    )


def graph_command(
    *,
    root: Path,
    dataset_index: Path,
    splits: Path,
    split_name: str,
    residual_geometry: Path,
    output_dir: Path,
    run_role: str,
    physics_mode: str,
    seed: int,
    factorized: bool = False,
) -> list[str | Path]:
    command: list[str | Path] = [
        "python3",
        root / "code/train_hccb_p418_spatiotemporal_regional_operator.py",
        "--dataset-index",
        dataset_index,
        "--splits",
        splits,
        "--split-name",
        split_name,
        "--residual-geometry",
        residual_geometry,
        "--output-dir",
        output_dir,
        "--run-role",
        run_role,
        "--physics-mode",
        physics_mode,
        "--temperature-output-mode",
        TEMPERATURE_OUTPUT_MODE,
    ]
    if physics_mode == "energy_and_flux":
        # Keep the formal physics models on the same residual-computation
        # device as the completed pair-disjoint reference model.
        command.extend(["--physics-device", "cuda"])
    if factorized:
        command.extend(["--spatial-temporal-mode", "factorized_static_spatial"])
    command.extend(["--seed", str(seed), "--resume"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "results/hccb_p418_physical_steps_12",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=ROOT / "parameters/hccb_p418_step_response_splits.json",
    )
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=(
            ROOT
            / "results/hccb_p418_physical_steps_12/regional_sequences/dataset_index.json"
        ),
    )
    parser.add_argument(
        "--observables",
        type=Path,
        default=(
            ROOT
            / "results/hccb_p418_physical_steps_12/"
            "hccb_p418_transient_observables.npz"
        ),
    )
    parser.add_argument(
        "--residual-geometry",
        type=Path,
        default=(
            ROOT
            / "results/hccb_p418_subface_residual_geometry_r2/"
            "subface_residual_geometry.npz"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    split_names = list(split_payload["splits"])
    expected_splits = [
        "direction_down_test",
        "direction_up_test",
        ROBUSTNESS_SPLIT,
    ]
    if split_names != expected_splits:
        raise ValueError(
            f"formal split order changed: {split_names}; expected {expected_splits}"
        )

    root = args.root.resolve()
    result_dir = args.result_dir.resolve()
    jobs: list[dict] = []

    def add_energy_job(model_job_id: str, model_output: Path) -> str:
        energy_id = f"energy__{model_job_id}"
        add_job(
            jobs,
            job_id=energy_id,
            stage="energy_evaluation",
            model_family="common_energy_balance",
            split_name=None,
            seed=None,
            device="cpu",
            depends_on=[model_job_id],
            output_dir=model_output,
            command=[
                "python3",
                root / "code/evaluate_hccb_p418_temporal_energy_balance.py",
                "--model-summary",
                model_output / "summary.json",
                "--dataset-index",
                args.dataset_index.resolve(),
                "--residual-geometry",
                args.residual_geometry.resolve(),
                "--output",
                model_output / "energy_balance_summary.json",
                "--device",
                "cpu",
            ],
            completion_file=model_output / "energy_balance_summary.json",
        )
        return energy_id

    final_dependencies: list[str] = []
    robustness_dependencies: list[str] = []
    observable_comparison_dependencies: list[str] = []
    regional_primary_comparison_dependencies: list[str] = []
    primary_physics_id: str | None = None
    primary_physics_output: Path | None = None

    for split_name in split_names:
        suffix = split_name
        transformer_id = f"transformer__{suffix}__seed{PRIMARY_SEED}"
        transformer_output = result_dir / f"transformer_{suffix}"
        add_job(
            jobs,
            job_id=transformer_id,
            stage="independent_training",
            model_family="observable_transformer",
            split_name=split_name,
            seed=PRIMARY_SEED,
            device="cpu",
            depends_on=[],
            output_dir=transformer_output,
            command=[
                "python3",
                root / "code/train_hccb_p418_transient_observable_transformer.py",
                "--data",
                args.observables.resolve(),
                "--splits",
                args.splits.resolve(),
                "--split-name",
                split_name,
                "--output-dir",
                transformer_output,
                "--run-role",
                "formal",
                "--history-kind",
                "physical_step_response",
                "--seed",
                str(PRIMARY_SEED),
            ],
        )
        final_dependencies.append(transformer_id)
        observable_comparison_dependencies.append(transformer_id)

        observable_dmdc_id = f"observable_dmdc__{suffix}"
        observable_dmdc_output = result_dir / f"observable_dmdc_{suffix}"
        add_job(
            jobs,
            job_id=observable_dmdc_id,
            stage="independent_training",
            model_family="observable_dmdc",
            split_name=split_name,
            seed=None,
            device="cpu",
            depends_on=[],
            output_dir=observable_dmdc_output,
            command=[
                "python3",
                root / "code/train_hccb_p418_observable_dmdc.py",
                "--data",
                args.observables.resolve(),
                "--splits",
                args.splits.resolve(),
                "--split-name",
                split_name,
                "--output-dir",
                observable_dmdc_output,
            ],
        )
        observable_comparison_dependencies.append(observable_dmdc_id)

        dmdc_id = f"dmdc__{suffix}"
        dmdc_output = result_dir / f"regional_dmdc_{suffix}"
        add_job(
            jobs,
            job_id=dmdc_id,
            stage="independent_training",
            model_family="regional_dmdc",
            split_name=split_name,
            seed=None,
            device="cpu",
            depends_on=[],
            output_dir=dmdc_output,
            command=[
                "python3",
                root / "code/train_hccb_p418_regional_dmdc.py",
                "--dataset-index",
                args.dataset_index.resolve(),
                "--splits",
                args.splits.resolve(),
                "--split-name",
                split_name,
                "--output-dir",
                dmdc_output,
            ],
        )
        final_dependencies.extend([dmdc_id, add_energy_job(dmdc_id, dmdc_output)])

        persistence_id = f"persistence__{suffix}"
        persistence_output = result_dir / f"regional_persistence_{suffix}"
        add_job(
            jobs,
            job_id=persistence_id,
            stage="independent_training",
            model_family="initial_temperature_persistence",
            split_name=split_name,
            seed=None,
            device="cpu",
            depends_on=[],
            output_dir=persistence_output,
            command=[
                "python3",
                root / "code/evaluate_hccb_p418_persistence_baseline.py",
                "--dataset-index",
                args.dataset_index.resolve(),
                "--splits",
                args.splits.resolve(),
                "--split-name",
                split_name,
                "--output-dir",
                persistence_output,
            ],
        )
        final_dependencies.extend(
            [
                persistence_id,
                add_energy_job(persistence_id, persistence_output),
            ]
        )

        data_id = f"graph_data_only__{suffix}__seed{PRIMARY_SEED}"
        data_output = (
            result_dir / f"regional_graph_transformer_bounded_data_only_{suffix}"
        )
        add_job(
            jobs,
            job_id=data_id,
            stage="independent_training",
            model_family="graph_transformer_data_only",
            split_name=split_name,
            seed=PRIMARY_SEED,
            device="cuda",
            depends_on=[],
            output_dir=data_output,
            command=graph_command(
                root=root,
                dataset_index=args.dataset_index.resolve(),
                splits=args.splits.resolve(),
                split_name=split_name,
                residual_geometry=args.residual_geometry.resolve(),
                output_dir=data_output,
                run_role="formal_data_only",
                physics_mode="data_only",
                seed=PRIMARY_SEED,
            ),
            measured_peak_gpu_gb=6.19,
        )
        final_dependencies.extend([data_id, add_energy_job(data_id, data_output)])

        physics_id = f"graph_physics__{suffix}__seed{PRIMARY_SEED}"
        physics_output = (
            result_dir / f"regional_graph_transformer_bounded_physics_{suffix}"
        )
        add_job(
            jobs,
            job_id=physics_id,
            stage="independent_training",
            model_family="graph_transformer_energy_flux",
            split_name=split_name,
            seed=PRIMARY_SEED,
            device="cuda",
            depends_on=[],
            output_dir=physics_output,
            command=graph_command(
                root=root,
                dataset_index=args.dataset_index.resolve(),
                splits=args.splits.resolve(),
                split_name=split_name,
                residual_geometry=args.residual_geometry.resolve(),
                output_dir=physics_output,
                run_role="formal",
                physics_mode="energy_and_flux",
                seed=PRIMARY_SEED,
            ),
            measured_peak_gpu_gb=19.64,
        )
        final_dependencies.extend(
            [physics_id, add_energy_job(physics_id, physics_output)]
        )

        factorized_id = f"graph_factorized__{suffix}__seed{PRIMARY_SEED}"
        factorized_output = (
            result_dir / f"regional_graph_transformer_bounded_factorized_{suffix}"
        )
        add_job(
            jobs,
            job_id=factorized_id,
            stage="independent_training",
            model_family="graph_transformer_factorized_energy_flux",
            split_name=split_name,
            seed=PRIMARY_SEED,
            device="cuda",
            depends_on=[],
            output_dir=factorized_output,
            command=graph_command(
                root=root,
                dataset_index=args.dataset_index.resolve(),
                splits=args.splits.resolve(),
                split_name=split_name,
                residual_geometry=args.residual_geometry.resolve(),
                output_dir=factorized_output,
                run_role="formal_factorized",
                physics_mode="energy_and_flux",
                seed=PRIMARY_SEED,
                factorized=True,
            ),
            measured_peak_gpu_gb=18.99,
        )
        final_dependencies.extend(
            [factorized_id, add_energy_job(factorized_id, factorized_output)]
        )

        low_rank_id = f"low_rank__{suffix}__seed{PRIMARY_SEED}"
        low_rank_output = result_dir / f"low_rank_temperature_residual_{suffix}"
        add_job(
            jobs,
            job_id=low_rank_id,
            stage="dependent_correction",
            model_family="low_rank_temperature_residual",
            split_name=split_name,
            seed=PRIMARY_SEED,
            device="cpu",
            depends_on=[physics_id],
            output_dir=low_rank_output,
            command=[
                "python3",
                root / "code/train_hccb_p418_low_rank_temperature_residual.py",
                "--prediction-dir",
                physics_output,
                "--output-dir",
                low_rank_output,
                "--split-name",
                split_name,
                "--run-role",
                "formal",
            ],
        )
        final_dependencies.extend(
            [low_rank_id, add_energy_job(low_rank_id, low_rank_output)]
        )

        diffusion_id = f"diffusion__{suffix}__seed{PRIMARY_SEED}"
        diffusion_output = result_dir / f"temporal_diffusion_{suffix}"
        add_job(
            jobs,
            job_id=diffusion_id,
            stage="dependent_correction",
            model_family="temporal_diffusion_residual",
            split_name=split_name,
            seed=PRIMARY_SEED,
            device="cuda",
            depends_on=[physics_id],
            output_dir=diffusion_output,
            command=[
                "python3",
                root / "code/train_hccb_p418_temporal_temperature_diffusion.py",
                "--prediction-dir",
                physics_output,
                "--residual-geometry",
                args.residual_geometry.resolve(),
                "--output-dir",
                diffusion_output,
                "--run-role",
                "computed_residual_benchmark",
                "--microbatch-size",
                "1",
                "--activation-precision",
                "bfloat16",
                "--device",
                "cuda",
                "--seed",
                str(PRIMARY_SEED),
                "--resume",
            ],
            measured_peak_gpu_gb=18.75,
        )
        final_dependencies.extend(
            [diffusion_id, add_energy_job(diffusion_id, diffusion_output)]
        )
        if split_name == ROBUSTNESS_SPLIT:
            primary_physics_id = physics_id
            primary_physics_output = physics_output
            robustness_dependencies.extend(
                [transformer_id, data_id, physics_id, low_rank_id, diffusion_id]
            )
            regional_primary_comparison_dependencies.extend(
                [persistence_id, dmdc_id, data_id, physics_id, factorized_id]
            )

    for seed in ROBUSTNESS_SEEDS:
        suffix = f"{ROBUSTNESS_SPLIT}_seed{seed}"
        transformer_id = f"transformer__{suffix}"
        transformer_output = result_dir / f"transformer_{suffix}"
        add_job(
            jobs,
            job_id=transformer_id,
            stage="random_seed_repeat",
            model_family="observable_transformer",
            split_name=ROBUSTNESS_SPLIT,
            seed=seed,
            device="cpu",
            depends_on=[],
            output_dir=transformer_output,
            command=[
                "python3",
                root / "code/train_hccb_p418_transient_observable_transformer.py",
                "--data",
                args.observables.resolve(),
                "--splits",
                args.splits.resolve(),
                "--split-name",
                ROBUSTNESS_SPLIT,
                "--output-dir",
                transformer_output,
                "--run-role",
                "formal",
                "--history-kind",
                "physical_step_response",
                "--seed",
                str(seed),
            ],
        )

        data_id = f"graph_data_only__{suffix}"
        data_output = (
            result_dir / f"regional_graph_transformer_bounded_data_only_{suffix}"
        )
        add_job(
            jobs,
            job_id=data_id,
            stage="random_seed_repeat",
            model_family="graph_transformer_data_only",
            split_name=ROBUSTNESS_SPLIT,
            seed=seed,
            device="cuda",
            depends_on=[],
            output_dir=data_output,
            command=graph_command(
                root=root,
                dataset_index=args.dataset_index.resolve(),
                splits=args.splits.resolve(),
                split_name=ROBUSTNESS_SPLIT,
                residual_geometry=args.residual_geometry.resolve(),
                output_dir=data_output,
                run_role="formal_data_only",
                physics_mode="data_only",
                seed=seed,
            ),
            measured_peak_gpu_gb=6.19,
        )

        physics_id = f"graph_physics__{suffix}"
        physics_output = (
            result_dir / f"regional_graph_transformer_bounded_physics_{suffix}"
        )
        add_job(
            jobs,
            job_id=physics_id,
            stage="random_seed_repeat",
            model_family="graph_transformer_energy_flux",
            split_name=ROBUSTNESS_SPLIT,
            seed=seed,
            device="cuda",
            depends_on=[],
            output_dir=physics_output,
            command=graph_command(
                root=root,
                dataset_index=args.dataset_index.resolve(),
                splits=args.splits.resolve(),
                split_name=ROBUSTNESS_SPLIT,
                residual_geometry=args.residual_geometry.resolve(),
                output_dir=physics_output,
                run_role="formal",
                physics_mode="energy_and_flux",
                seed=seed,
            ),
            measured_peak_gpu_gb=19.64,
        )

        low_rank_id = f"low_rank__{suffix}"
        low_rank_output = result_dir / f"low_rank_temperature_residual_{suffix}"
        add_job(
            jobs,
            job_id=low_rank_id,
            stage="dependent_correction",
            model_family="low_rank_temperature_residual",
            split_name=ROBUSTNESS_SPLIT,
            seed=seed,
            device="cpu",
            depends_on=[physics_id],
            output_dir=low_rank_output,
            command=[
                "python3",
                root / "code/train_hccb_p418_low_rank_temperature_residual.py",
                "--prediction-dir",
                physics_output,
                "--output-dir",
                low_rank_output,
                "--split-name",
                ROBUSTNESS_SPLIT,
                "--run-role",
                "formal",
            ],
        )

        diffusion_id = f"diffusion__{suffix}"
        diffusion_output = result_dir / f"temporal_diffusion_{suffix}"
        add_job(
            jobs,
            job_id=diffusion_id,
            stage="dependent_correction",
            model_family="temporal_diffusion_residual",
            split_name=ROBUSTNESS_SPLIT,
            seed=seed,
            device="cuda",
            depends_on=[physics_id],
            output_dir=diffusion_output,
            command=[
                "python3",
                root / "code/train_hccb_p418_temporal_temperature_diffusion.py",
                "--prediction-dir",
                physics_output,
                "--residual-geometry",
                args.residual_geometry.resolve(),
                "--output-dir",
                diffusion_output,
                "--run-role",
                "computed_residual_benchmark",
                "--microbatch-size",
                "1",
                "--activation-precision",
                "bfloat16",
                "--device",
                "cuda",
                "--seed",
                str(seed),
                "--resume",
            ],
            measured_peak_gpu_gb=18.75,
        )

        seed_jobs = [
            transformer_id,
            data_id,
            physics_id,
            low_rank_id,
            diffusion_id,
        ]
        robustness_dependencies.extend(seed_jobs)
        final_dependencies.extend(seed_jobs)
        for model_id, output_dir in (
            (data_id, data_output),
            (physics_id, physics_output),
            (low_rank_id, low_rank_output),
            (diffusion_id, diffusion_output),
        ):
            final_dependencies.append(add_energy_job(model_id, output_dir))

    robustness_id = "summarize_seed_robustness"
    robustness_output = result_dir / f"seed_robustness_{ROBUSTNESS_SPLIT}"
    add_job(
        jobs,
        job_id=robustness_id,
        stage="final_summary",
        model_family="seed_robustness_summary",
        split_name=ROBUSTNESS_SPLIT,
        seed=None,
        device="cpu",
        depends_on=sorted(set(robustness_dependencies)),
        output_dir=robustness_output,
        command=[
            "python3",
            root / "code/summarize_hccb_p418_step_seed_robustness.py",
            "--result-dir",
            result_dir,
            "--splits",
            args.splits.resolve(),
            "--split-name",
            ROBUSTNESS_SPLIT,
            "--primary-seed",
            str(PRIMARY_SEED),
            "--seeds",
            str(PRIMARY_SEED),
            *[str(seed) for seed in ROBUSTNESS_SEEDS],
            "--output-dir",
            robustness_output,
        ],
    )
    final_dependencies.append(robustness_id)

    observable_comparison_id = "build_observable_model_comparison"
    observable_comparison_output = result_dir / "observable_model_comparison"
    add_job(
        jobs,
        job_id=observable_comparison_id,
        stage="paper_results",
        model_family="observable_model_comparison",
        split_name=None,
        seed=None,
        device="cpu",
        depends_on=sorted(set(observable_comparison_dependencies)),
        output_dir=observable_comparison_output,
        command=[
            "python3",
            root / "code/build_hccb_p418_observable_model_table.py",
            "--result-root",
            result_dir,
            "--csv",
            observable_comparison_output / "comparison.csv",
            "--summary",
            observable_comparison_output / "summary.json",
            "--tex",
            root / "manuscript/generated_observable_dynamics.tex",
            "--text",
            root / "manuscript/generated_observable_dynamics_text.tex",
        ],
    )

    regional_comparison_id = "build_regional_model_comparison"
    regional_comparison_output = result_dir / "regional_model_comparison"
    add_job(
        jobs,
        job_id=regional_comparison_id,
        stage="paper_results",
        model_family="regional_model_comparison",
        split_name=ROBUSTNESS_SPLIT,
        seed=None,
        device="cpu",
        depends_on=sorted(set(regional_primary_comparison_dependencies)),
        output_dir=regional_comparison_output,
        command=[
            "python3",
            root / "code/build_hccb_p418_regional_model_table.py",
            "--result-root",
            result_dir,
            "--csv",
            regional_comparison_output / "comparison.csv",
            "--summary",
            regional_comparison_output / "summary.json",
            "--tex",
            root / "manuscript/generated_regional_dynamics.tex",
            "--text",
            root / "manuscript/generated_regional_dynamics_text.tex",
        ],
    )

    comparison_id = "summarize_model_comparison"
    comparison_output = result_dir / "model_comparison"
    add_job(
        jobs,
        job_id=comparison_id,
        stage="paper_results",
        model_family="formal_model_comparison",
        split_name=None,
        seed=None,
        device="cpu",
        depends_on=sorted(set(final_dependencies)),
        output_dir=comparison_output,
        command=[
            "python3",
            root / "code/summarize_hccb_p418_step_model_comparison.py",
            "--result-dir",
            result_dir,
            "--step-root",
            root / "hccb_p418_physical_steps_12",
            "--splits",
            args.splits.resolve(),
            "--split-names",
            *split_names,
            "--seed-robustness-summary",
            robustness_output / "summary.json",
            "--output-dir",
            comparison_output,
        ],
    )

    performance_id = "build_transient_performance_table"
    performance_summary = comparison_output / "transient_performance_table.json"
    add_job(
        jobs,
        job_id=performance_id,
        stage="paper_results",
        model_family="paper_performance_table",
        split_name=None,
        seed=None,
        device="cpu",
        depends_on=[comparison_id],
        output_dir=comparison_output,
        completion_file=performance_summary,
        command=[
            "python3",
            root / "code/build_hccb_p418_transient_performance_table.py",
            "--metrics-csv",
            comparison_output / "physical_step_model_metrics.csv",
            "--output",
            root / "manuscript/generated_transient_performance.tex",
            "--summary",
            performance_summary,
        ],
    )

    cost_id = "build_transient_cost_table"
    cost_summary = comparison_output / "transient_cost_table.json"
    add_job(
        jobs,
        job_id=cost_id,
        stage="paper_results",
        model_family="paper_cost_table",
        split_name=None,
        seed=None,
        device="cpu",
        depends_on=[comparison_id],
        output_dir=comparison_output,
        completion_file=cost_summary,
        command=[
            "python3",
            root / "code/build_hccb_p418_transient_cost_table.py",
            "--speed-csv",
            comparison_output / "physical_step_model_speedup.csv",
            "--output",
            root / "manuscript/generated_transient_cost.tex",
            "--summary",
            cost_summary,
        ],
    )

    result_text_id = "build_transient_result_text"
    generated_result_text = root / "manuscript/generated_transient_result_text.tex"
    add_job(
        jobs,
        job_id=result_text_id,
        stage="paper_results",
        model_family="paper_generated_result_text",
        split_name=None,
        seed=None,
        device="cpu",
        depends_on=[comparison_id, cost_id],
        output_dir=root / "manuscript",
        completion_file=generated_result_text,
        command=[
            "python3",
            root / "code/build_hccb_p418_transient_result_text.py",
            "--summary",
            comparison_output / "summary.json",
            "--metrics",
            comparison_output / "physical_step_model_metrics.csv",
            "--cost-summary",
            cost_summary,
            "--output",
            generated_result_text,
        ],
    )

    figure_id = "plot_transient_model_comparison"
    figure_validation_marker = (
        root / "manuscript/generated_transient_model_comparison_validated.tex"
    )
    add_job(
        jobs,
        job_id=figure_id,
        stage="paper_results",
        model_family="paper_model_comparison_figure",
        split_name=None,
        seed=None,
        device="cpu",
        depends_on=[comparison_id],
        output_dir=root / "figures",
        completion_file=figure_validation_marker,
        command=[
            "python3",
            root / "code/plot_hccb_p418_transient_model_comparison.py",
            "--result-dir",
            result_dir,
            "--splits",
            args.splits.resolve(),
            "--output-dir",
            root / "figures",
        ],
    )

    field_figure_id = "plot_openfoam_model_field_comparison"
    field_figure_validation_marker = (
        root / "manuscript/generated_openfoam_model_field_comparison_validated.tex"
    )
    add_job(
        jobs,
        job_id=field_figure_id,
        stage="paper_results",
        model_family="paper_openfoam_model_field_figure",
        split_name=ROBUSTNESS_SPLIT,
        seed=None,
        device="cpu",
        depends_on=[comparison_id],
        output_dir=root / "figures",
        completion_file=field_figure_validation_marker,
        command=[
            "bash",
            root / "code/build_hccb_p418_selected_field_figure.sh",
        ],
    )

    completed_job_ids = [
        str(job["job_id"])
        for job in jobs
        if valid_completion(Path(str(job["completion_file"])))
    ]
    if len(completed_job_ids) == len(jobs):
        status = "completed_p418_formal_transient_training_job_set"
        execution_state = "all_registered_jobs_complete"
    elif completed_job_ids:
        status = "p418_formal_transient_training_jobs_in_progress"
        execution_state = "partially_completed_on_registered_local_or_workstation_results"
    else:
        status = "formal_p418_transient_training_jobs_prepared_not_started"
        execution_state = "prepared_not_started"
    manifest = {
        "status": status,
        "source_runner": str(root / "code/run_hccb_p418_step_responses.sh"),
        "split_names": split_names,
        "primary_seed": PRIMARY_SEED,
        "robustness_split": ROBUSTNESS_SPLIT,
        "robustness_seeds": [PRIMARY_SEED, *ROBUSTNESS_SEEDS],
        "formal_training_settings": {
            "graph_epochs": 500,
            "temperature_output_mode": TEMPERATURE_OUTPUT_MODE,
            "fluid_temperature_range_K": [300.0, 1000.0],
            "solid_temperature_range_K": [298.0, 1300.0],
            "diffusion_microbatch_size": 1,
            "diffusion_activation_precision": "bfloat16",
            "history_kind": "physical_step_response",
            "complete_curve_splits_only": True,
        },
        "job_count": len(jobs),
        "cuda_job_count": sum(job["device"] == "cuda" for job in jobs),
        "cpu_job_count": sum(job["device"] == "cpu" for job in jobs),
        "preferred_gpu_vram_GB": 24,
        "minimum_known_working_gpu": "RTX 4000 Ada 20.6 GB, one large job at a time",
        "execution_state": execution_state,
        "completed_job_count": len(completed_job_ids),
        "remaining_job_count": len(jobs) - len(completed_job_ids),
        "completed_job_ids": completed_job_ids,
        "final_summary_dependencies": [
            observable_comparison_id,
            regional_comparison_id,
            comparison_id,
            performance_id,
            cost_id,
            result_text_id,
            figure_id,
            field_figure_id,
        ],
        "jobs": jobs,
        "new_physical_parameters": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in (
        "status",
        "job_count",
        "cuda_job_count",
        "cpu_job_count",
        "execution_state",
        "completed_job_count",
        "remaining_job_count",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
