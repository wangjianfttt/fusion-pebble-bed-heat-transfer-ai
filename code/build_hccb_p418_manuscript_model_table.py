#!/usr/bin/env python3
"""Build the manuscript model-setting table from registered source files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _architecture_map(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: item for item in data["architectures"]}


def _setting_map(path: Path) -> dict[tuple[str, str], str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["model"], row["setting"]): row["value"] for row in rows}


def _required(mapping: dict, key):
    if key not in mapping:
        raise KeyError(f"missing registered model setting: {key}")
    return mapping[key]


def build_table(registry_path: Path, settings_path: Path) -> str:
    architectures = _architecture_map(registry_path)
    settings = _setting_map(settings_path)

    pinn = _required(architectures, "PINO-paper coordinate PINN control")["source_settings"]
    rigno = _required(architectures, "RIGNO-style regional graph operator")["source_settings"]
    transolver = _required(architectures, "Transolver")["source_settings"]
    graph = _required(
        architectures, "Published-component spatiotemporal regional operator"
    )["source_settings"]
    regional_dmdc = _required(
        architectures, "Volume-weighted DMDc baseline"
    )["source_settings"]
    observable_dmdc = _required(
        architectures, "Observable DMDc baseline"
    )["source_settings"]
    pod = _required(
        architectures, "Snapshot-POD low-rank temperature-residual correction"
    )["source_settings"]
    diffusion = _required(
        architectures, "PDE-Refiner-style diffusion refinement"
    )["source_settings"]

    graph_epochs = _required(settings, ("图-Transformer", "epochs"))
    graph_lr = _required(settings, ("图-Transformer", "learning_rate"))
    graph_fluid_temperature_range = _required(
        settings, ("图-Transformer", "fluid_temperature_output_range_K")
    )
    graph_solid_temperature_range = _required(
        settings, ("图-Transformer", "solid_temperature_output_range_K")
    )
    diffusion_epochs = _required(settings, ("扩散剩余误差修正", "epochs"))

    rows = [
        (
            r"Coordinate PINN \cite{raissi2019pinn,li2024pino,wang2024hccb_pinn}",
            rf"{pinn['hidden_layers']} hidden layers $\times$ {pinn['hidden_width']}; "
            rf"{pinn['activation']} activation",
            rf"{pinn['optimizer']}; {pinn['epochs']} epochs; learning rate {pinn['learning_rate']}",
            "Minimum validation total loss; independent conditions used after selection",
        ),
        (
            r"Regional graph operator \cite{mousavi2025rigno}",
            rf"{rigno['regional_mesh_levels']} regional levels; latent width "
            rf"{rigno['node_latent_size']}; {rigno['processor_steps']} processor steps",
            rf"{rigno['optimizer']}; {rigno['epochs']} epochs; effective batch "
            rf"{rigno['effective_batch_size']}",
            "Minimum validation total loss on the common condition split",
        ),
        (
            r"Steady Physics-Attention \cite{wu2024transolver}",
            rf"{transolver['layers']} layers $\times$ {transolver['hidden_size']}; "
            rf"{transolver['attention_heads']} heads; {transolver['physics_slices']} slices",
            rf"{transolver['optimizer']}; {transolver['epochs']} epochs; effective batch "
            rf"{transolver['effective_batch_size']}",
            "Minimum validation total loss on the common condition split",
        ),
        (
            r"Graph--Transformer \cite{pfaff2021meshgraphnets,mousavi2025rigno,wu2024transolver,vaswani2017attention,iparraguirre2026mgnt}",
            rf"{graph['preprocessor_mpnn_iterations']} local pre-processing graph blocks + "
            rf"{graph['physics_attention_blocks']} Physics-Attention blocks "
            rf"({graph['physical_tokens']} slices) + {graph['temporal_layers']} temporal layers + "
            rf"{graph['refinement_mpnn_iterations']} local refinement graph blocks; "
            rf"hidden path {graph['hidden_path']}; bounded-residual temperature output "
            rf"(fluid {graph_fluid_temperature_range} K, solid "
            rf"{graph_solid_temperature_range} K)",
            rf"AdamW; {graph_epochs} epochs; learning rate {graph_lr}",
            "Data-only checkpoint: minimum validation normalized temperature MSE; "
            "fixed and ReLoBRaLo physics-loss candidates: common unweighted mean "
            "of the three dimensionless validation groups, followed by one "
            "independent-test evaluation",
        ),
        (
            r"Observable DMDc \cite{proctor2016dmdc}",
            rf"$\dot z=Az+Bu$; six outlet/aggregate trajectories; candidate ranks "
            + ", ".join(str(value) for value in observable_dmdc["candidate_ranks"]),
            "Closed-form continuous-time dynamics on the original nonuniform times",
            "Rank selected by validation normalized trajectory MSE on the same split as the observable Transformer",
        ),
        (
            r"Regional DMDc \cite{proctor2016dmdc}",
            rf"$\dot z=Az+Bu$; volume-weighted fluid and solid regional temperatures; candidate ranks "
            + ", ".join(
                str(value) for value in regional_dmdc["candidate_pod_ranks"]
            ),
            "Closed-form continuous-time reduced-field dynamics on the original nonuniform times",
            "Rank selected by validation solid-temperature RMSE",
        ),
        (
            r"Snapshot POD \cite{sirovich1987pod}",
            "Volume-weighted decomposition of the deterministic-model temperature residual",
            "Ranks from zero to the rank supported by training residuals",
            "Validation RMSE; lower rank retained for an exact tie",
        ),
        (
            r"Diffusion correction \cite{lippe2023pderefiner}",
            rf"{diffusion['num_refinement_steps']} residual steps; temperature only; "
            rf"EMA {diffusion['ema_decay']}; {diffusion['formal_stochastic_samples_per_curve']} samples",
            rf"AdamW; {diffusion_epochs} epochs; effective batch "
            rf"{diffusion['effective_batch_size']}",
            r"Validation checkpoint fixed before independent prediction; all independent results retained; joint improvement denotes temperature RMSE $\downarrow$ without a larger projection-aware energy RMSE",
        ),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{0.90}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Numerical settings used for the model comparison. These values define network architecture, numerical training and model selection; they are not pebble-bed material properties or operating conditions.}",
        r"\label{tab:model_settings}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}p{5.2cm}>{\raggedright\arraybackslash}p{5.0cm}}",
        r"\toprule",
        r"Method & Structure & Training and model selection \\",
        r"\midrule",
    ]
    for method, structure, training, selection in rows:
        lines.append(f"{method} & {structure} & {training}. {selection} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("parameters/hccb_p418_ai_architecture_sources.json"),
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("parameters/hccb_p418_model_numerical_settings.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("manuscript/generated_model_settings.tex")
    )
    args = parser.parse_args()
    text = build_table(args.registry, args.settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
