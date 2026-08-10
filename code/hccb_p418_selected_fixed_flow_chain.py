#!/usr/bin/env python3
"""Resolve the validation-selected fixed-flow model chain for final outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


STRICT_SPLIT = "pair_disjoint_stress_test"
LOSS_ROOT_NAME = "fixed_flow_loss_balancing_pair_disjoint_stress_test"
INTEGRATION_RECORD_NAME = "selected_downstream_integration.json"
MODEL_KEYS = (
    "graph_transformer_energy_flux",
    "graph_transformer_factorized_energy_flux",
    "low_rank_residual_correction",
    "diffusion_residual_correction",
)
EXPECTED_STATUS = {
    "graph_transformer_energy_flux": (
        "completed_p418_spatiotemporal_regional_operator"
    ),
    "graph_transformer_factorized_energy_flux": (
        "completed_p418_spatiotemporal_regional_operator"
    ),
    "low_rank_residual_correction": (
        "completed_p418_low_rank_temperature_residual"
    ),
    "diffusion_residual_correction": (
        "completed_p418_temporal_temperature_diffusion"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def selected_chain_record_path(result_dir: Path) -> Path:
    return result_dir.resolve() / LOSS_ROOT_NAME / INTEGRATION_RECORD_NAME


def selected_model_directories(
    result_dir: Path,
    split_name: str,
    *,
    allow_registered_preselection: bool = False,
) -> dict[str, Path]:
    """Return model directories, requiring validation selection for final strict results."""
    result_dir = result_dir.resolve()
    if split_name != STRICT_SPLIT:
        return {
            "graph_transformer_energy_flux": (
                result_dir / f"regional_graph_transformer_bounded_physics_{split_name}"
            ),
            "graph_transformer_factorized_energy_flux": (
                result_dir / f"regional_graph_transformer_bounded_factorized_{split_name}"
            ),
            "low_rank_residual_correction": (
                result_dir / f"low_rank_temperature_residual_{split_name}"
            ),
            "diffusion_residual_correction": (
                result_dir / f"temporal_diffusion_{split_name}"
            ),
        }

    record_path = selected_chain_record_path(result_dir)
    if allow_registered_preselection and not record_path.is_file():
        return {
            "graph_transformer_energy_flux": (
                result_dir
                / f"regional_graph_transformer_bounded_physics_{split_name}"
            ),
            "graph_transformer_factorized_energy_flux": (
                result_dir
                / f"regional_graph_transformer_bounded_factorized_{split_name}"
            ),
            "low_rank_residual_correction": (
                result_dir / f"low_rank_temperature_residual_{split_name}"
            ),
            "diffusion_residual_correction": (
                result_dir / f"temporal_diffusion_{split_name}"
            ),
        }
    record = load_json(record_path)
    if record.get("status") != "completed_p418_selected_loss_balancing_downstream":
        raise ValueError("the validation-selected fixed-flow model chain is incomplete")
    if record.get("split_name") != split_name:
        raise ValueError("the selected fixed-flow chain uses another split")
    if record.get("independent_test_read_after_validation_selection") is not True:
        raise ValueError("the selected chain does not record the one-time final test read")
    if record.get("new_physical_parameters") != []:
        raise ValueError("the selected chain introduced an unregistered physical parameter")
    selected_id = record.get("selected_candidate_id")
    if not isinstance(selected_id, str) or not selected_id:
        raise ValueError("the selected chain does not identify the chosen loss method")
    loss_root = record_path.parent
    selection_path = loss_root / "selected_loss_balancing_method.json"
    selection = load_json(selection_path)
    if (
        selection.get("status")
        != "p418_loss_balancing_selected_on_validation_only"
        or selection.get("selected_candidate_id") != selected_id
        or selection.get("independent_test_read") is not False
    ):
        raise ValueError("the selected chain differs from validation-only loss selection")
    if sha256(selection_path) != record.get("selection_record_sha256"):
        raise ValueError("the loss-selection record changed after downstream training")

    entries = record.get("model_paths", {})
    if set(entries) != set(MODEL_KEYS):
        raise ValueError("the selected chain does not contain the four required models")
    resolved: dict[str, Path] = {}
    for model_name in MODEL_KEYS:
        entry = entries[model_name]
        relative = Path(str(entry["directory_relative_to_result_root"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe selected-model path for {model_name}")
        directory = (result_dir / relative).resolve()
        try:
            directory.relative_to(result_dir)
        except ValueError as exc:
            raise ValueError(f"selected-model path escapes result root: {model_name}") from exc
        summary_path = directory / "summary.json"
        summary = load_json(summary_path)
        if summary.get("status") != EXPECTED_STATUS[model_name]:
            raise ValueError(f"unexpected selected-model status for {model_name}")
        if summary.get("split_name") != split_name:
            raise ValueError(f"selected model uses another split: {model_name}")
        if summary.get("new_physical_parameters") != []:
            raise ValueError(f"selected model added a physical parameter: {model_name}")
        if sha256(summary_path) != entry.get("summary_sha256"):
            raise ValueError(f"selected-model summary changed: {model_name}")
        resolved[model_name] = directory

    physics_dir = resolved["graph_transformer_energy_flux"]
    physics = load_json(physics_dir / "summary.json")
    factorized = load_json(
        resolved["graph_transformer_factorized_energy_flux"] / "summary.json"
    )
    for model_name, summary in (
        ("graph_transformer_energy_flux", physics),
        ("graph_transformer_factorized_energy_flux", factorized),
    ):
        if (
            summary.get("evaluation_stage") != "final"
            or summary.get("test_evaluated") is not True
            or summary.get("loss_balancing", {}).get("candidate_id") != selected_id
        ):
            raise ValueError(f"{model_name} is not the selected final model")
    if physics.get("loss_weights") != factorized.get("loss_weights"):
        raise ValueError("selected repeated-query and factorized models use different weights")
    for model_name in (
        "low_rank_residual_correction",
        "diffusion_residual_correction",
    ):
        summary = load_json(resolved[model_name] / "summary.json")
        recorded = Path(str(summary.get("deterministic_prediction_dir", "")))
        if recorded.name != physics_dir.name:
            raise ValueError(f"{model_name} was not built from the selected physics model")
    return resolved
