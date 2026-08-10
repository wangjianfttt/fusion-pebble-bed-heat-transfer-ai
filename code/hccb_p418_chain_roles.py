"""Role labels shared by the steady-to-transient P418 model chain."""

from __future__ import annotations

import numpy as np


DETERMINISTIC_CHAIN_STATUS = "completed_p418_steady_PINN_to_graph_transformer_chain"
FUSED_CHAIN_STATUS = "completed_p418_steady_PINN_graph_transformer_diffusion_chain"


def steady_condition_roles(summary: dict[str, object]) -> dict[str, str]:
    """Map every steady condition to the role used for steady-PINN fitting."""
    split = summary.get("split_case_ids")
    if not isinstance(split, dict):
        raise ValueError("steady PINN summary lacks its condition split")
    role_by_condition: dict[str, str] = {}
    for role in ("train", "validation", "test"):
        identifiers = split.get(role)
        if not isinstance(identifiers, list):
            raise ValueError(f"steady PINN summary lacks the {role} condition list")
        for identifier in identifiers:
            key = str(identifier)
            if key in role_by_condition:
                raise ValueError(f"steady condition {key} appears in multiple roles")
            role_by_condition[key] = role
    return role_by_condition


def endpoint_novelty_class(source_role: str, target_role: str) -> str:
    """Describe whether a transient test also uses unseen steady endpoints."""
    unseen = sum(role != "train" for role in (source_role, target_role))
    if unseen == 2:
        return "both_steady_endpoints_unseen"
    if unseen == 1:
        return "one_steady_endpoint_unseen"
    return "steady_endpoints_seen_transient_held_out"


def summarize_endpoint_groups(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    """Keep the strict end-to-end subset separate from easier held-out curves."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["endpoint_novelty_class"]), []).append(row)
    summary: dict[str, dict[str, float | int]] = {}
    for name, selected in grouped.items():
        exact = np.asarray(
            [float(row["exact_initial_solid_temperature_RMSE_K"]) for row in selected]
        )
        chained = np.asarray(
            [float(row["steady_PINN_initial_solid_temperature_RMSE_K"]) for row in selected]
        )
        initial = np.asarray(
            [float(row["source_solid_initial_temperature_RMSE_K"]) for row in selected]
        )
        summary[name] = {
            "curve_count": len(selected),
            "mean_source_initial_temperature_RMSE_K": float(initial.mean()),
            "exact_initial_mean_solid_temperature_RMSE_K": float(exact.mean()),
            "steady_PINN_initial_mean_solid_temperature_RMSE_K": float(chained.mean()),
            "mean_error_amplification": float(
                chained.mean() / max(exact.mean(), np.finfo(np.float64).eps)
            ),
        }
    return summary
