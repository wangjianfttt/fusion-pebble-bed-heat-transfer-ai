#!/usr/bin/env python3
"""Train a Transolver-sized Transformer on P418 time-response curves.

The model maps one published operating condition and the complete reported
time coordinate to a thermal-response trajectory.  It is a direct trajectory
operator, not an autoregressive accident-sequence model. Formal runs require
all 60 steady-solver histories or all 12 designed physical step responses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "results/hccb_p418_transient_observables_live/hccb_p418_transient_observables.npz"
DEFAULT_SPLITS = ROOT / "parameters/hccb_p418_model_splits.json"
SOURCE_REGISTRY = ROOT / "parameters/hccb_p418_ai_architecture_sources.json"
TARGET_NAMES = [
    "outlet_temperature_K",
    "mass_flow_magnitude_kg_s",
    "inlet_enthalpy_flow_W",
    "outlet_enthalpy_flow_W",
    "cooling_wall_power_W",
    "maximum_solid_temperature_K",
]

FORMAL_ARCHITECTURE = {
    "d_model": 256,
    "heads": 8,
    "layers": 5,
}
FORMAL_TRAINING = {
    "epochs": 500,
    "batch_size": 8,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
}


def validate_numerical_settings(args: argparse.Namespace) -> None:
    values = {
        "d_model": args.d_model,
        "heads": args.heads,
        "layers": args.layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    if any(values[name] <= 0 for name in ["d_model", "heads", "layers", "epochs", "batch_size"]):
        raise SystemExit("Transformer dimensions, epochs and batch size must be positive")
    if args.d_model % args.heads != 0:
        raise SystemExit("d_model must be divisible by heads")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise SystemExit("learning rate must be positive and weight decay cannot be negative")
    if args.run_role != "formal":
        return

    expected = {**FORMAL_ARCHITECTURE, **FORMAL_TRAINING}
    changed = {
        name: {"expected": expected[name], "received": values[name]}
        for name in expected
        if values[name] != expected[name]
    }
    if changed:
        raise SystemExit(
            "formal temporal Transformer settings must match the registered Transolver-sized "
            f"configuration; changed values: {changed}"
        )


def select_columns(data: np.lib.npyio.NpzFile) -> np.ndarray:
    names = [str(value) for value in data["signal_names"]]
    values = data["values"].astype("float32")

    def column(name: str) -> np.ndarray:
        return values[:, :, names.index(name)]

    mass_magnitude = 0.5 * (column("outlet_mass_flow_kg_s") - column("inlet_mass_flow_kg_s"))
    return np.stack(
        [
            column("outlet_temperature_K"),
            mass_magnitude,
            column("inlet_enthalpy_flow_W"),
            column("outlet_enthalpy_flow_W"),
            column("cooling_wall_power_W"),
            column("maximum_solid_temperature_K"),
        ],
        axis=-1,
    )


def split_indices(
    case_ids: list[str],
    split_path: Path,
    split_name: str,
    *,
    require_complete: bool = False,
) -> dict[str, list[int]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))["splits"][split_name]
    roles = {
        role: [str(value) for value in split[role]]
        for role in ["train", "validation", "test"]
    }
    role_sets = {role: set(values) for role, values in roles.items()}
    if any(len(role_sets[role]) != len(roles[role]) for role in roles):
        raise ValueError(f"duplicate complete curves inside split roles: {split_name}")
    if any(
        role_sets[left] & role_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError(f"complete curves overlap across split roles: {split_name}")
    present = set(case_ids)
    declared = set().union(*role_sets.values())
    if len(present) != len(case_ids):
        raise ValueError("input data contain duplicate curve identifiers")
    if require_complete and present != declared:
        raise ValueError(
            f"formal physical-step data and split differ: missing={sorted(declared-present)}, "
            f"extra={sorted(present-declared)}"
        )
    if not present.issubset(declared):
        raise ValueError(
            f"input data contain curves outside the selected split: {sorted(present-declared)}"
        )
    by_id = {case_id: i for i, case_id in enumerate(case_ids)}
    return {
        role: [by_id[case_id] for case_id in roles[role] if case_id in by_id]
        for role in ["train", "validation", "test"]
    }


def metric_rows(
    case_ids: list[str],
    indices: list[int],
    truth: np.ndarray,
    prediction: np.ndarray,
    time_mask: np.ndarray,
    split: str,
) -> list[dict[str, object]]:
    rows = []
    for local_i, case_i in enumerate(indices):
        valid = time_mask[case_i]
        for target_i, name in enumerate(TARGET_NAMES):
            residual = prediction[local_i, valid, target_i] - truth[case_i, valid, target_i]
            rows.append(
                {
                    "split": split,
                    "condition_id": case_ids[case_i],
                    "target": name,
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "maximum_absolute_error": float(np.max(np.abs(residual))),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--split-name", default="interleaved_all_ranges")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-role", choices=["smoke", "formal"], default="formal")
    parser.add_argument(
        "--history-kind",
        choices=["solver_relaxation", "physical_step_response"],
        default="solver_relaxation",
    )
    parser.add_argument("--epochs", type=int, default=FORMAL_TRAINING["epochs"])
    parser.add_argument("--d-model", type=int, default=FORMAL_ARCHITECTURE["d_model"])
    parser.add_argument("--heads", type=int, default=FORMAL_ARCHITECTURE["heads"])
    parser.add_argument("--layers", type=int, default=FORMAL_ARCHITECTURE["layers"])
    parser.add_argument("--batch-size", type=int, default=FORMAL_TRAINING["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=FORMAL_TRAINING["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=FORMAL_TRAINING["weight_decay"])
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    validate_numerical_settings(args)

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit(f"PyTorch is required: {exc}")

    data = np.load(args.data, allow_pickle=True)
    case_ids_all = [str(value) for value in data["case_id"]]
    complete_all = data["complete"].astype(bool)
    fully_reported = data["time_mask"].astype(bool).sum(axis=1) == data["time_mask"].shape[1]
    available = np.flatnonzero(complete_all & fully_reported)
    required_formal_curves = 12 if args.history_kind == "physical_step_response" else 60
    if args.run_role == "formal" and len(available) != required_formal_curves:
        raise SystemExit(
            f"formal {args.history_kind} training requires {required_formal_curves} completed curves, "
            f"found {len(available)}"
        )
    if len(available) < 3:
        raise SystemExit(f"at least three completed curves are required, found {len(available)}")

    conditions_all = data["conditions"].astype("float32")
    condition_names = [str(value) for value in data["condition_names"]]
    if args.history_kind == "physical_step_response":
        required_step_inputs = {
            "source_inlet_velocity_m_s",
            "source_inlet_temperature_K",
            "source_solid_heat_source_MW_m3",
            "target_inlet_velocity_m_s",
            "target_inlet_temperature_K",
            "target_solid_heat_source_MW_m3",
        }
        missing = required_step_inputs.difference(condition_names)
        if missing:
            raise SystemExit(f"physical step data omit source/target inputs: {sorted(missing)}")
    time_all = data["time_s"].astype("float32")
    mask_all = data["time_mask"].astype(bool)
    targets_all = select_columns(data)
    case_ids = [case_ids_all[i] for i in available]
    conditions = conditions_all[available]
    time_s = time_all[available]
    time_mask = mask_all[available]
    targets = targets_all[available]
    require_complete_curves = (
        args.run_role == "formal" and args.history_kind == "physical_step_response"
    )
    split = split_indices(
        case_ids,
        args.splits,
        args.split_name,
        require_complete=require_complete_curves,
    )
    if any(not split[role] for role in ["train", "validation", "test"]):
        raise SystemExit(f"available curves do not populate every split: {split}")

    train_idx = split["train"]
    train_mask = time_mask[train_idx]
    condition_mean = conditions[train_idx].mean(axis=0)
    condition_std = conditions[train_idx].std(axis=0)
    condition_std[condition_std < 1.0e-12] = 1.0
    target_train = targets[train_idx][train_mask]
    target_mean = target_train.mean(axis=0)
    target_std = target_train.std(axis=0)
    target_std[target_std < 1.0e-12] = 1.0
    maximum_time = float(np.nanmax(time_s[train_idx]))
    if maximum_time <= 0.0:
        raise ValueError("training time coordinate has no positive value")

    condition_norm = (conditions - condition_mean) / condition_std
    target_norm = (targets - target_mean) / target_std
    time_norm = time_s / maximum_time
    input_values = np.concatenate(
        [
            np.broadcast_to(condition_norm[:, None, :], (*time_s.shape, conditions.shape[-1])),
            time_norm[:, :, None],
        ],
        axis=-1,
    ).astype("float32")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TrajectoryTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(input_values.shape[-1], args.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.heads,
                dim_feedforward=args.d_model,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=args.layers)
            self.output_projection = nn.Linear(args.d_model, len(TARGET_NAMES))

        def forward(self, x):
            return self.output_projection(self.encoder(self.input_projection(x)))

    model = TrajectoryTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_idx) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )
    input_tensor = torch.tensor(input_values, device=device)
    target_tensor = torch.tensor(target_norm, device=device)
    mask_tensor = torch.tensor(time_mask, device=device)
    rng = np.random.default_rng(args.seed)
    history = []
    selection_history = []
    best_validation = math.inf
    best_epoch = None
    best_state = None
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), args.batch_size):
            index_np = order[start : start + args.batch_size]
            index = torch.tensor(index_np, dtype=torch.long, device=device)
            prediction = model(input_tensor[index])
            valid = mask_tensor[index].unsqueeze(-1)
            loss = ((prediction - target_tensor[index]) ** 2)[valid.expand_as(prediction)].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_index = torch.tensor(
                split["validation"], dtype=torch.long, device=device
            )
            validation_prediction = model(input_tensor[validation_index])
            validation_valid = mask_tensor[validation_index].unsqueeze(-1)
            validation_mse = float(
                (
                    (validation_prediction - target_tensor[validation_index]) ** 2
                )[validation_valid.expand_as(validation_prediction)]
                .mean()
                .cpu()
            )
        selection_history.append(
            {"epoch": epoch + 1, "validation_normalized_mse": validation_mse}
        )
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % max(args.epochs // 10, 1) == 0 or epoch + 1 == args.epochs:
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_normalized_mse": float(np.mean(losses)),
                    "validation_normalized_mse": validation_mse,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    training_seconds = time.perf_counter() - start_time
    if best_state is None or best_epoch is None:
        raise RuntimeError("training did not produce a validation-selected model")
    model.load_state_dict(best_state)

    prediction_by_split: dict[str, np.ndarray] = {}
    inference_seconds_by_split: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for role, indices in split.items():
            begin = time.perf_counter()
            pred_norm = model(input_tensor[indices]).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds_by_split[role] = time.perf_counter() - begin
            prediction_by_split[role] = pred_norm * target_std + target_mean

    rows = []
    for role, indices in split.items():
        rows.extend(metric_rows(case_ids, indices, targets, prediction_by_split[role], time_mask, role))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "casewise_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save(best_state, args.output_dir / "model_state.pt")

    test_rows = [row for row in rows if row["split"] == "test"]
    registry = json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))
    transolver_source = next(item for item in registry["architectures"] if item["name"] == "Transolver")
    summary = {
        "status": f"completed_p418_{args.history_kind}_transformer_{args.run_role}",
        "run_role": args.run_role,
        "history_kind": args.history_kind,
        "data": str(args.data),
        "split_name": args.split_name,
        "seed": args.seed,
        "available_complete_curves": len(available),
        "split_case_counts": {role: len(indices) for role, indices in split.items()},
        "split_case_ids": {
            role: [case_ids[index] for index in indices] for role, indices in split.items()
        },
        "complete_curve_split_verified": bool(require_complete_curves),
        "time_points_are_never_split_across_roles": True,
        "trajectory_points": int(time_s.shape[1]),
        "target_names": TARGET_NAMES,
        "condition_names": condition_names,
        "architecture_source": {
            "paper": transolver_source["paper"],
            "paper_url": transolver_source["paper_url"],
            "official_code_url": transolver_source["official_code_url"],
        },
        "architecture": {
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "feedforward_width": args.d_model,
            "dropout": 0.0,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate_peak": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "OneCycleLR",
            "epochs": args.epochs,
        },
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "compute_device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "training_seconds": training_seconds,
        "selection_split": "validation",
        "selection_metric": "masked normalized trajectory MSE across all observable targets",
        "selected_epoch": best_epoch,
        "best_validation_normalized_mse": best_validation,
        "validation_selection_history": selection_history,
        "inference_seconds_by_split": inference_seconds_by_split,
        "test_mean_rmse_by_target": {
            name: float(np.mean([row["rmse"] for row in test_rows if row["target"] == name]))
            for name in TARGET_NAMES
        },
        "mass_conservation_representation": (
            "one oriented mass-flow magnitude target is learned; inlet=-m and outlet=+m "
            "are reconstructed when engineering outputs are required"
        ),
        "new_physical_parameters": [],
        "scientific_scope": (
            "Full-curve prediction of computed coupled thermal step responses between exact published P418 endpoints. "
            "The target hydrodynamic field is fixed while fluid and solid energy equations evolve. Source and target "
            "conditions are both model inputs; complete curves, rather than time points, are held out."
            if args.history_kind == "physical_step_response"
            else
            "Direct fixed-condition solver-relaxation trajectory prediction from 3D OpenFOAM histories. "
            "This can test steady-solver acceleration and is not physical transient evidence."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
