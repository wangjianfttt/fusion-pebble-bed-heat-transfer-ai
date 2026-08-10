#!/usr/bin/env python3
"""Train PDE-Refiner-style correction of deterministic P418 regional fields."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from hccb_p418_parametric_regional_operator import (
    collapse_mesh_to_level,
    load_p418_regional_mesh,
)
from hccb_p418_regional_diffusion_refiner import (
    P418RegionalResidualRefiner,
    make_velocity_training_pair,
    sample_residual,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as loaded:
        required = {
            "condition_id",
            "condition_normalized",
            "baseline_state_normalized",
            "target_state_normalized",
            "node_type",
            "node_volume_m3",
        }
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"prediction file lacks fields: {sorted(missing)}")
        return {name: loaded[name].copy() for name in required}


def active_mask(node_type: np.ndarray) -> np.ndarray:
    mask = np.zeros((len(node_type), 5), dtype=np.float32)
    mask[node_type == 0] = 1.0
    mask[:, 4] = 1.0
    return mask


def residual_channel_scale(
    residual: np.ndarray, node_type: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    mask = active_mask(node_type)
    scale = np.ones(5, dtype=np.float32)
    for channel in range(5):
        active = mask[:, channel] > 0.0
        weights = volume[active]
        mean_square = np.sum(
            residual[:, active, channel] ** 2 * weights[None], dtype=np.float64
        ) / (len(residual) * np.sum(weights))
        scale[channel] = max(float(math.sqrt(mean_square)), 1.0e-6)
    return scale


def structural_features(
    regional_topology: Path,
    model_geometry: Path,
    regional_level: int,
    expected_type: np.ndarray,
    expected_volume: np.ndarray,
) -> np.ndarray:
    mesh = collapse_mesh_to_level(
        load_p418_regional_mesh(regional_topology, model_geometry), regional_level
    )
    level = mesh.levels[0]
    node_type = level.node_type.cpu().numpy()
    volume = level.volume_m3.cpu().numpy()
    if not np.array_equal(node_type, expected_type):
        raise ValueError("diffusion prediction nodes and regional mesh types differ")
    if not np.allclose(volume, expected_volume, rtol=1.0e-5, atol=0.0):
        raise ValueError("diffusion prediction nodes and regional mesh volumes differ")
    centre = mesh.coordinate_center_m.cpu().numpy()
    coordinate_scale = np.maximum(mesh.coordinate_scale_m.cpu().numpy(), 1.0e-12)
    coordinates = (level.centroid_m.cpu().numpy() - centre) / coordinate_scale
    volume_scale = max(float(mesh.volume_scale_m3.cpu()), 1.0e-30)
    log_volume = np.log(np.maximum(volume / volume_scale, 1.0e-30))[:, None]
    one_hot = np.column_stack((node_type == 0, node_type == 1)).astype(np.float32)
    return np.concatenate(
        (coordinates, log_volume, one_hot, level.boundary_fraction.cpu().numpy()), axis=1
    ).astype(np.float32)


def weighted_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    square = (prediction - target).square() * mask[None] * volume[None, :, None]
    denominator = (mask * volume[:, None]).sum() * prediction.shape[0]
    return square.sum() / denominator


def state_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    node_type: np.ndarray,
    volume: np.ndarray,
) -> dict[str, object]:
    mask = active_mask(node_type)
    channel_rmse: list[float] = []
    for channel in range(5):
        active = mask[:, channel] > 0.0
        mse = np.sum(
            (prediction[:, active, channel] - target[:, active, channel]) ** 2
            * volume[active][None],
            dtype=np.float64,
        ) / (len(prediction) * np.sum(volume[active]))
        channel_rmse.append(float(math.sqrt(mse)))
    return {
        "state_normalized_rmse": float(math.sqrt(np.mean(np.square(channel_rmse)))),
        "state_channel_rmse": channel_rmse,
    }


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--model-geometry", type=Path, required=True)
    parser.add_argument("--regional-level", type=int, default=5)
    parser.add_argument("--architecture-sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--physics-slices", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--num-refinement-steps", type=int, default=3)
    parser.add_argument("--min-noise-std", type=float, default=4.0e-7)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--software-smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")

    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = {
        name: load_predictions(args.prediction_dir / f"{name}_regional_predictions.npz")
        for name in ("train", "validation", "test")
    }
    node_type = splits["train"]["node_type"].astype(np.int64)
    node_volume = splits["train"]["node_volume_m3"].astype(np.float32)
    for name, data in splits.items():
        if not np.array_equal(data["node_type"], node_type):
            raise ValueError(f"node types differ in {name}")
        if not np.allclose(data["node_volume_m3"], node_volume):
            raise ValueError(f"node volumes differ in {name}")
    structure_np = structural_features(
        args.regional_topology.resolve(),
        args.model_geometry.resolve(),
        args.regional_level,
        node_type,
        node_volume,
    )
    train_residual = (
        splits["train"]["target_state_normalized"]
        - splits["train"]["baseline_state_normalized"]
    ).astype(np.float32)
    residual_scale_np = residual_channel_scale(train_residual, node_type, node_volume)

    source_payload = json.loads(args.architecture_sources.resolve().read_text(encoding="utf-8"))
    pde_source = next(
        item for item in source_payload["architectures"] if item["name"].startswith("PDE-Refiner")
    )
    transolver_source = next(
        item for item in source_payload["architectures"] if item["name"] == "Transolver"
    )
    expected = pde_source["source_settings"]
    if (
        args.num_refinement_steps != expected["num_refinement_steps"]
        or args.min_noise_std != expected["minimum_noise_standard_deviation"]
        or args.ema_decay != expected["ema_decay"]
    ):
        raise ValueError("diffusion schedule differs from the registered PDE-Refiner settings")
    transolver_expected = transolver_source["source_settings"]
    if not args.software_smoke and (
        args.hidden_dim != transolver_expected["hidden_size"]
        or args.layers != transolver_expected["layers"]
        or args.attention_heads != transolver_expected["attention_heads"]
        or args.physics_slices != transolver_expected["physics_slices"]
        or args.batch_size != transolver_expected["effective_batch_size"]
        or args.epochs != transolver_expected["epochs"]
        or args.learning_rate != transolver_expected["learning_rate_peak"]
        or args.weight_decay != transolver_expected["weight_decay"]
    ):
        raise ValueError(
            "full diffusion training must retain the registered Transolver backbone settings"
        )

    model = P418RegionalResidualRefiner(
        structural_dim=structure_np.shape[1],
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        attention_heads=args.attention_heads,
        physics_slices=args.physics_slices,
        num_refinement_steps=args.num_refinement_steps,
    ).to(device)
    ema = copy.deepcopy(model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * math.ceil(len(train_residual) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=total_steps
    )

    mask = torch.as_tensor(active_mask(node_type), device=device)
    volume = torch.as_tensor(node_volume, device=device)
    structure = torch.as_tensor(structure_np, device=device)
    scale = torch.as_tensor(residual_scale_np, device=device).view(1, 1, 5)
    arrays: dict[str, dict[str, torch.Tensor]] = {}
    for name, data in splits.items():
        baseline = torch.as_tensor(data["baseline_state_normalized"], device=device)
        target = torch.as_tensor(data["target_state_normalized"], device=device)
        arrays[name] = {
            "baseline": baseline,
            "target": target,
            "residual": (target - baseline) / scale,
            "condition": torch.as_tensor(data["condition_normalized"], device=device),
        }

    history: list[dict[str, float | int]] = []
    best_validation = math.inf
    started = time.time()
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_residual), generator=generator, device=device)
        train_sum = 0.0
        train_count = 0
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            residual = arrays["train"]["residual"][index]
            step = torch.randint(
                0,
                args.num_refinement_steps + 1,
                (len(index),),
                generator=generator,
                device=device,
            )
            noise = torch.randn(residual.shape, generator=generator, device=device)
            noised, target_velocity = make_velocity_training_pair(
                residual,
                step,
                noise=noise,
                num_refinement_steps=args.num_refinement_steps,
                min_noise_std=args.min_noise_std,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                arrays["train"]["baseline"][index],
                noised,
                arrays["train"]["condition"][index],
                structure,
                step,
            )
            loss = weighted_velocity_loss(prediction, target_velocity, mask, volume)
            loss.backward()
            optimizer.step()
            scheduler.step()
            update_ema(ema, model, args.ema_decay)
            train_sum += float(loss.detach()) * len(index)
            train_count += len(index)

        model.eval()
        with torch.no_grad():
            residual = arrays["validation"]["residual"]
            step = torch.arange(len(residual), device=device) % (args.num_refinement_steps + 1)
            validation_generator = torch.Generator(device=device).manual_seed(args.seed + epoch + 1000)
            noise = torch.randn(residual.shape, generator=validation_generator, device=device)
            noised, target_velocity = make_velocity_training_pair(
                residual,
                step,
                noise=noise,
                num_refinement_steps=args.num_refinement_steps,
                min_noise_std=args.min_noise_std,
            )
            prediction = ema(
                arrays["validation"]["baseline"],
                noised,
                arrays["validation"]["condition"],
                structure,
                step,
            )
            validation_loss = float(
                weighted_velocity_loss(prediction, target_velocity, mask, volume)
            )
        row = {
            "epoch": epoch + 1,
            "training_velocity_loss": train_sum / train_count,
            "validation_velocity_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(
                {
                    "model": ema.state_dict(),
                    "epoch": epoch + 1,
                    "residual_channel_scale": residual_scale_np,
                },
                output / "best.pt",
            )

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    ema.load_state_dict(checkpoint["model"])
    ema.eval()
    evaluations: dict[str, object] = {}
    prediction_files: dict[str, str] = {}
    for split_index, (name, data) in enumerate(splits.items()):
        initial_generator = torch.Generator(device=device).manual_seed(args.seed + 2000 + split_index)
        initial_noise = torch.randn(
            arrays[name]["baseline"].shape, generator=initial_generator, device=device
        )
        normalized_residual = sample_residual(
            ema,
            arrays[name]["baseline"],
            arrays[name]["condition"],
            structure,
            initial_noise=initial_noise,
            num_refinement_steps=args.num_refinement_steps,
            min_noise_std=args.min_noise_std,
        )
        refined = arrays[name]["baseline"] + normalized_residual * scale
        refined[:, node_type == 1, :4] = 0.0
        refined_np = refined.cpu().numpy()
        baseline_np = data["baseline_state_normalized"].astype(np.float32)
        target_np = data["target_state_normalized"].astype(np.float32)
        prediction_path = output / f"{name}_refined_predictions.npz"
        np.savez_compressed(
            prediction_path,
            condition_id=data["condition_id"],
            condition_normalized=data["condition_normalized"],
            baseline_state_normalized=baseline_np,
            refined_state_normalized=refined_np,
            target_state_normalized=target_np,
            node_type=node_type,
            node_volume_m3=node_volume,
        )
        prediction_files[name] = prediction_path.name
        evaluations[name] = {
            "baseline": state_metrics(baseline_np, target_np, node_type, node_volume),
            "refined": state_metrics(refined_np, target_np, node_type, node_volume),
        }

    summary = {
        "status": "regional_diffusion_refiner_complete",
        "role": "correction of a deterministic regional CHT prediction",
        "software_smoke": args.software_smoke,
        "base_prediction_directory": str(args.prediction_dir.resolve()),
        "best_epoch": int(checkpoint["epoch"]),
        "epochs": args.epochs,
        "training_seconds": time.time() - started,
        "settings": {
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "attention_heads": args.attention_heads,
            "physics_slices": args.physics_slices,
            "num_refinement_steps": args.num_refinement_steps,
            "min_noise_std": args.min_noise_std,
            "ema_decay": args.ema_decay,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "OneCycleLR",
        },
        "algorithm_sources": {
            "diffusion_schedule": pde_source,
            "regional_attention_and_optimizer": transolver_source,
        },
        "input_sha256": {
            name: sha256(args.prediction_dir / f"{name}_regional_predictions.npz")
            for name in splits
        },
        "residual_channel_scale": residual_scale_np.tolist(),
        "evaluations": evaluations,
        "prediction_files": prediction_files,
        "physical_statement": (
            "The refiner changes no material property, operating condition or boundary "
            "condition. It learns only the training-set residual of a deterministic CHT predictor."
        ),
        "new_physical_parameters": [],
    }
    (output / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
