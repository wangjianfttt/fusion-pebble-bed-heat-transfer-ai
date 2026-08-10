#!/usr/bin/env python3
"""Evaluate one trained P418 model on independent packed-bed meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from hccb_p418_parametric_regional_operator import (
    collapse_mesh_to_level,
    load_p418_regional_mesh,
    validate_mesh,
)
from train_hccb_p418_regional_operator import (
    build_model,
    evaluate,
    load_engineering_geometry,
    load_scales,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def validate_protocol(payload: dict[str, object]) -> None:
    normalization = payload["normalization"]
    packings = payload["evaluation_packings"]
    seeds = [int(item["seed"]) for item in packings]
    if len(seeds) != len(set(seeds)):
        raise ValueError("packing seeds must be unique")
    final = [
        item for item in packings if item["role"] == "final_zero_shot_packing"
    ]
    if len(final) != 1:
        raise ValueError("the protocol must define one final zero-shot packing")
    if int(normalization["packing_seed"]) == int(final[0]["seed"]):
        raise ValueError("final zero-shot packing cannot define normalization")
    for item in packings:
        conditions = list(item["condition_ids"])
        if len(conditions) != 9 or len(set(conditions)) != 9:
            raise ValueError("each independent packing must contain nine unique cases")


def write_or_verify_fixed_model_record(
    *,
    path: Path,
    checkpoint: Path,
    statistics: Path,
    protocol: Path,
    architecture: str,
    normalization_seed: int,
) -> dict[str, object]:
    payload = {
        "status": "model_fixed_before_final_packing_evaluation",
        "architecture": architecture,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "training_statistics_path": str(statistics),
        "training_statistics_sha256": sha256(statistics),
        "protocol_path": str(protocol),
        "protocol_sha256": sha256(protocol),
        "normalization_packing_seed": normalization_seed,
        "statement": (
            "Architecture, weights and normalization are fixed before the "
            "final-packing field files are loaded."
        ),
    }
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        comparable = {
            key: previous.get(key)
            for key in (
                "architecture",
                "checkpoint_sha256",
                "training_statistics_sha256",
                "protocol_sha256",
                "normalization_packing_seed",
            )
        }
        expected = {key: payload[key] for key in comparable}
        if comparable != expected:
            raise ValueError("the existing fixed-model record describes another model")
        return previous
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("pinn", "graph", "transolver"), required=True
    )
    parser.add_argument("--packing-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    protocol_path = args.protocol.resolve()
    checkpoint_path = args.checkpoint.resolve()
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(payload)
    normalization = payload["normalization"]
    statistics_path = resolve(project_root, str(normalization["training_statistics"]))
    reference_topology = resolve(project_root, str(normalization["regional_topology"]))
    reference_geometry = resolve(project_root, str(normalization["model_geometry"]))
    required_common = [
        checkpoint_path,
        statistics_path,
        reference_topology,
        reference_geometry,
    ]
    missing_common = [str(path) for path in required_common if not path.is_file()]
    if missing_common:
        raise FileNotFoundError("missing reference files: " + ", ".join(missing_common))

    packings = {int(item["seed"]): item for item in payload["evaluation_packings"]}
    missing_seed = sorted(set(args.packing_seeds) - set(packings))
    if missing_seed:
        raise ValueError(f"packing seeds are absent from the protocol: {missing_seed}")
    selected = [packings[seed] for seed in args.packing_seeds]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_selected = any(
        item["role"] == "final_zero_shot_packing" for item in selected
    )
    if final_selected:
        write_or_verify_fixed_model_record(
            path=output_dir / "model_fixed_before_seed303.json",
            checkpoint=checkpoint_path,
            statistics=statistics_path,
            protocol=protocol_path,
            architecture=args.architecture,
            normalization_seed=int(normalization["packing_seed"]),
        )

    device = torch.device(args.device)
    reference_mesh = collapse_mesh_to_level(
        load_p418_regional_mesh(reference_topology, reference_geometry),
        int(normalization["regional_level"]),
    )
    boundary_role_count = int(reference_mesh.fine_boundary_role.shape[1])
    model, settings = build_model(args.architecture, boundary_role_count)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    scales = load_scales(statistics_path, str(normalization["split_name"]))

    summaries: list[dict[str, object]] = []
    for item in selected:
        seed = int(item["seed"])
        dataset_path = resolve(project_root, str(item["dataset_index"]))
        topology_path = resolve(project_root, str(item["regional_topology"]))
        geometry_path = resolve(project_root, str(item["model_geometry"]))
        required = [dataset_path, topology_path, geometry_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"seed{seed} files are incomplete: " + ", ".join(missing))
        output_path = output_dir / f"seed{seed}_{item['role']}.json"
        if item["role"] == "final_zero_shot_packing" and output_path.exists():
            raise FileExistsError(
                "the first seed303 zero-shot result already exists and will not be replaced"
            )

        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        records = {
            str(record["condition_id"]): record for record in dataset["conditions"]
        }
        condition_ids = [str(value) for value in item["condition_ids"]]
        missing_cases = sorted(set(condition_ids) - set(records))
        if missing_cases:
            raise ValueError(f"seed{seed} is missing cases: {missing_cases}")
        mesh = collapse_mesh_to_level(
            load_p418_regional_mesh(topology_path, geometry_path),
            int(item["regional_level"]),
        ).to(device)
        if hasattr(model, "validate_runtime_mesh"):
            model.validate_runtime_mesh(mesh)
        else:
            validate_mesh(mesh)
            if int(mesh.fine_boundary_role.shape[1]) != boundary_role_count:
                raise ValueError(
                    "packing boundary-role count differs from the trained PINN"
                )
        engineering_geometry = load_engineering_geometry(dataset_path.parent, dataset)
        result = evaluate(
            model=model,
            mesh=mesh,
            records=records,
            case_ids=condition_ids,
            dataset_root=dataset_path.parent,
            scales=scales,
            engineering_geometry=engineering_geometry,
            chunk_size=args.chunk_size,
            device=device,
        )
        summary = {
            "status": "independent_packing_evaluation_complete",
            "packing_seed": seed,
            "packing_role": item["role"],
            "architecture": args.architecture,
            "checkpoint_sha256": sha256(checkpoint_path),
            "normalization_packing_seed": int(normalization["packing_seed"]),
            "model_settings": settings,
            "result": result,
        }
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)

    combined = {
        "status": "selected_cross_packing_evaluations_complete",
        "packing_count": len(summaries),
        "packing_seeds": [item["packing_seed"] for item in summaries],
        "results": summaries,
    }
    (output_dir / "cross_packing_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
