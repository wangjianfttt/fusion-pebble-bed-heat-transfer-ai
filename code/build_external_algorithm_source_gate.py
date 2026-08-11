#!/usr/bin/env python3
"""Build an external algorithm source gate for the pebble-bed heat AI route.

This script records mature external algorithm families that may be reused or
adapted. It does not import physical parameters and it does not train models.
The purpose is to keep architecture borrowing tied to the current APD routes,
data contracts and claim boundaries.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "external_algorithm_source_gate"
ALGORITHM_MANIFEST = ROOT / "algorithms" / "algorithm_candidate_manifest.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    algorithm_ids = {row.get("algorithm_id", "") for row in read_csv(ALGORITHM_MANIFEST)}

    rows: list[dict[str, object]] = [
        {
            "source_id": "EXT001",
            "matched_algorithm_id": "A001",
            "external_method": "NeuralOperator / Fourier Neural Operator",
            "source_type": "official_open_source_library",
            "source_url": "https://github.com/neuraloperator/neuraloperator",
            "source_basis": (
                "Official PyTorch neural-operator library; README states it contains official FNO implementations "
                "and learns mappings between function spaces."
            ),
            "project_use": "reduced multi-condition heat-transfer operator and FNO/PINO baseline",
            "route_gate": "APD001",
            "current_decision": "use_as_reduced_forward_baseline",
            "must_not_claim": "does not prove resolved pebble-scale CFD/CFD-DEM fidelity",
            "next_condition": "compare against FVM/FDM and boundary/PDE residual on literature-parameter support",
        },
        {
            "source_id": "EXT002",
            "matched_algorithm_id": "A002",
            "external_method": "PINO physics-informed neural operator",
            "source_type": "official_research_code",
            "source_url": "https://github.com/neuraloperator/physics_informed",
            "source_basis": (
                "PINO combines operator learning with function optimization and PDE residuals over parametric PDE families."
            ),
            "project_use": "PDE-residual-constrained operator learning for HCCB heat-transfer fields",
            "route_gate": "APD001;APD005",
            "current_decision": "use_for_reduced_pde_residual_route_block_resolved_claim",
            "must_not_claim": "does not repair missing APD005 physical RTD or resolved velocity packages",
            "next_condition": "use correct heat equation residual, boundary conditions and fair classical baselines",
        },
        {
            "source_id": "EXT003",
            "matched_algorithm_id": "A003",
            "external_method": "NVIDIA PhysicsNeMo",
            "source_type": "open_source_sciml_framework",
            "source_url": "https://github.com/NVIDIA/physicsnemo",
            "source_basis": (
                "PhysicsNeMo provides PyTorch-based SciML modules including neural operators, diffusion models, "
                "Transformers, data pipelines and symbolic PDE residual tools."
            ),
            "project_use": "possible infrastructure reference for scalable PINN/operator/diffusion experiments",
            "route_gate": "APD001;APD003;APD004;APD005",
            "current_decision": "infrastructure_reference_not_dependency_yet",
            "must_not_claim": "do not add a heavy framework dependency without environment and reproducibility audit",
            "next_condition": "only adopt after a minimal local/remote installation smoke test and one reproduced example",
        },
        {
            "source_id": "EXT004",
            "matched_algorithm_id": "A004",
            "external_method": "PDE-Refiner / diffusion-inspired neural PDE solver",
            "source_type": "published_method",
            "source_url": "https://www.microsoft.com/en-us/research/publication/pde-refiner-achieving-accurate-long-rollouts-with-neural-pde-solvers/",
            "source_basis": (
                "PDE-Refiner draws on diffusion-model ideas for multistep refinement, long rollouts and uncertainty estimates."
            ),
            "project_use": "candidate residual/refinement and uncertainty branch after deterministic heat-transfer prior",
            "route_gate": "APD004",
            "current_decision": "candidate_control_not_mainline",
            "must_not_claim": "not a full diffusion solver for blanket heat transfer and not validated for fusion pebble beds",
            "next_condition": "must beat Transformer/set posterior reference under RMSE, PDE residual and physical split gates",
        },
        {
            "source_id": "EXT005",
            "matched_algorithm_id": "A008",
            "external_method": "Conformalized operator uncertainty calibration",
            "source_type": "uncertainty_method_candidate",
            "source_url": "https://arxiv.org/abs/2402.15406",
            "source_basis": (
                "Conformal operator calibration can attach statistical coverage to operator predictions."
            ),
            "project_use": "calibration wrapper for sparse inverse or module surrogate uncertainty bands",
            "route_gate": "APD003;APD004;APD006",
            "current_decision": "candidate_uq_wrapper_only",
            "must_not_claim": "coverage calibration is not heat-equation correctness or material-parameter validation",
            "next_condition": "combine coverage with PDE residual, boundary checks and literature-parameter support gates",
        },
        {
            "source_id": "EXT006",
            "matched_algorithm_id": "A009",
            "external_method": "Porous-DeepONet / structure-aware porous operator",
            "source_type": "porous_media_operator_paper",
            "source_url": "https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2024.07.002",
            "source_basis": "Porous-media operator-learning route for mapping structure descriptors to solution fields.",
            "project_use": "future wall-porosity, packing-structure and velocity-field conditioned heat operator",
            "route_gate": "APD002;APD005",
            "current_decision": "hold_until_traceable_structure_fields",
            "must_not_claim": "cannot use invented porosity, contact or wall-effect fields",
            "next_condition": "requires DEM/CFD/literature-derived structure descriptors and holdout physical splits",
        },
        {
            "source_id": "EXT007",
            "matched_algorithm_id": "A011",
            "external_method": "AURORA OpenMC-MOOSE multiphysics",
            "source_type": "official_open_source_fusion_multiphysics_code",
            "source_url": "https://github.com/aurora-multiphysics/aurora",
            "source_basis": (
                "The official repository couples OpenMC mesh tallies to MOOSE heat conduction and publishes generic thermal tests plus supplementary input-generation scripts. "
                "The repository does not expose a ready-made paper HCPB volumetric heat-field dataset."
            ),
            "project_use": "deterministic APD006 source-field and temperature-baseline route before any module neural surrogate",
            "route_gate": "APD006",
            "current_decision": "use_as_deterministic_multiphysics_reference_data_still_required",
            "must_not_claim": "do not treat repository availability or the paper neutron rate as a validated module W/m3 heat field",
            "next_condition": "obtain/reproduce geometry, nuclear data, OpenMC energy-deposition tally and MOOSE temperature field, then pass the APD006 package contract",
        },
        {
            "source_id": "EXT008",
            "matched_algorithm_id": "A012",
            "external_method": "GINO irregular-geometry neural operator",
            "source_type": "official_open_source_library_module",
            "source_url": "https://github.com/neuraloperator/neuraloperator",
            "source_basis": "Maintained NeuralOperator implementation for irregular input/output geometry through a latent regular grid.",
            "project_use": "primary geometry-aware baseline for DEM/CFD-derived pebble-bed or module fields",
            "route_gate": "APD002;APD006",
            "current_decision": "preferred_irregular_geometry_operator_after_data_gate",
            "must_not_claim": "latent-grid mapping does not preserve wall flux or energy automatically",
            "next_condition": "audit interpolation, wall boundary and energy balance on deterministic fields",
        },
        {
            "source_id": "EXT009",
            "matched_algorithm_id": "A013",
            "external_method": "GNOT multi-input transformer neural operator",
            "source_type": "ICML_published_official_code",
            "source_url": "https://proceedings.mlr.press/v202/hao23c.html",
            "source_basis": "Published multi-input operator for heterogeneous geometry, material and condition inputs.",
            "project_use": "Transformer operator control with the same inputs and physical splits as GINO",
            "route_gate": "APD002;APD006",
            "current_decision": "fair_transformer_control_after_data_gate",
            "must_not_claim": "architecture novelty or superiority before same-data comparison",
            "next_condition": "same split, parameter count, error, energy residual and timing protocol as GINO",
        },
        {
            "source_id": "EXT010",
            "matched_algorithm_id": "A014",
            "external_method": "Transolver physics-attention transformer",
            "source_type": "ICML_published_official_code",
            "source_url": "https://proceedings.mlr.press/v235/wu24r.html",
            "source_basis": "Published attention architecture for PDE fields on irregular meshes.",
            "project_use": "second irregular-mesh Transformer control",
            "route_gate": "APD002;APD006",
            "current_decision": "candidate_control_after_data_gate",
            "must_not_claim": "cannot replace deterministic geometry and boundary validation",
            "next_condition": "run only after the same validated 3D field package exists",
        },
        {
            "source_id": "EXT011",
            "matched_algorithm_id": "A017",
            "external_method": "FunDPS function-space guided diffusion",
            "source_type": "NeurIPS_published_official_code",
            "source_url": "https://github.com/neuraloperator/FunDPS",
            "source_basis": "Function-space diffusion posterior sampling with PDE and observation guidance.",
            "project_use": "sparse-temperature posterior after a calibrated deterministic/operator prior",
            "route_gate": "APD003;APD004",
            "current_decision": "posterior_candidate_not_forward_mainline",
            "must_not_claim": "generated samples are not deterministic heat-transfer truth",
            "next_condition": "require posterior coverage, PDE residual, boundary and energy checks",
        },
        {
            "source_id": "EXT012",
            "matched_algorithm_id": "A023",
            "external_method": "MeshGraphNet-Transformer",
            "source_type": "arXiv_preprint_local_pdf",
            "source_url": "https://arxiv.org/abs/2601.23177",
            "source_basis": (
                "The paper reports a local message-passing, global Physics-Attention and local refinement sequence "
                "with explicit graph and attention dimensions."
            ),
            "project_use": "fixed numerical structure for the regional graph-Transformer comparison",
            "route_gate": "APD002",
            "current_decision": "use_as_fixed_architecture_source",
            "must_not_claim": "solid-mechanics performance is not evidence for breeder-bed heat transfer",
            "next_condition": "test the fixed structure on the complete HCCB steady and thermal-step fields",
        },
        {
            "source_id": "EXT013",
            "matched_algorithm_id": "A024",
            "external_method": "Adjacency-masked Graph Transformer for mesh simulations",
            "source_type": "arXiv_preprint_official_code",
            "source_url": "https://arxiv.org/abs/2508.18051",
            "source_basis": (
                "The paper evaluates graph Transformers on three-dimensional CFD meshes up to 300,000 nodes "
                "and three million edges."
            ),
            "project_use": "direct mesh-CFD literature support for the selected graph-Transformer family",
            "route_gate": "APD002",
            "current_decision": "use_as_3d_cfd_method_source",
            "must_not_claim": "published CFD errors and speedups do not transfer to conjugate pebble-bed heat transfer",
            "next_condition": "compare on identical HCCB fields, condition splits and finite-volume quantities",
        },
    ]

    for row in rows:
        row["algorithm_id_in_manifest"] = row["matched_algorithm_id"] in algorithm_ids
        if not row["algorithm_id_in_manifest"]:
            row["current_decision"] = "blocked_missing_algorithm_manifest_id"

    fieldnames = [
        "source_id",
        "matched_algorithm_id",
        "algorithm_id_in_manifest",
        "external_method",
        "source_type",
        "source_url",
        "source_basis",
        "project_use",
        "route_gate",
        "current_decision",
        "must_not_claim",
        "next_condition",
    ]
    write_csv(OUT_DIR / "external_algorithm_source_gate.csv", rows, fieldnames)

    blocked_or_candidate = [
        row for row in rows if any(x in str(row["current_decision"]) for x in ["candidate", "hold", "not_dependency"])
    ]
    summary = {
        "status": "completed_external_algorithm_source_gate",
        "num_sources": len(rows),
        "num_candidate_or_hold": len(blocked_or_candidate),
        "manifest_ids_valid": all(row["algorithm_id_in_manifest"] for row in rows),
        "primary_decision": (
            "Use PINN, the fixed regional graph-Transformer, DMDc/POD and residual diffusion on the same HCCB "
            "data splits; do not add another architecture before the registered comparisons are complete."
        ),
        "csv": str((OUT_DIR / "external_algorithm_source_gate.csv").relative_to(ROOT)),
        "markdown_cn": str((OUT_DIR / "external_algorithm_source_gate_CN.md").relative_to(ROOT)),
        "claim_boundary": (
            "This table records algorithm sources only. It imports no physical parameters, performs no training and "
            "does not establish accuracy without the corresponding HCCB calculations."
        ),
    }

    lines = [
        "# 外部成熟算法来源对照",
        "",
        "本表记录可借鉴的 PINN、神经算子、Transformer 和扩散模型来源，并说明每种方法在当前球床换热研究中具体做什么、不做什么。",
        "",
        f"- status: `{summary['status']}`",
        f"- sources: `{summary['num_sources']}`",
        f"- manifest ids valid: `{summary['manifest_ids_valid']}`",
        f"- primary decision: {summary['primary_decision']}",
        "",
        "| source | method | route | decision | allowed use | blocked claim |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {source_id} | {external_method} | `{route_gate}` | `{current_decision}` | {project_use} | {must_not_claim} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 当前路线解释",
            "",
            "1. FNO/PINO/PINN 是 reduced HCCB 多工况换热正演的当前主线，但不能替代 FVM/FDM 或 resolved CFD/CFD-DEM 验证。",
            "2. Transformer/set posterior 只作为稀疏传感器到残差系数空间的参考边界，不能写成 Transformer 普遍优越。",
            "3. 扩散/PDE-Refiner 分支只用于修正确定性模型剩余的温度误差，必须同时检查温度误差、方程误差和未参加训练的完整工况。",
            "4. PhysicsNeMo 是成熟工程框架候选，但在本项目中还只是基础设施参考；采用前必须先做环境和最小例子复现。",
            "5. Porous-DeepONet或其他结构感知模型，只有在孔隙率、近壁结构、接触或速度场来自文献、DEM或CFD后才考虑加入。",
            "6. AURORA/OpenMC-MOOSE 是 APD006 的确定性多物理参考路线；官方仓库公开了工作流和通用测试，但没有可直接导入的论文 HCPB 热源场，因此仍需真实运行或数据归档。",
            "",
            "## 参数边界",
            "",
            "本表不导入任何物理参数。所有换热、流动、扩散和结构参数仍以 `parameters/literature_parameter_manifest.csv` 及其原始文献为准。",
            "",
        ]
    )
    (OUT_DIR / "external_algorithm_source_gate_CN.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
