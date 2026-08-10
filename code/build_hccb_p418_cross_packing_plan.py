#!/usr/bin/env python3
"""Build the exact-P418 screening plan for transfer across three packings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_hccb_gmsh_cht_smoke_case import matrix_condition_id, parse_p418_matrix
from hccb_p418_source_contract import ALL_STEADY_PHYSICAL_PARAMETER_IDS


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "parameters/literature_parameter_manifest.csv"
PACKING_SUMMARY = (
    ROOT / "data/apd006_hccb_source_sequence_target_packings/packing_set_summary.json"
)
DEFAULT_PLAN = ROOT / "parameters/hccb_p418_cross_packing_plan.json"
DEFAULT_CSV = ROOT / "parameters/hccb_p418_cross_packing_screening_matrix.csv"
DEFAULT_NOTE = ROOT / "CROSS_PACKING_STUDY_CN.md"
PACKING_PARAMETER_IDS = ("P048", "P049", "P050", "P390")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def p418_conditions(value: str) -> dict[str, tuple[float, float, float]]:
    velocities, temperatures, sources = parse_p418_matrix(value)
    return {
        matrix_condition_id(velocity, temperature, source): (
            velocity,
            temperature,
            source,
        )
        for velocity in velocities
        for temperature in temperatures
        for source in sources
    }


def screening_condition_ids(value: str) -> list[str]:
    """Return eight envelope corners and one published interior condition."""
    velocities, temperatures, sources = parse_p418_matrix(value)
    identifiers = [
        matrix_condition_id(velocity, temperature, source)
        for velocity in (min(velocities), max(velocities))
        for temperature in (min(temperatures), max(temperatures))
        for source in (min(sources), max(sources))
    ]
    interior = matrix_condition_id(0.15, 700.0, 6.85)
    identifiers.append(interior)
    return identifiers


def load_packings(root: Path, path: Path = PACKING_SUMMARY) -> list[dict[str, object]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if [int(record["seed"]) for record in records] != [101, 202, 303]:
        raise ValueError("packing summary must contain seeds 101, 202 and 303 in order")

    invariant_fields = (
        "diameter_growth_stages",
        "crop_placement_id",
        "reconstruction_mode",
        "physical_particle_diameter_m",
        "meshing_particle_diameter_m",
        "box_lengths_m",
        "cooled_wall_face",
        "symmetry_faces",
    )
    reference = records[0]
    validated: list[dict[str, object]] = []
    for original in records:
        record = dict(original)
        if not all(record.get("checks", {}).values()):
            raise ValueError(f"packing seed {record['seed']} has a failed geometry check")
        for field in invariant_fields:
            if record[field] != reference[field]:
                raise ValueError(f"packing seed {record['seed']} differs in {field}")
        seed = int(record["seed"])
        packing_path = (
            root
            / "data/apd006_hccb_source_sequence_target_packings"
            / f"seed{seed}_s80_xlo_ycentre/packing.npz"
        )
        if not packing_path.is_file():
            raise FileNotFoundError(packing_path)
        actual_hash = sha256(packing_path)
        if actual_hash != record["packing_npz_sha256"]:
            raise ValueError(f"packing checksum differs for seed {seed}")
        record["packing_path"] = str(packing_path.relative_to(root))
        validated.append(record)
    return validated


def build_plan(root: Path = ROOT) -> dict[str, object]:
    rows = load_manifest(root / "parameters/literature_parameter_manifest.csv")
    required = tuple(
        dict.fromkeys((*ALL_STEADY_PHYSICAL_PARAMETER_IDS, *PACKING_PARAMETER_IDS))
    )
    for parameter_id in required:
        if parameter_id not in rows:
            raise KeyError(parameter_id)
        if rows[parameter_id]["status"] != "extracted":
            raise ValueError(f"{parameter_id} is not an extracted literature value")

    available = p418_conditions(rows["P418"]["value"])
    if len(available) != 60:
        raise ValueError(f"P418 should contain 60 conditions, found {len(available)}")
    selected_ids = screening_condition_ids(rows["P418"]["value"])
    if len(selected_ids) != 9 or len(set(selected_ids)) != 9:
        raise ValueError("cross-packing screening must contain nine unique conditions")
    unknown = sorted(set(selected_ids).difference(available))
    if unknown:
        raise ValueError(f"screening conditions are outside P418: {unknown}")

    packings = load_packings(
        root,
        root / "data/apd006_hccb_source_sequence_target_packings/packing_set_summary.json",
    )
    packing_by_seed = {int(item["seed"]): item for item in packings}
    conditions = [
        {
            "condition_id": condition_id,
            "inlet_velocity_m_s": available[condition_id][0],
            "inlet_temperature_K": available[condition_id][1],
            "solid_heat_source_MW_m3": available[condition_id][2],
            "selection_role": (
                "published_interior_reference"
                if condition_id == matrix_condition_id(0.15, 700.0, 6.85)
                else "published_envelope_corner"
            ),
        }
        for condition_id in selected_ids
    ]

    few_shot_adaptation = [
        matrix_condition_id(0.05, 300.0, 4.85),
        matrix_condition_id(0.15, 700.0, 6.85),
        matrix_condition_id(0.25, 900.0, 8.85),
    ]
    few_shot_test = [
        condition_id
        for condition_id in selected_ids
        if condition_id not in few_shot_adaptation
    ]
    source = rows["P418"]
    return {
        "status": "hccb_p418_cross_packing_screening_plan_ready",
        "purpose": (
            "Test whether a model trained on one particle arrangement retains accuracy "
            "on two independent particle arrangements within the current 60+9+9 paper route."
        ),
        "source_condition_matrix": {
            "parameter_id": "P418",
            "source_title": source["source_title"],
            "source_doi": source["source_url_or_doi"],
            "published_case_count": len(available),
        },
        "physical_parameter_ids": list(required),
        "new_physical_parameter_values_added": [],
        "packing_realisations": [
            {
                "seed": seed,
                "role": role,
                "packing_path": packing_by_seed[seed]["packing_path"],
                "packing_npz_sha256": packing_by_seed[seed]["packing_npz_sha256"],
                "particle_count": packing_by_seed[seed]["particle_count"],
                "geometric_porosity": packing_by_seed[seed][
                    "crop_porosity_geometric"
                ],
            }
            for seed, role in (
                (101, "current_full_60_condition_base"),
                (202, "nine_condition_development_packing"),
                (303, "nine_condition_final_zero_shot_packing"),
            )
        ],
        "screening_design": {
            "case_count_per_new_packing": len(conditions),
            "selection_rule": (
                "Eight exact corners of the published P418 condition envelope plus "
                "the exact published interior condition u0p15_T700_q6p85."
            ),
            "conditions": conditions,
            "execution_case_count_for_seeds_202_and_303": 2 * len(conditions),
        },
        "model_use": {
            "seed101": (
                "Fit the fixed-packing models with the complete 60-condition matrix."
            ),
            "seed202": (
                "Use the nine screening cases to compare architecture choices and quantify "
                "the first independent-packing loss of accuracy."
            ),
            "seed303_zero_shot": (
                "Freeze architecture, weights and preprocessing before opening any seed303 "
                "field; predict all nine cases without seed303 fitting."
            ),
            "seed303_optional_few_shot_after_zero_shot": {
                "adaptation_conditions": few_shot_adaptation,
                "test_conditions": few_shot_test,
                "reporting_rule": (
                    "Record the all-nine zero-shot result first. The optional three-case "
                    "adaptation result is separate and cannot replace it."
                ),
            },
        },
        "reported_quantities": [
            "pressure_drop",
            "outlet_temperature",
            "maximum_solid_temperature",
            "cooling_wall_heat_flow",
            "fluid_temperature_field_error",
            "solid_temperature_field_error",
            "hotspot_location_error",
            "mass_and_energy_residuals",
        ],
        "later_full_extension": {
            "case_count": 180,
            "protocol": "three_fold_leave_one_complete_packing_out",
            "statement": (
                "A later three-packing 180-case study is an optional extension. It is not "
                "required for the current 60+9+9 paper and must not be presented as completed evidence."
            ),
        },
    }


def write_outputs(
    plan: dict[str, object], plan_path: Path, csv_path: Path, note_path: Path
) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    conditions = plan["screening_design"]["conditions"]  # type: ignore[index]
    rows = []
    for seed, role in ((202, "development"), (303, "final_zero_shot")):
        for condition in conditions:
            rows.append({"packing_seed": seed, "packing_role": role, **condition})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    note_path.write_text(
        """# HCCB球床不同颗粒排列的迁移研究

## 为什么必须做

当前60个P418工况都使用seed101颗粒排列。神经网络可能学到这套排列中特有的流道和热点，却不一定能适应另一套随机装填。因此，固定装填上的未见工况预测完成后，还必须更换颗粒排列。

## 三套颗粒排列怎样使用

- seed101：完成全部60个P418工况，建立基础模型；
- seed202：先计算9个代表性工况，用来比较模型结构并测量更换装填后的精度下降；
- seed303：模型结构、权重和数据处理全部固定后再打开，9个工况均不参与训练。

seed303的9个直接预测结果记录完成后，才允许另做3个工况的小样本适配。这个结果必须单独报告，不能代替完全不看seed303数据时的结果。

## 9个工况从哪里来

选取P418公开60工况范围的8个角点，再加入公开的内部工况`u0p15_T700_q6p85`。流速、入口温度和颗粒体积发热率都来自P418，没有插值，也没有增加新的物理参数。具体清单见`parameters/hccb_p418_cross_packing_screening_matrix.csv`。

## 比较什么

每个工况同时比较压降、出口温度、颗粒最高温度、冷却壁热量、流体和颗粒三维温度场、热点位置以及质量和能量收支。只有平均温度准确而热点位置明显错误时，不能认为模型已经适应新装填。

## 与最终研究的关系

这18个新算例用于检验模型是否依赖seed101的特定颗粒排列，也是当前论文必须完成的跨装填计算。三套装填各60个工况、共180个三维计算属于后续扩展，不是当前论文形成完整结果的前提。当前远程机正在运行seed101的60工况，因此暂不启动seed202和seed303网格计算，避免与现有OpenFOAM任务抢占算力。
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args()
    plan = build_plan(args.root.resolve())
    write_outputs(plan, args.plan.resolve(), args.csv.resolve(), args.note.resolve())
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
