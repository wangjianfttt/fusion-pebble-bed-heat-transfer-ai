#!/usr/bin/env python3
"""Build the source-backed tables used by the P418 supplementary material."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PHYSICAL_NAMES = {
    "P048": "Pebble diameter",
    "P049": "Source-packing target porosity",
    "P050": "Initial packing dimensions",
    "P390": "Published extracted-region dimensions",
    "P404": "Meshing diameter correction",
    "P423": "Cooling-wall-adjacent crop rule",
    "P418": "Velocity, inlet-temperature and heat-source matrix",
    "P425": "Cooling-wall temperature",
    "P426": "Working pressure",
    "P427": "Published domain and boundary families",
    "P070": "Helium dynamic viscosity",
    "P071": "Helium thermal conductivity",
    "P388": "Helium constant-pressure heat capacity",
    "P389": "Helium density",
    "P092": "Li4SiO4 thermal conductivity",
    "P403": "Li4SiO4 density",
    "P406": "EU reference-pebble heat-capacity relation",
    "P428": "Pure Li4SiO4 enthalpy-increment relation",
    "P429": "Pure Li4SiO4 heat-capacity relation",
    "P430": "Pure Li4SiO4 molar-mass conversion",
    "P431": "Pure Li4SiO4 transition temperatures",
    "P424": "Helium property-table temperature range",
}

MODEL_NAMES = {
    "工程量时间Transformer": "Observable temporal Transformer",
    "图-Transformer": "Graph--Transformer",
    "扩散剩余误差修正": "Diffusion residual refinement",
    "共同能量算子": "Common finite-volume energy operator",
    "全耦合图-Transformer": "Fully coupled graph--Transformer",
    "稳态多输出模型": "Steady multi-output model",
    "稳态图网络": "Steady regional graph model",
    "稳态Transolver": "Steady Physics-Attention model",
    "POD低秩修正": "POD low-rank correction",
    "全耦合损失权重": "Fully coupled loss weighting",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def latex_escape(value: object) -> str:
    text = str(value or "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_\allowbreak{}",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(replacements.get(char, char) for char in text)
    for punctuation in (",", ";", "=", "/", "("):
        escaped = escaped.replace(punctuation, punctuation + r"\allowbreak{}")
    return escaped


def compact_text(value: object, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit and len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return latex_escape(text)


def path_list(value: str, limit: int = 3) -> str:
    parts = [item.strip() for item in value.split(";") if item.strip()]
    shown = [
        re.sub(r"[^\x00-\x7f]+", "non-ASCII-name", item)
        for item in parts[:limit]
    ]
    result = r"; ".join(r"\nolinkurl{" + item + "}" for item in shown)
    if len(parts) > limit:
        result += rf"; +{len(parts) - limit} additional files"
    return result or "---"


def url_text(value: str) -> str:
    value = value.strip()
    return r"\url{" + value + "}" if value else "---"


def longtable(
    *,
    caption: str,
    label: str,
    column_spec: str,
    header: str,
    rows: list[str],
    column_count: int,
    font_command: str = r"\scriptsize",
) -> str:
    return "\n".join(
        [
            r"\begingroup",
            font_command,
            r"\setlength{\tabcolsep}{3pt}",
            rf"\begin{{longtable}}{{{column_spec}}}",
            rf"\caption{{{caption}}}\label{{{label}}}\\",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            r"\endfirsthead",
            rf"\multicolumn{{{column_count}}}{{l}}{{\textit{{Table \thetable\ continued}}}}\\",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{column_count}}}{{r}}{{\textit{{Continued on next page}}}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
            *rows,
            r"\end{longtable}",
            r"\endgroup",
            "",
        ]
    )


def build_physical_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        evidence = (
            compact_text(row["文献"], 90)
            + r"\newline "
            + url_text(row["链接或DOI"])
        )
        body.append(
            " & ".join(
                [
                    compact_text(row["parameter_id"]),
                    compact_text(PHYSICAL_NAMES[row["parameter_id"]]),
                    compact_text(row["采用值或关系式"])
                    + " "
                    + compact_text(row["单位"]),
                    compact_text(row["原文位置说明"], 180),
                    evidence,
                ]
            )
            + r" \\"
        )
    return longtable(
        caption=(
            "Complete physical-parameter provenance. These entries define the "
            "geometry, materials and operating conditions; neural-network and "
            "numerical settings are listed separately."
        ),
        label="tab:supp_physical_parameters",
        column_spec=(
            r">{\raggedright\arraybackslash}p{1.05cm}"
            r">{\raggedright\arraybackslash}p{2.45cm}"
            r">{\raggedright\arraybackslash}p{2.65cm}"
            r">{\raggedright\arraybackslash}p{3.65cm}"
            r">{\raggedright\arraybackslash}p{5.25cm}"
        ),
        header=(
            r"ID & Physical quantity & Adopted value & Source location & "
            r"Literature source"
        ),
        rows=body,
        column_count=5,
    )


def build_equation_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        implementation = (
            path_list(row["OpenFOAM位置"], limit=2)
            + r"\newline "
            + path_list(row["Python实现"], limit=2)
        )
        body.append(
            " & ".join(
                [
                    compact_text(row["文献参数编号"]),
                    compact_text(row["符号或关系"]),
                    implementation,
                ]
            )
            + r" \\"
        )
    return longtable(
        caption=(
            "Mapping from source-backed inputs to the OpenFOAM reference "
            "calculation and the learning-model state. The listed paths identify "
            "the implementation rather than introducing additional physical values."
        ),
        label="tab:supp_equation_input_map",
        column_spec=(
            r">{\raggedright\arraybackslash}p{2.0cm}"
            r">{\raggedright\arraybackslash}p{4.0cm}"
            r">{\raggedright\arraybackslash}p{12.2cm}"
        ),
        header=(
            r"Source ID & Symbol/relation & OpenFOAM/Python implementation"
        ),
        rows=body,
        column_count=3,
    )


def build_model_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    compact_text(MODEL_NAMES.get(row["model"], row["model"])),
                    compact_text(row["setting"]),
                    compact_text(row["value"]),
                    compact_text(row["setting_type"]),
                    r"\textit{Source:} "
                    + path_list(row["source_path"], limit=2)
                    + r"\newline \textit{Code:} "
                    + path_list(row["implementation_path"], limit=2),
                ]
            )
            + r" \\"
        )
    return longtable(
        caption=(
            "Complete numerical and learning-model settings. Every entry is "
            "explicitly non-physical and therefore cannot alter the registered "
            "pebble-bed material properties or operating conditions."
        ),
        label="tab:supp_model_settings",
        column_spec=(
            r">{\raggedright\arraybackslash}p{2.7cm}"
            r">{\raggedright\arraybackslash}p{3.2cm}"
            r">{\raggedright\arraybackslash}p{1.5cm}"
            r">{\raggedright\arraybackslash}p{2.6cm}"
            r">{\raggedright\arraybackslash}p{8.0cm}"
        ),
        header=(
            r"Method & Setting & Value & Basis & Source and implementation"
        ),
        rows=body,
        column_count=5,
    )


def build_result_map(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    compact_text(row["result_or_section"]),
                    path_list(row["source_data"], limit=3),
                    path_list(row["program"], limit=3),
                    compact_text(row["status"]),
                ]
            )
            + r" \\"
        )
    return longtable(
        caption=(
            "Result-to-source map. ``Ready'' denotes an existing calculation or "
            "method artifact; pending rows remain absent from final scientific "
            "claims until the named formal calculation is complete."
        ),
        label="tab:supp_result_source_map",
        column_spec=(
            r">{\raggedright\arraybackslash}p{2.8cm}"
            r">{\raggedright\arraybackslash}p{6.6cm}"
            r">{\raggedright\arraybackslash}p{6.4cm}"
            r">{\raggedright\arraybackslash}p{2.4cm}"
        ),
        header=r"Result/section & Source data & Program & Status",
        rows=body,
        column_count=4,
    )


def build(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    physical = read_csv(root / "parameters/hccb_p418_physical_parameter_sources.csv")
    equations = read_csv(root / "parameters/hccb_p418_equation_input_map.csv")
    numerical = read_csv(root / "parameters/hccb_p418_model_numerical_settings.csv")
    result_map = read_csv(root / "manuscript/result_source_map.csv")

    expected = {"physical": 22, "equations": 31, "numerical": 78}
    actual = {
        "physical": len(physical),
        "equations": len(equations),
        "numerical": len(numerical),
    }
    if actual != expected:
        raise ValueError(f"Unexpected source-table counts: {actual}; expected {expected}")
    if len(result_map) < 40:
        raise ValueError(f"Result source map is unexpectedly short: {len(result_map)}")

    physical_ids = {row["parameter_id"] for row in physical}
    equation_ids = {
        item.strip()
        for row in equations
        for item in row["文献参数编号"].split(";")
        if item.strip()
    }
    if physical_ids != equation_ids:
        raise ValueError(
            "Physical/equation source IDs differ: "
            + json.dumps(
                {
                    "unused_physical": sorted(physical_ids - equation_ids),
                    "unknown_equation_ids": sorted(equation_ids - physical_ids),
                },
                ensure_ascii=False,
            )
        )
    if any(row["is_physical_parameter"] != "no" for row in numerical):
        raise ValueError("At least one model setting is incorrectly marked physical")

    outputs = {
        "generated_supp_physical_parameters.tex": build_physical_table(physical),
        "generated_supp_equation_input_map.tex": build_equation_table(equations),
        "generated_supp_model_settings.tex": build_model_table(numerical),
        "generated_supp_result_source_map.tex": build_result_map(result_map),
    }
    summary: dict[str, object] = {
        "status": "completed_p418_supplement_inputs",
        "physical_parameter_count": len(physical),
        "equation_input_count": len(equations),
        "model_numerical_setting_count": len(numerical),
        "result_source_map_count": len(result_map),
        "result_status_counts": dict(
            sorted(Counter(row["status"] for row in result_map).items())
        ),
        "all_physical_parameters_mapped": physical_ids == equation_ids,
        "all_model_settings_nonphysical": all(
            row["is_physical_parameter"] == "no" for row in numerical
        ),
        "generated_files": sorted(outputs),
        "new_physical_parameters": [],
    }
    return summary, outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "manuscript"
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=ROOT / "results/hccb_p418_supplement_inputs",
    )
    args = parser.parse_args()

    summary, outputs = build(args.project_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (args.output_dir / name).write_text(content, encoding="utf-8")
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    (args.summary_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
