#!/usr/bin/env python3
"""Summarize whether transient solid fields traverse Li4SiO4 transitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from hccb_source_backed_thermophysical import (
    MANIFEST,
    load_hccb_thermophysical_parameters,
)
from hccb_p418_li4sio4_transition_characterization import (
    DEFAULT_CHARACTERIZATION,
    load_transition_characterization,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 differs: {actual} != {expected}")


def trapezoidal_sample_weights(time_s: np.ndarray) -> np.ndarray:
    if time_s.ndim != 1 or len(time_s) < 2:
        raise ValueError("each thermal trajectory needs at least two time points")
    if np.any(~np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("physical times must be finite and strictly increasing")
    delta = np.diff(time_s)
    weights = np.empty(len(time_s), dtype=np.float64)
    weights[0] = 0.5 * delta[0]
    weights[-1] = 0.5 * delta[-1]
    if len(time_s) > 2:
        weights[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    return weights


def p431_source(manifest_path: Path) -> dict[str, str]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["parameter_id"] == "P431"]
    if len(rows) != 1:
        raise ValueError(f"expected one P431 row, found {len(rows)}")
    row = rows[0]
    return {
        "parameter_id": row["parameter_id"],
        "title": row["source_title"],
        "url": row["source_url_or_doi"],
        "value": row["value"],
        "status": row["status"],
    }


def latex_text(summary: dict[str, object]) -> str:
    transitions = summary["transition_temperatures_K"]
    regions = summary["transition_regions"]
    reached = summary["sequence_count_reaching_each_transition"]
    crossing = summary["sequence_count_with_regional_trajectory_crossing"]
    entering = summary["sequence_count_entering_each_transition_region"]
    below = summary["minimum_temperature_K"]
    above = summary["maximum_temperature_K"]
    validity = summary["all_temperatures_inside_smoothed_heat_capacity_range"]
    validity_text = (
        "All exported solid temperatures remain inside the published smoothed heat-capacity interval."
        if validity
        else "Some exported solid temperatures lie outside the published smoothed heat-capacity interval."
    )
    return (
        f"The {summary['sequence_count']} thermal trajectories span a solid-temperature range of "
        f"{below:.1f}--{above:.1f}~K. The published Li$_4$SiO$_4$ second-order "
        f"transition temperatures are {transitions[0]:.0f} and {transitions[1]:.0f}~K "
        "(parameter record P431)~\\cite{kleykamp1996enthalpy}. "
        f"The lower transition lies inside the computed range for {reached[0]} trajectories "
        f"and is crossed by at least one regional solid-temperature history in {crossing[0]}; "
        f"the corresponding counts for the upper transition are {reached[1]} and {crossing[1]}. "
        f"The measured transition regions are {regions[0]['temperature_range_K'][0]:.2f}--"
        f"{regions[0]['temperature_range_K'][1]:.2f}~K and "
        f"{regions[1]['temperature_range_K'][0]:.2f}--"
        f"{regions[1]['temperature_range_K'][1]:.2f}~K; {entering[0]} and "
        f"{entering[1]} trajectories enter these regions, respectively. "
        f"{validity_text} The heat-capacity relation used here is the published smoothed "
        "calorimetric relation and does not reproduce the sharp transition anomalies; the "
        "reported integrated enthalpy uptakes are not imposed because the source does not "
        "provide a unique analytic peak shape. The coverage counts therefore identify where "
        "that model limitation matters rather than claiming transition-resolved thermodynamics.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--transition-characterization",
        type=Path,
        default=DEFAULT_CHARACTERIZATION,
    )
    args = parser.parse_args()

    dataset_path = args.dataset_index.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset.get("status") != "p418_regional_thermal_step_sequences_ready":
        raise ValueError("regional thermal-step dataset is not complete")
    if dataset.get("waiting_sequence_count") != 0:
        raise ValueError("regional thermal-step dataset still contains waiting sequences")
    if dataset.get("sequence_count") != len(dataset.get("sequences", [])):
        raise ValueError("sequence count differs from the dataset records")

    geometry_path = dataset_path.parent / dataset["regional_geometry_file"]
    require_sha(
        geometry_path,
        dataset["regional_geometry_sha256"],
        "regional sequence geometry",
    )
    with np.load(geometry_path, allow_pickle=False) as loaded:
        node_type = loaded["node_type"].astype(np.int8)
        node_volume = loaded["node_volume_m3"].astype(np.float64)
    solid = node_type == 1
    if not np.any(solid) or np.any(node_volume[solid] <= 0.0):
        raise ValueError("solid regional nodes or their volumes are invalid")

    state_names = dataset["state_names"]
    if state_names.count("temperature_K") != 1:
        raise ValueError("dataset must contain one temperature_K state channel")
    temperature_channel = state_names.index("temperature_K")
    parameters = load_hccb_thermophysical_parameters(args.manifest.resolve())
    transitions = np.asarray(parameters.solid_transition_temperatures_k, dtype=np.float64)
    cp_low, cp_high = parameters.solid_cp_temperature_range_k
    source = p431_source(args.manifest.resolve())
    characterization, transition_regions = load_transition_characterization(
        args.transition_characterization.resolve()
    )
    region_critical = np.asarray(
        [item.critical_temperature_reported_k for item in transition_regions],
        dtype=np.float64,
    )
    if not np.array_equal(transitions, region_critical):
        raise ValueError("P431 critical temperatures differ from the transition regions")

    records: list[dict[str, object]] = []
    all_solid_temperatures = []
    region_weighted_exposure = np.zeros(len(transition_regions), dtype=np.float64)
    total_weighted_exposure = 0.0
    for sequence in dataset["sequences"]:
        sequence_path = dataset_path.parent / sequence["sequence_file"]
        require_sha(sequence_path, sequence["sequence_sha256"], sequence["sequence_id"])
        with np.load(sequence_path, allow_pickle=False) as loaded:
            time_s = loaded["time_s"].astype(np.float64)
            state = loaded["state_physical"].astype(np.float64)
        if state.ndim != 3 or state.shape[0] != len(time_s):
            raise ValueError(f"invalid state shape in {sequence['sequence_id']}")
        if state.shape[1] != len(node_type) or state.shape[2] <= temperature_channel:
            raise ValueError(f"state and regional geometry differ in {sequence['sequence_id']}")
        solid_temperature = state[:, solid, temperature_channel]
        if np.any(~np.isfinite(solid_temperature)):
            raise ValueError(f"non-finite solid temperature in {sequence['sequence_id']}")
        time_weight = trapezoidal_sample_weights(time_s)
        volume = node_volume[solid]
        volume_time_weight = time_weight[:, None] * volume[None, :]
        total_volume_time = float(np.sum(volume_time_weight))
        total_weighted_exposure += total_volume_time
        node_min = np.min(solid_temperature, axis=0)
        node_max = np.max(solid_temperature, axis=0)
        minimum = float(np.min(solid_temperature))
        maximum = float(np.max(solid_temperature))
        row: dict[str, object] = {
            "sequence_id": sequence["sequence_id"],
            "family": sequence["family"],
            "time_point_count": len(time_s),
            "solid_temperature_minimum_K": minimum,
            "solid_temperature_maximum_K": maximum,
            "solid_temperature_range_K": maximum - minimum,
            "temperature_outside_smoothed_cp_range_count": int(
                np.count_nonzero(
                    (solid_temperature < cp_low) | (solid_temperature > cp_high)
                )
            ),
        }
        for index, transition in enumerate(transitions, start=1):
            snapshot_spans = (np.min(solid_temperature, axis=1) <= transition) & (
                np.max(solid_temperature, axis=1) >= transition
            )
            node_crosses = (node_min <= transition) & (node_max >= transition)
            row.update(
                {
                    f"transition_{index}_temperature_K": float(transition),
                    f"transition_{index}_inside_sequence_temperature_range": bool(
                        minimum <= transition <= maximum
                    ),
                    f"transition_{index}_snapshot_span_count": int(
                        np.count_nonzero(snapshot_spans)
                    ),
                    f"transition_{index}_regional_trajectory_crossing_count": int(
                        np.count_nonzero(node_crosses)
                    ),
                    f"transition_{index}_solid_volume_fraction_with_trajectory_crossing": float(
                        np.sum(volume[node_crosses]) / np.sum(volume)
                    ),
                    f"transition_{index}_solid_volume_time_fraction_at_or_above": float(
                        np.sum(volume_time_weight * (solid_temperature >= transition))
                        / total_volume_time
                    ),
                    f"transition_{index}_minimum_absolute_distance_K": float(
                        np.min(np.abs(solid_temperature - transition))
                    ),
                }
            )
            region = transition_regions[index - 1]
            onset = region.onset_temperature_k
            end = region.end_temperature_k
            inside_region = (solid_temperature >= onset) & (solid_temperature <= end)
            snapshot_overlaps_region = (
                np.min(solid_temperature, axis=1) <= end
            ) & (np.max(solid_temperature, axis=1) >= onset)
            node_enters_region = (node_min <= end) & (node_max >= onset)
            exposure = float(np.sum(volume_time_weight * inside_region))
            region_weighted_exposure[index - 1] += exposure
            distance_to_region = np.maximum(
                np.maximum(onset - solid_temperature, solid_temperature - end),
                0.0,
            )
            row.update(
                {
                    f"transition_{index}_region_onset_K": onset,
                    f"transition_{index}_region_end_K": end,
                    f"transition_{index}_additional_enthalpy_uptake_J_mol": (
                        region.additional_enthalpy_uptake_j_mol
                    ),
                    f"transition_{index}_sequence_enters_reported_region": bool(
                        np.any(node_enters_region)
                    ),
                    f"transition_{index}_snapshot_overlap_count": int(
                        np.count_nonzero(snapshot_overlaps_region)
                    ),
                    f"transition_{index}_regional_trajectory_entry_count": int(
                        np.count_nonzero(node_enters_region)
                    ),
                    f"transition_{index}_solid_volume_fraction_with_trajectory_entry": float(
                        np.sum(volume[node_enters_region]) / np.sum(volume)
                    ),
                    f"transition_{index}_solid_volume_time_fraction_inside_region": (
                        exposure / total_volume_time
                    ),
                    f"transition_{index}_minimum_distance_to_region_K": float(
                        np.min(distance_to_region)
                    ),
                }
            )
        records.append(row)
        all_solid_temperatures.append(solid_temperature.reshape(-1))

    if not records:
        raise ValueError("no complete thermal trajectories were found")
    combined = np.concatenate(all_solid_temperatures)
    reached = [
        sum(row[f"transition_{index}_inside_sequence_temperature_range"] for row in records)
        for index in range(1, len(transitions) + 1)
    ]
    crossing = [
        sum(
            row[f"transition_{index}_regional_trajectory_crossing_count"] > 0
            for row in records
        )
        for index in range(1, len(transitions) + 1)
    ]
    entering = [
        sum(
            row[f"transition_{index}_sequence_enters_reported_region"]
            for row in records
        )
        for index in range(1, len(transitions) + 1)
    ]
    region_records = [
        {
            "transition_id": item.transition_id,
            "temperature_range_K": [
                item.onset_temperature_k,
                item.end_temperature_k,
            ],
            "critical_temperature_reported_K": item.critical_temperature_reported_k,
            "additional_enthalpy_uptake_J_mol": (
                item.additional_enthalpy_uptake_j_mol
            ),
        }
        for item in transition_regions
    ]
    summary = {
        "status": "p418_li4sio4_transition_temperature_coverage_complete",
        "dataset_index": str(dataset_path),
        "dataset_index_sha256": sha256(dataset_path),
        "sequence_count": len(records),
        "transition_temperatures_K": transitions.tolist(),
        "transition_parameter_source": source,
        "transition_characterization_source": characterization["source"],
        "transition_regions": region_records,
        "smoothed_heat_capacity_temperature_range_K": [cp_low, cp_high],
        "minimum_temperature_K": float(np.min(combined)),
        "maximum_temperature_K": float(np.max(combined)),
        "sequence_count_reaching_each_transition": reached,
        "sequence_count_with_regional_trajectory_crossing": crossing,
        "sequence_count_entering_each_transition_region": entering,
        "combined_solid_volume_time_fraction_in_each_transition_region": (
            region_weighted_exposure / total_weighted_exposure
        ).tolist(),
        "all_temperatures_inside_smoothed_heat_capacity_range": bool(
            np.all((combined >= cp_low) & (combined <= cp_high))
        ),
        "sequence_results": records,
        "interpretation": (
            "The exact P431 critical temperatures and the source-reported onset/end "
            "temperatures are compared with computed regional solid-temperature histories. "
            "No fitted tolerance or analytic heat-capacity peak shape is introduced. The "
            "P428-P429 heat-capacity relation is smoothed and does not resolve the sharp "
            "transition anomalies. The reported additional enthalpy uptakes are retained "
            "as literature characterization and are not imposed on the OpenFOAM solution."
        ),
        "new_physical_parameters": [],
        "new_model_physical_parameters": [],
    }

    csv_path = output / "transition_temperature_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Li4SiO4相变温度覆盖情况",
        "",
        f"- 瞬态曲线数量：{len(records)}",
        f"- 固体温度总范围：{summary['minimum_temperature_K']:.1f}--{summary['maximum_temperature_K']:.1f} K",
        f"- 文献相变温度（P431）：{transitions[0]:.0f} K、{transitions[1]:.0f} K",
        (
            f"- 原文给出的相变影响区："
            f"{region_records[0]['temperature_range_K'][0]:.2f}--"
            f"{region_records[0]['temperature_range_K'][1]:.2f} K、"
            f"{region_records[1]['temperature_range_K'][0]:.2f}--"
            f"{region_records[1]['temperature_range_K'][1]:.2f} K"
        ),
        (
            f"- 原文积分得到的额外焓吸收："
            f"{region_records[0]['additional_enthalpy_uptake_J_mol']:.0f}、"
            f"{region_records[1]['additional_enthalpy_uptake_J_mol']:.0f} J/mol"
        ),
        f"- 温度范围达到两处相变温度的曲线数：{reached[0]}、{reached[1]}",
        f"- 至少有一个固体区域随时间跨过相变温度的曲线数：{crossing[0]}、{crossing[1]}",
        f"- 至少有一个固体区域进入两段原文温区的曲线数：{entering[0]}、{entering[1]}",
        f"- 全部温度是否位于P428--P429平滑热容关系的适用范围内：{'是' if summary['all_temperatures_inside_smoothed_heat_capacity_range'] else '否'}",
        "",
        "这里没有人为规定相变附近多少K，而是直接采用原文给出的起始温度和终止温度。",
        "原文没有给出唯一的解析热容峰形，因此900和630 J/mol只作为量热特征记录，不加入OpenFOAM比热曲线或神经网络目标。",
        "P428--P429是平滑热容关系，不能再现相变处的尖锐热容异常；本表用于指出哪些瞬态结果会受到这一物性近似影响。",
    ]
    (output / "transition_temperature_coverage_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if args.latex_output:
        args.latex_output.parent.mkdir(parents=True, exist_ok=True)
        args.latex_output.write_text(latex_text(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
