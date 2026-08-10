#!/usr/bin/env python3
"""Load the source-backed Li4SiO4 second-order transition regions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHARACTERIZATION = (
    ROOT / "parameters/hccb_p418_li4sio4_transition_characterization.json"
)


@dataclass(frozen=True)
class TransitionRegion:
    transition_id: str
    onset_temperature_k: float
    critical_temperature_reported_k: float
    end_temperature_k: float
    additional_enthalpy_uptake_j_mol: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_transition_characterization(
    path: Path = DEFAULT_CHARACTERIZATION,
) -> tuple[dict[str, object], tuple[TransitionRegion, ...]]:
    """Read and validate literature temperatures without fitting a peak shape."""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("status") != "source_backed_li4sio4_transition_characterization":
        raise ValueError("unexpected Li4SiO4 transition-characterization status")
    if payload.get("material") != "pure Li4SiO4":
        raise ValueError("transition characterization is not for pure Li4SiO4")

    source = payload["source"]
    pdf_path = ROOT / str(source["local_pdf"])
    if not pdf_path.is_file():
        raise FileNotFoundError(f"transition source PDF is missing: {pdf_path}")
    if sha256(pdf_path) != source["local_pdf_sha256"]:
        raise ValueError("transition source PDF SHA-256 differs")
    if int(source["evidence_page_in_pdf"]) != 3 or int(source["printed_page"]) != 102:
        raise ValueError("transition source page differs from the archived evidence")

    use = payload["use_in_this_project"]
    if not use["temperature_history_classification"]:
        raise ValueError("transition regions are not enabled for temperature classification")
    if use["openfoam_heat_capacity_modified"]:
        raise ValueError("transition characterization must not modify OpenFOAM heat capacity")
    if use["neural_network_target_modified"]:
        raise ValueError("transition characterization must not modify training targets")
    if use["analytic_peak_shape_assumed"]:
        raise ValueError("an analytic transition peak shape was not reported by the source")

    regions: list[TransitionRegion] = []
    for item in payload["transition_regions"]:
        onset_c = float(item["onset_temperature_degC"])
        critical_c = float(item["critical_temperature_degC"])
        end_c = float(item["end_temperature_degC"])
        onset_k = float(item["onset_temperature_K"])
        critical_converted_k = float(item["critical_temperature_converted_K"])
        critical_reported_k = float(item["critical_temperature_reported_K"])
        end_k = float(item["end_temperature_K"])
        if abs(onset_k - (onset_c + 273.15)) > 1.0e-10:
            raise ValueError("transition onset Celsius-to-kelvin conversion differs")
        if abs(critical_converted_k - (critical_c + 273.15)) > 1.0e-10:
            raise ValueError("critical-temperature Celsius-to-kelvin conversion differs")
        if abs(end_k - (end_c + 273.15)) > 1.0e-10:
            raise ValueError("transition end Celsius-to-kelvin conversion differs")
        if not onset_k < critical_converted_k < end_k:
            raise ValueError("transition onset, critical and end temperatures are unordered")
        if abs(critical_converted_k - critical_reported_k) > 0.2:
            raise ValueError("reported and Celsius-converted critical temperatures differ")
        enthalpy = float(item["additional_enthalpy_uptake_J_mol"])
        if enthalpy <= 0.0:
            raise ValueError("additional transition enthalpy must be positive")
        regions.append(
            TransitionRegion(
                transition_id=str(item["transition_id"]),
                onset_temperature_k=onset_k,
                critical_temperature_reported_k=critical_reported_k,
                end_temperature_k=end_k,
                additional_enthalpy_uptake_j_mol=enthalpy,
            )
        )
    if len(regions) != 2:
        raise ValueError("exactly two Li4SiO4 transition regions are required")
    if regions[0].end_temperature_k >= regions[1].onset_temperature_k:
        raise ValueError("the two reported transition regions overlap")
    return payload, tuple(regions)


if __name__ == "__main__":
    data, values = load_transition_characterization()
    print(
        json.dumps(
            {
                "status": data["status"],
                "transition_regions": [
                    {
                        "transition_id": item.transition_id,
                        "temperature_range_K": [
                            item.onset_temperature_k,
                            item.end_temperature_k,
                        ],
                        "critical_temperature_reported_K": (
                            item.critical_temperature_reported_k
                        ),
                        "additional_enthalpy_uptake_J_mol": (
                            item.additional_enthalpy_uptake_j_mol
                        ),
                    }
                    for item in values
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
