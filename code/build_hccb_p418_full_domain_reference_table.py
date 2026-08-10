#!/usr/bin/env python3
"""Write the one-condition local/full-domain CHT comparison for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def number(value: float, digits: int = 3) -> str:
    return f"\\num{{{float(value):.{digits}g}}}"


def percent(value: float, digits: int = 3) -> str:
    return number(100.0 * float(value), digits)


def build_table(comparison: dict) -> str:
    if comparison.get("status") != "hccb_p418_full_and_local_domain_compared":
        raise ValueError("full/local-domain comparison is incomplete")
    if comparison.get("new_physical_parameters") != []:
        raise ValueError("full-domain comparison introduced an unregistered physical parameter")

    geometry = comparison["geometry"]
    local = comparison["local_domain"]
    full = comparison["full_domain"]
    delta = comparison["local_relative_to_full"]
    condition = comparison["same_operating_condition"]

    local_length = float(geometry["local_flow_length_dp"])
    full_length = float(geometry["full_total_flow_length_dp"])
    rows = [
        ("Flow length ($d_p$)", number(local_length), number(full_length), number(full_length)),
        ("Pressure drop (Pa)", number(local["pressure_drop_Pa"]), number(full["pressure_drop_Pa"]), number(full["published_pressure_drop_Pa"])),
        ("Domain-average $|\\Delta p|/L$ (kPa m$^{-1}$)", number(local["domain_average_pressure_gradient_Pa_m"] / 1.0e3), number(full["domain_average_pressure_gradient_Pa_m"] / 1.0e3), "--"),
        ("Outlet temperature (K)", number(local["outlet_temperature_K"]), number(full["outlet_temperature_K"]), "--"),
        ("Maximum solid temperature (K)", number(local["maximum_solid_temperature_K"]), number(full["maximum_solid_temperature_K"]), number(full["published_maximum_temperature_K"])),
        ("$(T_{s,\\max}-T_w)/(q'''d_p^2/k_s)$", number(local["dimensionless_maximum_temperature"]), number(full["dimensionless_maximum_temperature"]), "--"),
        ("Cooling-wall heat / generated heat", number(local["cooling_wall_heat_over_generated_power"]), number(full["cooling_wall_heat_over_generated_power"]), "--"),
        ("Relative mass difference", number(local["relative_mass_imbalance"]), number(full["relative_mass_imbalance"]), "--"),
        ("Relative energy difference", number(local["relative_energy_imbalance"]), number(full["relative_energy_imbalance"]), "--"),
    ]
    body = "\n".join("{} & {} & {} & {} \\\\".format(*row) for row in rows)

    return rf"""\begin{{table*}}[htbp]
\centering
\small
\caption{{One-condition comparison between the fine local crop and the source-sized pore-resolved domain at $u_{{\rm in}}={number(condition['inlet_velocity_m_s'])}$ m s$^{{-1}}$, $T_{{\rm in}}={number(condition['inlet_temperature_K'])}$ K and $q'''={number(condition['solid_heat_source_MW_m3'])}$ MW m$^{{-3}}$. The source-sized pressure gradient uses the complete $30d_p$ flow length, including the published $10d_p$ inlet and outlet extensions, and is not a packed-region-only gradient. The published values are the P391 source-domain references \cite{{wang2023pore}}. This is a domain-size check at one condition, not a replacement for the 60-condition local training set.}}
\label{{tab:full_domain_reference}}
\begin{{tabular}}{{lrrr}}
\toprule
Quantity & Fine local crop & Source-sized domain & Published reference \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}

At the common operating condition, the local-to-source-sized differences are {percent(delta['domain_average_pressure_gradient_relative_difference'])}\% for the reported domain-average pressure gradient, {percent(delta['outlet_temperature_rise_relative_difference'])}\% for the outlet-temperature change relative to the inlet, {percent(delta['dimensionless_maximum_temperature_relative_difference'])}\% for the dimensionless maximum temperature and {percent(delta['cooling_wall_heat_fraction_relative_difference'])}\% for cooling-wall heat normalized by generated heat. These values quantify the effect of enlarging the resolved domain under the same physical inputs; they are not used to tune either calculation.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.comparison.read_text(encoding="utf-8"))
    table = build_table(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    summary = {
        "status": "hccb_p418_full_domain_manuscript_table_written",
        "comparison": str(args.comparison.resolve()),
        "output": str(args.output.resolve()),
        "new_physical_parameters": [],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
