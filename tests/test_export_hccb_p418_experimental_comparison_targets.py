#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/export_hccb_p418_experimental_comparison_targets.py"
SPEC = importlib.util.spec_from_file_location("experimental_targets", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(name: str) -> dict[str, str]:
    return {
        "观测量": name,
        "测量方法": "method",
        "获得方式": "direct" if name != "气固界面净换热" else "由稳态热量收支反推",
        "主要检验内容": "physics",
        "文献参数编号": "P001",
        "限制": "limitation",
    }


def test_build_targets_preserves_measurement_roles_and_energy_balance() -> None:
    physics = [
        {
            "condition_id": "case_a",
            "inlet_velocity_m_s": "0.05",
            "inlet_temperature_K": "300",
            "outlet_temperature_K": "500",
            "pressure_drop_Pa": "4",
            "cooling_wall_temperature_K": "635",
            "solid_maximum_temperature_K": "650",
            "generated_power_W": "2",
            "net_outward_enthalpy_flow_W": "1.2",
            "cooling_wall_heat_into_fluid_W": "0.8",
        }
    ]
    dimensionless = [
        {
            "condition_id": "case_a",
            "openfoam_solid_wall_heat_into_solid_W": "-0.5",
            "openfoam_interphase_heat_into_fluid_W": "1.5",
            "openfoam_interface_flux_and_phase_temperature_sign_agree": "True",
        }
    ]
    observables = [row(item[0]) for item in MODULE.OBSERVABLES]
    long_rows, balance_rows = MODULE.build_targets(
        physics, dimensionless, observables
    )
    assert len(long_rows) == len(MODULE.OBSERVABLES)
    assert balance_rows[0]["interphase_heat_from_solid_balance_W"] == 1.5
    assert balance_rows[0]["absolute_difference_over_generated"] == 0.0
    interface = [
        item for item in long_rows if item["observable"] == "气固界面净换热"
    ]
    assert interface[0]["experimental_obtaining_method"] == "由稳态热量收支反推"
    assert interface[0]["value_role"] == "steady_solid_energy_balance"


def test_actual_completed_cases_export(tmp_path: Path) -> None:
    physics = ROOT / "results/hccb_p418_mixed_endpoint_smoke_completed_physics/completed_case_physics.csv"
    heat = ROOT / "results/hccb_p418_mixed_endpoint_smoke_dimensionless_heat_transfer/dimensionless_heat_transfer.csv"
    if not physics.is_file() or not heat.is_file():
        return
    long_rows, balance_rows = MODULE.build_targets(
        MODULE.read_csv(physics),
        MODULE.read_csv(heat),
        MODULE.read_csv(ROOT / "parameters/hccb_p418_experimental_observable_matrix.csv"),
    )
    assert len(balance_rows) == 8
    assert len(long_rows) == 8 * len(MODULE.OBSERVABLES)
    assert max(float(row["absolute_difference_over_generated"]) for row in balance_rows) < 3e-4
    MODULE.write_csv(tmp_path / "targets.csv", long_rows)
    with (tmp_path / "targets.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == len(long_rows)
