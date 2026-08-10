import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from export_hccb_p418_transient_observables import SIGNALS, export_matrix
from configure_hccb_p418_transient_snapshots import maximin_indices, normalized_points, replace_purge_write


class P418TransientObservableExportTests(unittest.TestCase):
    def test_snapshot_selection_and_purge_update_are_deterministic(self):
        rows = [
            {"inlet_velocity_m_s": 0.05, "inlet_temperature_K": 300, "solid_heat_source_MW_m3": 4.85},
            {"inlet_velocity_m_s": 0.15, "inlet_temperature_K": 700, "solid_heat_source_MW_m3": 6.85},
            {"inlet_velocity_m_s": 0.25, "inlet_temperature_K": 900, "solid_heat_source_MW_m3": 8.85},
        ]
        self.assertEqual(maximin_indices(normalized_points(rows), 2), [1, 0])
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp) / "controlDict"
            control.write_text("writeInterval 25;\npurgeWrite 2;\n")
            replace_purge_write(control)
            self.assertIn("purgeWrite 0;", control.read_text())

    def test_export_joins_direct_function_object_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "matrix"
            case = root / "u0p05_T300_q4p85"
            root.mkdir()
            (root / "matrix_manifest.json").write_text(
                json.dumps(
                    {
                        "source_title": "source",
                        "source_doi": "https://doi.org/example",
                        "published_conditions": [
                            {
                                "condition_id": case.name,
                                "inlet_velocity_m_s": 0.05,
                                "inlet_temperature_K": 300.0,
                                "solid_heat_source_MW_m3": 4.85,
                            }
                        ],
                    }
                )
            )
            case.mkdir()
            (case / "cht_smoke_metadata.json").write_text(
                json.dumps(
                    {
                        "inlet_velocity_m_s": 0.05,
                        "inlet_temperature_K": 300.0,
                        "solid_heat_source_W_m3": 4.85e6,
                        "outlet_pressure_Pa": 1.2e5,
                        "cooling_wall_temperature_K": 635.0,
                    }
                )
            )
            values_by_name = {
                "inlet_temperature_K": [300.0, 300.0],
                "outlet_temperature_K": [300.0, 350.0],
                "inlet_pressure_Pa": [120010.0, 120020.0],
                "outlet_pressure_Pa": [120000.0, 120000.0],
                "inlet_mass_flow_kg_s": [-2.0, -2.0],
                "outlet_mass_flow_kg_s": [0.0, 1.9],
                "inlet_enthalpy_flow_W": [-3.0, -3.0],
                "outlet_enthalpy_flow_W": [0.0, 4.0],
                "cooling_wall_power_W": [10.0, 5.0],
                "maximum_solid_temperature_K": [300.0, 500.0],
                "volume_average_fluid_temperature_K": [300.0, 325.0],
                "volume_average_solid_temperature_K": [300.0, 420.0],
            }
            for name, (relative_dir, filename) in SIGNALS.items():
                path = case / "postProcessing" / relative_dir / "0" / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "# Time value\n"
                    + "\n".join(f"{time} {value}" for time, value in zip([0, 1], values_by_name[name]))
                    + "\n"
                )
            output = Path(tmp) / "out"
            summary = export_matrix(root, output)
            self.assertEqual(summary["case_count_with_time_histories"], 1)
            data = np.load(output / "hccb_p418_transient_observables.npz", allow_pickle=True)
            names = [str(value) for value in data["signal_names"]]
            pressure_index = names.index("pressure_drop_Pa")
            mass_index = names.index("signed_mass_residual_kg_s")
            self.assertEqual(data["values"][0, 1, pressure_index], 20.0)
            self.assertAlmostEqual(data["values"][0, 1, mass_index], -0.1)
            with (output / "hccb_p418_transient_observables_long.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)

            (case / "step_response_complete.json").write_text("{}\n")
            (case / "step_case_metadata.json").write_text(
                json.dumps(
                    {
                        "source_parameters": {
                            "inlet_velocity_m_s": 0.05,
                            "inlet_temperature_K": 900.0,
                            "solid_heat_source_MW_m3": 4.85,
                        },
                        "target_parameters": {
                            "inlet_velocity_m_s": 0.05,
                            "inlet_temperature_K": 300.0,
                            "solid_heat_source_MW_m3": 4.85,
                        },
                    }
                )
            )
            step_output = Path(tmp) / "step_out"
            step_summary = export_matrix(root, step_output, history_kind="physical_step_response")
            self.assertEqual(step_summary["history_kind"], "physical_step_response")
            self.assertIn("exact published P418 endpoints", step_summary["scientific_scope"])
            self.assertIn("source_inlet_temperature_K", step_summary["condition_names"])
            self.assertIn("target_inlet_temperature_K", step_summary["condition_names"])

            (case / "fully_coupled_step_response_complete.json").write_text("{}\n")
            (case / "fully_coupled_step_metadata.json").write_text(
                json.dumps(
                    {
                        "source_parameters": {
                            "inlet_velocity_m_s": 0.05,
                            "inlet_temperature_K": 900.0,
                            "solid_heat_source_MW_m3": 4.85,
                        },
                        "target_parameters": {
                            "inlet_velocity_m_s": 0.25,
                            "inlet_temperature_K": 900.0,
                            "solid_heat_source_MW_m3": 4.85,
                        },
                    }
                )
            )
            coupled_output = Path(tmp) / "coupled_out"
            coupled_summary = export_matrix(
                root,
                coupled_output,
                history_kind="fully_coupled_flow_heat_response",
            )
            self.assertEqual(
                coupled_summary["history_kind"],
                "fully_coupled_flow_heat_response",
            )
            self.assertEqual(coupled_summary["completed_case_count"], 1)
            self.assertIn(
                "Velocity, pressure, face mass flux",
                coupled_summary["scientific_scope"],
            )


if __name__ == "__main__":
    unittest.main()
