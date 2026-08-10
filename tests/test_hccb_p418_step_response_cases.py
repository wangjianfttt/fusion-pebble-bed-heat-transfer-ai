#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "code/build_hccb_p418_step_response_cases.py"
RUNNER = ROOT / "code/run_hccb_p418_step_responses.sh"
FINALIZER = ROOT / "code/finalize_hccb_p418_step_response.sh"
HIGH_RE_RECOVERY = ROOT / "scripts/recover_hccb_p418_high_re_postprocess.sh"
HIGH_RE_RECOVERY_SBATCH = (
    ROOT / "cloud_migration/run_p418_high_re_postprocess_recovery_n96p.sbatch"
)
FORMAL_RECOVERY = ROOT / "scripts/recover_hccb_p418_formal_fixed_flow_postprocess.sh"
FORMAL_RECOVERY_SBATCH = (
    ROOT / "cloud_migration/run_p418_formal_fixed_flow_postprocess_recovery_n96p.sbatch"
)
TRANSIENT_SMOKE = ROOT / "code/run_hccb_openfoam13_true_thermal_transient_smoke.sh"


class P418StepResponseCaseTests(unittest.TestCase):
    def make_endpoint(
        self,
        matrix: Path,
        condition_id: str,
        velocity: float,
        temperature: float,
        source: float,
        final_time: int,
    ) -> None:
        case = matrix / condition_id
        for path in (
            case / "0/fluid",
            case / "0/solid",
            case / "system/fluid",
            case / "system/solid",
            case / "constant/fluid/polyMesh",
            case / "constant/solid/polyMesh",
        ):
            path.mkdir(parents=True, exist_ok=True)
        for region in ("fluid", "solid"):
            for name in ("points", "faces", "owner", "neighbour", "boundary"):
                (case / "constant" / region / "polyMesh" / name).write_text(f"{region}-{name}\n")
            (case / "constant" / region / "polyMesh/cellProc").write_text(
                f"{region}-endpoint-cellProc\n"
            )
        for name in ("U", "T", "p", "p_rgh"):
            (case / "0/fluid" / name).write_text(f"field {name}\n")
        (case / "0/solid/T").write_text("solid T\n")
        (case / "constant/solid/fvModels").write_text(f"q {source * 1e6:.0f};\n")
        (case / "constant/solid/physicalProperties").write_text("steady thermo placeholder\n")
        (case / "system/controlDict").write_text(
            "startTime 0;\nendTime 300;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 25;\npurgeWrite 2;\n"
        )
        for region in ("fluid", "solid"):
            (case / f"system/{region}/fvSchemes").write_text(
                "ddtSchemes { default steadyState; }\n"
            )
        (case / "system/fluid/fvSolution").write_text(
            "solvers\n{\n    h { solver PBiCGStab; }\n}\n"
            "PIMPLE\n{\n    flow yes;\n    momentumPredictor yes;\n}\n"
        )
        (case / "cht_smoke_metadata.json").write_text(
            json.dumps(
                {
                    "inlet_velocity_m_s": velocity,
                    "pore_opening_boundary_velocity_m_s": velocity / 0.4,
                    "inlet_open_area_fraction": 0.4,
                    "inlet_temperature_K": temperature,
                    "solid_heat_source_W_m3": source * 1e6,
                }
            )
        )
        (case / "formal_sample_complete.json").write_text(
            json.dumps({"time": str(final_time)}) + "\n"
        )

    def test_builder_copies_target_and_records_one_published_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = root / "matrix"
            matrix.mkdir()
            source_id = "u0p05_T300_q6p85"
            target_id = "u0p05_T900_q6p85"
            self.make_endpoint(matrix, source_id, 0.05, 300.0, 6.85, 200)
            self.make_endpoint(matrix, target_id, 0.05, 900.0, 6.85, 300)
            conditions = [
                {"condition_id": source_id, "inlet_velocity_m_s": 0.05, "inlet_temperature_K": 300.0, "solid_heat_source_MW_m3": 6.85},
                {"condition_id": target_id, "inlet_velocity_m_s": 0.05, "inlet_temperature_K": 900.0, "solid_heat_source_MW_m3": 6.85},
            ]
            (matrix / "matrix_manifest.json").write_text(
                json.dumps({"source_title": "P418 source", "source_doi": "doi", "published_conditions": conditions})
            )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "source_parameter_id": "P418",
                        "source_title": "P418 source",
                        "source_doi": "doi",
                        "numerical_time_design": {
                            "duration_s": 300,
                            "delta_t_s": 1,
                            "field_write_interval_s": 25,
                            "field_write_schedule": [
                                {"start_s": 0, "end_s": 25, "interval_s": 1},
                                {"start_s": 25, "end_s": 300, "interval_s": 25},
                            ],
                        },
                        "scientific_scope": "computed physical step",
                        "sequences": [
                            {
                                "sequence_id": "temperature_up",
                                "family": "inlet_temperature_step",
                                "source_condition_id": source_id,
                                "target_condition_id": target_id,
                            }
                        ],
                    }
                )
            )
            output = root / "steps"
            result = subprocess.run(
                [sys.executable, str(BUILDER), "--matrix-root", str(matrix), "--output-root", str(output), "--plan", str(plan), "--require-all-ready"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            metadata = json.loads((output / "temperature_up/step_case_metadata.json").read_text())
            self.assertEqual(list(metadata["changed_physical_input"]), ["inlet_temperature_K"])
            self.assertEqual(metadata["new_physical_parameters"], [])
            control = (output / "temperature_up/system/controlDict").read_text()
            self.assertIn("purgeWrite 0;", control)
            self.assertIn("deltaT 1.0;", control)
            self.assertIn("writeControl timeStep;", control)
            self.assertIn("endTime 1.0;", control)
            self.assertIn("writeInterval 1;", control)
            for region in ("fluid", "solid"):
                schemes = (output / f"temperature_up/system/{region}/fvSchemes").read_text()
                self.assertIn("default Euler;", schemes)
                self.assertNotIn("steadyState", schemes)
            solution = (output / "temperature_up/system/fluid/fvSolution").read_text()
            self.assertIn('"(rho|rhoFinal)"', solution)
            self.assertIn("solver diagonal;", solution)
            self.assertIn("flow no;", solution)
            self.assertIn("momentumPredictor no;", solution)
            self.assertEqual(
                metadata["transient_model"],
                "thermal_step_with_quasi_steady_target_hydrodynamics",
            )
            self.assertEqual(metadata["snapshot_times_s"], list(range(0, 26)) + list(range(50, 301, 25)))
            self.assertEqual(len(metadata["snapshot_times_s"]), 37)
            self.assertEqual(metadata["field_write_schedule"][0]["interval_s"], 1.0)
            self.assertEqual(len(metadata["execution_schedule"]), 36)
            self.assertTrue(
                all(
                    abs(
                        (row["end_s"] - row["start_s"])
                        - row["write_interval_s"]
                    )
                    < 1.0e-12
                    for row in metadata["execution_schedule"]
                )
            )
            self.assertEqual(
                [row["end_s"] for row in metadata["execution_schedule"]],
                metadata["snapshot_times_s"][1:],
            )
            self.assertEqual(metadata["source_final_time_s"], "200")
            self.assertEqual(metadata["target_final_time_s"], "300")
            solid_thermo = (output / "temperature_up/constant/solid/physicalProperties").read_text()
            self.assertIn("thermo          eIcoTabulated", solid_thermo)
            self.assertNotIn("steady thermo placeholder", solid_thermo)
            self.assertNotIn("P406", solid_thermo)
            self.assertEqual(
                metadata["transient_thermo_parameter_ids"],
                ["P092", "P403", "P428", "P429", "P430", "P431"],
            )
            self.assertIn("P429", metadata["transient_thermo_parameter_ids"])
            self.assertIn("P430", metadata["transient_thermo_parameter_ids"])
            thermo_metadata = json.loads(
                (output / "temperature_up/transient_solid_thermo.json").read_text()
            )
            self.assertEqual(
                thermo_metadata["parameter_ids"],
                ["P092", "P403", "P428", "P429", "P430", "P431"],
            )
            self.assertEqual(
                (matrix / target_id / "constant/solid/physicalProperties").read_text(),
                "steady thermo placeholder\n",
            )
            for region in ("fluid", "solid"):
                self.assertEqual(
                    (
                        matrix
                        / target_id
                        / f"constant/{region}/polyMesh/cellProc"
                    ).read_text(),
                    f"{region}-endpoint-cellProc\n",
                )
                self.assertFalse(
                    (
                        output
                        / "temperature_up"
                        / f"constant/{region}/polyMesh/cellProc"
                    ).exists()
                )

    def test_runner_uses_exact_field_copy_and_shell_syntax(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('cp "${target_case}/${target_time}/fluid/${field}"', text)
        self.assertIn("for field in U p p_rgh phi", text)
        self.assertIn('cp "${source_case}/${source_time}/fluid/T"', text)
        self.assertIn('cp "${source_case}/${source_time}/solid/T"', text)
        self.assertIn("verify_hccb_p418_step_initialization.py", text)
        self.assertIn('json.load(open(sys.argv[1]))["execution_schedule"]', text)
        self.assertIn("-entry startFrom -set latestTime", text)
        self.assertIn("transient stage", text)
        self.assertIn('-entry deltaT -set "${delta_t}"', text)
        self.assertIn("latest_complete_parallel_time()", text)
        self.assertIn("time_greater_or_equal()", text)
        self.assertIn("integer_write_steps()", text)
        self.assertIn("parallel_common_time_index()", text)
        self.assertIn("recover_incomplete_stage_end()", text)
        self.assertIn("compute_hccb_resume_write_interval.py", text)
        self.assertIn("-entry writeControl -set timeStep", text)
        self.assertIn('-entry writeInterval -set "${write_target_index}"', text)
        self.assertIn(
            '-entry writeInterval -set "${recovery_target_index}"',
            text,
        )
        self.assertIn("did not produce a complete", text)
        self.assertIn("([eE][+-]?[0-9]+)?", text)
        self.assertIn("|| return 1", text)
        self.assertIn("bash -euo pipefail -c", text)
        for field in (
            "fluid/T",
            "fluid/U",
            "fluid/p",
            "fluid/p_rgh",
            "fluid/phi",
            "solid/T",
            "uniform/time",
        ):
            self.assertIn(field, text)
        self.assertIn("resume ${sequence} from ${restart_time} s", text)
        self.assertIn("skip completed transient stage", text)
        self.assertIn('-entry startTime -set "${restart_time}"', text)
        self.assertIn(': > "${case_dir}/log.foamMultiRun.step"', text)
        self.assertIn("OPENFOAM_BASHRC=${OPENFOAM_BASHRC:-/opt/openfoam13/etc/bashrc}", text)
        self.assertIn('source "${OPENFOAM_BASHRC}"', text)
        self.assertIn('rm -rf "${case_dir}"/processor*', text)
        self.assertIn("-entry numberOfSubdomains -set \"${NP_PER_CASE}\"", text)
        self.assertIn("processor_count=$(find", text)
        self.assertIn("expected ${NP_PER_CASE}", text)
        self.assertIn("foamMultiRun failed during stage", text)
        self.assertIn('OPENFOAM_BASHRC="${OPENFOAM_BASHRC}"', text)
        finalizer_text = FINALIZER.read_text(encoding="utf-8")
        self.assertGreaterEqual(
            finalizer_text.count('format(float(value), ".12g")'),
            2,
        )
        self.assertIn('if [[ -f ${OPENFOAM_BASHRC} ]]', finalizer_text)
        self.assertIn("SKIP_RECONSTRUCT_IF_COMPLETE", finalizer_text)
        self.assertIn("reusing", finalizer_text)
        recovery_text = HIGH_RE_RECOVERY.read_text(encoding="utf-8")
        self.assertIn("SKIP_RECONSTRUCT_IF_COMPLETE=1", recovery_text)
        self.assertIn("REMOVE_PROCESSORS_AFTER_EXPORT=0", recovery_text)
        self.assertIn("postprocess_recovered_from_complete_300_s_fields", recovery_text)
        self.assertIn("allowed_for_model_fitting", recovery_text)
        formal_recovery_text = FORMAL_RECOVERY.read_text(encoding="utf-8")
        self.assertIn("SKIP_RECONSTRUCT_IF_COMPLETE=1", formal_recovery_text)
        self.assertIn("REMOVE_PROCESSORS_AFTER_EXPORT=0", formal_recovery_text)
        self.assertIn("postprocess_recovered_from_complete_300_s_fields", formal_recovery_text)
        self.assertIn('"allowed_for_model_fitting": True', formal_recovery_text)
        self.assertIn("completed_p418_formal_fixed_flow_sequence", formal_recovery_text)
        formal_sbatch_text = FORMAL_RECOVERY_SBATCH.read_text(encoding="utf-8")
        self.assertIn("formal_fixed_steps_300s_r10b_front6_resume_20260729", formal_sbatch_text)
        self.assertIn("formal_fixed_steps_300s_r10_recovery_20260729", formal_sbatch_text)
        self.assertIn("WORK_SET=${2:-back}", formal_sbatch_text)
        self.assertIn("summarize_hccb_p418_step_endpoint_readiness.py", text)
        self.assertIn('summary["ready_sequence_count"]', text)
        self.assertIn("--require-all-ready", text)
        self.assertIn("export_hccb_p418_step_regional_sequences.py", text)
        self.assertIn("--steady-dataset-index", text)
        self.assertIn("train_hccb_p418_spatiotemporal_regional_operator.py", text)
        chained = (
            ROOT / "code/run_hccb_p418_chained_initial_state_evaluation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("evaluate_hccb_p418_chained_initial_state.py", chained)
        self.assertIn("evaluate_hccb_p418_temporal_energy_balance.py", chained)
        self.assertIn("evaluate_hccb_p418_chained_diffusion.py", chained)
        self.assertIn("summarize_hccb_p418_fused_chain.py", chained)
        self.assertIn("--roles test", chained)
        self.assertIn("hccb_p418_60_pinn_", chained)
        self.assertIn("--role test", chained)
        self.assertGreaterEqual(text.count("--resume"), 5)
        self.assertIn("train_hccb_p418_regional_dmdc.py", text)
        self.assertIn("--physics-mode energy_and_flux", text)
        self.assertIn("--run-role formal_data_only", text)
        self.assertIn("--physics-mode data_only", text)
        self.assertIn(
            "TEMPERATURE_OUTPUT_MODE=${TEMPERATURE_OUTPUT_MODE:-literature_bounded_residual}",
            text,
        )
        self.assertGreaterEqual(
            text.count('--temperature-output-mode "${TEMPERATURE_OUTPUT_MODE}"'),
            5,
        )
        self.assertIn("regional_graph_transformer_bounded_data_only_", text)
        self.assertIn("regional_graph_transformer_bounded_physics_", text)
        self.assertIn("regional_graph_transformer_bounded_factorized_", text)
        self.assertIn(
            "regional_graph_transformer_bounded_physics_${split_name}",
            chained,
        )
        self.assertIn("--run-role formal_factorized", text)
        self.assertIn("--spatial-temporal-mode factorized_static_spatial", text)
        self.assertIn("train_hccb_p418_low_rank_temperature_residual.py", text)
        self.assertIn("evaluate_hccb_p418_temporal_energy_balance.py", text)
        self.assertIn("energy_balance_summary.json", text)
        self.assertIn('low_rank_temperature_residual_${split_name}', text)
        self.assertGreaterEqual(text.count('--residual-geometry "${SUBFACE_GEOMETRY}"'), 3)
        self.assertIn("RUN_DIFFUSION_BENCHMARK=${RUN_DIFFUSION_BENCHMARK:-1}", text)
        self.assertIn("pair_disjoint_stress_test", text)
        self.assertIn("STEP_SPLIT_NAMES", text)
        self.assertIn("--split-names ${STEP_SPLIT_NAMES}", text)
        self.assertIn("PRIMARY_MODEL_SEED=${PRIMARY_MODEL_SEED:-20260717}", text)
        self.assertIn("RUN_SEED_ROBUSTNESS=${RUN_SEED_ROBUSTNESS:-1}", text)
        self.assertIn("ROBUSTNESS_SPLIT=${ROBUSTNESS_SPLIT:-pair_disjoint_stress_test}", text)
        self.assertIn('ROBUSTNESS_MODEL_SEEDS=${ROBUSTNESS_MODEL_SEEDS:-"20260717 20260718 20260719"}', text)
        self.assertIn("summarize_hccb_p418_step_seed_robustness.py", text)
        self.assertIn("--seeds ${ROBUSTNESS_MODEL_SEEDS}", text)
        self.assertIn("DIFFUSION_DEVICE=${DIFFUSION_DEVICE:-cuda}", text)
        self.assertIn(
            "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}",
            text,
        )
        self.assertIn("export PYTORCH_CUDA_ALLOC_CONF", text)
        self.assertIn(
            "DIFFUSION_MICROBATCH_SIZE=${DIFFUSION_MICROBATCH_SIZE:-1}", text
        )
        self.assertIn(
            "DIFFUSION_ACTIVATION_PRECISION=${DIFFUSION_ACTIVATION_PRECISION:-bfloat16}",
            text,
        )
        self.assertIn("REQUIRE_GRAPH_CUDA=${REQUIRE_GRAPH_CUDA:-1}", text)
        self.assertIn("check_hccb_p418_gpu_training_ready.py", text)
        self.assertIn("hccb_dense_cht_p418_60_sourceflow_r3_dataset", text)
        self.assertIn("hccb_p418_60_sourceflow_r3_postprocess_summary.json", text)
        self.assertIn("validate_hccb_p418_steady_dataset_ready.py", text)
        self.assertIn("--expected-cases 60", text)
        self.assertIn("summarize_hccb_p418_step_model_comparison.py", text)
        self.assertIn("build_hccb_p418_transient_performance_table.py", text)
        self.assertIn("generated_transient_performance.tex", text)
        self.assertIn('--step-root "${STEP_ROOT}"', text)
        self.assertIn("RUN_DIFFUSION_BENCHMARK", text)
        self.assertIn("train_hccb_p418_temporal_temperature_diffusion.py", text)
        self.assertIn("--run-role computed_residual_benchmark", text)
        self.assertIn('--device "${DIFFUSION_DEVICE}"', text)
        self.assertIn('--microbatch-size "${DIFFUSION_MICROBATCH_SIZE}"', text)
        self.assertIn(
            '--activation-precision "${DIFFUSION_ACTIVATION_PRECISION}"', text
        )
        self.assertNotIn("--device cuda", text)
        self.assertNotIn("mapFields", text)
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        subprocess.run(["bash", "-n", str(FINALIZER)], check=True)
        subprocess.run(["bash", "-n", str(HIGH_RE_RECOVERY)], check=True)
        subprocess.run(["bash", "-n", str(HIGH_RE_RECOVERY_SBATCH)], check=True)
        subprocess.run(["bash", "-n", str(FORMAL_RECOVERY)], check=True)
        subprocess.run(["bash", "-n", str(FORMAL_RECOVERY_SBATCH)], check=True)
        subprocess.run(["bash", "-n", str(TRANSIENT_SMOKE)], check=True)

    def test_curve_splits_are_disjoint_and_cover_all_planned_sequences(self) -> None:
        plan = json.loads((ROOT / "parameters/hccb_p418_transient_step_plan.json").read_text())
        planned = {row["sequence_id"] for row in plan["sequences"]}
        payload = json.loads((ROOT / "parameters/hccb_p418_step_response_splits.json").read_text())
        for name, split in payload["splits"].items():
            train, validation, test = map(set, (split["train"], split["validation"], split["test"]))
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)
            self.assertEqual(train | validation | test, planned)
            self.assertEqual(len(train), 6)
            self.assertEqual(len(train | validation | test), 12)
            if name.startswith("direction_"):
                self.assertEqual((len(validation), len(test)), (3, 3))
            else:
                self.assertEqual((len(validation), len(test)), (2, 4))

        sequence = {row["sequence_id"]: row for row in plan["sequences"]}
        stress = payload["splits"]["pair_disjoint_stress_test"]
        role_pairs = {}
        for role in ("train", "validation", "test"):
            role_pairs[role] = {
                tuple(sorted((sequence[name]["source_condition_id"], sequence[name]["target_condition_id"])))
                for name in stress[role]
            }
        self.assertFalse(role_pairs["train"] & role_pairs["validation"])
        self.assertFalse(role_pairs["train"] & role_pairs["test"])
        self.assertFalse(role_pairs["validation"] & role_pairs["test"])


if __name__ == "__main__":
    unittest.main()
