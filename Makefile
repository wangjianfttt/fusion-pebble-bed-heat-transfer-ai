.PHONY: reproduce check params references dataset validate-dataset pinn-status pinn-summary operator-cases convective-cases operator-dataset fno-status fno-timed-status fno-residual fno-repair pino-status pino-timed-status pino-residual operator-baselines sparse-task sparse-baseline sparse-refiner sparse-refiner-repair sequence-task transformer-status transient-sequence transient-transformer transient-baselines transient-forecast-comparison transient-sparse-task transient-sparse-baseline transient-sparse-refiner transient-sparse-residual hccb-3d-transient-velocity-sequence hccb-3d-transient-velocity-baselines hccb-3d-transient-event-sequence hccb-3d-transient-event-baselines hccb-3d-transient-dense-forward-operator hccb-3d-transient-event-delta-library hccb-3d-transient-dense-forward-prior-gate hccb-3d-transient-sparse-field-event-delta-global-residual-prior hccb-3d-transient-sparse-field-event-delta-local-residual-prior hccb-3d-transient-sparse-field-event-delta-patch-residual-prior hccb-3d-transient-sparse-field-task hccb-3d-transient-sparse-field-baselines hccb-3d-transient-sparse-field-refiner hccb-3d-transient-sparse-field-physics hccb-3d-transient-sparse-field-residual-gate hccb-3d-transient-sparse-field-pde-refiner hccb-3d-transient-sparse-field-boundary-pde-operator hccb-3d-transient-sparse-field-boundary-pde-curriculum hccb-3d-transient-sparse-field-boundary-pino-operator hccb-3d-transient-sparse-field-diffusion-posterior hccb-3d-transient-sparse-field-fno-distilled hccb-3d-transient-sparse-field-dense-prior-bridge hccb-3d-transient-sparse-field-event-delta-prior-bridge hccb-3d-transient-sparse-field-event-delta-residual-structure hccb-3d-transient-sparse-field-event-delta-residual-sensors hccb-3d-transient-sparse-field-event-delta-diverse-residual-sensors hccb-3d-transient-sparse-field-event-delta-residual-calibrator hccb-3d-transient-sparse-field-event-delta-physics-gate hccb-3d-transient-sparse-field-event-delta-physics-constrained-residual hccb-3d-transient-sparse-field-event-delta-pde-sensor-encoder hccb-3d-transient-sparse-field-event-delta-transformer-posterior-head hccb-3d-transient-sparse-field-event-delta-score-posterior-head hccb-3d-transient-sparse-field-event-delta-basis-pde-guided hccb-3d-transient-sparse-field-event-delta-field-pde-projection hccb-3d-transient-sparse-field-event-delta-constrained-latent-denoiser hccb-3d-transient-sparse-field-event-delta-constrained-latent-denoiser-weight-sensitivity hccb-3d-transient-sparse-field-event-delta-trust-region-pde-proposal hccb-3d-transient-sparse-field-event-delta-observation-aware-pde-proposal hccb-3d-transient-sparse-field-event-delta-pareto-gate apd003-apd004-sparse-inverse-promotion-gate structure-gaps structure-activation nonuniform-3d-structure-support structure-support-gate cfd-dem-velocity-support-gate prepare-cfd-dem-velocity-package-templates prepare-cfd-dem-velocity-execution-packet prepare-cfd-dem-velocity-pilot-case-skeletons prepare-cfd-dem-velocity-remote-run-packet prepare-cfd-dem-velocity-solver-deck-templates cfd-dem-velocity-solver-deck-completion-plan cfd-dem-velocity-postprocess-packaging-plan validate-cfd-dem-velocity-solver-case-decks import-resolved-velocity-csv-package validate-cfd-dem-velocity-package resolved-cfd-dem-velocity-operator-task hccb-3d-wall-porosity-stress hccb-3d-wall-porosity-baselines hccb-3d-size-dispersion-stress hccb-3d-size-dispersion-baselines hccb-3d-permeability-velocity-proxy hccb-3d-permeability-velocity-baselines hccb-3d-support-expanded-operator hccb-3d-support-residual-operator hccb-3d-support-residual-observability hccb-3d-support-residual-token-task hccb-3d-support-residual-transformer-posterior hccb-3d-support-residual-coefficient-diffusion hccb-3d-support-residual-coefficient-diffusion-calibration anisotropic-keff-digitize anisotropic-keff-reconcile anisotropic-structure-task anisotropic-structure-operator anisotropic-structure-comparison structure-contrast-task structure-contrast-operator structure-contrast-boundary-operator structure-contrast-support-design structure-contrast-augmented-task structure-contrast-supported-operator hccb-3d-porous-heat hccb-3d-porous-baselines hccb-3d-porous-operator hccb-transport-closure hccb-modified-closure-screen hccb-modified-closure-dataset closure-aware-fno closure-aware-fno-keff closure-aware-pino closure-aware-pino-sweep closure-specific-fno closure-specific-pino closure-physics-gated-moe closure-ood-rejected-moe closure-support-aware-moe closure-support-coverage-design coverage-calibrated-closure coverage-calibrated-closure-pino coverage-calibrated-boundary-satisfying coverage-calibrated-closure-comparison coverage-error-localization coverage-boundary-projection closure-operator-baselines closure-operator-comparison architecture-gate research-route-gate hybrid-architecture-route-gate architecture-parameter-data-gate reduced-forward-timing reduced-forward-efficiency-gate training-cost-coverage-gate efficiency-timing-protocol structure-task structure-deeponet apd005-jpcrd-h2-he-primary-import-candidate apd005-isotope-hto-mapping-gate apd005-resolved-rtd-unlock-gate official-neuraloperator-fno-smoke apd006-module-source-audit apd006-aurora-repository-audit apd006-remote-aurora-staging-audit apd006-remote-solver-stack-audit apd006-hcpb-open-benchmark-candidate-audit prepare-apd006-module-benchmark-templates apd006-module-benchmark-contract apd006-premux-external-heat-benchmark operator-readiness

PYTHON ?= python3
GEOMETRY_PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,$(PYTHON))
TORCH_PYTHON ?= $(if $(wildcard /opt/anaconda3/bin/python3),/opt/anaconda3/bin/python3,$(PYTHON))
P418_ROOT ?= $(CURDIR)
P418_MATRIX_ROOT ?= $(P418_ROOT)/hccb_dense_cht_p418_60_sourceflow_r3
P418_PREFLIGHT_ROOT ?= $(P418_ROOT)/hccb_dense_cht_p418_sourceflow_preflight
P418_RESULT_ROOT ?= $(P418_ROOT)/results
P418_CONCURRENT_CASES ?= 1
P418_MPI_RANKS ?= 32
P418_PYTHON ?= $(if $(wildcard /data2/wangjian/venv/bin/python3),/data2/wangjian/venv/bin/python3,$(TORCH_PYTHON))

.PHONY: p418-progress p418-fused-preflight p418-research-route-check p418-parameter-evidence p418-local-transport-support p418-local-transport-sensitivity p418-scientific-findings p418-model-comparison-protocol p418-fully-coupled-step-plan p418-end-to-end-plan p418-fully-coupled-model-plan p418-high-re-evaluation-plan p418-formal-plan p418-formal-run p418-manuscript-refresh p418-reproducibility p418-public-test p418-physical-model-figure p418-field-cloud-figure

P418_TRANSIENT_RESULT_ROOT ?= $(P418_RESULT_ROOT)/hccb_p418_physical_steps_12

p418-physical-model-figure:
	$(PYTHON) code/plot_hccb_p418_physical_model_domain.py

p418-field-cloud-figure:
	ROOT="$(P418_ROOT)" RESULT_ROOT="$(P418_TRANSIENT_RESULT_ROOT)" \
		bash code/build_hccb_p418_selected_field_figure.sh

p418-parameter-evidence:
	$(PYTHON) code/verify_hccb_p418_parameter_evidence_files.py \
		--output "$(P418_RESULT_ROOT)/hccb_p418_parameter_evidence/summary.json"
	$(PYTHON) code/build_hccb_p418_parameter_evidence_summary.py

p418-local-transport-support:
	$(PYTHON) code/build_hccb_p418_local_transport_model_support.py

p418-local-transport-sensitivity:
	$(TORCH_PYTHON) code/check_hccb_p418_local_transport_model_sensitivity.py

p418-scientific-findings:
	$(PYTHON) code/build_hccb_p418_local_transport_model_support.py
	$(PYTHON) code/build_current_scientific_findings_cn.py

p418-model-comparison-protocol:
	$(PYTHON) code/verify_hccb_p418_model_comparison_protocol.py

p418-fully-coupled-step-plan:
	$(PYTHON) code/verify_hccb_p418_fully_coupled_step_plan.py

p418-fused-preflight:
	$(P418_PYTHON) code/preflight_hccb_p418_fused_research.py \
		--matrix-root "$(P418_MATRIX_ROOT)" \
		--step-root "$(P418_ROOT)/hccb_p418_physical_steps_12" \
		--output-dir "$(P418_RESULT_ROOT)/hccb_p418_fused_preflight"

p418-research-route-check:
	$(PYTHON) code/check_hccb_p418_research_route_completeness.py \
		--output-dir "$(P418_RESULT_ROOT)/hccb_p418_research_route_completeness"

p418-end-to-end-plan:
	$(PYTHON) code/plan_hccb_p418_end_to_end_research.py \
		--project-root "$(P418_ROOT)"

p418-fully-coupled-model-plan:
	ROOT="$(P418_ROOT)" EXECUTE=0 DEVICE=cpu \
		bash code/run_hccb_p418_fully_coupled_model_stage.sh

p418-high-re-evaluation-plan:
	ROOT="$(P418_ROOT)" EXECUTE=0 MODE=fixed \
		bash code/run_hccb_p418_high_re_independent_evaluation.sh
	ROOT="$(P418_ROOT)" EXECUTE=0 MODE=fully_coupled \
		bash code/run_hccb_p418_high_re_independent_evaluation.sh

p418-progress:
	@if test -f "$(P418_RESULT_ROOT)/hccb_p418_sourceflow_watch_status.txt"; then \
		head -n 1 "$(P418_RESULT_ROOT)/hccb_p418_sourceflow_watch_status.txt"; \
	fi
	@if test -d "$(P418_PREFLIGHT_ROOT)"; then \
		$(PYTHON) code/report_hccb_p418_runtime_progress.py \
			--matrix-root "$(P418_PREFLIGHT_ROOT)" \
			--concurrent-cases 1 \
			--parallel-ranks "$(P418_MPI_RANKS)"; \
	fi
	$(PYTHON) code/report_hccb_p418_runtime_progress.py \
		--matrix-root "$(P418_MATRIX_ROOT)" \
		--concurrent-cases "$(P418_CONCURRENT_CASES)" \
		--parallel-ranks "$(P418_MPI_RANKS)" \
		--output "$(P418_RESULT_ROOT)/hccb_p418_sourceflow_runtime_progress.json"

p418-formal-plan:
	ROOT="$(P418_ROOT)" RESULT_ROOT="$(P418_RESULT_ROOT)" \
		P418_PYTHON="$(P418_PYTHON)" EXECUTE=0 \
		bash code/run_hccb_p418_formal_calculations.sh

p418-formal-run:
	ROOT="$(P418_ROOT)" RESULT_ROOT="$(P418_RESULT_ROOT)" \
		P418_PYTHON="$(P418_PYTHON)" EXECUTE=1 \
		NP_PER_CASE="$(P418_MPI_RANKS)" CONCURRENT_CASES="$(P418_CONCURRENT_CASES)" \
		bash code/run_hccb_p418_formal_calculations.sh

p418-manuscript-refresh:
	ROOT="$(P418_ROOT)" RESULT_ROOT="$(P418_RESULT_ROOT)" \
		bash code/run_hccb_p418_manuscript_refresh.sh

p418-reproducibility:
	$(PYTHON) code/build_hccb_p418_public_figure_data.py \
		--project-root "$(P418_ROOT)" \
		--output-dir "$(P418_RESULT_ROOT)/hccb_p418_public_figure_data"
	$(PYTHON) code/build_hccb_p418_public_training_manifest.py \
		--input "$(P418_RESULT_ROOT)/hccb_p418_physical_steps_12/formal_training_jobs_workstation.json" \
		--output "$(P418_RESULT_ROOT)/hccb_p418_public_data_release_preflight/formal_training_manifest_public.json" \
		--source-root "$(P418_ROOT)"
	$(PYTHON) code/build_hccb_p418_public_data_release.py \
		--project-root "$(P418_ROOT)" \
		--output-dir "$(P418_RESULT_ROOT)/hccb_p418_public_data_release_preflight"
	$(PYTHON) code/build_hccb_p418_reproducibility_manifest.py \
		--project-root "$(P418_ROOT)" \
		--output-dir "$(P418_RESULT_ROOT)/hccb_p418_reproducibility_manifest" \
		--require-source-complete
	$(PYTHON) code/package_hccb_p418_reproducibility_source.py \
		--project-root "$(P418_ROOT)" \
		--manifest "$(P418_RESULT_ROOT)/hccb_p418_reproducibility_manifest/manifest.json" \
		--output "$(P418_RESULT_ROOT)/hccb_p418_reproducibility_manifest/p418_reproduction_source.tar.gz" \
		--record "$(P418_RESULT_ROOT)/hccb_p418_reproducibility_manifest/source_archive_record.json"

p418-public-test:
	PYTHON="$(P418_PYTHON)" bash scripts/test_p418_public_package.sh

reproduce:
	bash scripts/reproduce_current.sh

check:
	$(PYTHON) code/check_parameter_manifest.py

apd006-premux-external-heat-benchmark:
	$(PYTHON) code/extract_premux_steady_thermocouples.py
	$(PYTHON) code/build_apd006_premux_external_heat_benchmark_gate.py

.PHONY: apd006-premux-deterministic-thermal-contract
apd006-premux-deterministic-thermal-contract:
	$(PYTHON) code/build_premux_nominal_2d_geometry.py
	$(PYTHON) code/build_apd006_premux_deterministic_thermal_gate.py

.PHONY: apd006-premux-vhi-evidence
apd006-premux-vhi-evidence:
	$(PYTHON) code/extract_premux_meplas_vhi_predictions.py
	$(PYTHON) code/compare_premux_vhi_solver_evidence.py

.PHONY: apd006-premux-cooling-coordinate-sensitivity
apd006-premux-cooling-coordinate-sensitivity:
	$(PYTHON) code/build_premux_cooling_coordinate_sensitivity.py
	$(PYTHON) code/summarize_premux_cooling_coordinate_sensitivity.py

.PHONY: apd006-premux-vhi-literature-uncertainty
apd006-premux-vhi-literature-uncertainty:
	$(PYTHON) code/build_premux_vhi_literature_uncertainty_ensemble.py
	$(PYTHON) code/summarize_premux_vhi_literature_uncertainty_ensemble.py

.PHONY: apd006-premux-chapter6-full-thermal-gate
apd006-premux-chapter6-full-thermal-gate:
	$(PYTHON) code/build_apd006_premux_chapter6_full_thermal_gate.py

.PHONY: apd006-premux-source-loss-placement
apd006-premux-source-loss-placement:
	$(PYTHON) code/build_premux_source_loss_placement_sensitivity.py
	$(PYTHON) code/summarize_premux_source_loss_placement_sensitivity.py

.PHONY: apd006-premux-chapter6-open-thermal
apd006-premux-chapter6-open-thermal:
	$(GEOMETRY_PYTHON) code/build_premux_chapter6_open_thermal_deck.py
	$(PYTHON) code/summarize_premux_chapter6_open_thermal.py

.PHONY: apd006-tesomex-transient-gate
apd006-tesomex-transient-gate:
	$(PYTHON) code/build_apd006_tesomex_transient_gate.py

.PHONY: apd006-tesomex-curve-acquisition-gate
apd006-tesomex-curve-acquisition-gate:
	$(PYTHON) code/build_apd006_tesomex_curve_acquisition_gate.py

.PHONY: apd006-tesomex-figure-5-6-digitize
apd006-tesomex-figure-5-6-digitize:
	$(PYTHON) code/digitize_tesomex_figure_5_6.py

.PHONY: apd006-tesomex-priority-histories-digitize
apd006-tesomex-priority-histories-digitize:
	$(PYTHON) code/digitize_tesomex_figures_5_8_5_9.py
	$(PYTHON) code/digitize_tesomex_figure_5_8_radial_profiles.py

.PHONY: apd006-tesomex-1d-transient-baseline
apd006-tesomex-1d-transient-baseline:
	$(PYTHON) code/run_tesomex_1d_transient_convergence.py

.PHONY: apd006-tesomex-digitization-sensitivity
apd006-tesomex-digitization-sensitivity:
	$(PYTHON) code/audit_tesomex_digitization_sensitivity.py

.PHONY: apd006-tesomex-thermomechanical-gate
apd006-tesomex-thermomechanical-gate:
	$(PYTHON) code/build_apd006_tesomex_thermomechanical_gate.py

.PHONY: apd006-tesomex-creep-constitutive-gate
apd006-tesomex-creep-constitutive-gate:
	$(PYTHON) code/digitize_tesomex_figure_6_2_creep.py
	$(PYTHON) code/build_tesomex_creep_mapping_gate.py

.PHONY: apd006-tesomex-thermomech-solver-gate
apd006-tesomex-thermomech-solver-gate:
	$(PYTHON) code/build_apd006_tesomex_thermomech_solver_gate.py

.PHONY: apd006-tesomex-creep-material-regression
apd006-tesomex-creep-material-regression:
	$(PYTHON) code/summarize_tesomex_creep_material_regression.py

.PHONY: apd006-tesomex-creep-unit-contract
apd006-tesomex-creep-unit-contract:
	$(PYTHON) code/audit_tesomex_creep_unit_contract.py

.PHONY: apd006-tesomex-cap-creep-equivalence-gate
apd006-tesomex-cap-creep-equivalence-gate:
	$(PYTHON) code/build_tesomex_cap_creep_equivalence_gate.py

.PHONY: apd006-tesomex-tdem-support-gate
apd006-tesomex-tdem-support-gate:
	$(PYTHON) code/build_apd006_tesomex_tdem_support_gate.py

.PHONY: apd006-tdem-pair-conductance-unit-gate
apd006-tdem-pair-conductance-unit-gate:
	$(PYTHON) code/run_tdem_pair_conductance_unit_gate.py
	$(PYTHON) code/run_desu_pair_conductance_unit_gate.py
	$(PYTHON) code/run_helium_pair_conductance_unit_gate.py

.PHONY: apd006-desu2020-figure3-pf0631-digitization
apd006-desu2020-figure3-pf0631-digitization:
	$(PYTHON) code/digitize_desu2020_figure3_pf0631.py
	$(PYTHON) code/repeat_digitize_desu2020_figure3_pf0631.py

.PHONY: apd006-panchal2020-external-curve
apd006-panchal2020-external-curve:
	$(PYTHON) code/build_panchal2020_li2tio3_external_curve.py

.PHONY: apd006-tdem-mechanical-packing-gate
apd006-tdem-mechanical-packing-gate:
	$(PYTHON) code/build_tdem_mechanical_packing_gate.py

.PHONY: apd006-periodic-hertz-rve-smoke-case
apd006-periodic-hertz-rve-smoke-case:
	$(PYTHON) code/generate_periodic_hertz_rve_smoke.py

.PHONY: apd006-periodic-hertz-rve-smoke-summary
apd006-periodic-hertz-rve-smoke-summary:
	$(PYTHON) code/summarize_periodic_hertz_rve_smoke.py

.PHONY: apd006-periodic-hertz-stress-sweep-case
apd006-periodic-hertz-stress-sweep-case:
	$(PYTHON) code/generate_periodic_hertz_stress_sweep.py

.PHONY: apd006-periodic-hertz-stress-sweep-summary
apd006-periodic-hertz-stress-sweep-summary:
	$(PYTHON) code/summarize_periodic_hertz_stress_sweep.py

.PHONY: apd006-periodic-hertz-mpi-consistency
apd006-periodic-hertz-mpi-consistency:
	$(PYTHON) code/audit_periodic_hertz_mpi_consistency.py

.PHONY: apd006-periodic-hertz-relaxation-extension-case
apd006-periodic-hertz-relaxation-extension-case:
	$(PYTHON) code/generate_periodic_hertz_relaxation_extension.py

.PHONY: apd006-periodic-hertz-relaxation-extension-summary
apd006-periodic-hertz-relaxation-extension-summary:
	$(PYTHON) code/summarize_periodic_hertz_relaxation_extension.py

.PHONY: apd006-tdem-20mm-thermal-rve-case
apd006-tdem-20mm-thermal-rve-case:
	$(PYTHON) code/generate_tdem_20mm_thermal_rve.py

.PHONY: apd006-tdem-20mm-thermal-rve-summary
apd006-tdem-20mm-thermal-rve-summary:
	$(PYTHON) code/summarize_tdem_20mm_thermal_rve.py

.PHONY: apd006-tdem-20mm-relaxation-extension-case
apd006-tdem-20mm-relaxation-extension-case:
	$(PYTHON) code/generate_tdem_20mm_relaxation_extension.py

.PHONY: apd006-tdem-20mm-relaxation-extension-summary
apd006-tdem-20mm-relaxation-extension-summary:
	$(PYTHON) code/summarize_tdem_20mm_relaxation_extension.py

.PHONY: apd006-tdem-20mm-box-shrink-rve-case
apd006-tdem-20mm-box-shrink-rve-case:
	$(PYTHON) code/generate_tdem_20mm_box_shrink_rve.py

.PHONY: apd006-tdem-20mm-promote-relaxed-geometry
apd006-tdem-20mm-promote-relaxed-geometry:
	$(PYTHON) code/promote_tdem_20mm_relaxed_geometry.py

.PHONY: apd006-tdem-20mm-air-network
apd006-tdem-20mm-air-network:
	$(PYTHON) code/solve_tdem_20mm_air_network.py

.PHONY: apd006-desu2020-figure9-digitization
apd006-desu2020-figure9-digitization:
	$(PYTHON) code/digitize_desu2020_figure9_lmt_air.py
	$(PYTHON) code/repeat_digitize_desu2020_figure9_lmt_air.py

.PHONY: apd006-desu2020-zero-stress-comparison
apd006-desu2020-zero-stress-comparison:
	$(PYTHON) code/compare_tdem_network_desu2020_figure9.py

.PHONY: apd006-tdem-20mm-wall-stress-sweep-case
apd006-tdem-20mm-wall-stress-sweep-case:
	$(PYTHON) code/generate_tdem_20mm_wall_stress_sweep.py

.PHONY: apd006-tdem-20mm-wall-stress-sweep-summary
apd006-tdem-20mm-wall-stress-sweep-summary:
	$(PYTHON) code/summarize_tdem_20mm_wall_stress_sweep.py
	$(PYTHON) code/solve_tdem_20mm_wall_stress_network.py
	$(PYTHON) code/build_tdem_external_evidence_manifest.py

.PHONY: apd006-tdem-20mm-pf-sensitivity-case
apd006-tdem-20mm-pf-sensitivity-case:
	$(PYTHON) code/generate_tdem_20mm_pf_sensitivity.py

.PHONY: apd006-tdem-20mm-pf-sensitivity-summary
apd006-tdem-20mm-pf-sensitivity-summary:
	$(PYTHON) code/evaluate_tdem_20mm_pf_sensitivity.py

.PHONY: apd006-nscp3d-wall-relaxation-summary
apd006-nscp3d-wall-relaxation-summary:
	$(PYTHON) code/summarize_nscp3d_wall_relaxation.py

.PHONY: apd006-nscp3d-wall-stress-sweep-case
apd006-nscp3d-wall-stress-sweep-case:
	$(PYTHON) code/generate_nscp3d_wall_stress_sweep.py

.PHONY: apd006-nscp3d-wall-stress-sweep-summary
apd006-nscp3d-wall-stress-sweep-summary:
	$(PYTHON) code/summarize_nscp3d_wall_stress_sweep.py
	$(PYTHON) code/solve_nscp3d_wall_stress_network.py
	$(PYTHON) code/solve_nscp3d_wall_stress_network_helium.py

.PHONY: apd006-nscp3d-periodic-seed
apd006-nscp3d-periodic-seed:
	$(PYTHON) code/convert_nscp3d_to_lammps_periodic.py

.PHONY: apd006-nscp3d-periodic-stress-sweep-case
apd006-nscp3d-periodic-stress-sweep-case:
	$(PYTHON) code/generate_nscp3d_periodic_stress_sweep.py

.PHONY: apd006-nscp3d-periodic-stress-extension-case
apd006-nscp3d-periodic-stress-extension-case:
	$(PYTHON) code/generate_nscp3d_periodic_stress_extension.py from_N5000_pf0.630780_d1mm_ppp_seed_E210_stages20_deps0.0005_adaptive20x50000

.PHONY: apd006-nscp3d-periodic-stress-summary
apd006-nscp3d-periodic-stress-summary:
	$(PYTHON) code/summarize_nscp3d_periodic_stress_sweep.py
	$(PYTHON) code/solve_nscp3d_periodic_helium_network.py --source-id N5000_pf0.630780_d1mm_ppp_seed --output-id base

.PHONY: apd006-nscp3d-pf0631-realization-gate
apd006-nscp3d-pf0631-realization-gate:
	$(PYTHON) code/summarize_nscp3d_pf0631_realization_gate.py

.PHONY: apd006-nscp3d-periodic-helium-realization-comparison
apd006-nscp3d-periodic-helium-realization-comparison:
	$(PYTHON) code/summarize_nscp3d_periodic_helium_realizations.py

params:
	$(PYTHON) code/audit_hccb_geometry_source.py
	$(PYTHON) code/audit_hccb_low_re_validation_boundary.py
	$(PYTHON) code/audit_mpnm_algorithm_candidate.py
	$(PYTHON) code/audit_mpnm_hccb_mapping_readiness.py
	$(PYTHON) code/export_extracted_parameters.py
	$(PYTHON) code/thermal_properties_from_manifest.py

.PHONY: apd006-hccb-low-re-boundary-audit apd006-hccb-low-re-correlation-evaluation apd006-mpnm-source-audit apd006-mpnm-smoke-audit apd006-mpnm-hccb-mapping-audit
apd006-hccb-low-re-boundary-audit:
	$(PYTHON) code/audit_hccb_low_re_validation_boundary.py

apd006-hccb-low-re-correlation-evaluation:
	PYTHONPATH=code $(PYTHON) code/evaluate_hccb_internal_heat_low_re_correlations.py

apd006-mpnm-source-audit:
	$(PYTHON) code/audit_mpnm_algorithm_candidate.py

apd006-mpnm-smoke-audit:
	$(PYTHON) code/audit_mpnm_smoke_output.py \
		--log results/apd006_mpnm_algorithm_audit/remote_flow_smoke.log \
		--environment results/apd006_mpnm_algorithm_audit/remote_environment.json

apd006-mpnm-hccb-mapping-audit:
	$(PYTHON) code/audit_mpnm_hccb_mapping_readiness.py

references:
	$(PYTHON) code/solve_steady_2d_conduction_reference.py
	$(PYTHON) code/solve_hccb_heat_source_reference.py
	PYTHONPATH=code $(PYTHON) code/evaluate_hccb_internal_heat_low_re_correlations.py

dataset:
	$(PYTHON) code/export_operator_dataset.py

validate-dataset:
	$(PYTHON) code/validate_operator_dataset.py

pinn-status:
	$(PYTHON) code/train_pinn_hccb_reference.py

pinn-summary:
	$(PYTHON) code/summarize_pinn_runs.py

operator-cases:
	$(PYTHON) code/build_hccb_operator_case_manifest.py

convective-cases:
	$(PYTHON) code/solve_hccb_convective_cases.py

operator-dataset:
	$(PYTHON) code/export_convective_operator_dataset.py

fno-status:
	$(PYTHON) code/train_fno_hccb_operator.py

fno-timed-status:
	FNO_RUN_ID=timed_fno_e300 FNO_EPOCHS=300 $(PYTHON) code/train_fno_hccb_operator.py

fno-residual:
	$(PYTHON) code/evaluate_fno_physics_residual.py

fno-repair:
	$(PYTHON) code/repair_fno_prediction_physics.py

pino-status:
	$(PYTHON) code/train_pino_hccb_operator.py

pino-timed-status:
	PINO_RUN_ID=timed_pino_pre_timed_fno_e300_curr_res0to0p001_bc0to0p005_e300 PINO_PRETRAIN_RUN_ID=timed_fno_e300 PINO_CURRICULUM=linear PINO_RESIDUAL_WEIGHT_START=0 PINO_RESIDUAL_WEIGHT_END=0.001 PINO_BOUNDARY_WEIGHT_START=0 PINO_BOUNDARY_WEIGHT_END=0.005 PINO_EPOCHS=300 PINO_LR=0.0005 $(PYTHON) code/train_pino_hccb_operator.py

pino-residual:
	OPERATOR_PREDICTION_FAMILY=pino_hccb_operator OPERATOR_RESIDUAL_FAMILY=pino_physics_residual $(PYTHON) code/evaluate_fno_physics_residual.py

operator-baselines:
	$(PYTHON) code/summarize_operator_baselines.py

apd001-minimal-experiment:
	$(PYTHON) code/run_apd001_minimal_experiment.py

apd001-promotion-gate:
	$(PYTHON) code/build_apd001_promotion_gate.py

apd001-selected-route-timing-gate:
	$(PYTHON) code/build_apd001_selected_route_timing_gate.py

apd002-structure-promotion-gate:
	$(PYTHON) code/build_apd002_structure_promotion_gate.py

sparse-task:
	$(PYTHON) code/build_sparse_observation_task.py

sparse-baseline:
	$(PYTHON) code/reconstruct_sparse_temperature_baseline.py

sparse-refiner:
	$(PYTHON) code/train_sparse_denoising_refiner.py

sparse-refiner-repair:
	$(PYTHON) code/repair_sparse_refiner_prediction.py

sequence-task:
	$(PYTHON) code/build_operating_sequence_task.py

transformer-status:
	$(PYTHON) code/train_transformer_operating_sequence.py

transient-sequence:
	$(PYTHON) code/build_transient_heat_sequence_task.py

transient-transformer:
	$(PYTHON) code/train_transformer_transient_heat.py
	TRANSIENT_TRANSFORMER_RUN_ID=horizon5 TRANSIENT_HORIZON=5 $(PYTHON) code/train_transformer_transient_heat.py

transient-baselines:
	$(PYTHON) code/train_transient_forecast_baselines.py
	TRANSIENT_BASELINE_RUN_ID=horizon5 TRANSIENT_BASELINE_HORIZON=5 $(PYTHON) code/train_transient_forecast_baselines.py

transient-forecast-comparison:
	$(PYTHON) code/summarize_transient_forecast_models.py

transient-sparse-task:
	$(PYTHON) code/build_transient_sparse_observation_task.py

transient-sparse-baseline:
	$(PYTHON) code/reconstruct_transient_sparse_baseline.py

transient-sparse-refiner:
	$(PYTHON) code/train_transient_sparse_refiner.py

transient-sparse-residual:
	$(PYTHON) code/evaluate_transient_sparse_physics.py

hccb-3d-transient-velocity-sequence:
	$(PYTHON) code/build_hccb_3d_transient_velocity_sequence_task.py

hccb-3d-transient-velocity-baselines:
	$(PYTHON) code/train_hccb_3d_transient_velocity_forecast_baselines.py
	HCCB_3D_TRANSIENT_BASELINE_RUN_ID=horizon5 HCCB_3D_TRANSIENT_HORIZON=5 $(PYTHON) code/train_hccb_3d_transient_velocity_forecast_baselines.py

hccb-3d-transient-event-sequence:
	$(PYTHON) code/build_hccb_3d_transient_event_sequence_task.py

hccb-3d-transient-event-baselines:
	HCCB_3D_TRANSIENT_TASK_PATH=data/hccb_3d_transient_event_sequence_tasks/hccb_3d_transient_event_sequences.npz HCCB_3D_TRANSIENT_OUT_ROOT=results/hccb_3d_transient_event_sequence_tasks/forecast_baselines $(PYTHON) code/train_hccb_3d_transient_velocity_forecast_baselines.py
	HCCB_3D_TRANSIENT_TASK_PATH=data/hccb_3d_transient_event_sequence_tasks/hccb_3d_transient_event_sequences.npz HCCB_3D_TRANSIENT_OUT_ROOT=results/hccb_3d_transient_event_sequence_tasks/forecast_baselines HCCB_3D_TRANSIENT_BASELINE_RUN_ID=horizon5 HCCB_3D_TRANSIENT_HORIZON=5 $(PYTHON) code/train_hccb_3d_transient_velocity_forecast_baselines.py

hccb-3d-transient-dense-forward-operator:
	$(PYTHON) code/train_hccb_3d_transient_dense_forward_operator.py

hccb-3d-transient-event-delta-library:
	$(PYTHON) code/evaluate_hccb_3d_transient_event_delta_library.py

hccb-3d-transient-dense-forward-prior-gate:
	$(PYTHON) code/gate_hccb_3d_transient_dense_forward_prior.py

hccb-3d-transient-sparse-field-event-delta-global-residual-prior:
	HCCB_3D_EVENT_DELTA_GLOBAL_RESIDUAL_PRIOR_RANKS=4,8 HCCB_3D_EVENT_DELTA_GLOBAL_RESIDUAL_PRIOR_ALPHAS=0.001,0.01,0.1,1.0 HCCB_3D_EVENT_DELTA_GLOBAL_RESIDUAL_PRIOR_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0 $(PYTHON) code/train_hccb_3d_event_delta_global_residual_prior.py

hccb-3d-transient-sparse-field-event-delta-local-residual-prior:
	HCCB_3D_EVENT_DELTA_LOCAL_RESIDUAL_PRIOR_ALPHAS=0.001,0.01,0.1,1.0 HCCB_3D_EVENT_DELTA_LOCAL_RESIDUAL_PRIOR_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0 HCCB_3D_EVENT_DELTA_LOCAL_RESIDUAL_PRIOR_MAX_ROWS=250000 $(PYTHON) code/train_hccb_3d_event_delta_local_residual_prior.py

hccb-3d-transient-sparse-field-event-delta-patch-residual-prior:
	HCCB_3D_EVENT_DELTA_PATCH_RESIDUAL_PRIOR_ALPHAS=0.01,0.1,1.0,10.0 HCCB_3D_EVENT_DELTA_PATCH_RESIDUAL_PRIOR_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0 HCCB_3D_EVENT_DELTA_PATCH_RESIDUAL_PRIOR_MAX_ROWS=180000 $(PYTHON) code/train_hccb_3d_event_delta_patch_residual_prior.py

hccb-3d-transient-sparse-field-task:
	$(PYTHON) code/build_hccb_3d_transient_sparse_field_task.py

hccb-3d-transient-sparse-field-baselines:
	$(PYTHON) code/reconstruct_hccb_3d_transient_sparse_field_baselines.py

hccb-3d-transient-sparse-field-refiner:
	$(PYTHON) code/train_hccb_3d_transient_sparse_field_linear_refiner.py

hccb-3d-transient-sparse-field-physics:
	$(PYTHON) code/evaluate_hccb_3d_transient_sparse_field_physics.py

hccb-3d-transient-sparse-field-residual-gate:
	$(PYTHON) code/sweep_hccb_3d_sparse_refiner_residual_gate.py

hccb-3d-transient-sparse-field-pde-refiner:
	$(PYTHON) code/train_hccb_3d_sparse_field_pde_refiner.py

hccb-3d-transient-sparse-field-boundary-pde-operator:
	$(PYTHON) code/train_hccb_3d_sparse_field_boundary_pde_operator.py

hccb-3d-transient-sparse-field-boundary-pde-curriculum:
	HCCB_3D_BOUNDARY_PDE_OPERATOR_OUT_NAME=boundary_pde_operator_curriculum HCCB_3D_BOUNDARY_PDE_OPERATOR_WEIGHTS=0.02 HCCB_3D_BOUNDARY_PDE_OPERATOR_PRETRAIN_EPOCHS=18 HCCB_3D_BOUNDARY_PDE_OPERATOR_EPOCHS=18 $(PYTHON) code/train_hccb_3d_sparse_field_boundary_pde_operator.py

hccb-3d-transient-sparse-field-boundary-pino-operator:
	HCCB_3D_BOUNDARY_PDE_OPERATOR_OUT_NAME=boundary_pino_operator HCCB_3D_BOUNDARY_PDE_OPERATOR_MODEL=fno HCCB_3D_BOUNDARY_PDE_OPERATOR_WIDTH=6 HCCB_3D_BOUNDARY_PDE_OPERATOR_WEIGHTS=0.0,0.02 HCCB_3D_BOUNDARY_PDE_OPERATOR_EPOCHS=18 $(PYTHON) code/train_hccb_3d_sparse_field_boundary_pde_operator.py

hccb-3d-transient-sparse-field-diffusion-posterior:
	$(PYTHON) code/sample_hccb_3d_sparse_field_diffusion_posterior.py

hccb-3d-transient-sparse-field-fno-distilled:
	$(PYTHON) code/train_hccb_3d_sparse_field_fno_distilled.py

hccb-3d-transient-sparse-field-dense-prior-bridge:
	$(PYTHON) code/evaluate_hccb_3d_dense_forward_prior_sparse_bridge.py

hccb-3d-transient-sparse-field-event-delta-prior-bridge:
	$(PYTHON) code/evaluate_hccb_3d_event_delta_prior_sparse_bridge.py

hccb-3d-transient-sparse-field-event-delta-residual-structure:
	$(PYTHON) code/analyze_hccb_3d_event_delta_residual_structure.py

hccb-3d-transient-sparse-field-event-delta-residual-sensors:
	$(PYTHON) code/design_hccb_3d_event_delta_residual_sensors.py

hccb-3d-transient-sparse-field-event-delta-diverse-residual-sensors:
	HCCB_3D_EVENT_DELTA_DIVERSE_SENSOR_POWER=0.5 $(PYTHON) code/design_hccb_3d_event_delta_diverse_residual_sensors.py

hccb-3d-transient-sparse-field-event-delta-residual-calibrator:
	$(PYTHON) code/train_hccb_3d_event_delta_residual_calibrator.py

hccb-3d-transient-sparse-field-event-delta-residual-physics-gate:
	$(PYTHON) code/gate_hccb_3d_event_delta_residual_calibrator_physics.py

hccb-3d-transient-sparse-field-event-delta-physics-constrained-residual:
	$(PYTHON) code/train_hccb_3d_event_delta_physics_constrained_residual.py

hccb-3d-transient-sparse-field-event-delta-pde-sensor-encoder:
	HCCB_3D_EVENT_DELTA_PDE_ENCODER_EPOCHS=40 HCCB_3D_EVENT_DELTA_PDE_ENCODER_WEIGHTS=0.001,0.01 HCCB_3D_EVENT_DELTA_PDE_ENCODER_GAMMAS=0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0 HCCB_3D_EVENT_DELTA_PDE_ENCODER_BUDGETS=36 HCCB_3D_EVENT_DELTA_PDE_ENCODER_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors $(PYTHON) code/train_hccb_3d_event_delta_pde_sensor_encoder.py

hccb-3d-transient-sparse-field-event-delta-structured-residual-basis:
	HCCB_3D_EVENT_DELTA_BASIS_RANKS=4,8,16,32,64 HCCB_3D_EVENT_DELTA_BASIS_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0 HCCB_3D_EVENT_DELTA_BASIS_BUDGETS=36 HCCB_3D_EVENT_DELTA_BASIS_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors $(PYTHON) code/gate_hccb_3d_event_delta_structured_residual_basis.py

hccb-3d-transient-sparse-field-event-delta-basis-posterior:
	HCCB_3D_EVENT_DELTA_BASIS_POSTERIOR_RANKS=4,8 HCCB_3D_EVENT_DELTA_BASIS_POSTERIOR_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75 HCCB_3D_EVENT_DELTA_BASIS_POSTERIOR_NOISE=0.0,0.25,0.5,1.0 HCCB_3D_EVENT_DELTA_BASIS_POSTERIOR_SAMPLES=24 HCCB_3D_EVENT_DELTA_BASIS_POSTERIOR_BUDGETS=36 HCCB_3D_EVENT_DELTA_BASIS_POSTERIOR_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors $(PYTHON) code/sample_hccb_3d_event_delta_basis_posterior.py

hccb-3d-transient-sparse-field-event-delta-basis-posterior-head:
	HCCB_3D_EVENT_DELTA_BASIS_HEAD_RANKS=8 HCCB_3D_EVENT_DELTA_BASIS_HEAD_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75 HCCB_3D_EVENT_DELTA_BASIS_HEAD_SIGMA=0.0,0.5,1.0 HCCB_3D_EVENT_DELTA_BASIS_HEAD_SAMPLES=16 HCCB_3D_EVENT_DELTA_BASIS_HEAD_EPOCHS=90 HCCB_3D_EVENT_DELTA_BASIS_HEAD_BUDGETS=36 HCCB_3D_EVENT_DELTA_BASIS_HEAD_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors $(PYTHON) code/train_hccb_3d_event_delta_basis_posterior_head.py

hccb-3d-transient-sparse-field-event-delta-transformer-posterior-head:
	HCCB_3D_EVENT_DELTA_TRANSFORMER_HEAD_RANKS=8 HCCB_3D_EVENT_DELTA_TRANSFORMER_HEAD_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75 HCCB_3D_EVENT_DELTA_TRANSFORMER_HEAD_SIGMA=0.0,0.5,1.0 HCCB_3D_EVENT_DELTA_TRANSFORMER_HEAD_SAMPLES=16 HCCB_3D_EVENT_DELTA_TRANSFORMER_HEAD_EPOCHS=80 HCCB_3D_EVENT_DELTA_TRANSFORMER_HEAD_BUDGETS=36 HCCB_3D_EVENT_DELTA_TRANSFORMER_HEAD_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors,diverse_residual_sensors $(PYTHON) code/train_hccb_3d_event_delta_transformer_posterior_head.py

hccb-3d-transient-sparse-field-event-delta-score-posterior-head:
	HCCB_3D_EVENT_DELTA_SCORE_HEAD_RANKS=8 HCCB_3D_EVENT_DELTA_SCORE_HEAD_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75 HCCB_3D_EVENT_DELTA_SCORE_HEAD_SIGMA=0.0,0.25,0.5,1.0 HCCB_3D_EVENT_DELTA_SCORE_HEAD_SAMPLES=16 HCCB_3D_EVENT_DELTA_SCORE_HEAD_EPOCHS=80 HCCB_3D_EVENT_DELTA_SCORE_HEAD_BUDGETS=36 HCCB_3D_EVENT_DELTA_SCORE_HEAD_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors $(PYTHON) code/train_hccb_3d_event_delta_score_posterior_head.py

hccb-3d-transient-sparse-field-event-delta-basis-pde-guided:
	HCCB_3D_EVENT_DELTA_BASIS_PDE_GUIDED_RANKS=8 HCCB_3D_EVENT_DELTA_BASIS_PDE_GUIDED_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75 HCCB_3D_EVENT_DELTA_BASIS_PDE_GUIDED_STEPS=-1.0,-0.5,0.0,0.5,1.0 HCCB_3D_EVENT_DELTA_BASIS_PDE_GUIDED_BUDGETS=36 HCCB_3D_EVENT_DELTA_BASIS_PDE_GUIDED_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors $(PYTHON) code/optimize_hccb_3d_event_delta_basis_pde_guided.py

hccb-3d-transient-sparse-field-event-delta-field-pde-projection:
	HCCB_3D_EVENT_DELTA_FIELD_PDE_BUDGETS=36 HCCB_3D_EVENT_DELTA_FIELD_PDE_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors HCCB_3D_EVENT_DELTA_FIELD_PDE_TRUST_SCALES=0.05,0.1,0.2 HCCB_3D_EVENT_DELTA_FIELD_PDE_STEPS=12 $(PYTHON) code/optimize_hccb_3d_event_delta_field_pde_projection.py

hccb-3d-transient-sparse-field-event-delta-pde-regularized-basis-head:
	HCCB_3D_EVENT_DELTA_PDE_BASIS_HEAD_RANKS=8 HCCB_3D_EVENT_DELTA_PDE_BASIS_HEAD_WEIGHTS=0.0,0.0001,0.001,0.01 HCCB_3D_EVENT_DELTA_PDE_BASIS_HEAD_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75 HCCB_3D_EVENT_DELTA_PDE_BASIS_HEAD_BUDGETS=36 HCCB_3D_EVENT_DELTA_PDE_BASIS_HEAD_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors HCCB_3D_EVENT_DELTA_PDE_BASIS_HEAD_EPOCHS=60 $(PYTHON) code/train_hccb_3d_event_delta_pde_regularized_basis_head.py

hccb-3d-transient-sparse-field-event-delta-latent-pde-refiner:
	HCCB_3D_EVENT_DELTA_LATENT_PDE_RANKS=8 HCCB_3D_EVENT_DELTA_LATENT_PDE_ALPHAS=0.01 HCCB_3D_EVENT_DELTA_LATENT_PDE_GAMMAS=0.0,0.01,0.02,0.05,0.1 HCCB_3D_EVENT_DELTA_LATENT_PDE_BUDGETS=36 HCCB_3D_EVENT_DELTA_LATENT_PDE_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors HCCB_3D_EVENT_DELTA_LATENT_PDE_STEPS=18 HCCB_3D_EVENT_DELTA_LATENT_PDE_PDE_WEIGHT=1.0 HCCB_3D_EVENT_DELTA_LATENT_PDE_SENSOR_WEIGHT=0.1 HCCB_3D_EVENT_DELTA_LATENT_PDE_TRUST_WEIGHT=0.005 $(PYTHON) code/optimize_hccb_3d_event_delta_latent_pde_refiner.py

hccb-3d-transient-sparse-field-event-delta-constrained-latent-denoiser:
	HCCB_3D_EVENT_DELTA_CONSTRAINED_DENOISER_RANKS=8 HCCB_3D_EVENT_DELTA_CONSTRAINED_DENOISER_ALPHAS=0.01 HCCB_3D_EVENT_DELTA_CONSTRAINED_DENOISER_GAMMAS=0.0,0.01,0.02,0.05,0.1 HCCB_3D_EVENT_DELTA_CONSTRAINED_DENOISER_BUDGETS=36 HCCB_3D_EVENT_DELTA_CONSTRAINED_DENOISER_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors HCCB_3D_EVENT_DELTA_CONSTRAINED_DENOISER_EPOCHS=50 HCCB_3D_EVENT_DELTA_CONSTRAINED_DENOISER_PDE_WEIGHT=0.1 HCCB_3D_EVENT_DELTA_CONSTRAINED_DENOISER_SENSOR_WEIGHT=0.05 HCCB_3D_EVENT_DELTA_CONSTRAINED_DENOISER_TRUST_WEIGHT=0.02 $(PYTHON) code/train_hccb_3d_event_delta_constrained_latent_denoiser.py

hccb-3d-transient-sparse-field-event-delta-constrained-latent-denoiser-weight-sensitivity:
	PYTHON=$(PYTHON) HCCB_3D_EVENT_DELTA_CONSTRAINED_DENOISER_AUDIT_WEIGHTS=0.001,0.01,0.1 $(PYTHON) code/audit_hccb_3d_event_delta_constrained_denoiser_weight_sensitivity.py

hccb-3d-transient-sparse-field-event-delta-trust-region-pde-proposal:
	HCCB_3D_EVENT_DELTA_TRUST_PDE_RANKS=8 HCCB_3D_EVENT_DELTA_TRUST_PDE_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75 HCCB_3D_EVENT_DELTA_TRUST_PDE_STEPS=-1.0,-0.5,0.0,0.5,1.0 HCCB_3D_EVENT_DELTA_TRUST_PDE_TARGET_QUANTILES=0.25,0.5,0.75 HCCB_3D_EVENT_DELTA_TRUST_PDE_LAMBDAS=0.0,0.05,0.1,0.2,0.5,1.0 HCCB_3D_EVENT_DELTA_TRUST_PDE_BUDGETS=36 HCCB_3D_EVENT_DELTA_TRUST_PDE_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors $(PYTHON) code/optimize_hccb_3d_event_delta_trust_region_pde_proposal.py

hccb-3d-transient-sparse-field-event-delta-observation-aware-pde-proposal:
	HCCB_3D_EVENT_DELTA_OBS_PDE_PROPOSAL_RANKS=8 HCCB_3D_EVENT_DELTA_OBS_PDE_PROPOSAL_GAMMAS=0.0,0.02,0.05,0.1,0.2,0.35,0.5,0.75 HCCB_3D_EVENT_DELTA_OBS_PDE_PROPOSAL_BUDGETS=36 HCCB_3D_EVENT_DELTA_OBS_PDE_PROPOSAL_SENSOR_FAMILIES=default_sparse_sensors,train_residual_sensors,diverse_residual_sensors $(PYTHON) code/optimize_hccb_3d_event_delta_observation_aware_pde_proposal.py

hccb-3d-transient-sparse-field-event-delta-pareto-gate:
	$(PYTHON) code/build_event_delta_pareto_promotion_gate.py

hccb-3d-transient-sparse-field-event-delta-route-comparison:
	$(PYTHON) code/build_event_delta_sparse_inverse_route_comparison.py

apd003-apd004-sparse-inverse-promotion-gate:
	$(PYTHON) code/build_apd003_apd004_sparse_inverse_promotion_gate.py

structure-gaps:
	$(PYTHON) code/build_structure_parameter_gap_manifest.py

structure-activation:
	$(PYTHON) code/build_structure_input_activation_manifest.py

nonuniform-3d-structure-support:
	$(PYTHON) code/design_nonuniform_3d_structure_support.py

structure-support-gate:
	$(PYTHON) code/build_structure_support_evidence_gate.py

cfd-dem-velocity-support-gate:
	$(PYTHON) code/build_cfd_dem_velocity_support_gate.py

prepare-cfd-dem-velocity-package-templates:
	$(PYTHON) code/prepare_cfd_dem_velocity_package_templates.py

prepare-cfd-dem-velocity-execution-packet:
	$(PYTHON) code/prepare_cfd_dem_velocity_execution_packet.py

prepare-cfd-dem-velocity-pilot-case-skeletons:
	$(PYTHON) code/prepare_cfd_dem_velocity_pilot_case_skeletons.py

prepare-cfd-dem-velocity-remote-run-packet:
	$(PYTHON) code/prepare_cfd_dem_velocity_remote_run_packet.py

prepare-cfd-dem-velocity-solver-deck-templates:
	$(PYTHON) code/prepare_cfd_dem_velocity_solver_deck_templates.py

validate-cfd-dem-velocity-solver-case-decks:
	$(PYTHON) code/validate_cfd_dem_velocity_solver_case_decks.py

cfd-dem-velocity-solver-deck-completion-plan:
	$(PYTHON) code/build_cfd_dem_velocity_solver_deck_completion_plan.py

cfd-dem-velocity-postprocess-packaging-plan:
	$(PYTHON) code/build_cfd_dem_velocity_postprocess_packaging_plan.py

cfd-dem-velocity-remote-staging-audit:
	$(PYTHON) code/audit_cfd_dem_velocity_remote_staging.py

cfd-dem-velocity-package-staged-outputs:
	$(PYTHON) code/package_cfd_dem_velocity_staged_outputs.py

import-resolved-velocity-csv-package:
	$(PYTHON) code/import_resolved_velocity_csv_package.py

validate-cfd-dem-velocity-package:
	$(PYTHON) code/validate_cfd_dem_velocity_field_package.py

resolved-cfd-dem-velocity-operator-task:
	$(PYTHON) code/build_resolved_cfd_dem_velocity_operator_task.py

apd005-resolved-velocity-readiness-delta:
	$(PYTHON) code/build_apd005_resolved_velocity_readiness_delta.py

apd005-remote-packet-deploy:
	$(PYTHON) code/deploy_apd005_remote_packet.py

apd005-remote-openfoam13-smoke:
	$(PYTHON) code/run_apd005_remote_openfoam13_smoke.py

apd005-openfoam13-deck-preflight:
	$(PYTHON) code/build_apd005_openfoam13_deck_preflight.py

apd005-remote-openfoam13-deck-preflight:
	$(PYTHON) code/run_apd005_remote_openfoam13_deck_preflight.py

apd005-velocity-solver-parameter-gap-gate:
	$(PYTHON) code/build_apd005_velocity_solver_parameter_gap_gate.py

apd005-openfoam-solver-strategy-gate:
	$(PYTHON) code/build_apd005_openfoam_solver_strategy_gate.py

apd005-openfoam13-solver-deck-candidate:
	$(PYTHON) code/build_apd005_openfoam13_solver_deck_candidate.py

apd005-remote-openfoam13-solver-deck-candidate:
	$(PYTHON) code/run_apd005_remote_openfoam13_solver_deck_candidate.py

apd005-remote-openfoam13-pressure-drop-audit:
	$(PYTHON) code/audit_apd005_remote_openfoam13_pressure_drop.py

apd005-remote-openfoam13-tracer-proxy-smoke:
	$(PYTHON) code/run_apd005_remote_openfoam13_tracer_proxy_smoke.py

apd005-rtd-parameter-gap-gate:
	$(PYTHON) code/build_apd005_rtd_parameter_gap_gate.py

apd005-rtd-literature-candidate-audit:
	$(PYTHON) code/audit_apd005_rtd_literature_candidates.py

apd005-rtd-screening-parameter-table:
	$(PYTHON) code/build_apd005_rtd_screening_parameter_table.py

apd005-rtd-effective-dispersion-screening-table:
	$(PYTHON) code/build_apd005_rtd_effective_dispersion_screening_table.py

apd005-jpcrd-diffusion-reference-audit:
	$(PYTHON) code/audit_apd005_jpcrd_diffusion_reference.py

apd005-jpcrd-render-reference-pages:
	$(PYTHON) code/render_apd005_jpcrd_reference_pages.py

apd005-jpcrd-h2-he-reference-equation-audit:
	$(PYTHON) code/build_apd005_jpcrd_h2_he_reference_equation_audit.py

apd005-nist-tn2279-diffusion-candidate-audit:
	$(PYTHON) code/audit_apd005_nist_tn2279_diffusion_candidates.py

apd005-helium-hydrogen-isotope-source-audit:
	$(PYTHON) code/audit_apd005_helium_hydrogen_isotope_sources.py

apd005-amdur-1965-he-t2-he-th-abstract-equation-audit:
	$(PYTHON) code/audit_apd005_amdur_1965_he_t2_he_th_abstract_equation.py

apd005-amdur-1965-unit-assumption-audit:
	$(PYTHON) code/audit_apd005_amdur_1965_unit_assumption.py

apd005-primary-diffusion-import-gate:
	$(PYTHON) code/validate_apd005_primary_diffusion_import.py

apd005-primary-source-acquisition-gate:
	$(PYTHON) code/build_apd005_primary_source_acquisition_gate.py

apd005-online-source-access-audit:
	$(PYTHON) code/build_apd005_online_source_access_audit.py

apd005-jpcrd-h2-he-primary-import-candidate:
	$(PYTHON) code/build_apd005_jpcrd_h2_he_primary_import_candidate.py
	APD005_PRIMARY_DIFFUSION_INPUT=data/apd005_primary_diffusion_candidates_jpcrd_h2_he.csv \
	APD005_PRIMARY_DIFFUSION_OUT_DIR=cases/apd005_jpcrd_h2_he_primary_import_candidate_gate \
	$(PYTHON) code/validate_apd005_primary_diffusion_import.py

apd005-isotope-hto-mapping-gate:
	$(PYTHON) code/build_apd005_isotope_hto_mapping_gate.py

apd005-physical-rtd-training-gate:
	$(PYTHON) code/build_apd005_physical_rtd_training_gate.py

prepare-apd005-physical-rtd-package-templates:
	$(PYTHON) code/prepare_apd005_physical_rtd_package_templates.py

apd005-physical-rtd-data-contract:
	$(PYTHON) code/validate_apd005_physical_rtd_data_contract.py

apd005-resolved-rtd-unlock-gate:
	$(PYTHON) code/build_apd005_resolved_rtd_unlock_gate.py

apd006-module-source-audit:
	$(PYTHON) code/build_apd006_module_source_audit.py

apd006-aurora-repository-audit:
	$(PYTHON) code/build_apd006_aurora_repository_audit.py

apd006-remote-aurora-staging-audit:
	$(PYTHON) code/audit_apd006_remote_aurora_staging.py

apd006-remote-solver-stack-audit:
	$(PYTHON) code/build_apd006_remote_solver_stack_audit.py

apd006-hcpb-open-benchmark-candidate-audit:
	$(PYTHON) code/build_apd006_hcpb_open_benchmark_candidate_audit.py

.PHONY: apd006-iaea-hcpb-openmc-audit
apd006-iaea-hcpb-openmc-audit:
	$(PYTHON) code/build_apd006_iaea_hcpb_openmc_audit.py

.PHONY: apd006-hcpb-heating-components
apd006-hcpb-heating-components:
	$(PYTHON) code/build_apd006_hcpb_heating_components.py

prepare-apd006-module-benchmark-templates:
	$(PYTHON) code/prepare_apd006_module_benchmark_templates.py

apd006-module-benchmark-contract:
	$(PYTHON) code/validate_apd006_module_benchmark_contract.py

apd005-rtd-lightweight-gates:
	$(PYTHON) code/build_apd005_rtd_parameter_gap_gate.py
	$(PYTHON) code/audit_apd005_rtd_literature_candidates.py
	$(PYTHON) code/build_apd005_rtd_screening_parameter_table.py
	$(PYTHON) code/build_apd005_rtd_effective_dispersion_screening_table.py
	$(PYTHON) code/audit_apd005_jpcrd_diffusion_reference.py
	$(PYTHON) code/build_apd005_jpcrd_h2_he_reference_equation_audit.py
	$(PYTHON) code/audit_apd005_nist_tn2279_diffusion_candidates.py
	$(PYTHON) code/audit_apd005_helium_hydrogen_isotope_sources.py
	$(PYTHON) code/audit_apd005_amdur_1965_he_t2_he_th_abstract_equation.py
	$(PYTHON) code/audit_apd005_amdur_1965_unit_assumption.py
	$(PYTHON) code/build_apd005_primary_source_acquisition_gate.py
	$(PYTHON) code/build_apd005_online_source_access_audit.py
	$(PYTHON) code/validate_apd005_primary_diffusion_import.py
	$(PYTHON) code/build_apd005_physical_rtd_training_gate.py
	$(PYTHON) code/prepare_apd005_physical_rtd_package_templates.py
	$(PYTHON) code/validate_apd005_physical_rtd_data_contract.py

apd005-remote-openfoam13-tracer-screening-diffusion:
	$(PYTHON) code/run_apd005_remote_openfoam13_tracer_screening_diffusion.py

apd005-remote-openfoam13-tracer-effective-dispersion:
	APD005_TRACER_DIFFUSION_KIND=effective_dispersion $(PYTHON) code/run_apd005_remote_openfoam13_tracer_screening_diffusion.py

apd005-remote-openfoam13-tracer-history:
	$(PYTHON) code/postprocess_apd005_remote_openfoam13_tracer_history.py

apd005-inertial-resistance-literature-candidates:
	$(PYTHON) code/build_apd005_inertial_resistance_literature_candidates.py

hccb-3d-wall-porosity-stress:
	$(PYTHON) code/build_hccb_3d_wall_porosity_stress_task.py

hccb-3d-wall-porosity-baselines:
	$(PYTHON) code/train_hccb_3d_wall_porosity_baselines.py

hccb-3d-size-dispersion-stress:
	$(PYTHON) code/build_hccb_3d_size_dispersion_stress_task.py

hccb-3d-size-dispersion-baselines:
	$(PYTHON) code/train_hccb_3d_size_dispersion_baselines.py

hccb-3d-permeability-velocity-proxy:
	$(PYTHON) code/build_hccb_3d_permeability_velocity_proxy_task.py

hccb-3d-permeability-velocity-baselines:
	$(PYTHON) code/train_hccb_3d_permeability_velocity_proxy_baselines.py

hccb-3d-support-expanded-operator:
	HCCB_3D_SUPPORT_OPERATOR_EPOCHS=40 HCCB_3D_SUPPORT_OPERATOR_WIDTH=8 $(PYTHON) code/train_hccb_3d_support_expanded_operator.py

hccb-3d-support-residual-operator:
	HCCB_3D_SUPPORT_RESIDUAL_EPOCHS=60 HCCB_3D_SUPPORT_RESIDUAL_WIDTH=16 $(PYTHON) code/train_hccb_3d_support_residual_operator.py

hccb-3d-support-residual-observability:
	$(PYTHON) code/analyze_hccb_3d_support_residual_observability.py

hccb-3d-support-residual-token-task:
	HCCB_3D_SUPPORT_RESIDUAL_BASIS_RANK=8 $(PYTHON) code/build_hccb_3d_support_residual_token_task.py

hccb-3d-support-residual-transformer-posterior:
	HCCB_3D_SUPPORT_RESIDUAL_TRANSFORMER_EPOCHS=180 HCCB_3D_SUPPORT_RESIDUAL_TRANSFORMER_DMODEL=32 $(PYTHON) code/train_hccb_3d_support_residual_transformer_posterior.py

hccb-3d-support-residual-coefficient-diffusion:
	HCCB_3D_SUPPORT_COEFF_DIFF_EPOCHS=220 HCCB_3D_SUPPORT_COEFF_DIFF_SAMPLES=64 $(PYTHON) code/sample_hccb_3d_support_residual_coefficient_diffusion.py

hccb-3d-support-residual-coefficient-diffusion-calibration:
	HCCB_3D_SUPPORT_COEFF_DIFF_CAL_EPOCHS=220 HCCB_3D_SUPPORT_COEFF_DIFF_CAL_SAMPLES=96 $(PYTHON) code/calibrate_hccb_3d_support_residual_coefficient_diffusion.py

anisotropic-keff-digitize:
	$(PYTHON) code/digitize_anisotropic_thin_bed_keff.py

anisotropic-keff-reconcile:
	$(PYTHON) code/reconcile_fig10_directional_keff_with_table3.py

anisotropic-structure-task:
	$(PYTHON) code/build_anisotropic_thin_bed_operator_task.py

anisotropic-structure-operator:
	$(PYTHON) code/train_anisotropic_structure_operator.py

anisotropic-structure-comparison:
	$(PYTHON) code/compare_anisotropic_vs_isotropic_structure_tasks.py

structure-contrast-task:
	$(PYTHON) code/build_structure_contrast_operator_task.py

structure-contrast-operator:
	$(PYTHON) code/train_structure_contrast_operator.py

structure-contrast-boundary-operator:
	$(PYTHON) code/train_structure_contrast_boundary_operator.py

structure-contrast-support-design:
	$(PYTHON) code/design_structure_contrast_support_expansion.py

structure-contrast-augmented-task:
	$(PYTHON) code/build_structure_contrast_augmented_task.py

structure-contrast-supported-operator:
	$(PYTHON) code/train_structure_contrast_supported_operator.py

hccb-3d-porous-heat:
	$(PYTHON) code/build_hccb_3d_porous_heat_task.py

hccb-3d-porous-baselines:
	$(PYTHON) code/train_hccb_3d_porous_baselines.py

hccb-3d-porous-operator:
	$(PYTHON) code/train_hccb_3d_porous_operator.py

hccb-transport-closure:
	$(PYTHON) code/export_hccb_transport_closure_candidates.py

hccb-modified-closure-screen:
	$(PYTHON) code/screen_hccb_modified_transport_closure.py

hccb-modified-closure-dataset:
	$(PYTHON) code/export_hccb_modified_closure_operator_dataset.py

closure-aware-fno:
	CLOSURE_FNO_SPLIT=interleaved CLOSURE_FNO_RUN_ID=interleaved $(PYTHON) code/train_closure_aware_fno.py
	CLOSURE_FNO_SPLIT=leave_temperature_max CLOSURE_FNO_RUN_ID=leave_temperature_max $(PYTHON) code/train_closure_aware_fno.py
	CLOSURE_FNO_SPLIT=leave_velocity_max CLOSURE_FNO_RUN_ID=leave_velocity_max $(PYTHON) code/train_closure_aware_fno.py
	CLOSURE_FNO_SPLIT=leave_closure_hccb_modified CLOSURE_FNO_RUN_ID=leave_closure_hccb_modified $(PYTHON) code/train_closure_aware_fno.py
	$(PYTHON) code/summarize_closure_aware_fno.py

closure-aware-fno-keff:
	CLOSURE_KEFF_FNO_SPLIT=interleaved CLOSURE_KEFF_FNO_RUN_ID=interleaved $(PYTHON) code/train_closure_aware_fno_keff.py
	CLOSURE_KEFF_FNO_SPLIT=leave_temperature_max CLOSURE_KEFF_FNO_RUN_ID=leave_temperature_max $(PYTHON) code/train_closure_aware_fno_keff.py
	CLOSURE_KEFF_FNO_SPLIT=leave_velocity_max CLOSURE_KEFF_FNO_RUN_ID=leave_velocity_max $(PYTHON) code/train_closure_aware_fno_keff.py
	CLOSURE_KEFF_FNO_SPLIT=leave_closure_hccb_modified CLOSURE_KEFF_FNO_RUN_ID=leave_closure_hccb_modified $(PYTHON) code/train_closure_aware_fno_keff.py
	$(PYTHON) code/summarize_closure_aware_fno_keff.py

closure-aware-pino:
	CLOSURE_PINO_SPLIT=interleaved CLOSURE_PINO_RUN_ID=interleaved $(PYTHON) code/train_closure_aware_pino.py
	CLOSURE_PINO_SPLIT=leave_temperature_max CLOSURE_PINO_RUN_ID=leave_temperature_max $(PYTHON) code/train_closure_aware_pino.py
	CLOSURE_PINO_SPLIT=leave_velocity_max CLOSURE_PINO_RUN_ID=leave_velocity_max $(PYTHON) code/train_closure_aware_pino.py
	CLOSURE_PINO_SPLIT=leave_closure_hccb_modified CLOSURE_PINO_RUN_ID=leave_closure_hccb_modified $(PYTHON) code/train_closure_aware_pino.py
	$(PYTHON) code/summarize_closure_aware_pino.py

closure-aware-pino-sweep:
	$(PYTHON) code/run_closure_pino_weight_sweep.py

closure-specific-fno:
	CLOSURE_SPECIFIC_FNO_SPLIT=interleaved CLOSURE_SPECIFIC_FNO_RUN_ID=interleaved $(PYTHON) code/train_closure_specific_fno.py
	CLOSURE_SPECIFIC_FNO_SPLIT=leave_temperature_max CLOSURE_SPECIFIC_FNO_RUN_ID=leave_temperature_max $(PYTHON) code/train_closure_specific_fno.py
	CLOSURE_SPECIFIC_FNO_SPLIT=leave_velocity_max CLOSURE_SPECIFIC_FNO_RUN_ID=leave_velocity_max $(PYTHON) code/train_closure_specific_fno.py
	CLOSURE_SPECIFIC_FNO_SPLIT=leave_closure_hccb_modified CLOSURE_SPECIFIC_FNO_RUN_ID=leave_closure_hccb_modified $(PYTHON) code/train_closure_specific_fno.py
	$(PYTHON) code/summarize_closure_specific_fno.py

closure-specific-pino:
	CLOSURE_SPECIFIC_PINO_SPLIT=interleaved CLOSURE_SPECIFIC_PINO_RUN_ID=interleaved $(PYTHON) code/train_closure_specific_pino.py
	CLOSURE_SPECIFIC_PINO_SPLIT=leave_temperature_max CLOSURE_SPECIFIC_PINO_RUN_ID=leave_temperature_max $(PYTHON) code/train_closure_specific_pino.py
	CLOSURE_SPECIFIC_PINO_SPLIT=leave_velocity_max CLOSURE_SPECIFIC_PINO_RUN_ID=leave_velocity_max $(PYTHON) code/train_closure_specific_pino.py
	CLOSURE_SPECIFIC_PINO_SPLIT=leave_closure_hccb_modified CLOSURE_SPECIFIC_PINO_RUN_ID=leave_closure_hccb_modified $(PYTHON) code/train_closure_specific_pino.py
	$(PYTHON) code/summarize_closure_specific_pino.py

closure-physics-gated-moe:
	$(PYTHON) code/run_closure_physics_gated_moe.py

closure-ood-rejected-moe:
	$(PYTHON) code/run_closure_ood_rejected_moe.py

closure-support-aware-moe:
	$(PYTHON) code/run_closure_support_aware_moe.py

closure-support-coverage-design:
	$(PYTHON) code/design_closure_support_coverage.py

coverage-calibrated-closure:
	$(PYTHON) code/train_coverage_calibrated_closure_fno.py
	$(PYTHON) code/train_coverage_calibrated_closure_baselines.py

coverage-calibrated-closure-pino:
	$(PYTHON) code/train_coverage_calibrated_closure_pino.py

coverage-calibrated-boundary-satisfying:
	$(PYTHON) code/train_coverage_calibrated_boundary_satisfying_operator.py

coverage-calibrated-closure-comparison:
	$(PYTHON) code/summarize_coverage_calibrated_closure.py

coverage-error-localization:
	$(PYTHON) code/analyze_coverage_calibrated_error_localization.py

coverage-boundary-projection:
	$(PYTHON) code/apply_coverage_calibrated_boundary_projection.py

closure-operator-baselines:
	$(PYTHON) code/train_closure_operator_baselines.py

closure-operator-comparison:
	$(PYTHON) code/summarize_closure_operator_models.py

architecture-gate:
	$(PYTHON) code/build_architecture_evidence_gate.py

research-route-gate:
	$(PYTHON) code/build_research_route_decision_gate.py

hybrid-architecture-route-gate:
	$(PYTHON) code/build_hybrid_architecture_route_gate.py

architecture-parameter-data-gate:
	$(PYTHON) code/build_architecture_parameter_data_gate.py

external-algorithm-source-gate:
	$(PYTHON) code/build_external_algorithm_source_gate.py

external-framework-smoke-gate:
	$(PYTHON) code/build_external_framework_smoke_gate.py

official-neuraloperator-fno-smoke:
	$(PYTHON) code/run_official_neuraloperator_fno_smoke.py

reproducible-experiment-matrix:
	$(PYTHON) code/build_reproducible_experiment_matrix.py

reduced-forward-timing:
	$(PYTHON) code/time_reduced_forward_models.py

reduced-forward-efficiency-gate:
	$(PYTHON) code/build_reduced_forward_efficiency_gate.py

training-cost-coverage-gate:
	$(PYTHON) code/build_training_cost_coverage_gate.py

efficiency-timing-protocol:
	$(PYTHON) code/build_efficiency_timing_protocol.py

structure-task:
	$(PYTHON) code/build_structure_aware_thin_bed_task.py

structure-deeponet:
	$(PYTHON) code/train_structure_aware_deeponet.py

operator-readiness:
	$(PYTHON) code/check_operator_model_readiness.py

apd006-tdem-training-admission-gate:
	$(PYTHON) code/build_apd006_tdem_training_admission_gate.py

apd006-tdem-graph-operator-pilot-dataset:
	$(PYTHON) code/build_tdem_graph_operator_pilot_dataset.py
	$(PYTHON) code/validate_tdem_graph_operator_pilot_dataset.py

apd006-tdem-graph-scalar-baselines:
	$(PYTHON) code/train_tdem_graph_pilot_scalar_baselines.py

apd006-tdem-graph-direct-solver-benchmark:
	$(PYTHON) code/benchmark_tdem_graph_direct_solver.py --holdout base --repeats 3
	$(PYTHON) code/benchmark_tdem_graph_direct_solver.py --holdout omit1 --repeats 3
	$(PYTHON) code/benchmark_tdem_graph_direct_solver.py --holdout omit2 --repeats 3

apd006-tdem-graph-solver-value-gate:
	$(PYTHON) code/summarize_tdem_graph_solver_value_gate.py

apd006-tdem-transient-graph-smoke:
	$(PYTHON) code/validate_tdem_transient_pair_vectorization.py
	$(PYTHON) code/build_tdem_transient_graph_dataset.py --output-id smoke --realizations base --max-states 1 --pressures 0.105 --protocols hot_wall_step
	$(PYTHON) code/validate_tdem_transient_graph_dataset.py --output-id smoke
	$(PYTHON) code/evaluate_tdem_transient_1d_analytical_baseline.py --output-id smoke

apd006-tdem-transient-graph-full:
	$(PYTHON) code/build_tdem_transient_graph_dataset.py --output-id full --workers 1
	$(PYTHON) code/validate_tdem_transient_graph_dataset.py --output-id full
	$(PYTHON) code/evaluate_tdem_transient_1d_analytical_baseline.py --output-id full

apd006-tdem-temporal-graph-transformer-physics-gate:
	$(PYTHON) code/validate_tdem_temporal_graph_transformer_physics.py

apd006-tdem-temporal-graph-transformer-base:
	$(PYTHON) code/train_tdem_temporal_graph_transformer_pino.py --holdout base --run-id temporal_graph_transformer_pino_v1

apd006-tdem-official-architecture-config-audit:
	$(PYTHON) code/audit_official_architecture_configs.py

apd006-tdem-temporal-graph-transformer-formal-base:
	$(PYTHON) code/train_tdem_temporal_graph_transformer_pino.py --holdout base --run-id temporal_graph_transformer_pino_formal_sourcebacked_cycle_v1 --seed 20260713 --hidden 64 --graph-layers 3 --transformer-layers 3 --heads 1 --temporal-mixer transformer --epochs 100 --learning-rate 0.001 --weight-decay 0.00005 --lr-method cycle --grad-clip 1000 --data-weight 5 --flux-weight 1 --physics-weight 1 --validation-every 5 --threads 16 --device auto

apd006-tdem-transient-direct-solver-clean:
	$(PYTHON) code/benchmark_tdem_transient_direct_solver.py --realization base --trajectories 1 --workers 1 --run-id base_serial_clean

apd006-tdem-temporal-architecture-gate:
	$(PYTHON) code/summarize_tdem_temporal_architecture_gate.py

apd006-tdem-temporal-sourcebacked-gate:
	$(PYTHON) code/summarize_tdem_temporal_sourcebacked_gate.py

apd006-tdem-sourcebacked-threefold-gate:
	$(PYTHON) code/repair_tdem_sourcebacked_provenance_metadata.py
	$(PYTHON) code/summarize_tdem_sourcebacked_threefold.py --require-complete

apd006-tdem-temporal-operator-figure:
	$(PYTHON) code/plot_tdem_temporal_operator_evidence.py

apd006-tdem-numerical-design-provenance:
	$(PYTHON) code/audit_tdem_numerical_design_provenance.py

apd006-tdem-architecture-comparison-figure:
	$(PYTHON) code/plot_tdem_architecture_comparison.py

apd006-tdem-mgnt-route-gate:
	$(PYTHON) code/audit_tdem_mgnt_route.py

apd006-graph-physics-candidate-audit:
	$(PYTHON) code/audit_graph_physics_candidate.py

apd006-tdem-mgnt-implementation-gate:
	$(PYTHON) code/validate_tdem_mgnt_temporal_pino_implementation.py
	$(PYTHON) code/audit_tdem_mgnt_route.py

apd006-tdem-mgnt-formal-base:
	$(PYTHON) code/audit_tdem_mgnt_route.py
	$(PYTHON) code/train_tdem_mgnt_temporal_pino.py --holdout base --run-id mgnt_temporal_pino_sourcebacked_v1 --device auto
	$(PYTHON) code/evaluate_tdem_mgnt_temporal_pino_detailed.py --holdout base --run-id mgnt_temporal_pino_sourcebacked_v1 --device auto
	$(PYTHON) code/audit_tdem_mgnt_comparison_contract.py

apd006-tdem-transolver-attention-port-gate:
	$(PYTHON) code/validate_transolver_physics_attention_port.py

apd006-tdem-transolver-pino-implementation-gate:
	$(PYTHON) code/validate_tdem_transolver_pino_implementation.py

apd006-tdem-transolver-pino-formal-base:
	$(PYTHON) code/train_tdem_transolver_pino.py --holdout base --run-id transolver_pino_sourcebacked_v1
	$(PYTHON) code/evaluate_tdem_transolver_pino_detailed.py --holdout base --run-id transolver_pino_sourcebacked_v1 --device auto
	$(PYTHON) code/audit_tdem_transolver_comparison_contract.py

apd006-tdem-diffusion-route-gate:
	$(PYTHON) code/audit_tdem_diffusion_route_readiness.py

apd006-sparse-thermal-observation-source-audit:
	$(PYTHON) code/audit_sparse_thermal_observation_sources.py

apd006-flow-coupled-heat-route-audit:
	$(PYTHON) code/audit_cheng2024_pressure_drop_points.py
	$(PYTHON) code/reproduce_hccb_flow_energy_reference.py
	$(PYTHON) code/reproduce_li2tio3_pressure_drop_correlation.py
	$(PYTHON) code/audit_li2tio3_flowing_helium_aggregate_gate.py
	$(PYTHON) code/audit_flow_coupled_heat_route.py

.PHONY: apd006-li2tio3-flowing-helium-aggregate-gate
apd006-li2tio3-flowing-helium-aggregate-gate:
	$(PYTHON) code/audit_li2tio3_flowing_helium_aggregate_gate.py

.PHONY: apd006-li2tio3-internal-heating-source-audit
apd006-li2tio3-internal-heating-source-audit:
	$(PYTHON) code/audit_li2tio3_internal_heating_sources.py

.PHONY: apd006-internal-heat-low-re-pore-scale-boundary
apd006-internal-heat-low-re-pore-scale-boundary:
	$(PYTHON) code/audit_internal_heat_low_re_pore_scale_boundary.py

.PHONY: apd006-hccb-variable-property-ablation
apd006-hccb-variable-property-ablation:
	$(PYTHON) code/analyze_hccb_variable_property_ablation.py

apd006-tdem-flow-solid-coupling-smoke:
	$(PYTHON) code/build_tdem_flow_solid_coupled_dataset.py --output-id smoke --matrix smoke --gas-bins 16 --cfl 0.4 --maximum-fraction 0.02

apd006-tdem-flow-solid-coupling-convergence:
	$(PYTHON) code/build_tdem_flow_solid_coupled_dataset.py --output-id reference_Nz16_cfl04 --matrix smoke --gas-bins 16 --cfl 0.4 --maximum-fraction 2.0
	$(PYTHON) code/build_tdem_flow_solid_coupled_dataset.py --output-id reference_Nz32_cfl04 --matrix smoke --gas-bins 32 --cfl 0.4 --maximum-fraction 2.0
	$(PYTHON) code/build_tdem_flow_solid_coupled_dataset.py --output-id reference_Nz16_cfl02 --matrix smoke --gas-bins 16 --cfl 0.2 --maximum-fraction 2.0
	$(PYTHON) code/audit_tdem_flow_solid_coupling.py

apd006-tdem-interphase-closure-sensitivity:
	$(PYTHON) code/analyze_tdem_interphase_closure_sensitivity.py
	$(PYTHON) code/audit_tdem_flow_solid_model_form_gate.py

.PHONY: apd006-hccb-pore-resolved-cht-route
apd006-hccb-pore-resolved-cht-route:
	$(PYTHON) code/audit_remote_openfoam13_cht_capability.py
	$(PYTHON) code/audit_hccb_pore_resolved_cht_rve_route.py
	$(PYTHON) code/build_hccb_openfoam_helium_property_table.py
	$(PYTHON) code/audit_hccb_openfoam_thermo_smoke.py --log results/apd006_hccb_openfoam_helium_property_table/thermo_smoke_passed.log --table-summary results/apd006_hccb_openfoam_helium_property_table/summary.json --output results/apd006_hccb_openfoam_helium_property_table/thermo_smoke_summary.json

HCCB_PACKING ?= data/apd006_hccb_openmc_packing/seed101_ipf035_cr1e5/packing.npz
HCCB_OPENMC_OUT ?= data/apd006_hccb_openmc_packing/seed101_ipf035_cr1e5
HCCB_PACKING_AUDIT ?= $(HCCB_OPENMC_OUT)/independent_audit.json
HCCB_MESH_CASE ?= cases/apd006_hccb_pore_resolved_cht/seed101_smoke
HCCB_P405_MESH_LEVEL ?= G2
HCCB_P405_MESH_CASE ?= cases/apd006_hccb_pore_resolved_cht/seed101_G2
HCCB_ACTIVE_CHT_CASE ?= $(HCCB_P405_MESH_CASE)
HCCB_PACKING_SEED ?= 101
HCCB_CHT_CASE_ID ?= seed101_T0700_u0p20
HCCB_SOLID_REGION ?= solid
HCCB_P418_MESH_CASE ?= runs/hccb_dense_cht_native_r2
HCCB_P418_MESH_MANIFEST ?= runs/hccb_dense_snappy_g2_nativezone_r2/case_manifest.json
HCCB_P418_SINGLE_CASE_ROOT ?= runs/hccb_dense_cht_p418_single
HCCB_P418_CONDITION_ID ?= u0p20_T700_q6p85
PACKMOL ?= packmol
HCCB_PACKMOL_OUT ?= data/apd006_hccb_packmol_packing/seed101

.PHONY: apd006-hccb-packmol-packing
apd006-hccb-packmol-packing:
	$(PYTHON) code/generate_hccb_packmol_packing.py --packmol $(PACKMOL) --seed 101 --output-dir $(HCCB_PACKMOL_OUT)

.PHONY: apd006-hccb-packmol-candidate-audit
apd006-hccb-packmol-candidate-audit:
	$(PYTHON) code/audit_hccb_packmol_candidate.py --candidate-dir $(HCCB_PACKMOL_OUT) --output $(HCCB_PACKMOL_OUT)/independent_audit.json

.PHONY: apd006-hccb-openmc-candidate-audit
apd006-hccb-openmc-candidate-audit:
	$(PYTHON) code/audit_hccb_openmc_candidate.py --candidate-dir $(HCCB_OPENMC_OUT) --output $(HCCB_OPENMC_OUT)/independent_audit.json --expect-seed $(HCCB_PACKING_SEED) --expect-initial-packing-fraction 0.35 --expect-contraction-rate 1e-5

.PHONY: apd006-hccb-openmc-watcher-contract
apd006-hccb-openmc-watcher-contract:
	$(PYTHON) code/audit_hccb_openmc_watcher_contract.py

.PHONY: apd006-hccb-pore-resolved-mesh-build
apd006-hccb-pore-resolved-mesh-build:
	$(PYTHON) code/build_hccb_pore_resolved_openfoam_mesh.py --packing $(HCCB_PACKING) --packing-audit $(HCCB_PACKING_AUDIT) --output-dir $(HCCB_MESH_CASE) --mesh-level smoke --cells-per-diameter 2 --sphere-subdivisions 1 --surface-refinement 1

.PHONY: apd006-hccb-pore-resolved-mesh-audit
apd006-hccb-pore-resolved-mesh-audit:
	$(PYTHON) code/audit_hccb_pore_resolved_openfoam_mesh.py --case $(HCCB_MESH_CASE) --run-mesh

.PHONY: apd006-hccb-p405-mesh-resolution-contract
apd006-hccb-p405-mesh-resolution-contract:
	$(PYTHON) code/audit_hccb_icosphere_geometry_resolution.py
	$(PYTHON) code/audit_hccb_p405_mesh_resolution_contract.py

.PHONY: apd006-hccb-p405-mesh-build-and-audit
apd006-hccb-p405-mesh-build-and-audit:
	$(PYTHON) code/build_hccb_pore_resolved_openfoam_mesh.py --packing $(HCCB_PACKING) --packing-audit $(HCCB_PACKING_AUDIT) --output-dir $(HCCB_P405_MESH_CASE) --mesh-level $(HCCB_P405_MESH_LEVEL)
	$(PYTHON) code/audit_hccb_pore_resolved_openfoam_mesh.py --case $(HCCB_P405_MESH_CASE) --run-mesh

.PHONY: apd006-hccb-steady-builder-contract
apd006-hccb-steady-builder-contract:
	$(PYTHON) code/audit_hccb_pore_resolved_openfoam_steady_builder_contract.py
	$(PYTHON) code/audit_openfoam13_cht_raw_field_write.py
	$(PYTHON) code/audit_hccb_pore_resolved_raw_field_export_contract.py
	$(PYTHON) code/audit_openfoam13_multiregion_interface_pairs.py

.PHONY: apd006-hccb-pore-resolved-steady-case
apd006-hccb-pore-resolved-steady-case:
	$(PYTHON) code/build_hccb_dense_cht_p418_matrix.py --mesh-case $(HCCB_P418_MESH_CASE) --mesh-manifest $(HCCB_P418_MESH_MANIFEST) --output-root $(HCCB_P418_SINGLE_CASE_ROOT) --mode selected --condition-id $(HCCB_P418_CONDITION_ID)

.PHONY: apd006-hccb-pore-resolved-steady-result-audit
apd006-hccb-pore-resolved-steady-result-audit:
	$(PYTHON) code/audit_hccb_pore_resolved_openfoam_steady_result.py --case $(HCCB_ACTIVE_CHT_CASE)

.PHONY: apd006-hccb-pore-resolved-raw-field-register
apd006-hccb-pore-resolved-raw-field-register:
	$(PYTHON) code/export_openfoam_multiregion_interface_pairs.py --case $(HCCB_ACTIVE_CHT_CASE) --fluid-region fluid --solid-region $(HCCB_SOLID_REGION) --patch-types mappedWall --output-dir $(HCCB_ACTIVE_CHT_CASE)/interface_pairs
	$(PYTHON) code/export_hccb_pore_resolved_cht_raw_fields.py --case $(HCCB_ACTIVE_CHT_CASE) --run-postprocess --openfoam-bashrc /opt/openfoam13/etc/bashrc --interface-summary $(HCCB_ACTIVE_CHT_CASE)/interface_pairs/summary.json

.PHONY: apd006-hccb-p391-reference
apd006-hccb-p391-reference:
	$(PYTHON) code/build_hccb_pore_resolved_openfoam_steady_case.py --case $(HCCB_ACTIVE_CHT_CASE)
	bash -lc 'source /opt/openfoam13/etc/bashrc >/dev/null 2>&1; cd "$(CURDIR)/$(HCCB_ACTIVE_CHT_CASE)"; ./Allrun.steady'
	$(PYTHON) code/audit_hccb_pore_resolved_openfoam_steady_result.py --case $(HCCB_ACTIVE_CHT_CASE)
	$(PYTHON) code/export_openfoam_multiregion_interface_pairs.py --case $(HCCB_ACTIVE_CHT_CASE) --fluid-region fluid --solid-region $(HCCB_SOLID_REGION) --patch-types mappedWall --output-dir $(HCCB_ACTIVE_CHT_CASE)/interface_pairs
	$(PYTHON) code/export_hccb_pore_resolved_cht_raw_fields.py --case $(HCCB_ACTIVE_CHT_CASE) --run-postprocess --openfoam-bashrc /opt/openfoam13/etc/bashrc --interface-summary $(HCCB_ACTIVE_CHT_CASE)/interface_pairs/summary.json

.PHONY: apd006-hccb-pore-resolved-dataset-contract
apd006-hccb-pore-resolved-dataset-contract:
	$(PYTHON) code/audit_hccb_pore_resolved_cht_dataset_contract.py
	$(PYTHON) code/audit_hccb_pore_resolved_ml_tensor_contract.py

.PHONY: apd006-hccb-pore-resolved-case-matrix
apd006-hccb-pore-resolved-case-matrix:
	$(PYTHON) code/build_hccb_pore_resolved_cht_case_matrix.py
	$(PYTHON) code/audit_hccb_pore_resolved_cht_case_matrix.py

.PHONY: apd006-hccb-published-pinn-baseline-gate
apd006-hccb-published-pinn-baseline-gate:
	$(PYTHON) code/audit_hccb_published_pinn_baseline.py

apd006-tdem-flow-solid-coupling-full:
	$(PYTHON) code/build_tdem_flow_solid_coupled_dataset.py --output-id full_matrix_Nz16_cfl04 --matrix full --gas-bins 16 --cfl 0.4 --maximum-fraction 2.0

apd006-tdem-evidence-manifest:
	$(PYTHON) code/build_tdem_external_evidence_manifest.py
.PHONY: apd006-rigno-source-audit
apd006-rigno-source-audit:
	python3 code/audit_rigno_algorithm_candidate.py

.PHONY: apd006-hccb-multiregion-native-graph-contract
apd006-hccb-multiregion-native-graph-contract:
	python3 code/audit_openfoam13_multiregion_native_graph.py

.PHONY: apd006-hccb-multiregion-boundary-face-contract
apd006-hccb-multiregion-boundary-face-contract:
	python3 code/audit_openfoam13_multiregion_boundary_faces.py

.PHONY: apd006-hccb-multiregion-boundary-condition-contract
apd006-hccb-multiregion-boundary-condition-contract:
	python3 code/audit_openfoam13_multiregion_boundary_conditions.py

.PHONY: apd006-hccb-multiregion-fv-balance-smoke
apd006-hccb-multiregion-fv-balance-smoke:
	$(TORCH_PYTHON) code/smoke_multiregion_finite_volume_balance.py

.PHONY: apd006-hccb-openfoam13-face-flux-smoke
apd006-hccb-openfoam13-face-flux-smoke:
	$(TORCH_PYTHON) code/smoke_openfoam13_face_flux_reconstruction.py

.PHONY: apd006-openfoam13-ascii-field-smoke
apd006-openfoam13-ascii-field-smoke:
	$(TORCH_PYTHON) code/smoke_openfoam_ascii_field.py

.PHONY: apd006-openfoam13-solved-mass-flux-comparison
apd006-openfoam13-solved-mass-flux-comparison:
	$(TORCH_PYTHON) code/compare_openfoam13_solved_mass_flux.py

.PHONY: apd006-openfoam13-solved-heat-flux-comparison
apd006-openfoam13-solved-heat-flux-comparison:
	$(TORCH_PYTHON) code/compare_openfoam13_solved_heat_flux.py

.PHONY: apd006-hccb-source-backed-thermophysical-smoke
apd006-hccb-source-backed-thermophysical-smoke:
	$(TORCH_PYTHON) code/smoke_hccb_source_backed_thermophysical.py

.PHONY: apd006-hccb-multiregion-steady-cht-residual-smoke
apd006-hccb-multiregion-steady-cht-residual-smoke:
	$(TORCH_PYTHON) code/smoke_hccb_multiregion_steady_cht_residual.py

.PHONY: apd006-hccb-multiregion-steady-cht-native-mesh-smoke
apd006-hccb-multiregion-steady-cht-native-mesh-smoke:
	$(TORCH_PYTHON) code/smoke_hccb_multiregion_steady_cht_native_mesh.py

.PHONY: apd006-hccb-steady-momentum-residual-smoke
apd006-hccb-steady-momentum-residual-smoke:
	$(TORCH_PYTHON) code/smoke_hccb_steady_momentum_residual.py

.PHONY: apd006-hccb-conservative-all-equation-smoke
apd006-hccb-conservative-all-equation-smoke:
	$(TORCH_PYTHON) code/smoke_hccb_conservative_physics_residual.py

.PHONY: apd006-hccb-conservative-solved-phi-check
apd006-hccb-conservative-solved-phi-check:
	$(TORCH_PYTHON) code/check_hccb_conservative_physics_solved_phi.py

.PHONY: apd006-hccb-steady-momentum-native-mesh-smoke
apd006-hccb-steady-momentum-native-mesh-smoke:
	$(TORCH_PYTHON) code/smoke_hccb_steady_momentum_native_mesh.py

apd006-hccb-multiregion-regional-coarsening-contract:
	python3 code/audit_openfoam13_multiregion_regional_coarsening.py

apd006-hccb-multiregion-regional-hierarchy-contract:
	python3 code/audit_openfoam13_multiregion_regional_hierarchy.py

.PHONY: apd006-hccb-multiregion-hierarchy-geometry-contract
apd006-hccb-multiregion-hierarchy-geometry-contract:
	python3 code/audit_openfoam13_multiregion_hierarchy_geometry.py

.PHONY: apd006-hccb-multiregion-p2r-r2p-support-contract
apd006-hccb-multiregion-p2r-r2p-support-contract:
	python3 code/audit_openfoam13_multiregion_p2r_r2p_support.py

.PHONY: apd006-hccb-rigno-target-sensitivity-contract
apd006-hccb-rigno-target-sensitivity-contract:
	python3 code/audit_rigno_target_sensitivity_contract.py

.PHONY: apd006-hccb-multiregion-regional-operator-smoke
apd006-hccb-multiregion-regional-operator-smoke:
	$(TORCH_PYTHON) code/smoke_hccb_multiregion_regional_operator.py

.PHONY: apd006-hccb-conservative-face-flux-operator-smoke
apd006-hccb-conservative-face-flux-operator-smoke:
	$(TORCH_PYTHON) code/smoke_hccb_conservative_face_flux_operator.py

.PHONY: apd006-hccb-compare-packing-realizations
apd006-hccb-compare-packing-realizations:
	$(PYTHON) code/compare_hccb_packing_realizations.py \
		--result data/apd006_hccb_openmc_packing/seed101_ipf035_cr1e5/independent_audit.json \
		--result data/apd006_hccb_openmc_packing/seed202_ipf035_cr1e5/independent_audit.json \
		--result data/apd006_hccb_openmc_packing/seed303_ipf035_cr1e5/independent_audit.json \
		--output-dir results/apd006_hccb_pore_resolved_cht/three_packing_comparison
