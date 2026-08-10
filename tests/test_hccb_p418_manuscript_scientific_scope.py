from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_manuscript_sources() -> str:
    files = (
        ROOT / "manuscript/main.tex",
        ROOT / "manuscript/methods_condensed.tex",
        ROOT / "manuscript/results_condensed.tex",
        ROOT / "manuscript/generated_scope_limits.tex",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    return re.sub(r"\s+", " ", text)


def test_steady_iteration_is_not_described_as_200_physical_seconds() -> None:
    text = read_manuscript_sources()
    assert "The steady labels 1--200 are nonlinear iterations, not physical seconds" in text
    assert "200th steady nonlinear" in text
    assert re.search(r"\b200\s*(?:s|seconds)\b", text, flags=re.IGNORECASE) is None


def test_main_and_independent_packings_keep_distinct_sample_roles() -> None:
    text = read_manuscript_sources()
    assert "database contains 60 steady" in text
    assert "second independently generated spherical-pebble arrangement" in text
    assert "nine matched conditions" in text
    assert re.search(r"\b69\s+(?:steady|conditions?|cases?|samples?)\b", text, re.I) is None


def test_crushable_dem_reference_defines_the_intact_sphere_scope() -> None:
    main = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    bibliography = (ROOT / "manuscript/references.bib").read_text(encoding="utf-8")
    verification = (
        ROOT / "literature/P418_INTRODUCTION_REFERENCE_VERIFICATION_CN.md"
    ).read_text(encoding="utf-8")
    assert main.count(r"\cite{wang2026crushable}") >= 2
    assert "Crushing evolution in pebble bed" in bibliography
    assert "10.1007/s41365-025-01806-0" in bibliography
    assert "10.1007/s41365-025-01806-0" in verification
    assert "不把破碎模型参数或碎片分布加入当前OpenFOAM算例" in verification


def test_transient_claims_remain_fixed_hydrodynamic() -> None:
    text = read_manuscript_sources()
    assert "12 fixed-hydrodynamics thermal-step trajectories" in text
    assert "not the initial momentum or pressure-wave transient" in text
    assert "Model claims therefore concern thermal evolution with a prescribed hydrodynamic field" in text
    assert "does not by itself establish a final model ranking" in text


def test_transient_energy_equations_and_fixed_flow_scope_are_explicit() -> None:
    text = read_manuscript_sources()
    assert r"\frac{\partial T_f}{\partial t}" in text
    assert r"\frac{\partial T_s}{\partial t}" in text
    assert "The steady calculations set both time derivatives to zero" in text
    assert (
        "the converged target velocity and pressure are held constant while the two "
        "energy equations remain time dependent"
    ) in text


def test_frozen_pinn_chain_is_distinguished_from_transient_training() -> None:
    text = read_manuscript_sources()
    assert "at inference, a frozen steady coordinate PINN supplies" in text
    assert "Physics-constrained variants incorporate finite-volume" in text
    assert "all models are evaluated using the same physical quantities" in text
    assert "The models use the same finite-volume" not in text
    assert "Training uses the exact \\OpenFOAM{} source state" in text
    assert "The strict end-to-end test replaces it with the frozen PINN prediction" in text
    assert "neither network is refitted" in text


def test_field_cloud_is_identified_as_a_regionalized_section() -> None:
    text = read_manuscript_sources()
    assert "Regionalized OpenFOAM--model temperatures" in text
    assert "Section interpolation is visual only" in text
    assert "metrics use the original regional nodes" in text


def test_domain_figure_uses_seed101_packing_and_local_crop() -> None:
    script = (ROOT / "code/plot_hccb_p418_physical_model_domain.py").read_text(
        encoding="utf-8"
    )
    manuscript = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")

    assert "seed101_s80_xlo_ycentre/packing.npz" in script
    assert 'len(source["centres_m"]) != 2039' in script
    assert "2039-pebble seed101 parent packing" in manuscript
    assert "The resolved 125-pebble crop" in manuscript
    assert "1799-pebble" not in manuscript
    assert '"diffusion refiner\\n3 $T$-residual steps"' in script
    assert '"diffusion_temperature_refinement_steps": 3' in script
    assert "the optional diffusion module refines only its temperature residual" in manuscript
