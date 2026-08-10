from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_defines_resolved_and_unresolved_heat_paths() -> None:
    text = " ".join(
        (ROOT / "manuscript/main.tex").read_text(encoding="utf-8").split()
    )
    assert "Contact conduction under stress and rarefied gas transport" in text
    assert "lie outside" in text
    assert "the present model" in text
    assert "Nor does it replace" in text
    assert "thermo-mechanical" in text
    assert "analyses of" in text
    assert "moscardini2018tdem" in text
    assert "peeketi2018compacted" in text
    assert "desu2020tdem" in text


def test_heat_mechanism_references_are_source_identified() -> None:
    references = (ROOT / "manuscript/references.bib").read_text(encoding="utf-8")
    assert "10.1016/j.fusengdes.2018.01.013" in references
    assert "10.1016/j.fusengdes.2018.02.088" in references
    assert "10.1016/j.fusengdes.2020.111767" in references

    manifest = (ROOT / "parameters/literature_parameter_manifest.csv").read_text(
        encoding="utf-8"
    )
    assert "tdem_contact_pair_cutoff_fraction_epsilon" in manifest
    assert "desu2020_helium_kinetic_molecular_diameter" in manifest
