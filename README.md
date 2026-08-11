# Reproducing the P418 pebble-bed heat-transfer study

This directory describes the reproducible route for the manuscript on
three-dimensional conjugate heat transfer and physics-constrained learning in
internally heated ceramic pebble beds.

## Scope

The paper is built from four distinct stages:

1. OpenFOAM 13 pore-resolved conjugate heat-transfer calculations.
2. Extraction of cell fields, face fluxes, pressure drop, wall heat transfer,
   hotspot temperature, and mass/energy balances.
3. Fair comparison of classical response surfaces, DMDc, PINN variants, graph
   models, Transformer models, and the residual temperature correction.
4. Regeneration of tables, figures and the main manuscript. The compact
   supplementary file is retained only as an optional editor-requested item.

The stages are deliberately separated. The reproducibility entry point does
not start a formal solver or model training by default.

## Recorded environment

- Operating system: Ubuntu 22.04, Linux 6.8.0-124-generic, x86_64
- OpenFOAM: OpenFOAM Foundation 13
- Multi-region entry point: the OpenFOAM 13 `foamMultiRun` command
- MPI: Open MPI 4.1.2
- Compiler: GCC 11.4.0
- Python: 3.10.12
- Formal OpenFOAM parallelism: 32 MPI ranks per case
- Python packages: `requirements-p418.txt`

The OpenFOAM extension used for direct helium-property evaluation is included
under `solver_extensions/hccbHeliumTransport/`. It evaluates the registered
P070 viscosity and P071 thermal-conductivity correlations directly instead of
interpolating a finite lookup table; the correlations and their coefficients
are unchanged. The directory also contains a pointwise comparison program.

The machine-readable copy is
`reproducibility/p418_environment.json`. The versions are transcribed from
the completed workstation environment record in
`cloud_migration/VERSION_DEPENDENCIES_CN.md`; they are not inferred or
invented.

Create the Python environment with Python 3.10. Install the registered CUDA
build of PyTorch from its official wheel index before installing the remaining
pinned packages:

```bash
python3.10 -m venv .venv-p418
source .venv-p418/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-p418.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The final command must report PyTorch `2.12.1+cu130`, CUDA `13.0`, and
`True` before CUDA training is started. CPU-only post-processing does not
require a CUDA device.

## Commands

From the project root:

```bash
# Read-only checks; no OpenFOAM solve and no model training.
bash scripts/reproduce_p418_paper.sh preflight

# Rebuild the file/checksum manifest only.
bash scripts/reproduce_p418_paper.sh manifest

# Build the deterministic small source archive after verifying every file hash.
bash scripts/reproduce_p418_paper.sh archive

# Run the self-contained scientific checks included in the small source package.
# This does not start OpenFOAM or model training.
make p418-public-test

# Use completed OpenFOAM fields to regenerate processed data.
# This fails if the required formal cases are incomplete.
bash scripts/reproduce_p418_paper.sh postprocess

# Regenerate final values, figures, and the main PDF.
# This fails unless every formal result required by the manuscript exists.
bash scripts/reproduce_p418_paper.sh paper

# Optional: build the retained supplementary file only when requested.
BUILD_SUPPLEMENT=1 bash scripts/reproduce_p418_paper.sh paper
```

Formal OpenFOAM calculations are intentionally kept behind the separate,
explicit command:

```bash
make p418-formal-plan
make p418-formal-run
```

The first command prints the run order. The second command starts or resumes
formal work and must only be used on an approved compute machine.

## Data layout

- Literature-traced parameters and model settings: `parameters/`
- Formal OpenFOAM cases: `hccb_dense_cht_p418_60_sourceflow_r3/` and the
  transient/cross-packing case directories listed in `README.md`
- Processed numerical results: `results/`
- Figure-generation programs: `code/plot_hccb_p418_*.py`
- Direct helium-property OpenFOAM extension and check program:
  `solver_extensions/hccbHeliumTransport/`
- Publication figures: `figures/`
- Manuscript and optional supplementary source: `manuscript/`
- File hashes and package description:
  `results/hccb_p418_reproducibility_manifest/`

Raw OpenFOAM fields are not copied into the small source package because they
are large. Their completion records, case identifiers, and downstream
processed artifacts remain part of the formal calculation route. A public
archive must either include those raw fields or state their archived location
and checksums before submission.

The public test target covers parameter provenance, nondimensional heat-transfer
definitions, pressure and fixed-flow checks, data splits, steady random-seed
statistics, thermodynamic consistency and reproducibility metadata. Development
tests that require raw cloud work directories, unpublished large arrays or
third-party source trees remain in the full research workspace and are not part
of the public-package test target.

The small package does include path-free, plot-ready tables under
`results/hccb_p418_public_figure_data/`. They regenerate the 60-condition
physical response, the nine-condition independent-packing comparison, and the
five-model by five-split steady comparison. Exact commands are given in that
directory's `README.md`. Domain rendering, three-dimensional cloud plots, and
the final transient prediction panels use larger geometry or prediction arrays
and therefore belong to the citable processed-data archive rather than the
small source package.

Copyrighted journal articles and third-party source repositories are not
redistributed in the small source archive. They are identified by DOI,
bibliographic metadata, source URL, and, where applicable, the checksum of the
copy used during development. Reproducing the literature checks therefore
requires the user to obtain those materials from their publishers or original
repositories. This does not affect regeneration of the three quantitative
manuscript figures whose compact processed tables are included in the small
archive.

The three-mesh engineering-observable table, GCI table and machine-readable
summary are small enough to be included directly in the source archive under
`results/hccb_p418_three_mesh_cht_sensitivity/`. They are the original
post-processing outputs recovered from the completed formal calculations, not
values reconstructed from the manuscript.

The current release scope is recorded in
`results/hccb_p418_public_data_release_preflight/`. Its machine-readable
summary distinguishes files that are already public-ready from final transient
prediction arrays that remain pending. The repository DOI remains unassigned
until the P418-specific Zenodo record is deposited; the DOI of the earlier
tritium-release study is not reused.

Repository citation metadata are prepared in the top-level `CITATION.cff` and
`reproducibility/repository_release_metadata_draft.json`. The latter is a
preparation record rather than a Zenodo deposition payload: author names,
affiliations, title, description, keywords and licences are fixed, while the
repository URL, DOI and release version remain unset until the final processed
prediction archive is complete.

The public code repository is
`https://github.com/wangjianfttt/fusion-pebble-bed-heat-transfer-ai`. The
processed-data DOI is added only after the final selected predictions and
figure records have been deposited.

## Licences

Original project software and reproduction scripts are released under the MIT
License in `LICENSE`. Processed numerical data, plot-ready tables, derived
figure data and released model predictions are licensed under CC BY 4.0 as
described in `DATA_LICENSE.md`. Third-party publications, publisher figures,
external data and third-party software are excluded from these licences.

The `archive` mode creates
`results/hccb_p418_reproducibility_manifest/p418_reproduction_source.tar.gz`
and a machine-readable record containing its SHA-256, size and member count.
The archive is deterministic: rebuilding it from unchanged files produces
identical bytes. It contains no symbolic links, model checkpoints or raw
OpenFOAM time directories.

## Result policy

No expected or illustrative value may replace a missing formal result.
`code/check_hccb_p418_final_scientific_requirements.py` reports the remaining
calculations and prevents a final manuscript refresh from being recorded as
complete while required results are absent.
