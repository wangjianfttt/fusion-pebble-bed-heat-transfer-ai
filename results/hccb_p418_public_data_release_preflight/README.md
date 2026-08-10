# P418 public data release preflight

This release accompanies the study of pore-resolved conjugate heat transfer and reduced-order prediction in an internally heated ceramic pebble bed. It is a preflight record: completed files are listed with size and SHA-256 in `summary.json`, whereas missing final model files remain explicitly absent.

## Release layers

1. **Compact source archive.** Code, registered inputs, tests, plot-ready tables and the three-mesh engineering/GCI results.
2. **Citable processed-data archive.** Regional geometry, selected model predictions, transient comparison tables and final figure records. This layer is not declared complete before the formal model comparison finishes.
3. **Large OpenFOAM archive.** Reconstructed and decomposed three-dimensional fields retained in the institutional archive; these are not duplicated in the compact public package.

## Reproduce the compact quantitative figures

From the project root, follow `results/hccb_p418_public_figure_data/README.md`. It provides the exact commands for the 60-condition response map, the seed101--seed202 integral comparison and the steady-model comparison. The three-mesh engineering observables and GCI values are supplied as CSV and JSON files rather than reconstructed from a PDF.

## Training and data splits

The public training manifest is provided as `formal_training_manifest_public.json`. It preserves the 75-job model, random-seed, data-split and dependency plan while replacing workstation paths with `${PROJECT_ROOT}`. It therefore documents the complete comparison plan without exposing machine-specific directories. Training, validation and independent-test trajectory identifiers are retained.

## Scientific scope

The successful transient database advances the energy equations on frozen, target-condition hydrodynamic fields. It is therefore a fixed-hydrodynamics thermal-step database, not evidence of a successful fully coupled flow-startup calculation. The independent seed202 results use a second intact spherical-pebble arrangement. Failed full-domain and fully coupled screening runs are reported only as applicability limits and are not included as successful training samples. The path-free direct-transport scope record is supplied with the compact files.

## Current release state

- Compact files ready: 11/11
- Final processed files currently ready: 2/8
- DOI: pending assignment for this P418 study
- Licences: MIT for original software; CC BY 4.0 for processed data
- Zenodo metadata draft: `zenodo_metadata_draft.json`
- Raw decomposed OpenFOAM fields: retained in the large institutional/NAS archive and not duplicated in the small source package

Use the SHA-256 values in `summary.json` to verify every listed file after download or transfer. No machine-local absolute path is required to reproduce the compact figures.
