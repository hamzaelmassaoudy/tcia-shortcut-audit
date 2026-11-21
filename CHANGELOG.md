# Changelog

Changes to this repository by version. Each GitHub release is archived on Zenodo; all versions:
https://doi.org/10.5281/zenodo.24247266.

## Unreleased

- `figures/make_figures.py`: the Fig. 5 legend now describes the silhouette as the body shape (the body mask at one
  uniform intensity), as defined in the methods; it previously read "body outline". No plotted value changed
  (2026-11-16).
- `README.md` and `CITATION.cff`: the Zenodo DOIs (all versions and version 1.1.0); the author's ORCID iD in
  `CITATION.cff` and `.zenodo.json` (2026-11-15).
- `CHANGELOG.md`: this file.

## 1.1.0 (2026-11-15)

Archived on Zenodo: https://doi.org/10.5281/zenodo.24247267.

- Plans 7m and 7n with checks R19–R21: contributing country and the CPTAC transfer, a positive control of the metadata
  screen (1680 synthetic tasks) and mask coverage by class (`results/14_screen_checks/`).
- Plan 7o with notebooks 12 and 13: four post hoc experiments on the source of the tissue-free signal in lung
  histology, namely lungs-only and tumour-only probes, body habitus and positioning, harmonized geometry, and a 3D
  ResNet-18 trained from scratch (`results/12_image_experiments/`, `results/13_cnn/`).
- Plan 7p with checks R22–R24: differences with confidence intervals, every Step 4 model within contributing site, and
  a placebo for the habitus measures (`results/15_source_checks/`).
- Figures redrawn in one style, with a workflow figure (Fig. 1) and one rule for "transfers" in every figure.
- README with the study at a glance; Zenodo metadata (`.zenodo.json`); citation version 1.1.0.
- The audit tool is unchanged (version 1.0.1).

## 1.0.1 (2026-11-06)

First snapshot of the study in this repository: the audit tool `tcia_shortcut_audit` 1.0.1 with its manual and
reproduction test, notebooks 01–11, the dated analysis plan with section 7l, the results of notebooks 01–11 and the
post hoc checks R1–R18 (`results/12_review_checks/`, `results/13_transfer_checks/`).
