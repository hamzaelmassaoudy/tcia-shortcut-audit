# tcia-shortcut-audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24247266.svg)](https://doi.org/10.5281/zenodo.24247266)

**Can public cancer CT benchmarks be solved from how the scans were acquired, rather than from the patient?** This
repository holds the open screening tool, the full analysis code, every result with its run record, and the frozen
foundation-model embeddings for the study *Screening public cancer CT benchmarks for non-tumour shortcuts: tissue-free
prediction and the limits of external validation*.

**Author:** Hamza El Massaoudy, School of Computer Science, Nanjing University of Information Science and Technology, Nanjing, China. Further authors will be listed when the paper is published.

## Study at a glance

| Item | Details |
|---|---|
| Status | Manuscript in preparation for journal submission; this repository is its release snapshot |
| Data | The Cancer Imaging Archive (TCIA), series metadata retrieved on 28 Oct 2026: 33 collections, 63,468 series |
| Tasks | 26 patient-level classification tasks (19 CT, 7 MRI), 2,775 patients |
| External cohorts | CPTAC lung, NSCLC Radiogenomics, NSCLC-Radiomics, Lung-PET-CT-Dx (lung); C4KC-KiTS (kidney) |
| Image models | Frozen CT foundation models CT-FM (revision `527b0d1c`) and Merlin (revision `66dd0758`) with linear probes; a 3D ResNet-18 (MONAI 1.6.1) trained from scratch |
| Segmentation | TotalSegmentator 2.10.0 (lungs and lung nodules), body masks from a fixed threshold rule |
| Analysis plans | `protocol/analysis_plan.md` and sections 7l–7p, each dated before its real-label run |
| Post hoc checks | R1–R24 (`results/12_review_checks/` to `results/15_source_checks/`) and experiments G1–G4 |
| Tool | `tool/tcia_shortcut_audit.py`, version 1.0.1 |
| Compute | Google Colab (CPU; A100 and L4 GPUs for the image notebooks); post hoc checks on a local CPU |
| Integrity | `MANIFEST.md5` for every tracked file; `run_manifest.json` with package versions and output checksums in each run folder |
| Licences | Code MIT; derived data CC BY 4.0; images remain under each TCIA collection's licence |

## Key findings

The study screened 26 patient-level classification tasks built from 33 collections of The Cancer Imaging Archive (TCIA):
63,468 series and 2,775 patients. The screen uses acquisition metadata only, retrieved without downloading any image.

- **Metadata alone predicted the label in 4 of 26 tasks** (AUC 0.79–0.85), including both tasks whose classes are
  separate collections (TCGA-LUAD vs TCGA-LUSC; TCGA-KIRC vs TCGA-KIRP).
- **With segmented body tissue replaced by air**, frozen CT foundation-model probes (CT-FM, Merlin) still predicted lung
  adenocarcinoma vs squamous cell carcinoma (AUC 0.82) and, less robustly, LIDC-IDRI primary vs metastatic cancer
  (AUC 0.72–0.73). In the 15 unflagged CT tasks, no background-only probe was flagged.
- **A probe without tissue intensities passed one of four external tests.** TCGA lung models did not transfer to
  three independent cohorts (714 patients; AUC 0.37–0.56). Two of four did transfer to CPTAC (Merlin; AUC 0.70 and
  0.74), which, like TCGA, takes each class from its own collection; the TCGA metadata model did not transfer (AUC 0.42),
  and sex did not explain the transfer (within-sex AUC 0.65–0.75). External discrimination alone therefore did not show
  that tumour information was used.
- **For lung histology in TCGA, post hoc experiments found the signal in the background and not in the tumour region**
  (plan 7o). Tumour-only inputs did not predict histology in TCGA (AUC 0.56–0.57). Seven measures of body habitus and
  positioning alone transferred to CPTAC (AUC 0.71); with them regressed out, the background-only transfer AUCs were
  0.56 (CT-FM) and 0.65 (Merlin), down from 0.68 and 0.74. Inside TCGA the signal remained at harmonized geometry
  (AUC 0.69–0.78), and a 3D network trained from scratch predicted histology better from the background (AUC 0.83) than
  from the full image (AUC 0.64). What carries the signal inside TCGA was not identified.
- **The screen is specific and its sensitivity is measured.** On 1680 synthetic tasks it flagged 1 of 240 tasks without
  an association and detected 80% of planted associations of AUC 0.72 (250 patients) to 0.79 (100 patients).
- **External AUC depends on how the test set is assembled.** When the two classes of a test set were drawn from
  different cohorts, the same fixed background-only probes gave external AUCs from 0.29 to 0.75, with apparent transfer
  and reversal even on sets built only from independent cohorts.

## Quick start: audit a TCIA task before downloading it

```bash
pip install -r tool/requirements.txt
python tool/tcia_shortcut_audit.py --collections TCGA-LUAD TCGA-LUSC --label-by-collection --out audit_lung
```

- **Your own labels:** pass a labels file (`--labels labels.csv`, with columns `PatientID,label` and optionally
  `collection,site`) instead of `--label-by-collection`.
- **Output:** the tool writes `report.md` with the verdict, the field groups that carry the signal, field distributions
  by class and, when sites are given, the site analyses.
- **Speed and scope:** it needs only the public TCIA interface and runs in minutes on a laptop.
- **Manual:** [`tool/README.md`](tool/README.md).

A flag means the task needs an audit before an image model is trusted, not that a model uses the shortcut. A task that is
not flagged is not shown to be free of shortcuts.

## Repository layout

| Folder | Content |
|---|---|
| `tool/` | The audit tool `tcia_shortcut_audit` (v1.0.1; v1.0.0 in `tool/versions/`), manual, requirements and a reproduction test |
| `notebooks/` | Analysis notebooks 01–13 as run in Google Colab; `notebooks/executed/` holds the copies with the outputs of the recorded runs |
| `results/NN_*/` | Outputs of notebook NN. Each run folder has `run_manifest.json` (package versions, settings, MD5 of every output). Image analyses keep the shuffled-label rehearsal (`rehearsal/`) next to the real run (`real/`) |
| `results/*/cache/embeddings/` | Frozen CT-FM and Merlin embeddings per patient (`.npz`): full image, background only, silhouette |
| `results/*/cache/api/` | Saved replies of the TCIA interface |
| `results/12_image_experiments/` | Notebook 12 (plan 7o, G1–G3; post hoc): lungs-only and tumour-only probes, habitus and positioning measures, habitus model and background-only probe with the measures removed, probes at harmonized geometry; rehearsal and real runs with run records; `cache/` holds the per-scan records, quality-control pictures and the embeddings of the new inputs; `posthoc_habitus_medians.py` adds medians by class (descriptive, written after the results were seen) |
| `results/13_cnn/` | Notebook 13 (plan 7o, G4; post hoc): 3D ResNet-18 trained from scratch on the harmonized volumes; rehearsal and real runs with run records; `checkpoints/` holds the predictions of each of the 48 trainings; `posthoc_background_minus_full.py` adds the paired difference between the two inputs (descriptive, written after the results were seen) |
| `results/16_strength_checks/` | Post hoc checks R25–R27 (plan `protocol/analysis_plan_7q.md`, written before they were run): whether each external cohort was large enough to detect a transfer of the CPTAC size, agreement of the metadata screen with the image flags, and the crossed-assembly check in numbers; read saved result files only, with a run record |
| `results/15_source_checks/` | Post hoc checks R22–R24 on the experiments of plan 7o (differences with confidence intervals, every Step 4 model within contributing site, a placebo for the habitus measures), fixed in `protocol/analysis_plan_7p.md` before they were run; rehearsal and real runs with run records |
| `results/14_screen_checks/` | Post hoc checks R19 (contributing country and the CPTAC transfer), R20 (positive control of the metadata screen: 1680 synthetic tasks) and R21 (mask coverage by class), fixed in `protocol/analysis_plan_7m.md` and `analysis_plan_7n.md` before they were run |
| `results/13_transfer_checks/` | Post hoc checks R17 (crossed assembly of external test sets) and R18 (sex and the CPTAC transfer), fixed in `protocol/analysis_plan_7l.md` before they were run; rehearsal and real runs with run records |
| `results/12_review_checks/` | Post hoc robustness checks R1–R16: seeds, a refitting sensitivity analysis, a strict external test, duplicate screening, site tables, mask quality control, sex and age by class (R1–R9); other main-series rules, single-property baselines, scans cut off by the field of view, kernel decoding within vendor, the AUC needed for a flag, multiplicity control and a planning record (R10–R16, fixed in plan section 7k before they were run) |
| `literature/RQ5/` | Literature search scripts and log, raw and de-duplicated records, screening decisions with reasons, extraction tables |
| `protocol/analysis_plan.md` | The dated analysis plan: research questions, data and methods, each step dated |
| `protocol/analysis_plan_7l.md` | Plan section 7l (checks R17 and R18), written on 6 Nov 2026 before they were run |
| `protocol/analysis_plan_7m.md`, `analysis_plan_7n.md` | Plan sections 7m and 7n (checks R19–R21), written on 7 Nov 2026 before they were run |
| `protocol/analysis_plan_7p.md` | Plan section 7p (checks R22–R24), written on 13 Nov 2026 before they were run |
| `protocol/analysis_plan_7o.md` | Plan section 7o (image experiments G1–G4, notebooks 12 and 13), written on 7 Nov 2026 before any of their images, masks or embeddings existed |
| `figures/make_figures.py` | Rebuilds all figures from `results/`, after checking each input against its run record |
| `environment/requirements.txt` | Package versions of the recorded runs |

## Reproducing the results

1. **Check integrity:** run `md5sum -c MANIFEST.md5`. The run records `results/*/run_manifest.json` give the MD5 of every
   output as it was written.
2. **Without images (CPU):**
   - Post hoc checks, from the saved results and embeddings:
     ```bash
     cd results/12_review_checks
     python r7_external_strict.py            # strict external test, about 30 s
     python r3_cv_seed_sensitivity.py        # 20 further cross-validation seeds
     python r8_refit_bootstrap.py            # refitting sensitivity analysis, about 80 min on 12 cores
     python r11_baselines.py                 # single-property baselines (plan 7k)
     cd ../13_transfer_checks
     python r17_r18_transfer_checks.py rehearsal   # labels permuted within each cohort
     python r17_r18_transfer_checks.py real        # crossed assembly and sex, a few minutes
     cd ../14_screen_checks
     python r19_country_cptac.py                   # contributing country (reads the saved GDC records)
     python r20_screen_positive_control.py real    # positive control of the screen, about 2 h on 12 cores
     python r21_mask_coverage.py                   # mask coverage by class
     cd ../12_image_experiments
     python posthoc_habitus_medians.py             # habitus measures by class, medians (descriptive)
     cd ../13_cnn
     python posthoc_background_minus_full.py       # 3D network: background only minus full image (descriptive)
     cd ../15_source_checks
     python r22_r24_source_checks.py rehearsal     # checks R22-R24, labels permuted within each cohort
     python r22_r24_source_checks.py real          # about 2 min
     ```
     `review_lib.py` (R1–R9) and `review_lib2.py` (R10–R16) execute the model code of notebook 05 verbatim, so these
     checks use the study's own functions.
   - Figures: `python figures/make_figures.py . figures/out`.
   - The tool's reproduction of the metadata audit: `python tool/tests/reproduce_notebook03.py .`
   - Metadata audit (notebooks 03–04): runs from the saved inputs in `results/02_tasks_features/`.
3. **With images (GPU):** notebooks 05–11 download the main CT series from TCIA (about 100 GB in total). They pin the
   encoder revisions (CT-FM `527b0d1c`, Merlin `66dd0758`) and run a shuffled-label rehearsal before the real labels.
   Notebook 12 downloads the 985 lung scans again, segments lungs and tumour with TotalSegmentator 2.10.0 in its own
   Python 3.12 environment (which the notebook creates) and embeds the new inputs (about 5.5 h on an A100). Notebook 13
   reads the harmonized volumes that notebook 12 saved in the project folder and trains the 3D network (48 trainings,
   about 8 h on an L4).

The image notebooks ran on Colab with Python 3.13 and the versions in `environment/requirements.txt`. Checks R1–R9
were rerun locally with the same scikit-learn, NumPy, SciPy and pandas versions; checks R10–R16 ran with Python
3.12.10, NumPy 2.4.2, pandas 2.2.3, SciPy 1.14.1 and scikit-learn 1.5.2. Image-probe AUCs reproduce within about
0.0012; the difference is consistent with floating-point differences between machines
(`results/12_review_checks/r4b_reproduction.csv`).

## Where each result comes from

| Result | Source file |
|---|---|
| Metadata audit of the 26 tasks | `results/03_audit/audit_primary.csv` |
| Field groups; site analyses | `results/03_audit/audit_feature_groups.csv`; `results/03_audit/audit_sites.csv` |
| Image probes of the flagged tasks | `results/05_image_check/real/image_probe.csv`, `results/06_image_posthoc/real/silhouette_probe.csv` |
| Screen validation (19 CT tasks) | `results/07_screen_validation/real/`, `results/10_figures_tables/robustness_screen_validation.csv` |
| External transfer | `results/08_transfer/real/transfer.csv`, `results/11_external_lung/real/transfer.csv`, `results/09_transfer_posthoc/real/` |
| Source of the tissue-free signal: lungs-only and tumour-only probes; habitus; harmonized geometry | `results/12_image_experiments/real/g1_*.csv`; `g2_*.csv`; `g3_*.csv` |
| 3D network trained from scratch | `results/13_cnn/real/cnn_internal.csv`, `cnn_transfer.csv`, `results/13_cnn/posthoc_background_minus_full.csv` |
| Acquisition decoding | `results/07_screen_validation/decoding/acquisition_decoding.csv` |
| Published studies on the flagged tasks | `literature/RQ5/extraction_v2.csv` |
| Post hoc checks | `results/12_review_checks/`, `results/13_transfer_checks/`, `results/14_screen_checks/` |
| Sensitivity of the screen; contributing country | `results/14_screen_checks/real/r20_detection_by_oracle_auc.csv`; `results/14_screen_checks/real/r19_within_country.csv` |
| Sex and the CPTAC transfer; crossed assembly | `results/13_transfer_checks/real/r18_within_sex.csv`; `results/13_transfer_checks/real/r17_crossed_assembly.csv` |

## How the analyses were run

- **Plans before data:** each analysis was written into a dated plan (`protocol/analysis_plan.md`,
  `protocol/analysis_plan_7l.md`, `7m`, `7n`, `7o`, `7p`, `7q`) before it was run on real labels. Analyses added after results were seen are labelled
  post hoc there and in the paper. The dates are the author's records, not external time stamps; this repository is a
  single release snapshot. `results/12_review_checks/r16_planning_record.csv` lists, for each analysis before plan 7o,
  the results already seen when it was planned.
- **Folder numbers:** `results/01_*` to `11_*`, `12_image_experiments` and `13_cnn` are the outputs of the notebooks with
  those numbers; `12_review_checks`, `13_transfer_checks` and `14_screen_checks` hold the script-based post hoc checks.
- **Rehearsals and records:** every image analysis ran first on shuffled labels, and every run logged the checksums of
  its outputs.
- **Research responsibility:** The author is responsible for study design, analysis plans, interpretation and reporting.

## Versions

- **1.0.1** (6 November 2026): first version of the study in this repository: the audit tool 1.0.1, notebooks 01–11,
  the dated analysis plan with section 7l and the post hoc checks R1–R18.
- **1.1.0** (15 November 2026; archived on Zenodo, https://doi.org/10.5281/zenodo.24247267): plans 7m–7p with checks
  R19–R24, notebooks 12 and 13 (four experiments on the source of the tissue-free signal), and the figures redrawn in
  one style. The audit tool is unchanged.
- **Since 1.1.0:** the Zenodo DOIs and the author's ORCID iD; the Fig. 5 legend now describes the silhouette as the body
  shape, not an outline (no plotted value changed).
  Checks R25–R27 (plan 7q, 25 November 2026) in `results/16_strength_checks/`.

## Data and licences

- **No images are included.** All images are public in TCIA, under each collection's licence. Labels came from the
  Genomic Data Commons, the collections' clinical files and the KiTS19 clinical file.
- **Code:** MIT licence (`LICENSE`).
- **Derived data** (tables, results, embeddings): CC BY 4.0. Please also cite the TCIA collections you use.
- **Citation:** see `CITATION.cff`. Each GitHub release is archived on Zenodo: all versions, https://doi.org/10.5281/zenodo.24247266; version 1.1.0, https://doi.org/10.5281/zenodo.24247267. The paper's reference will be added when it is published.
