# 7p. Checks on the experiments of plan 7o (R22–R24), written on 13 Nov 2026

**Status:** written on 13 Nov 2026 (local date, UTC+8), after every earlier result (notebooks 01–13, checks R1–R21,
experiments G1–G4) had been seen, and **before any quantity below was computed**. All three checks are **post hoc** and
descriptive. Nothing here changes an earlier number, flag or reading. They use only saved embeddings, saved per-patient
scores, the saved scan records of notebook 12 and the saved predictions of notebook 13; no image is read and no network
is trained. Each runs first with the labels permuted within each cohort (the rehearsal labels of notebook 12), then with
the real labels.

**Why.** (1) Several statements in the text compare a result that passes a rule with one that does not ("removed",
"weakened", "better", "only to") without an estimate of the difference. (2) Half of the TCGA lung patients come from
two tissue-source sites that hold almost only one class, and the experiments of plan 7o were not evaluated within site.
(3) The fall of the CPTAC transfer after the seven habitus measures are regressed out has no control that shows it to
be specific to these measures.

**Reproduction gate.** The script refits the probes of notebooks 05 and 12 from the saved embeddings with the same
settings and splits. Before any new quantity is written, each refitted AUC must equal its recorded value within 0.002
(the tolerance found earlier between machines); the recorded and refitted values are saved.

## R22. Differences with confidence intervals

Patient bootstrap, 2000 resamples, seed 2026, percentile 95% CI; resamples with a missing class are redrawn. Paired
differences use the same resampled patients for both models; unpaired differences resample each cohort on its own.

- (a) CPTAC lung, each encoder: external AUC of the TCGA background-only probe minus that of the same probe with the
  seven habitus measures regressed out (paired).
- (b) CPTAC lung, each encoder: native geometry minus harmonized geometry, for the background-only and the full-image
  probe (paired).
- (c) TCGA lung, cross-validated, each encoder: background only minus tumour only, on the patients with a detected
  tumour (paired); background only minus lungs only (paired).
- (d) TCGA lung, cross-validated, each encoder: background only minus full image (paired).
- (e) Each fixed TCGA probe (two encoders, full image and background only): external AUC on CPTAC lung minus external
  AUC on each of the three independent cohorts (unpaired).

**Reading (fixed now).** A difference is *shown* if its CI excludes 0; otherwise *not shown*. No correction for the
number of comparisons; all are reported.

## R23. The experiments of plan 7o within site (TCGA lung)

Site: the institution of the tissue-source-site code (as in check R4). For every model of plan 7o with out-of-fold
predictions in TCGA lung (lungs only, tumour only, habitus model, background only with the habitus measures removed,
harmonized full image and background, and the 3D network on both inputs), and for the native probes as a reference:

- the within-site AUC, the share of squamous–adenocarcinoma pairs from the same site that the model orders correctly
  (sites with both classes), with a patient-bootstrap CI;
- the AUC within the one site that holds both classes in similar numbers (University of Pittsburgh), with its CI.

The seven habitus measures are compared between classes within site in the same way.

**Reading (fixed now).** The signal of a model *holds within site* if the CI of its within-site AUC lies above 0.5;
otherwise *not shown within site*. The refitted within-site AUC of the native Merlin background probe must agree with
the recorded value (0.787) within 0.002.

## R24. Specificity of the habitus measures

- (a) Placebo. The rows of the habitus matrix are permuted among the TCGA patients and, separately, among the CPTAC
  patients (200 permutations, seed 2026). This keeps the number and distribution of the regressors and breaks their
  link to the scans. Each time, the background-only embeddings are residualized on the permuted measures and the probe
  is refitted on TCGA and applied to CPTAC. Reported: the median and the 2.5th and 97.5th percentiles of the placebo
  external AUCs, and the one-sided P = (1 + number of placebo AUCs at or below the AUC with the real measures)/(1 + 200).
- (b) The habitus model refitted without each measure in turn: external AUC on CPTAC lung and on NSCLC Radiogenomics;
  and the coefficients of the full model on the standardized measures.

**Reading (fixed now).** The fall is *specific to the measures* if P < 0.05 in (a); otherwise *not shown to be
specific*. Part (b) is descriptive.

## Records

Outputs in `results/15_source_checks/` (`rehearsal/` and `real/`), with `run_manifest.json` (package versions, settings,
MD5 of every output). Script: `results/15_source_checks/r22_r24_source_checks.py`.
