# 7l. Two post hoc checks of the external-transfer result (R17, R18), written on 6 Nov 2026

**Status:** written on 6 Nov 2026 (local date, UTC+8), after every transfer result of notebooks 08 and 11 and checks
R1–R16 had been seen, and **before any of the numbers below was computed**. Both checks are labeled **post hoc**.
Nothing here changes an earlier number, flag or reading.

**Inputs only (no model is refitted, no image is read):**
- the saved per-patient scores of the fixed TCGA lung probes, fitted on all 104 TCGA patients and applied unchanged to
  each target: `results/11_external_lung/real/transfer_scores.csv.gz` (columns pair, encoder, input, key, label,
  score; label 1 = squamous cell carcinoma);
- patient sex from the study records of the TCIA interface, `results/01_inventory/studies.csv.gz`, taken as in R9 (first
  study with a value per patient).

**Statistics:** AUC by ranks (ties one half), squamous = positive class; 95% CI = percentiles of 2000 patient bootstrap
resamples (seed 2026), resamples lacking a class redrawn, as in notebooks 05–11.

## R17. Crossed assembly (an illustration of what an assembled external test measures)

**Question.** If the classes of an external test set are drawn from different cohorts, how much can the external AUC of
one fixed model change with that choice alone?

**Design.** Four target cohorts: CPTAC lung (assembled like TCGA: class = collection), NSCLC Radiogenomics,
NSCLC-Radiomics and Lung-PET-CT-Dx (each a single source with both classes). For each encoder (CT-FM, Merlin) and input
(full image, background only, silhouette), and for each ordered pair of different cohorts (X, Y), a crossed set is
formed from the adenocarcinomas of X and the squamous cell carcinomas of Y (12 crossed sets), next to the 4 natural
cohorts. The AUC of the fixed TCGA probe is computed on each set, with its bootstrap CI.

**Reported:** per encoder and input, the AUC of each crossed set, their range, and the range over the 4 natural cohorts;
the mean score of each cohort and class (the cohort offset). Primary display: background only, both encoders.

**Reading (fixed now).** This check illustrates a property of the test design, not the mechanism of the probe: a crossed
set changes the cohort composition of the classes, so its AUC mixes histology with differences between cohorts. The
result is stated as "the external AUC of the same fixed model ranged from a to b depending only on the cohorts the
classes were drawn from". It is not read as evidence for or against acquisition.

**Rehearsal gate.** First run with the labels permuted within each cohort (seed 2026): the natural-cohort AUCs must lie
between 0.35 and 0.65. **Reproduction gate (real labels).** The 4 natural-cohort AUCs of each encoder and input must
equal the external AUCs recorded in `results/11_external_lung/real/transfer.csv` (full image and background) and
`silhouette_transfer.csv` within 1e-9; otherwise the check stops.

## R18. Sex and the TCGA → CPTAC transfer

**Question.** Is the transfer to CPTAC lung explained by a different sex mix of the classes?

**Design.** For CPTAC lung (and, for comparison, NSCLC Radiogenomics), patients with recorded sex. For each encoder and
input with a transfer in Table 4 (CT-FM background, Merlin background, Merlin full image), and the silhouettes:
(a) sex alone as a score (male = 1) for squamous; (b) the probe's AUC within men and within women; (c) a within-sex AUC:
the share of squamous–adenocarcinoma pairs of the same sex that the score orders correctly, with a bootstrap CI over
patients.

**Reading (fixed now), for the three transferred probes.** Sex is *not a sufficient explanation* if the within-sex AUC
(c) has a CI lower bound above 0.5. If (c) is at least 0.60 but its CI includes 0.5, the reading is *inconclusive at
this sample size*. Below 0.60, sex *cannot be excluded*. Sex-only AUC (a) is reported beside it.

**Rehearsal gate.** As for R17, with labels permuted within each cohort: within-sex AUCs between 0.35 and 0.65.

**Outputs:** `results/13_transfer_checks/` (r17_crossed_assembly.csv, r17_cohort_offsets.csv, r18_within_sex.csv,
run records with checksums). Scripts: `r17_crossed_assembly.py`, `r18_within_sex.py`.

**Note added after the runs (6 Nov 2026).** Both checks were implemented in one script,
`results/13_transfer_checks/r17_r18_transfer_checks.py`, rather than the two script names given above. After the first
rehearsal, the column name used by the reproduction gate for the silhouette AUCs (`silhouette_external_auc`) was
corrected; this affects only the gate of the real run, not any computed value. The final rehearsal and real runs used
the same script (SHA-256 in each `run_manifest.json`). No rule, input or reading above was changed.
