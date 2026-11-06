# 7m. Two further post hoc checks: contributing country and the CPTAC transfer (R19), and the sensitivity of the metadata screen (R20), written on 7 Nov 2026

**Status:** written on 7 Nov 2026 (local date, UTC+8), after every earlier result (including R1–R18) had been seen, and
**before any number below was computed**. Both checks are labeled **post hoc**. Nothing here changes an earlier number,
flag or reading.

**Statistics, as in notebooks 03–11:** AUC by ranks (ties one half); 95% CI = percentiles of 2000 patient bootstrap
resamples (seed 2026), resamples lacking a class redrawn.

## R19. Contributing country and the TCGA → CPTAC lung transfer

**Question.** TCGA lung classes came from partly different contributing institutions (site-only AUC 0.755). Did the
classes of CPTAC lung also come from different contributing sources, and does the transfer of the TCGA probes hold
within one source?

**Data.** CPTAC records no contributing institution in the Genomic Data Commons (GDC); the GDC demographic record gives
`country_of_birth` for CPTAC patients and, for some TCGA patients, `country_of_residence_at_enrollment`. These are
proxies for the contributing country, not imaging sites. For TCGA, the institution of the tissue-source-site code
(`results/12_review_checks/r4_site_mapping.csv`) and its country are used. Records are fetched once from the GDC API and
saved unchanged (`gdc_demographics.json`). Probe scores: the saved scores of the fixed TCGA probes on CPTAC lung
(`results/11_external_lung/real/transfer_scores.csv.gz`).

**Analyses.** (a) Class by contributing country (CPTAC) and class by institution and country (TCGA). (b) Country alone as
a score for squamous cell carcinoma in CPTAC lung (share of squamous per country, computed leave-one-patient-out), with
CI. (c) For the probes that transferred (CT-FM background, Merlin background, Merlin full image) and the silhouettes:
the within-country AUC on CPTAC lung (squamous–adenocarcinoma pairs from the same country), with CI. Patients with no
recorded country form their own group only if both classes are present; otherwise they are left out of (c).

**Reading (fixed now), for the three transferred probes.** The transfer is *not explained by contributing country* if
the within-country CI lies above 0.5. It is *consistent with contributing source* if the within-country AUC is below
0.60 and the country-only CI lies above 0.5. Otherwise the reading is *inconclusive*. If fewer than 20 same-country pairs
exist, the check is reported as *not informative*.

## R20. Sensitivity and false-flag rate of the metadata screen (positive control)

**Question.** How strong must an acquisition–label association be for the screen to flag it, and how often does it flag
a task with no association?

**Design (metadata only, no images).** Base cohorts: all patients with a main series in LIDC-IDRI (CT) and in
Duke-Breast-Cancer-MRI (MRI), with their real label cards built by the tool (`tool/tcia_shortcut_audit.py`, version
1.0.1, unchanged). Three acquisition sources, each a binary group G defined from the label card without any label:
(1) vendor (most frequent vendor against the rest); (2) image count above the cohort median; (3) one description word:
the most frequent word of `protocol_text` present in 25–75% of the cohort. Synthetic labels: y ~ Bernoulli(logistic(b0 +
b·G)) with odds ratio exp(b) in {1, 1.5, 2, 3, 5, 8, 15}, b0 set so that the expected prevalence of y is 0.35. Task sizes:
100 and 250 patients, sampled without replacement. 20 replicates per cell (2 cohorts × 3 sources × 7 odds ratios × 2
sizes = 84 cells, 1680 tasks), seeds fixed per replicate.

**Screen.** Exactly the study's screen: the tool's label card, model, 5 × 5 cross-validation, bootstrap CI and 999-label
permutation test. Flag: lower CI bound ≥ 0.60 and Benjamini–Hochberg q < 0.05, where q is computed for the synthetic task
added to the 26 real tasks of Table 1 (a family of 27). The permutation test is run only when the CI part is met (a task
failing it cannot be flagged).

**Reported.** For each task, the "oracle AUC" (AUC of G alone for y in the sampled task), the screen AUC and the flag.
Detection probability (with Wilson 95% CI) by oracle-AUC bin (<0.55, 0.55–0.60, …, ≥0.80) and size; the false-flag rate
at odds ratio 1; the oracle AUC at which detection reaches 80% for each size. These are compared with the simulation of
R14 (true AUC 0.654–0.793 needed at the study's class sizes).

**Reading (fixed now).** The flag rule is *specific* if the false-flag rate at odds ratio 1 is at most 5% (its Wilson
upper bound reported). Sensitivity is stated as the oracle AUC at which detection reaches 80%, per size. No pass/fail
threshold is set for sensitivity; the result is a measured operating characteristic. The synthetic signals are visible
to the interface by construction, so R20 measures detection of visible signals only; signals in fields the interface
does not report (for example reconstruction diameter, R11) are outside its reach.

**Rehearsal gate.** A first run with 3 replicates per cell at odds ratio 1 and 15 only: the false-flag rate at odds
ratio 1 must be at most 1 of the rehearsal tasks per cohort and source, and detection at odds ratio 15 with 250 patients
must exceed 50%, or the code is checked before the full run.

**Outputs:** `results/14_screen_checks/` (r19_*.csv, r20_*.csv, run records with checksums).

**Note added after the first R19 run (7 Nov 2026).** The first run of `r19_country_cptac.py` left the 17 CPTAC lung
patients without a recorded country out of analysis (c), contrary to the rule above (they hold both classes, 3 and 14, so
they form their own group). The script was corrected to follow the rule and rerun; only the corrected run is kept and
reported. In the first run the readings were the same (CT-FM and Merlin background: inconclusive; Merlin full image:
not explained by contributing country).
