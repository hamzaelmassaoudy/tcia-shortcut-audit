# Analysis plan, section 7q: checks R25–R27

Written on 25 November 2026, after every earlier result had been seen and before any of these checks was computed.
All three are post hoc. They read saved result files only: no model is refitted, no image is read and no label is
used beyond what the saved result files already contain. A shuffled-label rehearsal is therefore not needed; R27
summarizes the rehearsal of R17 that already exists. None of them can change an earlier number, flag or reading.

Script: `results/16_strength_checks/r25_r27_strength_checks.py`. Outputs in `results/16_strength_checks/` with a
`run_manifest.json` (package versions, settings and the MD5 of every input and output).

## R25: size of the external cohorts

Question: were the external cohorts large enough to detect a transfer of the size observed in CPTAC lung?

For each external target, with its class sizes (CPTAC lung 37 adenocarcinomas / 36 squamous cell carcinomas;
NSCLC Radiogenomics 166 / 33; NSCLC-Radiomics 51 / 152; Lung-PET-CT-Dx 251 / 61; C4KC-KiTS 143 clear cell / 21
papillary), compute the probability that a fixed probe whose true external AUC is *a* meets the CI part of the
transfer rule (lower bound of the 95% percentile patient-bootstrap CI above 0.50), for *a* = 0.65, 0.70 and 0.739 (the
CPTAC Merlin background-only estimate), and the true AUC at which this probability reaches 80% (bisection to 0.001).

Model and simulation as in R14: binormal scores with equal variances (true AUC = Φ(d/√2)); 1,000 simulated data sets
per AUC value, each with 1,000 bootstrap resamples of all patients; resamples missing a class are dropped; data set
*s* uses `default_rng([2026, s])`, so the probability rises monotonically with the AUC. Not modelled: the permutation
test and its Holm correction, which can only make the full criterion harder to meet.

Reading fixed in advance: a cohort is called large enough to detect a transfer of the CPTAC size if the probability at
*a* = 0.739 is at least 0.80.

## R26: agreement of the metadata screen with the image flags

Question: how often did a CT task without a metadata flag have a flagged background-only probe, and the reverse?

Over the 19 CT tasks, cross-tabulate the metadata flag (notebook 03) against the background-only flag of the image
probes (`results/05_image_check/real/image_probe.csv` for the four flagged tasks,
`results/07_screen_validation/real/image_probe_unflagged_tasks.csv` for the 15 others), for CT-FM, for Merlin and for
either encoder; the same for the full-image flag. Report the four counts, and the share of image-flagged tasks that
the screen flagged (sensitivity), the share of tasks without an image flag that the screen left unflagged
(specificity), and the shares among screen-flagged and screen-unflagged tasks (positive and negative predictive
value), each with an exact Clopper–Pearson 95% CI. Descriptive; the image flags of the two groups come from separate
testing families, as reported in the article.

## R27: crossed assembly in numbers

Question: how often does assembling a test set from two cohorts alone produce a CI above or below 0.50?

From `results/13_transfer_checks/real/r17_crossed_assembly.csv` and the rehearsal file of the same name (labels
permuted within each cohort), count for each encoder and input the test sets whose CI lies wholly above 0.50 and wholly
below 0.50, separately for the three cohorts with both classes in one collection, for CPTAC lung (one collection per
class) and for the 12 crossed test sets. Descriptive.
