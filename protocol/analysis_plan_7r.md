# Analysis plan, section 7r: checks R28 and R29

Written on 26 November 2026, after every earlier result had been seen and before either check was computed. Both are
post hoc. They read saved result files only: no model is refitted and no image is read. None of them can change an
earlier number, flag or reading.

Script: `results/16_strength_checks/r28_r29_checks.py`. Outputs in `results/16_strength_checks/` with
`run_manifest_r28_r29.json` (package versions, settings and the MD5 of every input and output).

## R28: how much of a crossed test set's AUC is the difference between its two source cohorts?

The labels-permuted arm of R17 was a single permutation. Its expected value is known exactly: when the labels are
permuted within each cohort, a crossed test set holds a random subset of the first cohort labelled adenocarcinoma and a
random subset of the second labelled squamous cell carcinoma, and the expected AUC is the AUC that separates the two
cohorts' scores, whatever the histology (between-cohort AUC).

From `results/11_external_lung/real/transfer_scores.csv.gz` (scores of the fixed TCGA lung probes, as in R17), for each
encoder, each input (full image, background only, silhouette) and each of the 12 ordered pairs of the four lung targets:

1. the between-cohort AUC (all patients of the first cohort scored 0, all of the second scored 1), with a 95% percentile
   patient-bootstrap CI (2,000 resamples, seed 2026, as in R17);
2. the distribution of the crossed-set AUC over 1,000 permutations of the labels within each cohort (seed 2026): mean,
   2.5th and 97.5th percentiles, and the share of permutations whose AUC is at least 0.60 and at most 0.40;
3. across the 12 crossed sets, the Spearman correlation between the crossed-set AUC with the recorded labels (R17) and
   the between-cohort AUC.

Reading, descriptive: the share of crossed sets whose between-cohort CI lies wholly above or below 0.50 measures how
often the difference between two source cohorts alone moves the AUC away from chance.

## R29: flag decisions at other thresholds

The flag rule (lower bound of the 95% CI at least 0.60 and q below 0.05) was fixed before the screen. For thresholds of
0.55, 0.60, 0.65 and 0.70 on the lower bound, with q below 0.05 unchanged, count the flagged tasks: for the metadata
screen (26 tasks, `results/03_audit/audit_primary.csv`) and for the image probes of the 19 CT tasks by encoder and input
(`results/05_image_check/real/image_probe.csv`, `results/07_screen_validation/real/image_probe_unflagged_tasks.csv`).
Descriptive; list the tasks that change status.
