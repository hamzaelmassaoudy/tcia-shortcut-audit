# 7n. Mask coverage by class (R21), written on 7 Nov 2026

**Status:** written on 7 Nov 2026 (local date, UTC+8), after all earlier results had been seen and **before R21 was
computed**. Post hoc and descriptive.

**Question.** Background-only inputs replace the body mask with air. If the mask covered a different share of the scan
in one class (for example because of residual tissue outside the largest component, or a different field of view), the
background input would carry that difference.

**Data.** The saved per-scan preparation records of the 525 scans of the four flagged tasks
(`results/05_image_check/cache/headers/*.json`): `body_fraction_dilated` (share of the volume inside the dilated mask) and
`body_area_cm2` (mean body cross-sectional area). Labels as in notebook 05.

**Analysis.** Per task: median (IQR) of each quantity by class, and its AUC for the second class (rank-based, bootstrap
95% CI, 2000 resamples, seed 2026). Benjamini–Hochberg over the 8 tests (4 tasks × 2 quantities).

**Reading (fixed now).** Mask coverage *differs by class* in a task if its CI excludes 0.5 and q < 0.05; such a task's
background result is then reported with this caveat. Otherwise the reading is *no detectable difference*.

**Output:** `results/14_screen_checks/real/r21_mask_coverage_by_class.csv`; script `r21_mask_coverage.py`.
