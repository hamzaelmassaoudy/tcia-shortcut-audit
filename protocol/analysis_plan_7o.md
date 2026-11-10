# 7o. Image experiments for revision 10 (G1–G4; notebooks 12 and 13), written on 7 Nov 2026

**Status:** written on 7 Nov 2026 (local date, UTC+8), after every earlier result (notebooks 01–11, checks R1–R21) had
been seen, and **before any image, mask or embedding below exists**. All four
experiments are **post hoc**. Nothing here changes an earlier number, flag or reading. Each runs first on shuffled
labels (rehearsal) with the gates of notebooks 05–11, then on the real labels.

**Why.** Four questions remain open: (1) whether the image models use tumour information at all, since all probes so far
pool a whole volume; (2) whether the tissue-free signal is body habitus or positioning rather than acquisition;
(3) whether harmonizing geometry removes it; and (4) whether an end-to-end network behaves like the linear probes.

**Cohorts and scans.** Exactly the scans already embedded in notebooks 05, 07, 08 and 11, identified by the
`SeriesInstanceUID` in their saved header records: TCGA lung (source; 104), CPTAC lung (73), NSCLC Radiogenomics (199),
NSCLC-Radiomics (203), Lung-PET-CT-Dx (312), and LIDC-IDRI primary versus metastatic (94). Each series is downloaded
again; the body mask and background-only image are rebuilt with notebook 08's code, and the rebuilt
`body_fraction_dilated` and `body_area_cm2` must equal the saved values within 1e-6 (reproduction gate; scans that fail
are listed and left out).

**Segmentation (label-free).** TotalSegmentator, pinned version 2.10.0 (Apache 2.0) with torch 2.5.1 and nnU-Net 2.6.2, run in its own Python environment so
that the analysis packages keep the versions of notebooks 05–11: (a) task `total` with `--fast`, the five lung lobes
united as the lung mask; (b) task `lung_nodules` (1.5 mm model), the largest connected nodule component as the tumour.
The nodule model's training data are not ours to choose; it may have seen public lung CT, never our labels. **Tumour
check:** in NSCLC-Radiomics, the manual GTV-1 contour (RTSTRUCT) of each patient is rasterized on the CT slice nearest to each contour by its recorded DICOM position; the share of patients
whose automatic tumour overlaps the GTV (Dice > 0.1) and the median Dice are reported. If fewer than 70% overlap, the
tumour-only results are reported as *unreliable segmentation* and are not interpreted.

## G1. Tumour-only and lungs-only probes (notebook 12)

**Inputs** (same geometry as the full image, so that only the kept region differs): *lungs only* (every voxel outside the
lung mask dilated by 5 mm set to −1000 HU) and *tumour only* (every voxel outside the tumour dilated by 10 mm set to
−1000 HU). Patients without a detected tumour are left out of the tumour-only analyses and counted by class. Embedded
with CT-FM and Merlin exactly as in notebook 05 (CT-FM's crop box from the input itself).

**Analyses.** (a) Internal 5 × 5 cross-validated AUC, bootstrap CI and permutation q in TCGA lung, in each target and in
LIDC-IDRI, with notebook 05's probe. (b) Transfer of the TCGA probes to the four targets with notebook 11's code
(external AUC and CI) and the strict test of R7 (target labels permuted, Holm across all comparisons of G1).

**Reading (fixed now)**, per encoder, for TCGA → targets:
- *tumour information that generalizes*: the tumour-only probe transfers (CI above 0.5 and Holm P < 0.05) to at least 2
  of the 3 independent cohorts;
- *behaves like the background*: it transfers to CPTAC lung only;
- *no transferable tumour signal*: neither.
The same reading is given for lungs only. Internal AUCs are descriptive.

## G2. Habitus and positioning (notebook 12)

**Measures per scan** (from the image and masks; label-free): (1) body AP-to-LR diameter ratio and (2) subcutaneous fat
area (−190 to −30 HU within 3 cm inside the body surface), both on the axial slice of largest lung area; (3) arms beside
the trunk (a body component of at least 10 cm² separate from the trunk on that slice); (4) table height: lowest body
voxel above the image bottom (mm); (5) offset of the body centroid from the image centre (mm); (6) scan length (mm;
header); (7) lung volume (litres).

**Analyses.** (a) Each measure by class in the five lung cohorts: AUC for squamous cell carcinoma having the higher value,
with CI. (b) A habitus model (logistic regression on the seven measures, notebook 05's settings): internal AUC in TCGA
and transfer to the four targets. (c) Background-only embeddings residualized on the seven measures (linear regression
of each embedding dimension, fitted on the training data only): internal AUC and transfer.

**Reading (fixed now).** Habitus *accounts for the CPTAC transfer* if the habitus model transfers to CPTAC lung (CI above
0.5) and the residualized background probe no longer does; *does not account for it* if the residualized probe still
transfers; otherwise *partly*.

## G3. Harmonized geometry (notebook 12)

**Inputs.** Every scan resampled to 1.0 × 1.0 mm in-plane and 5.0 mm slabs (z averaged), cropped to a fixed 360 × 360 mm
field centred on the body centroid and to the z-range from 20 mm above the lung apex to 20 mm below the lung base (from
the lung mask); full image and background only; embedded as above.

**Reading (fixed now).** Harmonization *removes the tissue-free signal* if the harmonized background probe's internal
TCGA CI includes 0.5 and it no longer transfers to CPTAC lung; *does not remove it* if it still transfers to CPTAC lung;
otherwise *reduces it*. The harmonized full-image probe is reported beside it.

## G4. End-to-end 3D network (notebook 13)

**Data.** The harmonized volumes of G3, saved at 2 × 2 × 5 mm (180 × 180 voxels in-plane), full image and background
only. **Model.** A 3D ResNet-18 (MONAI) trained from scratch, binary cross-entropy with class weights, AdamW (learning
rate 1e-4, weight decay 1e-2), 60 epochs, flips and ±10° rotations, batch 8; no tuning. Intensities clipped to −1000 to 1000 HU and scaled to 0–1
(Merlin's window); the volume depth is fixed at 64 slabs of 5 mm centred on the harmonized range (padded with air). The
rehearsal trains one seed per fold; the real run three. **Evaluation.** In TCGA lung,
5-fold stratified cross-validation (seed 2026), three training seeds per fold, out-of-fold AUC averaged over seeds with a
bootstrap CI; then three models trained on all TCGA patients applied unchanged to the four targets (external AUC, CI and
the strict test).

**Reading (fixed now).** As G1: *generalizes* if the full-image network transfers to at least 2 of the 3 independent
cohorts; *behaves like the background* if it transfers to CPTAC lung only (and the background network likewise);
otherwise *no transferable signal*. With 104 training patients the network is expected to be weak; its results are
reported whatever they are.

## Gates and records

- Rehearsal on labels permuted within each cohort (seed 2026): every internal permutation null mean between 0.35 and
  0.55, and at least 95% of scans with all inputs embedded, before the real labels are used.
- Run records (`run_manifest.json`) with package versions, the TotalSegmentator version and model weights' checksums, the
  MD5 of every output, and every scan left out with its reason.
- Outputs: `results/12_image_experiments/` (G1–G3) and `results/13_cnn/` (G4).
