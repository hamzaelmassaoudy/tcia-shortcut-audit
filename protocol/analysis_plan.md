# Dated analysis plan (sections 4, 6 and 7 of the project plan)

Extracted from the project plan, kept as written on the dates in each heading. Changes: the sanitization rules in SANITIZATION.md, and 8 personal or publication-planning passage(s) and the literature-screening logistics replaced or shortened in place; every method, rule and date is unchanged. Section 5 (planned contributions) is left out. The plan records, for each step, when it was proposed and when the author approved it before it was run.

**Dates and times (note added 5 Nov 2026).** Dates in this plan are local calendar dates, written without times; the run records (`results/*/run_manifest.json`, logs) give times in UTC. A run that started late in the UTC day therefore falls on the next local date: for example, the real-label run of notebook 11, given below as run on 3 Nov 2026, started on 3 Nov in UTC but on 4 Nov in local time. The local dates are China Standard Time (UTC+8). Sections 7g–7j were written after the first metadata and image results (notebooks 03–05) had been seen; each analysis in them was fixed before it was run. The git history of this repository starts on 5 Nov 2026 and does not time-stamp earlier entries. (Note added 7 Nov 2026: the public repository is a single release snapshot; the earlier version history is kept by the author.)

## 4. Research questions

- **RQ1 (prevalence):** across TCIA collections used for classification, how often can the label be predicted from acquisition metadata alone, well above chance?
- **RQ2 (which signal):** which metadata fields carry it: scanner manufacturer or model, slice count, protocol or series description, software version, acquisition year, or submitting site?
- **RQ3 (site or protocol):** does the signal remain *within* a single submitting site? If so, it is protocol–label coupling, not only site confounding. TCGA barcodes encode the tissue source site.
- **RQ4 (does it matter):** for one or two flagged tasks, does an image model trained the usual way get an inflated internal score? Does that score drop under acquisition-balanced or acquisition-matched evaluation, or under a counterfactual test (as in an earlier study)?
- **RQ5 (literature):** how many published studies using the flagged collections reported or controlled for acquisition?
- **Added in 7g (approved 29 Oct 2026):**
  - **RQ6 (screen validity):** does the metadata flag predict where patient-free images predict the label?
  - **RQ7 (consequence):** do models trained on a flagged benchmark transfer to an independent dataset with the same question?
  - **RQ8 (mechanism):** do CT foundation-model embeddings encode acquisition, even without the patient?

## 6. Data

**Source 1: the TCIA public API** (no account needed for public collections).
- **Test on 28 Oct 2026:** `getSeries?Collection=TCGA-LIHC` returned **1,688 series**.
- **Fields returned:** BodyPartExamined, Collection, DateReleased, FileSize, ImageCount, Manufacturer, ManufacturerModelName, Modality, PatientID, ProtocolName, SeriesDate, SeriesDescription, SeriesInstanceUID, SeriesNumber, SoftwareVersions, StudyDate, StudyDesc, StudyInstanceUID, and license fields.
- **Not returned:** slice thickness and reconstruction kernel are not separate fields. They often appear in SeriesDescription or ProtocolName text, so the notebooks parse them. ImageCount approximates the slice count.

**Source 2: labels.**
- **TCGA collections:** use the GDC API (standardized clinical and molecular fields).
- **Other collections:** use their own clinical spreadsheets.
- **To be confirmed per collection in notebook 01.**

**Candidate tasks** (to verify: label availability, counts, licenses). There are two kinds:

- **(a) Labels inside one collection:**
  - TCGA-LIHC (grade, stage);
  - TCGA-BRCA and Duke-Breast-Cancer-MRI (molecular subtype);
  - LIDC-IDRI (malignancy rating);
  - NSCLC-Radiogenomics (histology, mutations);
  - PROSTATEx (clinical significance);
  - UPENN-GBM (molecular markers);
  - CPTAC collections;
  - HCC-TACE-Seg.
- **(b) Labels defined by which collection a scan came from** (a classic pitfall: the class equals the source collection):
  - glioma grade: TCGA-GBM vs TCGA-LGG. **Likely unavailable:** a manual API check on 28 Oct 2026 found neither collection on the public TCIA API, and IDC holds only pathology and annotation series for them, with no CT or MR. Notebook 01 records this in `candidates_missing.csv` and `idc_availability.csv`;
  - renal subtype: TCGA-KIRC vs TCGA-KIRP vs TCGA-KICH;
  - lung histology: TCGA-LUAD vs TCGA-LUSC.

**Data use:** cite every collection's DOI as TCIA requires, and follow each license.

## 7. Methods (draft)

- **Unit of analysis:** the patient. Leakage-safe means one patient is never in both training and test.
- **Features per patient:**
  - manufacturer, model and software version;
  - number of series;
  - slice count (ImageCount) of the main series, with a selection rule fixed in advance;
  - tokens parsed from SeriesDescription and ProtocolName (e.g., "1.25mm", "B30f", "STANDARD");
  - study year;
  - submitting site (TCGA barcode).
- **Models:**
  - logistic regression and gradient boosting (LightGBM) on metadata only;
  - a "single field" baseline per field group.
- **Validation:**
  - repeated stratified cross-validation (5 × 5), patient-level;
  - site-grouped cross-validation where the site is known.
- **Statistics:**
  - AUC or macro-AUC with bootstrap 95% CIs;
  - permutation tests against chance;
  - Benjamini–Hochberg correction across collections;
  - every analysis fixed in advance in this file before it is run, and post hoc analyses labeled as such.
- **Within-site analysis** (RQ3): repeat the audit inside the largest submitting sites.
- **Field importance** (RQ2): permutation importance and leave-one-field-group-out.
- **Image-model demonstration** (RQ4, Colab GPU):
  - pick 1–2 flagged tasks and download only those images;
  - train a standard CNN;
  - compare its internal score with the score on an acquisition-balanced or acquisition-matched test set;
  - add a counterfactual test, as in an earlier study.
- **Literature check** (RQ5): a fixed search for papers using each flagged collection; record whether acquisition was reported or controlled.

### 7a. Design rules (fixed 28 Oct 2026, before any data was pulled)

These rules close gaps found in the first review of this plan. They must not be changed after metadata–label modelling starts (notebook 03). Any later change is logged as post hoc.

1. **Which tasks count (the RQ1 denominator).** A task enters the audit only if all of these hold:
   - the collection is public on TCIA and contains CT or MR;
   - the label is patient-level and comes from the GDC API or the collection's official clinical file;
   - each class has at least **20 patients with imaging**.

   Every candidate from notebook 01 is listed with its reason for inclusion or exclusion. The minimum-per-class threshold may be reset once, after notebook 01 and before notebook 03, and the change is logged. Within-collection tasks (6a) and collection-defined tasks (6b) are reported separately and never pooled into one prevalence number.
2. **What counts as a flag.** A task is flagged only if both hold:
   - the lower bound of the 95% bootstrap CI of the metadata-only AUC (macro-AUC for more than two classes) is **≥ 0.60**;
   - its Benjamini–Hochberg-adjusted permutation p-value is **< 0.05**.

   All tasks are reported, flagged or not.
3. **Features that are never used** because they leak the label or the source:
   - `Collection`, `CollectionURI`, `DateReleased`, `TimeStamp` (TCIA ingest time), every license field;
   - `PatientID`, `PatientName`, `StudyInstanceUID`, `SeriesInstanceUID`;
   - `AdmittingDiagnosesDescription`, a study-level field that can name the diagnosis;
   - the demographic fields `PatientAge`, `PatientSex` and `EthnicGroup`, which are not acquisition;
   - the TCGA tissue-source-site code. The site is used only for site-grouped validation and within-site analysis (RQ3). It is never an input to the acquisition-only model.

   Tokens from `SeriesDescription`, `ProtocolName` and `StudyDesc` pass through a label-word blocklist before use. The blocklist holds class names, diagnoses, "tumor", "mets", "grade" and similar words. The 50 most frequent tokens per collection are checked by hand, and both the list and the review are saved. `FileSize` is used in a sensitivity analysis only, not in the primary model.
4. **Two kinds of signal, kept apart.**
   - **Scanner/protocol fields (primary audit):** manufacturer, model, software version, slice count of the main series, and slice thickness and kernel parsed from text.
   - **Care-pathway fields (reported separately):** number of studies, number of series, modalities present, and time between studies. These can differ by disease for legitimate clinical reasons.

   Metadata results are called **"acquisition–label associations"**. The word **"shortcut"** is used only where RQ4 shows a model actually exploits the signal.
5. **Dates.** TCIA de-identification shifts dates in many collections. Study year is used only for collections whose dates are judged real (from the collection's TCIA page and a plausibility check in notebook 01). Otherwise it is dropped and the drop is logged. Some collections (e.g., the CPTAC ones) carry `LongitudinalTemporalOffsetFromEvent`, the days from diagnosis. That timing needs no real dates and counts as a care-pathway field (7a.4).
6. **Small samples.** Every result reports n per class. CIs are bootstrapped at the patient level, and wide CIs are reported as they are, never hidden.
7. **Main-series rule** (implemented in notebook 01 and frozen there):
   - the rule runs separately for CT and for MR. Each collection's **primary modality** is the one with more patients; a tie goes to CT;
   - per patient, take the **earliest study** of that modality;
   - within it, take the eligible series with the **most images** (ties: lowest SeriesNumber, then UID);
   - a series is **not eligible** if any of these holds:
     - it has fewer than **10 images**;
     - TCIA marks it `ThirdPartyAnalysis = yes`;
     - its SeriesDescription matches scout, localizer, "loc", topogram, survey, calibration, bolus tracking, dose report, screen save, secondary capture, derived, MPR/MIP reformat or fusion;
     - **for CT only**, its SeriesDescription marks a sagittal or coronal reformat.

     ProtocolName is matched only when SeriesDescription is empty. The exact patterns are in notebook 01;
   - if the earliest study has no eligible series, use the next study and record this.
8. **Recommended (Hamza's choice):** register sections 4, 7 and 7a on OSF before notebook 03, so that "fixed in advance" can be shown with a timestamp.
9. **Decisions after notebook 01** (28 Oct 2026, implemented and frozen in notebook 02):
   - **Dates (7a.5):** no collection's dates could be judged real. Every TCGA/CPTAC collection with a reference looks shifted, and the others have constant or unverifiable years. **Study year is never a feature.**
   - **Task list:** one label table built from these sources:
     - GDC grade and stage for every TCGA/CPTAC collection;
     - the collection-defined tasks: LUAD vs LUSC, and KIRC vs KIRP vs KICH plus its pairs;
     - Duke: molecular subtype, ER, PR, HER2;
     - UPENN-GBM: MGMT, IDH1 (baseline `_11` rows);
     - LIDC-IDRI: confirmed diagnosis, benign vs malignant and primary vs metastatic, for the 157 patients with diagnosis data;
     - PROSTATEx: patient-level clinical significance (training set);
     - NSCLC Radiogenomics: histology, EGFR, KRAS;
     - HCC-TACE-Seg: TNM stage.

     Rule 7a.1 then decides eligibility.
   - **Free-text words (7a.3, 7a.4):** every word is sorted into one of five groups:
     - label words: blocked;
     - hospital names: blocked;
     - processed-image markers: blocked;
     - care-pathway words (biopsy, screening, follow-up, outside reads ...): kept apart;
     - protocol words: used.

     The lists are in notebook 02 and were reviewed by hand from the 50 most frequent words per collection.
   - **Care-pathway counts** include acquisitions only. Annotation series (SEG, RTSTRUCT, SR ...) are left out.
   - **Site (RQ3):** a TCGA tissue-source-site code names one institution *for one cancer type*. Codes are therefore mapped to institutions with GDC's code table, and the within-site analysis of collection-defined tasks uses the institution. For NSCLC Radiogenomics, the site is the clinical file's "Patient affiliation" (Stanford or VA).
   - **Still open:** LIDC-IDRI's radiologist malignancy rating, the label most published work uses, needs the XML annotation files (8.6 MB). It will be added in a later notebook, with its rule fixed before notebook 03.

### 7b. Notebook 03 analysis, fixed on 28 Oct 2026 before it was run on real labels

The code was tested only on **label-shuffled data**, so no real result was seen before this section was fixed.

1. **Tasks:** the 26 eligible tasks in `results/02_tasks_features/tasks_final.csv` (notebook 02 run of 15:29:58 UTC) form the multiple-testing family. A task added later (e.g., the LIDC radiologist rating) is reported with BH recomputed over all tasks, and with the original family as well.
2. **Inputs per patient:** the main series of the task's modality.
3. **Feature groups:**
   - **Scanner/protocol groups (primary model):**
     1. scanner: vendor and model;
     2. software version;
     3. image count (log);
     4. slice thickness and kernel parsed from text;
     5. body part;
     6. description words: protocol words only, kept if used by ≥ 3 training patients.
   - **Care-pathway group (separate model):** numbers of studies and series, CT/MR/PET counts, number of modalities, study rank used, days from diagnosis where given, and care-pathway words.
   - **Never used:** site, demographics, dates, and blocked words.
   - **Encoding:** done inside each training fold (one-hot with categories seen ≥ 3 times; missing values become their own category or are median-imputed with an indicator).
4. **Primary model:** L2 logistic regression, C = 1, balanced class weights, with fixed settings and no tuning. Multinomial for more than two classes.
5. **Primary estimate:** 5 × 5 repeated stratified cross-validation at patient level (seed 2026).
   - **AUC:** the mean over the 5 repeats of the pooled out-of-fold AUC. Macro one-vs-rest AUC for more than two classes.
   - **95% CI:** 2,000 patient-level bootstrap resamples of the out-of-fold predictions (percentile).
6. **Test against chance:** a label permutation test with 999 permutations.
   - Each permutation refits the primary model on **one fixed 5-fold split that does not depend on the labels** (KFold, seed 2026).
   - The fold encodings are fitted once, because they use no labels.
   - The observed statistic is computed the same way, so the test is exact.
   - p = (1 + number of permutations ≥ observed) / 1000. BH correction across the family gives q.
   - *(Changed on 28 Oct 2026 from "stratified split, whole pipeline rerun" after a speed test on shuffled labels, before any real result. The new version is about 10× faster and exactly valid.)*
7. **Flag (rule 7a.2):** the CI lower bound ≥ 0.60 **and** q < 0.05, primary model only.
8. **Secondary analyses** (estimates with CIs only, no tests):
   - **(S1)** LightGBM with fixed settings (200 trees, learning rate 0.05, 15 leaves, at least 10 patients per leaf, row and column sampling 0.8, balanced class weights);
   - **(S2)** the primary model plus file size;
   - **(S3)** the care-pathway model alone, and scanner/protocol plus care pathway;
   - **(S4)** the null-distribution mean from the permutations, as a check that the pipeline does not leak (it should be ≈ 0.5).
9. **Which fields carry the signal (RQ2):**
   - each scanner/protocol group alone;
   - the AUC drop when each group is left out;
   - the AUC drop when one group's inputs are shuffled in the test folds (5 shuffles per fold).

   All of these use the primary model and the 5 × 5 CV.
10. **Site or protocol (RQ3)**, for tasks with known sites (TCGA institution; NSCLC Radiogenomics affiliation):
    - **(a) Site-only baseline:** logistic regression on the one-hot site.
    - **(b) Site-held-out CV:** each fold holds out whole institutions.
    - **(c) Within-site AUC (main RQ3 statistic):** from the primary out-of-fold scores, the share of positive–negative pairs *from the same institution* that the model orders correctly. A score driven only by the site gives ≈ 0.5. The CI comes from bootstrapping patients within each site.
    - **(d) Refit inside each site** with ≥ 10 patients per class (5 × 5 CV).
11. **Checks done on shuffled labels before the real run (28–29 Oct 2026):**
    - **Permutation test calibration:** two tasks (HCC-TACE-Seg stage, TCGA-LIHC grade), 20 fresh label shuffles each. The p-values spread over 0.08–0.98, none ≤ 0.05, and the null mean AUC was ≈ 0.49.
    - **Spread of AUC under no signal:** NSCLC Radiogenomics histology, 20 shuffles. Mean 0.53 (LR) and 0.53 (LightGBM), SD ≈ 0.06, and 15% of shuffles reached ≥ 0.60.

    **What this means for reading results:**
    - a point AUC is not evidence on its own; the flag needs the CI lower bound **and** the exact permutation test (7b.7);
    - under no signal, cross-validated AUC can fall clearly *below* 0.5 (a known downward bias), so values below 0.5 mean "no signal", not "reversed signal";
    - refits inside single sites (d) with about 10 patients per class are very noisy and are only descriptive.

### 7c. Post hoc checks (notebook 04), decided on 29 Oct 2026 after the notebook 03 results were seen

**Everything in this section is post hoc** and is reported as such. The notebook 03 results stay the pre-specified results. These checks only test how robust they are. The word lists were fixed from the word list alone (patient counts per task, no labels). Exception: for the four flagged tasks, the class differences of the top words had already been seen (`00_LOG.md` section 5). None of the newly blocked words was among them.

0. **Reproduction:** notebook 04 first recomputes notebook 03's primary model on the real labels. The fixed-split AUC and every out-of-fold probability must match. The package versions are compared too.
1. **Full word review (completes rule 7a.3):**
   - every protocol or care-pathway word used by ≥ 3 patients in at least one task was reviewed (381 words), not only the top 50 per collection;
   - newly blocked:
     - label words "guz" (Polish: tumor), "spn" (solitary pulmonary nodule), "lym" (lymphoma), "plc" (primary liver cancer);
     - site word "spsk1" (a Polish hospital);
     - processing word "cad" (computer-aided detection output);
   - moved to care pathway: "diagnostic", and the outside-study import words "cd", "forfile", "import", "navigation";
   - the primary analysis (7b.4–7b.7) is **rerun** with the corrected list: AUC, CI, permutation test, BH over the 26 tasks, and flags by rule 7a.2.
2. **Word meaning (RQ2 / 7a.4):** each kept protocol word is sorted by meaning into one class. Single letters and numbers are classed only where their raw-text use is clear (e.g., "a"/"p" from "A/P", "c"/"i" from "+C"/"I+").
   - **Clinical choice:**
     - body coverage;
     - contrast and timing;
     - purpose of the exam (e.g., PE protocol, PET/CT).
   - **Technique:** thickness, kernel, reconstruction, sequence, scanner settings, patient position.
   - **Unclear:** everything else.

   With the corrected list, two models are added:
   - **(i) no clinical-choice words:** the primary model with only technique and unclear words;
   - **(ii) strict technical:** vendor/model, software version, slice thickness and kernel, and technique words only (no body part, no image count).

   Each gets AUC, CI, permutation test, its own BH and the 7a.2 flag rule. Each word class alone gets AUC and CI.
3. **Chance test inside groups (RQ3):** the within-site AUC of 7b.10(c) gets an exact test.
   - **Method:** labels are permuted only within each group, 999 times, on notebook 03's inputs and model, with the fixed label-independent 5-fold split of 7b.6. p is one-sided (signal beyond the group); BH is applied within each grouping.
   - **Groupings:**
     - the institution for the 12 site tasks;
     - **the description language (Polish or other)** for the 5 CPTAC tasks. GDC has no tissue-source site or country for CPTAC-3 cases (checked 29 Oct 2026), so the language serves as a proxy for the contributing site.
   - **Reported alongside:** the 5 × 5 within-group AUC with its CI; for the site tasks, it must equal notebook 03.

### 7d. RQ4 image-model check (notebook 05), approved on 29 Oct 2026

Written after the notebook 03 and 04 results were seen and **before any image was downloaded**. Approved by Hamza on 29 Oct 2026 ("Yes, write 05"). Details settled while building the notebook, still before any image was downloaded, are listed under "Settled while building" at the end of this section. Changes after the real-label run are logged as post hoc.

1. **Tasks: all four flagged tasks**, so that no task is chosen by its result.

   | Task | Patients | Main-series DICOM size |
   |---|---|---|
   | LUAD vs LUSC | 104 (67 / 37) | 8.0 GB |
   | KIRC vs KIRP | 259 (236 / 23) | 21.4 GB |
   | CPTAC-PDA stage | 68 (47 / 21) | 10.0 GB |
   | LIDC primary vs metastatic | 94 (51 / 43) | 10.5 GB |

   The sizes come from TCIA's FileSize (`series_metadata.csv.gz`); about 50 GB in total.
2. **Images:**
   - the same main series the metadata audit used (7a.7);
   - downloaded from TCIA inside Colab to the runtime's own disk (not to Drive) and converted to HU volumes;
   - only small derived files go to Drive: embeddings, a DICOM-header table, and compressed resampled volumes for reruns (estimated 5–10 GB; Hamza to confirm).
3. **The image model, "the usual way":** a public CT foundation model used as a frozen feature extractor, with the notebook 03 logistic regression on top (a linear probe). This replaces "train a standard CNN" (section 7) because:
   - it is the usual practice now for datasets of 68–259 patients, where a CNN trained from scratch is unstable;
   - it is deterministic;
   - it allows the same exact permutation test as notebook 03.

   Encoders:
   - **main: CT-FM** (Pai et al., 2025; Apache-2.0; 512 features). It was self-supervised on 148,000 CT scans from the Imaging Data Commons, which includes TCIA collections. It may therefore have seen these scans, but never their labels. This is stated in the paper;
   - **second: Merlin** (Stanford; MIT license), trained on abdominal CT with reports from outside TCIA.

   Each encoder is used with its own documented preprocessing. Statistics as in 7b:
   - 5 × 5 CV;
   - bootstrap CI;
   - 999-permutation test on the fixed split;
   - BH over the four tasks per encoder;
   - the flag rule 7a.2.
4. **Acquisition-controlled evaluation of the same image scores:**
   - **(a) within-group AUC** with the within-group permutation test of 7c.3. Groups:
     - the institution for the TCGA tasks;
     - the description language for CPTAC-PDA;
     - the scanner vendor for LIDC, which has no site code;
   - **(b) institution-held-out CV** for the TCGA tasks, as in 7b.10(b);
   - **(c) beyond metadata:**
     - the AUC of metadata + image against metadata alone (paired bootstrap difference);
     - within each class, the rank correlation between the out-of-fold image score and metadata score.
5. **Counterfactual tests (inputs with the biology removed):**
   - **(a) background only:** every voxel inside the body is set to air, leaving the table, field of view and noise. The body mask is HU > −500, largest component, holes filled, dilated by 5 mm. Same probe and statistics.
     - The body outline still shows body size, so body size alone (mean body cross-section from the mask) gets its own AUC as a control.
   - **(b) acquisition swap** (as in an earlier study): the frozen probe scores each scan again after its slice spacing is resampled to the other class's median. A matched in-plane smoothing serves as the control. The rule, fixed before the run: shift ≥ 0.05, CI excludes 0, in both directions, and larger than the control.
6. **DICOM-header audit (secondary, RQ2):** the downloaded headers give fields the API lacks:
   - true slice thickness and spacing;
   - kernel, kVp and tube current;
   - reconstruction diameter;
   - contrast agent.

   The notebook 03 primary model is rerun with these fields added, to see whether the API-only audit underestimates the signal.
7. **Wording rule (7a.4):** "shortcut" is used for a task only if both hold:
   - the image probe is flagged (7a.2);
   - and the background-only probe is flagged too, or the acquisition swap meets its rule.

   Otherwise the result is called an "acquisition–label association". The within-group and held-out results are reported as estimates.
8. **Before the real run:**
   - the code is tested locally on 2–3 small series (C: has about 4 GB free) and with shuffled labels;
   - in Colab, the first pass runs on shuffled labels as a rehearsal; the real-label pass follows.

**Settled while building (29 Oct 2026, before any image was downloaded or any image result seen):**

1. **CT-FM input:** the authors' own embedding script (`scripts/3d_mir_generate_embeddings.py` in github.com/project-lighter/CT-FM):
   - 3 × 1 × 1 mm spacing;
   - foreground crop;
   - 24 × 128 × 128 patches with 0.625 overlap;
   - mean of the patch features.

   This replaces the model card's whole-volume example, because it matches the pretraining spacing (3 × 1 × 1 mm) and fits in GPU memory. The weights come from a pinned Hugging Face revision.
2. **Merlin input:** the package's own image transforms. The image encoder is loaded from the official checkpoint with strict loading, without the text half. It is checked once against the official `Merlin(ImageEmbedding=True)`. Gradient checkpointing, a memory saver for training, is switched off; this does not change the output.
3. **Background only, CT-FM:** the foreground crop box is taken from the original scan, so both versions cover the same field. CT-FM's crop keeps voxels above 0 HU, so on its own it would crop a background-only scan down to the table.
4. **Acquisition swap: one direction only (thin to thick).** This replaces "in both directions" in item 5(b).
   - **Why:** a thick scan cannot be made thin, because missing detail cannot be added. Both encoders also resample to 3 mm, so the reverse direction would change almost nothing by construction, and a rule that needs both directions could never be met.
   - **How:** the class with the thinner median slice spacing (from the DICOM headers) is slab-averaged to the other class's median spacing.
   - **When:** only if the two medians differ by ≥ 0.5 mm. Only scans thinner than the target by more than 0.1 mm are swapped.
   - **Control:** in-plane Gaussian smoothing. Its sigma is chosen per patient so that the mean absolute HU change inside the body equals the swap's.
   - **Scoring:** the fold models trained on the original scans score both the swapped and the control scans.
   - **Rule:** the mean shift toward the thick class is ≥ 0.05 with a CI that excludes 0, and the CI of the swap − control difference lies above 0.
5. **Storage:** no scan volumes are stored on Drive, only headers, embeddings, QC pictures and tables (well under 1 GB). Scans are downloaded again when needed. This replaces the 5–10 GB volume cache of item 2.
6. **Header-audit fields:**
   - **used:** slice thickness, slice spacing, pixel spacing, reconstruction diameter, kVp, tube current, exposure, CTDIvol, pitch, kernel, filter type and patient position;
   - **never used:** site, station name, device serial number, demographics, dates;
   - **reported by class but not used:** contrast agent, because it is a clinical choice (7c).
7. **One "Run all":** shuffled labels first, then an automatic gate, then the real labels. The gate requires that:
   - every analysis ran;
   - every null mean AUC lies between 0.35 and 0.55. The upper bound catches leakage. The lower bound is loose because, under no signal, cross-validated AUC sits below 0.5 (7b.11), and more so with 512–2,048 embedding features: a code test on random stand-in embeddings gave null means of 0.44–0.47 for Merlin;
   - at least 95% of each task's patients have embeddings.
8. **Missing scans:** scans that cannot be downloaded or read are excluded and listed. All image analyses use the same patients, and the metadata comparison is recomputed on those same patients.
9. **Body-size control:** the mean body cross-section (cm²) of the undilated body mask.
10. **Within-class rank correlation:** computed from the repeat-averaged out-of-fold probabilities, with a bootstrap CI within each class.

### 7e. RQ5 literature check, fixed on 29 Oct 2026 before any search was run

**Question:** studies that predicted a flagged task's label from that collection's images: did they report or control for acquisition, and how do their reported scores compare with our metadata-only AUC?

1. **Target studies:** original research (journal, conference or preprint) in any language with an English abstract. A study counts if it trains or evaluates a model, radiomics or deep learning, that predicts the flagged label from images of the flagged collection(s):
   - **LUAD vs LUSC:** lung adenocarcinoma vs squamous cell carcinoma on CT, using TCGA-LUAD and TCGA-LUSC images from TCIA, alone or pooled with other data;
   - **KIRC vs KIRP:** renal cell carcinoma subtype on CT (clear cell vs papillary, with or without chromophobe), using TCGA-KIRC, TCGA-KIRP or TCGA-KICH images;
   - **CPTAC-PDA stage:** any stage (or TNM) prediction from CPTAC-PDA images;
   - **LIDC primary vs metastatic:** LIDC-IDRI's patient-level diagnosis data (primary vs metastatic, or benign vs malignant).

   Not counted: reviews, editorials, theses, studies on other data, and genomics-only or pathology-only studies.
2. **Sources and dates:**
   - Europe PMC (titles, abstracts and open full texts; includes PubMed);
   - PubMed;
   - arXiv;
   - Semantic Scholar (conferences and preprints);
   - plus the reference lists of included studies (one round).

   Every query string, date and hit count goes into `literature/RQ5/search_log.md`.
3. **Screening:**
   - titles and abstracts first, then the full text;
   - every record keeps its decision and reason in `screening.csv`;
   - duplicates are merged by DOI, then by title;
   - every decision and its reason recorded for checking.
4. **Extraction per included study** (`extraction.csv`):
   - year, venue, collections used, patients per class, method, and validation (split, CV, external);
   - best reported AUC (or accuracy) on the flagged TCIA data;
   - **acquisition level**, the highest that applies:
     - **A0:** not mentioned;
     - **A1:** some acquisition parameters described;
     - **A2:** acquisition compared between the classes;
     - **A3:** a design that controls for it: site- or scanner-held-out testing, matching or stratifying by site or scanner, harmonization by scanner (e.g., ComBat), an acquisition-only baseline, or acquisition covariates;
   - recorded separately, but not counted as control: resampling or intensity normalization;
   - site or source confounding named as a limitation: yes or no;
   - external test on an independent institution: yes or no.

   If a full text is not freely available, the study is extracted from its abstract and marked so.
5. **Reporting (descriptive only):**
   - per task: the number of studies, the share at each acquisition level, the share with external testing;
   - their reported AUCs next to our metadata-only AUC for the same task (notebook 03), marked as different patients and different splits.

### 7f. Post hoc image checks (notebook 06), approved on 29 Oct 2026 after the notebook 05 results were seen

**Status:** proposed on 29 Oct 2026; **approved by Hamza on 29 Oct 2026** ("Approve and upload"), before notebook 06 was run. Every result is labeled **post hoc**. Nothing here changes a notebook 05 number, flag or the 7d.7 wording; these checks only explain them.

**Why.** Notebook 05 allowed "shortcut" for 2 of 8 task–encoder pairs (LIDC with CT-FM; LUAD vs LUSC with Merlin), and in both the background-only probe was flagged. A reviewer will ask three things:
- Does the background signal survive inside one hospital or vendor, and with whole hospitals held out? Notebook 05 ran these only for the full image.
- Is it just the body outline? The background-only scan still shows where the body was (the blanked region). Body size alone was weakly informative for LUAD vs LUSC (AUC 0.42, CI 0.34–0.49, i.e. about 0.58 the other way).
- Is it the same acquisition that the DICOM header records?

**Checks.** All four tasks and both encoders, with the probe, CV, bootstrap, permutation tests and BH of 7d.

1. **Background-only under acquisition control** (notebook 05's cached embeddings; nothing is downloaded):
   - **(a) within-group AUC** with the within-group permutation test, groups as in 7d.4(a); BH over the four tasks per encoder;
   - **(b) group-held-out AUC:** whole hospitals for the TCGA tasks, as in 7d.4(b). For LIDC (two vendors) and CPTAC-PDA (two description languages): train on one group, test on the other. For these two tasks this is also run for the full-image probe, which notebook 05 did not do.
2. **Body-outline control ("silhouette"):**
   - the scans are downloaded again, with notebook 05's code and settings;
   - the silhouette is exactly the region the background version blanked (the dilated body mask), filled with 0 HU; everything else is −1000 HU, so there is no table, noise or field-of-view edge;
   - CT-FM uses the original scan's crop box, as the background version did;
   - same probe and statistics as the background probe, the flag rule 7a.2, and the within-group and held-out checks of check 1;
   - a paired bootstrap difference: background AUC minus silhouette AUC.

   **Reading rule, fixed now:** if the silhouette probe is flagged where the background probe was, that background result is reported as "may reflect the body outline or positioning". If the silhouette is not flagged and the background-minus-silhouette CI excludes 0, the background signal is reported as "beyond the body outline". Otherwise it is "unresolved".
3. **Background versus the DICOM header** (the 7d.6 header fields; estimates only, no flags):
   - **(a)** header-only model versus header + background embedding: paired bootstrap difference ("added by the background beyond the header");
   - **(b)** within each class, the rank correlation between the out-of-fold background score and the header-only score, with a bootstrap CI (as 7d.4(c));
   - **(c)** descriptive: within each class, the Spearman correlation between the background score and each numeric header field, and a Kruskal–Wallis test for kernel, filter type and patient position.
4. **Re-embedding check:** the original scans are embedded again in the new session. The largest difference from notebook 05's cached embeddings is reported, since GPU arithmetic can differ slightly between sessions.

**How it runs:** as in 7d. One "Run all": a shuffled-label rehearsal with the same automatic gate, then the real labels. A GPU is needed for check 2 only. About 50 GB are downloaded again into Colab's own disk; only embeddings, QC pictures and tables go to Drive (`results/06_image_posthoc/`).

**Not done:** no new tasks, encoders or labels, and no change to notebook 05's flags or wording.

**Notebook numbering:** this becomes notebook 06; figures, tables and the audit tool move to notebook 07.

**Settled while building (29 Oct 2026, before any notebook 06 result):**
1. **Reproduction of notebook 05:** the notebook first recomputes notebook 05's image and background probes on the same patients, labels and splits. They count as reproduced if every AUC is within 0.001 of notebook 05's, and every difference is saved. Exact equality is not required: with up to 2,048 features the logistic regression's solution differs in the last digits between machines. A local test on notebook 05's rehearsal gave AUC differences of 0 for three tasks and 0.00004 for KIRC vs KIRP (23 against 236 patients). The tables report notebook 06's recomputed AUCs; the paper quotes notebook 05's.
2. **Mask check:** for every scan, the slice count and the dilated body fraction must equal notebook 05's recorded values; otherwise the notebook warns.
3. **Shuffled labels in the rehearsal:** the same seed as notebook 05 (999), so its rehearsal can be compared number for number.
4. **Noted on 29 Oct 2026, while testing notebook 07, before any notebook 06 result was read:** when the body is cut off by the reconstruction field of view, the body mask is the whole field-of-view circle. The background-only version is then nearly empty, and the silhouette shows the field-of-view circle, an acquisition setting, rather than the body outline.
   - This happened in a local QC picture (TCGA-UCEC, 250 mm reconstruction diameter).
   - It applies to 9 of notebook 05's 525 scans (1.7%), whose dilated body mask covers ≥ 75% of the image. Counted overall only, not by class.
   - The silhouette results will be read with this in mind.

### 7g. Reframed story and three new experiments, approved on 29 Oct 2026

**Status:** proposed on 29 Oct 2026 at Hamza's request ("can we reframe the story … if that take more experiments that okay?"; "Yes, write it"). **Approved by Hamza on 29 Oct 2026** ("Approve 7g"), before any data for experiments A–C was downloaded.
- **Timing:** written after the notebook 03–05 results were seen, while notebook 06 runs, and before any data for experiments A–C was downloaded or analyzed.
- **Labeling:** the paper labels A–C as added after the first results.
- **Nothing earlier changes:** notebooks 01–06 and sections 7a–7f stay as they are, including their negative results.

**Why reframe.** The archive-wide prevalence is modest (4 of 26 tasks flagged), so "how common are shortcuts?" makes a weak headline. Our strongest result is different: for two benchmarks, a CT model predicts the diagnosis **with the patient removed from the scan**. The reframed paper asks whether a cheap metadata screen finds such benchmarks, and what they do to generalization.

**Reframed question:** *Can public cancer CT benchmarks be solved without seeing the patient? A no-download screen for acquisition shortcuts, validated with images, and its consequence for generalization.*
**Working title (proposal):** *Diagnosis without the patient: screening public cancer CT benchmarks for acquisition shortcuts*

**The story in five steps:**
1. **Screen** (notebook 03, done): metadata alone flags 4 of 26 tasks, with no download.
2. **Validation** (experiment A): patient-free images predict the label where the screen says so, and not elsewhere.
3. **Mechanism** (notebooks 05–06, experiment C): the embeddings carry acquisition, even without the patient.
4. **Consequence** (experiment B): a model trained on a flagged benchmark loses its performance on an independent dataset that asks the same clinical question.
5. **Practice** (RQ5, the tool, the checklist): a "patient-free baseline" recommended for every benchmark.

**New research questions** (RQ1–RQ5 stay):
- **RQ6 (screen validity):** across the audit's CT tasks, does the notebook 03 metadata flag predict where patient-free images predict the label?
- **RQ7 (consequence):** do models trained on a flagged benchmark keep their performance on an independent dataset with the same clinical question?
- **RQ8 (mechanism, label-free):** do CT foundation-model embeddings encode acquisition parameters, even with the patient removed?

**Experiment A: does the screen work? (notebook 07)**
1. **Tasks, by rule:** every CT task of the notebook 03 audit that was **not** flagged, 15 tasks:
   - LIDC benign vs malignant;
   - TCGA-LIHC grade and stage;
   - TCGA-BLCA stage;
   - TCGA-KIRC stage and grade;
   - NSCLC Radiogenomics histology, EGFR and KRAS;
   - TCGA-UCEC grade;
   - HCC-TACE-Seg stage;
   - CPTAC-CCRCC grade and stage;
   - CPTAC-UCEC grade;
   - CPTAC-PDA grade (the same scans as the flagged CPTAC-PDA stage task).

   The 7 MR tasks are left out because both encoders are CT models; this is stated as a limitation.
2. **Images:** notebook 03's patients and the 7a.7 main series.
   - 725 of these patients are not yet downloaded (91.7 GB, from TCIA's FileSize).
   - The rest (TCGA-KIRC, CPTAC-PDA, part of LIDC) reuse notebook 05's cached embeddings.
3. **Per task and encoder:**
   - the full-image probe, the background-only probe (notebook 05's) and the silhouette probe (notebook 06's);
   - the same CV, bootstrap CI, permutation test and flag rule 7a.2;
   - BH over the 15 tasks, per encoder and input.
4. **Screen validation, over all 19 CT tasks** (the 4 flagged tasks' numbers are read from notebooks 05 and 06 as they ran):
   - **(a) primary:** the Spearman correlation between notebook 03's metadata AUC and the background-only AUC, for CT-FM (the main encoder, 7d.3). One-sided test by 99,999 random permutations of the task labels. Merlin is secondary.
   - **(b)** a concordance table of the metadata flag against the background-only flag, for each encoder. Any unflagged task whose background-only probe is flagged is named as **"missed by the metadata screen"**.
   - **(c)** the same for the full-image probe (descriptive).
   - **Reading rule, fixed now:**
     - "the screen is supported" if (a) has rho > 0 with p < 0.05 and no task is missed;
     - "supported, with misses" if (a) holds but a task is missed;
     - otherwise "not supported".
5. **Run:** shuffled-label rehearsal with the same gate as 7d, then the real labels.

**Experiment B: does a flagged benchmark generalize? (notebook 08)**
1. **Pairs, by clinical question.** These are the only flagged tasks for which TCIA holds an independent dataset with the same label:
   - **Lung histology.** The source is TCGA-LUAD vs TCGA-LUSC (flagged). There are two targets:
     - NSCLC Radiogenomics, adenocarcinoma vs squamous (199 patients; not flagged, metadata AUC 0.60);
     - CPTAC-LUAD vs CPTAC-LSCC (74 patients with a main series, 38 / 36; 18.6 GB). Here, too, each class is its own collection: a second benchmark built the same confounded way, from other hospitals.
   - **Renal subtype.** The source is TCGA-KIRC vs TCGA-KIRP (flagged). The target is **C4KC-KiTS**, a single institution: clear cell (143) vs papillary (21).
     - Labels come from the public KiTS19 clinical file (`kits.json` in github.com/neheller/kits19, pinned to a commit). `case_00000` is TCIA `KiTS-00000`; this was checked on 29 Oct 2026 (210 patients).
     - Chromophobe and the other subtypes are left out, as in the source task.
     - The main series is chosen with notebook 01's frozen 7a.7 code; the collection's CT is 37.7 GB in all phases.
   - **Not possible:** LIDC primary vs metastatic and CPTAC-PDA stage have no independent dataset with the same label (stated).
2. **Per pair, encoder and input** (full image; background-only):
   - the probe (the same logistic regression) is trained on all source patients and applied to all target patients; this gives the **external AUC**, with a bootstrap CI over target patients;
   - it is compared with the **source internal AUC** (notebook 05's CV) and the **target internal AUC** (CV inside the target dataset; for NSCLC Radiogenomics this comes from experiment A). The target internal AUC is the unconfounded estimate of what the image carries for this question;
   - **transfer gap** = source internal AUC minus external AUC, with a bootstrap CI (source and target resampled independently);
   - secondary: the reverse direction (train on the target, test on the source), and the metadata model's transfer where notebook 02 features exist (NSCLC Radiogenomics, CPTAC lung).
   - **Reading rule, fixed now:**
     - "does not transfer" if the external AUC's CI includes 0.5;
     - "transfers with a loss" if it excludes 0.5 and the gap's CI excludes 0;
     - otherwise "transfers".
3. **Run:** a rehearsal with shuffled source labels (the external AUC should look like chance), then the real labels.

**Experiment C: what do the embeddings encode? (label-free; in notebook 07)**
- **Scans:** every scan embedded in notebooks 05 and 07 (about 1,250 patients), for each input: full image, background-only and silhouette.
- **Targets, from the DICOM headers:**
  - manufacturer;
  - slice spacing in three bins (< 2 mm, 2 to < 4 mm, ≥ 4 mm);
  - reconstruction diameter, above vs below the median;
  - kVp, 120 vs other;
  - kernel (values with ≥ 20 scans).

  Contrast agent is a clinical choice, so it is reported separately.
- **Model:** the same logistic regression with 5 × 5 CV. The AUC (one-vs-rest mean for more than two values) is reported pooled and **within collection**, counting only pairs from the same collection (as in 7c.3), so that collection identity alone cannot explain it. Descriptive only.

**Not part of this proposal (to decide later):**
- **D:** LIDC radiologist malignancy ratings;
- **E:** "patient-free radiomics";
- **F:** the MR tasks with an MRI foundation model.

**Notebooks and compute:**
- **07 (A + C):** 91.7 GB downloaded again into Colab's own disk; L4 or A100; about 5–7 hours; resumable across sessions.
- **08 (B):** about 30–40 GB (the CPTAC lung and C4KC-KiTS main series); about 2–3 hours.
- **09:** figures, tables and the audit tool (previously 07).

**Settled while building notebook 07 (29 Oct 2026, before any data for A or C was downloaded):**
1. **Flags in the screen validation:**
   - for the 4 flagged tasks, the flags come from notebook 05 (full image, background; BH over its 4 tasks) and notebook 06 (silhouette);
   - for the 15 new tasks, BH runs over those 15, per encoder and input.
2. **Metadata AUC:** the screen validation uses notebook 03's AUC (all patients). Notebook 07 also recomputes it on the patients with images, and checks that it is reproduced within 0.001 when no patient is missing.
3. **Rehearsal:** the 15 new tasks use notebook 05's shuffle seed (999). The 4 flagged tasks enter the rehearsal's screen validation with notebooks 05 and 06's rehearsal numbers.
4. **Experiment C:**
   - **Scans:** every scan with all three inputs (notebooks 05–07), run once, label-free, after experiment A's real pass.
   - **Manufacturer:** the first word of the header value, so "PHILIPS" and "PHILIPS MEDICAL SYSTEMS" count as one vendor.
   - **Kernel:** names are kept as written; they are vendor-specific.
   - **Chance check:** each decoding is repeated once with the target shuffled (seed 999).
5. **Shared patients:** several tasks share patients (one main series per patient), so a patient is downloaded and embedded once.
6. **Work disk:** a scan's volumes are deleted from Colab's disk once it is embedded; only the embeddings and header records are kept.

**Settled while building notebook 08 (30 Oct 2026, before any image or result of experiment B):** Only TCIA series metadata and the KiTS label file's class counts have been read.
1. **CPTAC lung target:** every patient with a CT main series in notebook 01: 73 patients (CPTAC-LUAD 37, CPTAC-LSCC 36), 18.5 GB. The 74 (38/36) above counted one CPTAC-LUAD patient whose only main series is MR. No patient is in both collections.
2. **C4KC-KiTS target:**
   - **Labels:** `data/kits.json` of github.com/neheller/kits19, pinned to commit `456b293435b25303bbc7995f094f83a0afdff0eb`, MD5 `46c6e3266379fb6b82875be401fa071a`. This is the same file as read on 29 Oct.
   - **Classes:** `clear_cell_rcc` (143) vs `papillary` (21). `clear_cell_papillary_rcc` (4) and every other subtype are left out.
   - **Main series:** notebook 01's rule 7a.7 code, copied unchanged into one function. The notebook checks the function by reproducing notebook 01's `main_series.csv` exactly (all 4,883 patient–modality pairs; this check passed locally on 30 Oct). TCIA's series list is cached.
   - **Result:** all 164 labelled patients have a main series, 20.3 GB. The rule picks the arterial phase for 124, noncontrast for 22 and late for 18. Each scan's phase is recorded.
3. **Class direction:** class 1 is squamous (TCGA-LUSC, NSCLC Radiogenomics `squamous`, CPTAC-LSCC) or papillary (TCGA-KIRP, KiTS `papillary`). The notebook checks that this agrees with notebook 05's class order.
4. **Images:**
   - preparation as notebook 07: original, background-only and silhouette are all embedded and cached;
   - experiment B analyses the full image and background-only, as planned. The silhouette is not analysed, because it carries scan geometry (log, section 15);
   - source embeddings come from notebook 05's cache, and NSCLC Radiogenomics from notebook 07's.
5. **Source internal AUC:** recomputed from notebook 05's cached embeddings with notebook 05's code. It must reproduce notebook 05 within 0.001. Its out-of-fold predictions are used for the gap.
6. **External AUC:**
   - one model (scaling + logistic regression, the same settings) is fitted on all source patients and applied to all target patients;
   - 95% CI by bootstrap over target patients (2,000 resamples; a resample without both classes is drawn again);
   - also a permutation test: the source labels are permuted 999 times and the model refitted. Its p is descriptive and is not part of the reading rule; its null mean feeds the gate.
   - **Score:** the model's log-odds, which ranks patients exactly as the probability does (same AUC). A local test with stand-in embeddings showed that when target embeddings lie far from the source's, probabilities can round to exactly 0 or 1, and every patient then ties.
7. **Transfer gap:** source internal AUC (repeat-averaged, out-of-fold) minus external AUC. Its CI comes from 2,000 bootstrap pairs, resampling source patients and target patients independently.
8. **Target internal AUC:** 5 × 5 CV, bootstrap CI and permutation test, as in notebook 05. For NSCLC Radiogenomics it must reproduce notebook 07's experiment A within 0.001.
9. **Reverse direction (secondary, descriptive):** a model fitted on all target patients is applied to all source patients, with AUC, CI over source patients and permutation p. Its gap is target internal minus reverse external.
10. **Metadata transfer (secondary, descriptive; the two lung pairs only):**
    - notebook 03's primary model (scanner fields, the same pipeline) is fitted on all source patients and applied to the target, with AUC and CI;
    - the target's internal metadata AUC is computed by the same CV;
    - C4KC-KiTS has no notebook 02 features, so it is not included.
11. **Rehearsal:**
    - source **and** target labels are shuffled (seed 999), so every number should look like chance;
    - the source internal AUCs must reproduce notebook 05's rehearsal, and the NSCLC Radiogenomics target internal AUCs must reproduce notebook 07's rehearsal.
12. **Gate:**
    - every analysis ran;
    - every permutation null mean (target internal, external, reverse) lies in 0.35–0.55;
    - at least 95% of each target's patients are embedded.
13. **Multiplicity:** the reading rule uses each 95% CI as fixed in 7g, with no correction across the 12 primary readings (3 pairs × 2 encoders × 2 inputs). This is stated in the paper.
14. **Size and time:** 237 new patients, about 38.8 GB. On an L4 or A100, expect about 1.5–2.5 hours.
15. **Edge cases of the reading rule, fixed before any data:**
    - **"Does not transfer"** if the external AUC's CI includes 0.5 **or lies wholly below it**. A CI wholly below 0.5 is also marked "reversed".
    - **"Transfers with a loss"** if the CI lies wholly above 0.5 and the gap's CI lies wholly above 0.
    - **"Transfers"** otherwise. This includes a gap CI wholly below 0, which means the external AUC is higher than the source's own.
16. **Last cell:** stops the Colab runtime after a successful run, as Hamza asked for notebook 07.

**If approved, also updated:**
- section 1 (the question and the working title);
- section 4 (RQ6–RQ8);
- section 5 (contributions);
- section 8 (two new figures: metadata AUC against patient-free AUC over the 19 tasks, and internal against external AUC);
- section 9 (the timeline).

### 7h. Why do the TCGA lung models transfer to CPTAC lung? Post hoc checks (notebook 09), proposed on 30 Oct 2026 after the notebook 08 results were seen

**Status:**
- proposed on 30 Oct 2026 at Hamza's request ("Yes, write it"), after the notebook 08 results were read;
- **approved by Hamza on 30 Oct 2026** ("Approve 7h"), before notebook 09 was written or run;
- every result is labeled **post hoc**; nothing here changes a notebook 08 number or reading.

**Why.** In notebook 08 the TCGA-trained background-only probes transferred to CPTAC lung (Merlin 0.739, CT-FM 0.677) but not to NSCLC Radiogenomics (log, section 17). The background-only input has no anatomy. It does still show:
- the empty body outline;
- the table;
- the field of view;
- the noise texture.

These carry patient size and acquisition settings (experiment C, log section 15). A reviewer will ask which of them carries the transfer.

**Data:**
- only the cached embeddings and header records of notebooks 05–08, so nothing is downloaded and no GPU is needed;
- datasets: TCGA lung (source, 104 patients), CPTAC lung (73), and NSCLC Radiogenomics (199) for contrast;
- the same probe, CI, permutation test and reading function as notebook 08 (7g, settled items 6, 7 and 15).

**Checks:**
1. **Silhouette transfer.**
   - The silhouette probe is fitted on TCGA lung (notebook 06's silhouettes) and applied to CPTAC and NSCLC Radiogenomics, for both encoders: external AUC, CI and permutation p.
   - Also the paired difference, background external AUC minus silhouette external AUC, by bootstrap over target patients.
   - **Reading rule, fixed now** (per encoder, CPTAC):
     - "may reflect the body outline or scan geometry" if the silhouette transfers (7g rule) where the background did;
     - "beyond the body outline" if the silhouette does not transfer and the difference's CI lies above 0;
     - otherwise "unresolved".
   - The silhouette also carries scan geometry (log, section 15), so the first reading cannot separate patient size from field of view.
2. **Header-only and body-size-only transfer.**
   - **Header model:** the 7d.6 header fields, with the same column transformer, fitted on TCGA lung and applied to CPTAC and NSCLC Radiogenomics. Each field's missing share per dataset is reported, because the imputer's missing-value indicators could otherwise act as dataset labels.
   - **Body-size model:** the body cross-section area alone.
   - **Reading rule, fixed now** (CPTAC):
     - if the header model transfers (7g rule), the background transfer is reported as "consistent with recorded acquisition settings";
     - if the body-size model transfers, as "consistent with patient size";
     - both can hold;
     - if neither transfers, as "not explained by the recorded header fields or body size".
3. **Which fields differ by class, in each lung dataset (descriptive):**
   - **numeric fields** (slice thickness, slice spacing, reconstruction diameter, kVp, tube current, body area): the AUC for the class (squamous as class 1) with a bootstrap CI, per dataset;
   - **categorical fields** (manufacturer as its first word, kernel, patient position, contrast): class proportions with a χ² p, uncorrected;
   - the table shows which fields point the same way in TCGA and CPTAC but not in NSCLC Radiogenomics.
4. **What the transferred score follows (descriptive):** within each CPTAC class, the Spearman correlation between the TCGA background model's score and each numeric header field, and Kruskal–Wallis for the categorical ones, as in 7f check 3(c).
5. **Not possible, stated as a limitation:**
   - CPTAC's contributing hospitals are not in our data (notebook 02 has sites for TCGA and NSCLC Radiogenomics only);
   - the DICOM institution fields are never read (7d.6);
   - so whether TCGA and CPTAC share hospitals cannot be tested.

**How it runs:**
- as a short notebook 09 on CPU, taking minutes;
- one "Run all": a shuffled-label rehearsal (seed 999, source and target) with the same gate as notebook 08, then the real labels;
- figures, tables and the audit tool move to notebook 10.

**Not done:**
- no new data, encoders or tasks;
- no change to notebook 08's readings;
- C4KC-KiTS is not included, since it shows no transfer to explain;
- all readings are uncorrected and descriptive.

### 7i. Robustness checks of the screen validation (notebook 10), added on 31 Oct 2026

**Status:**
- proposed on 30 Oct 2026 when Hamza asked which extra experiments are worth doing; **added at Hamza's request on 31 Oct 2026** ("Yes, add them");
- written before any of these checks was computed. The per-task AUCs they reuse were already seen (log, section 15), so every result is labeled **post hoc sensitivity analysis**;
- nothing here changes the primary RQ6 reading of 7g (CT-FM: "the screen is supported"). The paper reports these checks next to it.

**Why.** Log section 15 named two weaknesses of the RQ6 correlation (19 tasks, Spearman ρ 0.51 for CT-FM background):
1. it may lean on the 4 flagged tasks;
2. several tasks share patients, so they are not independent for the task-permutation test. The 19 tasks form **11 groups with no shared patients** (checked on 31 Oct 2026 from notebook 02's patient lists; no labels or results were used):
   - TCGA-KIRC grade, TCGA-KIRC stage and KIRC vs KIRP;
   - the three NSCLC Radiogenomics tasks;
   - LIDC benign vs malignant and LIDC primary vs metastatic;
   - CPTAC-PDA grade and stage;
   - CPTAC-CCRCC grade and stage;
   - TCGA-LIHC grade and stage;
   - five single tasks: LUAD vs LUSC, TCGA-BLCA stage, TCGA-UCEC grade, CPTAC-UCEC grade, HCC-TACE-Seg stage.

**Data:** only notebook 07's `screen_validation_tasks.csv` (notebook 03's metadata AUC and the background-only AUC per task and encoder) and notebook 02's patient lists. Nothing is refitted.

**Checks** (both encoders; CT-FM is the main one, as in 7g):
1. **Without the flagged tasks:** Spearman ρ between metadata AUC and background AUC over the 15 unflagged tasks. One-sided p by 99,999 random permutations, as in 7g A.4(a).
2. **Patient-disjoint groups:** each group's mean metadata AUC and mean background AUC (11 points). Same ρ and permutation p.
3. **Leave one task out** (descriptive): ρ over the 19 tasks with each task left out in turn. Reports the range and the task whose removal moves ρ most.

**Wording rule, fixed now:**
- **check 1:** "the correlation holds without the flagged tasks" if ρ > 0 and p < 0.05; otherwise "the correlation depends on the flagged tasks";
- **check 2:** "holds over patient-disjoint task groups" if ρ > 0 and p < 0.05; otherwise "not shown over patient-disjoint task groups";
- whatever these show, the paper states RQ6 as: the 7g reading, plus "no task was missed by the screen", plus these two sentences.

**Not done:**
- no new tasks, encoders or images;
- the end-to-end CNN, a third encoder, patient-free radiomics and the MR tasks are left out of this paper. This scope was proposed on 30 Oct 2026 and accepted on 31 Oct 2026 ("Yes, add them"; "i want to finish this paper experiments and finish the paper"). The CNN and the third encoder are kept as answers if a reviewer asks; patient-free radiomics and the MR tasks go to later studies.

### 7j. two more independent lung cohorts (notebook 11), proposed on 3 Nov 2026

**Status:**
- proposed on 3 Nov 2026 at the author's request [publication-planning words removed];
- **approved by Hamza on 3 Nov 2026** ("Approve and upload"), after notebook 11 was written and tested locally on shuffled labels only, and before it ran on real labels;
- **written before any new result exists.** To check feasibility, only the class counts of the two new label files and the series lists from the TCIA interface were read. No image, embedding or model output exists.
- **Part A, notebook 11, run on 3 Nov 2026** (Colab, third run, after amendment A.9). Coverage was 100% and 60 of 60 reproduction checks passed. The A.5 reading is **"consistent across independent cohorts"**. Results, caveats and a post hoc sensitivity check are in log section 33.
- **A.7 run on 4 Nov 2026** (log section 34). Neither new task is flagged, as expected. Lung-PET-CT-Dx shows a weak signal below the flag (AUC 0.638, lower bound 0.565, q 0.032), carried by protocol words. Part A is complete.

**Why.** The paper's strongest claim, that external validation can confirm a shortcut instead of exposing it, rests on one independent lung cohort (NSCLC Radiogenomics) against one confounded cohort (CPTAC lung). A reviewer would call one independent cohort thin.

**A. Two more independent external lung cohorts (notebook 11).**
1. **Targets** (TCIA; each from a single source, labels within the collection, not one collection per class):
   - **NSCLC-Radiomics** (Lung1; DOI 10.7937/K9/TCIA.2015.PF0M9REI; version 4):
     - labels from `NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv`, field `Histology`;
     - adenocarcinoma = 51, squamous cell carcinoma = 152;
     - excluded: large cell (114), not otherwise specified (63), missing (42).
   - **Lung-PET-CT-Dx** (DOI 10.7937/TCIA.2020.NNC2-0461; version 5):
     - labels from the first letter of the patient ID, as documented by the collection: A, adenocarcinoma = 251; G, squamous cell carcinoma = 61;
     - excluded: B (small cell, 38) and E (large cell, 5).
   - These counts are fixed as the expected counts. Patients without an eligible CT main series are dropped and reported.
2. **Images:**
   - notebook 01's main-series rule (CT only);
   - then notebook 05–07's steps, unchanged: Hounsfield units, body mask, background-only image, silhouette, and CT-FM and Merlin embeddings with the same settings (EMB_KEY);
   - DICOM header fields as in 7d.6.
3. **Source models:**
   - the TCGA lung (LUAD vs LUSC) image and background-only probes of notebook 08, fitted on all 104 source patients;
   - the silhouette probe of notebook 09;
   - nothing is refitted on the source.
4. **Per target, encoder and input** (image, background only; silhouette secondary):
   - external AUC with a bootstrap 95% CI over target patients, and the permutation p;
   - the gap from the source's internal AUC;
   - the target's own cross-validated AUC (5 × 5);
   - the transfer rule of 7g (external CI above 0.5).
5. **Reading rule, fixed now** (background-only probes, both encoders). The acquisition hypothesis predicts no transfer to independent cohorts:
   - **"consistent across independent cohorts"** if no background-only probe transfers to either new cohort;
   - **"partly consistent"** if probes transfer to one cohort only;
   - **"not confirmed"** if background-only probes transfer to both.
   - The paper reports the result whichever way it falls, next to NSCLC Radiogenomics and CPTAC.
6. **Mechanism (descriptive, as 7h check 3):** in each new cohort, the AUC with a bootstrap CI for squamous cell carcinoma having the higher slice thickness, slice spacing, reconstruction diameter, kVp, tube current and body area. 7h found squamous slices thinner in TCGA and CPTAC but thicker in NSCLC Radiogenomics. The prediction stated now: a background-only probe can only transfer where the slice-thickness direction matches TCGA's.
7. **Metadata screen on the two new tasks:**
   - the open tool (metadata only) on each cohort's histology task, with the flag rule of 7b;
   - the expectation, stated now: not flagged, since each is a single-source collection.
   - **Run details, fixed on 4 Nov 2026 before the A.7 run.** Notebook 11's image results were known by then; A.7 uses
     metadata only.
     - **Tool:** `tool/tcia_shortcut_audit.py` 1.0.0, unchanged, with its defaults and `--modality CT`. The defaults
       are seed 2026, 5 × 5 CV, 2,000 bootstrap resamples and 999 permutations.
     - **Metadata:** notebook 11's saved getSeries replies (their MD5s are in its manifest), given as `--series-csv`.
       Both analyses therefore see the same metadata.
     - **Labels:** built as in notebook 11. NSCLC-Radiomics uses the pinned Lung1 file's `Histology` field.
       Lung-PET-CT-Dx uses the letter in the patient ID (A = adenocarcinoma, G = squamous cell carcinoma). Both tasks
       name their classes "adenocarcinoma" and "squamous cell carcinoma".
     - **Flag:** lower CI ≥ 0.60 and BH q < 0.05, with BH over the 26 original tasks plus these two (7b.1). The 26
       original flags are reported both unchanged and under the 28-task BH.
     - **Primary run:** the tool's own main-series rule. For 17 Lung-PET-CT-Dx patients it picks the `<ALPHA Range>`
       series, as notebook 11's first runs did.
     - **Secondary run:** those 17 series are set aside as in A.9, so the label cards come from the series notebook 11
       embedded.
8. **Run order:**
   - one "Run all": a shuffled-label rehearsal (seed 999) with notebook 08's gate, then the real labels;
   - images go to Drive or Colab disk only and are deleted after embedding; only embeddings and header records are kept;
   - nothing touches C: or D:.
9. **Amendment, 4 Nov 2026 (label-blind; made after two Colab runs stopped at the coverage gate, before any real-label
   result existed):**
   - **Problem:** for 17 Lung-PET-CT-Dx patients (13 A, 4 G), the main-series rule picks a Siemens `<ALPHA Range>`
     series. These hold Secondary Capture renderings (SOP class 1.2.840.10008.5.1.4.1.1.7) with no slice geometry, not
     CT slices. They passed the rule because their descriptions match none of its exclusion patterns.
   - **Effect:** coverage was 295 of 312 (94.6%), below the 95% gate, so the real-label pass never ran.
   - **Change:**
     - a series the rule picks that holds no CT slices ("no image slices in the download") is set aside, and the same
       rule picks again from that patient's remaining series;
     - at most 3 series are set aside per patient;
     - every replacement is recorded in `cache/series_set_aside.csv` and in the run manifest;
     - the image steps (notebook 08's `prepare_patient`) are unchanged.
   - **Scope:** only the two new cohorts.
     - No main series of notebook 01 (the 4,883 behind the 26 audited tasks) has a description containing "range".
     - The source and the old targets were all embedded already (TCGA 104/104, NSCLC Radiogenomics 199/199, CPTAC
       lung 73/73), so nothing published changes.

**B.** [literature-screening logistics removed; not part of the analyses]

**Not done here (kept for a follow-up paper):**
- an end-to-end trained network;
- acquisition harmonization as a mitigation;
- MR image probes;
- a third encoder.

**After A and B:**
- update the manuscript; the author list and contributions are reviewed when the work is finished;
- [publication-planning note removed]
- [publication-planning note removed]

### 7k. Post hoc robustness checks R10–R16, fixed on 6 Nov 2026 before any was run

**Status:**
- These seven checks were listed on 6 Nov 2026.
- Hamza approved them on 6 Nov 2026 ("Yes, run them").
- The details below were written before any of them was run.

All are **post hoc**:
- They reuse the paper's code: the tool for the metadata screen, and notebook 05's model code (cell 11) for every other model.
- They use saved files only: series metadata, DICOM headers, embeddings and study records. No image is downloaded.
- Scripts and outputs: `results/12_review_checks/r10_*` to `r16_*`.

1. **R10, series-selection rule.** For the four flagged tasks, the metadata screen is rerun with two other
   main-series rules. Everything else is unchanged: the first study with an eligible series, eligibility, label card,
   model and statistics.
   - **(a) Fewest images:** in that study, the eligible series with the fewest images (at least 10, as before); ties by
     series number, then UID.
   - **(b) Nearest 5 mm:** the eligible series whose slice thickness, parsed from its description by the label card's
     parser, is closest to 5 mm; ties by more images, then series number. If no eligible series of the study has a
     parsed thickness, the original series is kept.
   - **Check first:** with the original rule, the tool must reproduce notebook 03's AUC and CI for each task within
     1e-9. Otherwise the tool's own original-rule run is the comparison, and the difference is reported.
   - **Reported per task and rule:**
     - the share of patients whose series changed;
     - AUC (95% CI);
     - permutation P (999);
     - BH over the four tasks per rule.

     The flag "holds" if the lower CI bound is ≥ 0.60 and q < 0.05.
2. **R11, single-property baselines.** These use notebook 05's logistic regression, 5 × 5 CV, bootstrap CI
   and permutation test (999 permutations on the fixed split).
   - **(a)** scan length along the body axis (`hdr_coverage_mm`) alone;
   - **(b)** reconstruction diameter alone;
   - **(c)** both.

     For (a)–(c): the 19 CT tasks, using patients with an embedded scan (headers of notebooks 05 and 07).
   - **(d)** sex and age, for all 26 tasks:
     - from the study records as in R9 (the first study with a value);
     - sex as male 0/1 and age in years;
     - missing values imputed inside the training folds, with an indicator.
   - **Multiplicity and flags:** BH within each baseline over its tasks; flagged as in the screen (lower CI ≥ 0.60 and
     q < 0.05).
   - **Comparison:** for the four flagged tasks, the paired bootstrap difference (notebook 05's `boot_diff`) between
     the background-only probe and each baseline. The probe is refitted for each encoder with notebook 05's code and
     splits.
3. **R12, scans with the body cut off by the field of view.**
   - The 9 of notebook 05's 525 scans whose dilated body mask covers ≥ 75% of the image (7f, item 4) are counted by
     class.
   - For each task that contains one, the image, background-only and silhouette probes (both encoders) are refitted
     without these scans. They are compared with a refit on all scans with the same code.
   - CIs and permutation P are computed as in notebook 05. BH is over the four tasks per encoder and input; the other
     tasks keep their P values.
   - Flag decisions are compared.
4. **R13, kernel decoding within vendor.** Notebook 07's kernel decoding (experiment C) is repeated
   within each manufacturer that has at least two kernels with ≥ 20 scans (notebook 07's minimum).
   - Same code: macro one-vs-rest AUC, 5 × 5 CV, bootstrap CI, within-collection AUC and shuffled targets (seed 999).
   - Both encoders and all three inputs.
   - **Check first:** the pooled kernel decoding is recomputed and must agree with notebook 07's within 0.001, the
     tolerance used for notebook 06's reproduction. Otherwise the differences are reported.
5. **R14, detectable AUC (CLAIM item 21).** For each task's class sizes (the 26 metadata tasks, and the 19
   CT image tasks with their embedded patients): the smallest true AUC at which the CI part of the flag rule (lower
   bound of the 95% percentile patient-bootstrap CI ≥ 0.60) is met with 80% probability.
   - **Simulation:**
     - binormal scores with equal variances;
     - 1,000 simulated data sets per AUC value, with 1,000 bootstrap resamples each;
     - the AUC is found by bisection to 0.001, using the same random numbers at every AUC value (seed 2026).
   - **Not modelled:** the permutation q criterion and the averaging over CV repeats. The values therefore approximate
     the rule, and the q criterion can only raise them.
6. **R15, multiplicity across encoders and inputs.** The shortcut rule (7d.7) is reapplied with
   multiplicity control across the two encoders and two inputs:
   - **(a)** the permutation P values stored by notebook 05 are adjusted by BH over all 16 image tests (4 tasks × 2
     encoders × 2 inputs) instead of four;
   - **(b)** the CI criterion is applied to a 98.75% CI (95% with a Bonferroni adjustment for the four encoder–input
     tests of a task);
   - **(c)** the swap rule is applied with 97.5% CIs (two encoders).

   The CIs come from a refit with notebook 05's code, splits and bootstrap resamples. The 95% CI is reported too, to
   show that the refit reproduces notebook 05.
7. **R16, planning record.** A table, built from the dated plan and the run records only. For each
   analysis (notebooks 03–11 and checks R1–R16) it lists:
   - the plan section and its local date;
   - the UTC start of the real-label run;
   - which real-label results had been seen when the analysis was planned.

   No new computation.

**Reading:**
- These checks are reported in the supplement and summarized in the paper. Each is labeled post hoc.
- None of them changes a primary result or the flag rule.

**Addendum to 7k, 6 Nov 2026 (made after the R11 result was seen):**
- **R11b, a descriptive split of R11(d), was added.**
- **Why:** in LIDC-IDRI primary versus metastatic, the sex-and-age baseline predicted the class (AUC 0.81). Age was
  missing for 25 of 43 primary and 1 of 51 metastatic patients; sex was missing for 11 and 0.
- **What it does:** for the four flagged tasks, it fits the same model and statistics on:
  - the two missingness indicators alone;
  - sex and age among the patients with both recorded.
- **Status:** it is post hoc and was not part of the rules fixed above.
