# TCIA acquisition-shortcut audit

**Can a cancer-imaging benchmark be solved without looking at the patient?** This tool checks, before any image is
downloaded, whether the labels of a task built on [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
can be predicted from how the scans were acquired: scanner, software, number of images, slice thickness, kernel, body
part and the words of the series descriptions. If they can, an image model trained on the task may learn the acquisition
instead of the disease.

It is the metadata screen of the paper *Screening public cancer CT benchmarks for non-tumour shortcuts: tissue-free
prediction and the limits of external validation* (Hamza El Massaoudy, NUIST), packaged so that anyone can run it on any TCIA collection.

## What it does

1. Downloads the **series descriptions** of the collections from the public TCIA API. No images are downloaded, and
   replies are cached in `.tcia_cache/`, so a rerun gives identical numbers.
2. Picks **one main series per patient**: the earliest study, and in it the eligible series with the most images.
   Scouts, localisers, reformats, dose reports and analysis results are excluded.
3. Builds a **label card** per patient from that series:
   - vendor, model and software version;
   - image count;
   - slice thickness and kernel, parsed from the descriptions;
   - body part;
   - description words. Words that name a diagnosis, a class or a hospital are removed. Words about why or when the
     scan was taken are kept apart and not used.
4. Fits a **logistic regression on the label card only**:
   - 5-fold stratified cross-validation, repeated 5 times;
   - a patient-bootstrap 95% CI;
   - a permutation test against chance.
5. Reports:
   - which field groups carry the signal, alone and when left out;
   - how each field is distributed by class;
   - if you give contributing sites, whether the signal survives **within one site** and when **whole sites are held
     out** (a site you supply may not be where the images were acquired).

**Flag rule:** lower 95% CI bound of the AUC >= 0.60 **and** permutation p < 0.05. For a family of several tasks, apply
Benjamini-Hochberg to their p-values first.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is needed. Nothing else is required: no GPU and no TCIA account, since only public metadata is used.

**Tested versions.** `requirements.txt` gives minimum versions. The reproduction test (below) passed, with a largest
difference of 6e-17, on Python 3.12.10 with numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, scikit-learn 1.6.1, joblib 1.6.0
and requests 2.32.4, the package versions of the paper's runs. With other versions, results can differ in the last
digits. To install exactly these versions:

```bash
pip install numpy==2.1.3 pandas==2.2.3 scipy==1.16.3 scikit-learn==1.6.1 joblib==1.6.0 requests==2.32.4
```

## Use

**Class = source collection** (e.g. TCGA-LUAD vs TCGA-LUSC):

```bash
python tcia_shortcut_audit.py --collections TCGA-LUAD TCGA-LUSC --label-by-collection --out lung_report
```

**Your own labels** (CSV with `PatientID` and `label`; optional `collection` and `site`):

```bash
python tcia_shortcut_audit.py --labels my_labels.csv --collections NSCLC-Radiomics --out my_report
```

**Options:**

| Option | What it does |
|---|---|
| `--modality CT` or `--modality MR` | Choose the modality. Default: the one with more patients. |
| `--block-words ...` | Remove extra words, for example a hospital name you know appears in the descriptions. Class names and collection names are always removed. |
| `--n-perm`, `--n-boot` | Change the number of permutations and bootstrap resamples. |
| `--series-csv` | Run offline from a saved metadata table. |

`python tcia_shortcut_audit.py --help` lists everything. On a laptop, a task with a few hundred patients takes under a minute.

## Output (in the `--out` folder)

| File | Content |
|---|---|
| `report.md` | Readable summary: result, flag, field groups, fields by class, site analysis |
| `summary.json` | All numbers and settings |
| `field_groups.csv` | AUC of each field group alone (95% CI), and the change when it is left out |
| `fields_by_class.csv` | Share of each vendor, model, kernel, body part, ... per class |
| `label_card.csv` | The label card of every patient (what the model saw), with removed words listed |
| `main_series.csv` | The series chosen for each patient |
| `oof_scores.csv` | Out-of-fold scores per patient |

## How to read the result

- **Flagged:** acquisition alone predicts the label.
  - **What to do:** before trusting an image model on this task, test it with the segmented tissue removed (background
    only), within one site or scanner, and on an independent dataset that asks the same clinical question and was not
    assembled the same way.
  - **Caution:** in the paper, an external test on a dataset assembled the same way (one collection per class) still
    "passed" for two of four probes.
- **Not flagged:** at the thresholds used, the metadata did not predict the label. This does not show that the task is
  free of acquisition shortcuts. With synthetic labels (paper, positive control), the screen flagged 1 of 240 tasks
  without an association and detected 80% of planted associations only at an AUC of about 0.72 (250 patients) to 0.79
  (100 patients); weaker associations were mostly missed.
  - Fields that are stored only in the image headers are not tested, such as the exact reconstruction kernel or the tube
    current.
  - In the paper, no background-only probe met the flag criterion in the 15 unflagged CT tasks, but there were only one or
    two positive references, and the screen works as a yes/no flag, not as a graded risk score.

## Reproducing the paper

`tests/reproduce_notebook03.py` reruns the two flagged collection-defined tasks of the paper from the saved metadata.
It checks that every AUC, CI bound, permutation p and field-group number equals the paper's within 1e-9. On 31 Oct 2026
the largest difference was 6e-17. A live run from the TCIA API gave the same AUC (0.827269 for TCGA-LUAD vs
TCGA-LUSC). The saved metadata are not part of this archive: the test needs the study repository (folder `results/`),
whose path is its argument. For CPTAC-PDA stage, which is not part of the test, the tool gives 0.843 where the study's
notebook gave 0.840.

```bash
python tests/reproduce_notebook03.py /path/to/Acquisition-Shortcut-Audit
```

## Limits

- Only what the TCIA API returns is used. Slice thickness and kernel are often missing from the descriptions.
- Collections without patient-level labels need your own labels file.
- A high AUC shows that acquisition and label are linked in this dataset. It does not show that a given image model
  uses that link.

## Citation and licence

Please cite the paper (reference to be added at publication). Licence: MIT (the `LICENSE` file).

## Versions

- **1.0.1** (4 Nov 2026): wording of the report, the log and this manual only; every computation is that of 1.0.0.
  The manual was updated on 5 Nov 2026 (tested versions, licence) and on 12 Nov 2026 (measured sensitivity of the
  screen, where the reproduction test finds its data); the code is unchanged.
- **1.0.0** (31 Oct 2026): the version used for the paper.
