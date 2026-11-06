# Acquisition-shortcut audit

Collections: NSCLC-Radiomics · modality CT · 203 patients · classes: adenocarcinoma (51), squamous cell carcinoma (152)

## Result

Metadata-only AUC **0.556** (95% CI 0.474-0.635); permutation p = 0.0880 (null mean 0.491).

**Not flagged.** Acquisition metadata does not predict the label well above chance. This lowers, but does not remove, the risk of an acquisition shortcut (fields missing from the metadata, such as reconstruction details stored only in the image headers, are not tested).

Rule: lower CI >= 0.6 and permutation p < 0.05 (apply BH first for a family of tasks).

## Which fields carry the signal

| Field group | Filled | AUC alone (95% CI) | Change in AUC when left out |
|---|---|---|---|
| description words | 78% | 0.585 (0.502-0.665) | +0.107 |
| image count | 100% | 0.532 (0.447-0.615) | -0.008 |
| body part | 100% | 0.500 (0.500-0.500) | +0.000 |
| slice thickness and kernel | 6% | 0.478 (0.432-0.528) | -0.011 |
| scanner (vendor, model) | 100% | 0.450 (0.387-0.514) | -0.001 |
| software version | 100% | 0.421 (0.372-0.467) | -0.011 |

## Fields by class (share of patients, or median)

| Field | Value | adenocarcinoma | squamous cell carcinoma |
|---|---|---|---|
| vendor | SIEMENS | 0.77 | 0.78 |
| vendor | CMS | 0.23 | 0.22 |
| model | BIOGRAPH 40 | 0.47 | 0.49 |
| model | SENSATION OPEN | 0.26 | 0.22 |
| model | XIO | 0.23 | 0.22 |
| model | SENSATION 16 | 0.04 | 0.06 |
| model | SENSATION 10 | 0.00 | 0.01 |
| software_version | syngo CT 2006A | 0.69 | 0.67 |
| software_version | 4.34.02 | 0.23 | 0.22 |
| software_version | VA70C | 0.04 | 0.06 |
| software_version | syngo CT 2009E | 0.04 | 0.04 |
| software_version | syngo CT 2006G | 0.00 | 0.01 |
| kernel | missing | 1.00 | 1.00 |
| body_part | LUNG | 1.00 | 1.00 |
| image_count | median | 122.00 | 130.00 |
| slice_thickness_mm | median | 3.00 | 3.00 |

## How this was computed

One main series per patient (earliest study; the eligible series with the most images; scouts, localisers, reformats and
analysis results excluded). Label card: scanner vendor and model, software version, image count, slice thickness and
kernel parsed from the descriptions, body part, and description words (words naming a diagnosis, class or hospital are
removed; words about why or when the scan was taken are kept apart and not used). Model: logistic regression,
5-fold stratified cross-validation repeated 5 times; 95% CI from 2000 patient bootstrap resamples;
permutation test with 999 label permutations. Tool version 1.0.0.
