# Acquisition-shortcut audit

Collections: Lung-PET-CT-Dx · modality CT · 312 patients · classes: adenocarcinoma (251), squamous cell carcinoma (61)

## Result

Metadata-only AUC **0.638** (95% CI 0.565-0.708); permutation p = 0.0070 (null mean 0.495).

**Not flagged.** Acquisition metadata does not predict the label well above chance. This lowers, but does not remove, the risk of an acquisition shortcut (fields missing from the metadata, such as reconstruction details stored only in the image headers, are not tested).

Rule: lower CI >= 0.6 and permutation p < 0.05 (apply BH first for a family of tasks).

## Which fields carry the signal

| Field group | Filled | AUC alone (95% CI) | Change in AUC when left out |
|---|---|---|---|
| description words | 100% | 0.648 (0.574-0.718) | +0.050 |
| software version | 100% | 0.580 (0.502-0.656) | +0.002 |
| slice thickness and kernel | 94% | 0.564 (0.492-0.632) | -0.004 |
| body part | 57% | 0.527 (0.455-0.604) | -0.000 |
| scanner (vendor, model) | 100% | 0.523 (0.457-0.589) | -0.002 |
| image count | 100% | 0.511 (0.435-0.584) | -0.001 |

## Fields by class (share of patients, or median)

| Field | Value | adenocarcinoma | squamous cell carcinoma |
|---|---|---|---|
| vendor | SIEMENS | 0.51 | 0.53 |
| vendor | GE | 0.39 | 0.39 |
| vendor | PHILIPS | 0.10 | 0.08 |
| model | BIOGRAPH 64 | 0.33 | 0.36 |
| model | DISCOVERY CT750 HD | 0.23 | 0.29 |
| model | SENSATION 10 | 0.13 | 0.05 |
| model | LIGHTSPEED VCT | 0.12 | 0.08 |
| model | BIOGRAPH64 | 0.05 | 0.12 |
| model | BRILLIANCE16 | 0.04 | 0.07 |
| model | ICT 256 | 0.05 | 0.02 |
| model | LIGHTSPEED PRO 16 | 0.02 | 0.02 |
| model | OPTIMA CT660 | 0.01 | 0.00 |
| model | REVOLUTION CT | 0.00 | 0.00 |
| software_version | VG51C | 0.24 | 0.15 |
| software_version | sles_hde3.5 | 0.20 | 0.23 |
| software_version | syngo CT 2006G | 0.13 | 0.05 |
| software_version | VG60A | 0.09 | 0.21 |
| software_version | 07MW18.4 | 0.12 | 0.08 |
| software_version | 2.3.5 | 0.04 | 0.07 |
| software_version | 4.1 | 0.05 | 0.02 |
| software_version | VG62A | 0.04 | 0.08 |
| software_version | sles_hde.132 | 0.03 | 0.07 |
| software_version | 06MW03.5 | 0.02 | 0.02 |
| software_version | VG62B | 0.02 | 0.03 |
| software_version | cj.37 | 0.01 | 0.00 |
| software_version | revo_1.5_m3d.9 | 0.00 | 0.00 |
| software_version | 07MW11.10 | 0.00 | 0.00 |
| kernel | B70F | 0.27 | 0.25 |
| kernel | missing | 0.26 | 0.18 |
| kernel | STND | 0.20 | 0.28 |
| kernel | B31F | 0.17 | 0.12 |
| kernel | B30F | 0.07 | 0.15 |
| kernel | LUNG/STND | 0.02 | 0.02 |
| kernel | STD | 0.01 | 0.00 |
| kernel | LUNG | 0.00 | 0.00 |
| kernel | B31S | 0.00 | 0.00 |
| kernel | B40F | 0.00 | 0.02 |
| body_part | CHEST | 0.50 | 0.39 |
| body_part | missing | 0.42 | 0.46 |
| body_part | ABDOMEN | 0.08 | 0.15 |
| body_part | Thorax | 0.00 | 0.00 |
| image_count | median | 180.00 | 186.00 |
| slice_thickness_mm | median | 5.00 | 5.00 |

## How this was computed

One main series per patient (earliest study; the eligible series with the most images; scouts, localisers, reformats and
analysis results excluded). Label card: scanner vendor and model, software version, image count, slice thickness and
kernel parsed from the descriptions, body part, and description words (words naming a diagnosis, class or hospital are
removed; words about why or when the scan was taken are kept apart and not used). Model: logistic regression,
5-fold stratified cross-validation repeated 5 times; 95% CI from 2000 patient bootstrap resamples;
permutation test with 999 label permutations. Tool version 1.0.0.
