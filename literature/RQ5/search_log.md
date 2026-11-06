# RQ5 search log

Queries of plan 7e, run with `rq5_search.py`. Hits = total reported by the source (Semantic Scholar is a relevance search: only its first 100 are kept); Saved = records kept.

| Source | Query | Date | Hits | Saved | Query string |
|---|---|---|---|---|---|
| Europe PMC | lung_A | 2026-10-29 | 30 | 30 | `("TCGA-LUAD" OR "TCGA LUAD") AND ("TCGA-LUSC" OR "TCGA LUSC") AND ("Cancer Imaging Archive" OR "TCIA") AND ("CT" OR "computed tomography")` |
| Europe PMC | lung_B | 2026-10-29 | 341 | 341 | `"adenocarcinoma" AND "squamous cell carcinoma" AND ("Cancer Imaging Archive" OR "TCIA") AND ("CT" OR "computed tomography") AND ("radiomics" OR "deep learning" OR "machine learning" OR "convolutional neural network")` |
| Europe PMC | kidney_A | 2026-10-29 | 86 | 86 | `("TCGA-KIRC" OR "TCGA KIRC" OR "TCGA-KIRP" OR "TCGA KIRP" OR "TCGA-KICH" OR "TCGA KICH") AND ("Cancer Imaging Archive" OR "TCIA") AND ("CT" OR "computed tomography")` |
| Europe PMC | kidney_B | 2026-10-29 | 200 | 200 | `"renal cell carcinoma" AND ("subtype" OR "subtypes" OR "papillary" OR "chromophobe") AND ("Cancer Imaging Archive" OR "TCIA") AND ("CT" OR "computed tomography") AND ("radiomics" OR "deep learning" OR "machine learning" OR "convolutional neural network")` |
| Europe PMC | pda_A | 2026-10-29 | 37 | 37 | `("CPTAC-PDA" OR "CPTAC PDA")` |
| Europe PMC | lidc_A | 2026-10-29 | 45 | 45 | `("LIDC-IDRI" OR "LIDC") AND ("diagnosis data" OR "tcia-diagnosis-data" OR "patient diagnosis" OR "157 patients" OR "pathologically confirmed" OR "pathologically proven" OR "biopsy-proven") AND ("metastatic" OR "metastasis" OR "metastases")` |
| PubMed | lung_P | 2026-10-29 | 10 | 10 | `("adenocarcinoma"[tiab] AND "squamous"[tiab]) AND ("TCIA"[all] OR "Cancer Imaging Archive"[all] OR "TCGA"[tiab]) AND ("CT"[tiab] OR "computed tomography"[tiab]) AND ("radiomics"[tiab] OR "deep learning"[tiab] OR "machine learning"[tiab] OR "neural network"[tiab])` |
| PubMed | kidney_P | 2026-10-29 | 12 | 12 | `("renal cell carcinoma"[tiab]) AND ("subtype"[tiab] OR "subtypes"[tiab] OR "papillary"[tiab] OR "chromophobe"[tiab]) AND ("TCIA"[all] OR "Cancer Imaging Archive"[all] OR "TCGA"[tiab]) AND ("CT"[tiab] OR "computed tomography"[tiab])` |
| PubMed | pda_P | 2026-10-29 | 2 | 2 | `"CPTAC"[tiab] AND ("pancreatic"[tiab]) AND ("CT"[tiab] OR "computed tomography"[tiab] OR "imaging"[tiab])` |
| PubMed | lidc_P | 2026-10-29 | 8 | 8 | `("LIDC"[tiab] OR "LIDC-IDRI"[tiab]) AND ("metastatic"[tiab] OR "metastasis"[tiab] OR "metastases"[tiab])` |
| arXiv | lung_X | 2026-10-29 | 1 | 1 | `all:TCIA AND all:adenocarcinoma AND all:squamous` |
| arXiv | lung_X2 | 2026-10-29 | 10 | 10 | `all:"TCGA-LUAD"` |
| arXiv | kidney_X | 2026-10-29 | 1 | 1 | `all:TCIA AND all:"renal cell carcinoma"` |
| arXiv | kidney_X2 | 2026-10-29 | 6 | 6 | `all:"TCGA-KIRC"` |
| arXiv | pda_X | 2026-10-29 | 0 | 0 | `all:"CPTAC-PDA"` |
| arXiv | lidc_X | 2026-10-29 | 0 | 0 | `all:LIDC AND all:metastatic` |
| Semantic Scholar | lung_S | 2026-10-29 | 344 | 100 | `CT lung adenocarcinoma squamous cell carcinoma classification TCGA TCIA` |
| Semantic Scholar | kidney_S | 2026-10-29 | 219 | 100 | `CT renal cell carcinoma subtype classification TCGA TCIA` |
| Semantic Scholar | pda_S | 2026-10-29 | 107 | 100 | `CPTAC-PDA pancreatic CT` |
| Semantic Scholar | lidc_S | 2026-10-29 | 15 | 15 | `LIDC-IDRI primary lung cancer versus metastatic classification diagnosis data` |
