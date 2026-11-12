# Sanitization record

Build: public release. Text files were changed only by these rules; every other file is a byte copy of the project file. In executed notebooks, the Colab user fields of each cell's execution record (user, user_tz, status) were removed.

The search patterns are not listed, because they contain the strings that were removed.

| What was replaced | Replacement |
|---|---|
| Colab account e-mail | the Google account that holds the project Drive folder |
| e-mail addresses | [e-mail removed] |
| author attribution (authors listed in README) | (authors: see README.md) |
| collaborator name (privacy) | [a former collaborator] |
| local Windows paths | <local path> |
| local paths | <local path> |
| figure script: journal wording in comments | neutral wording (5 places) |
| analysis plan: publication-planning and personal passages | bracketed note in place (8 places) |
| analysis plan: literature-screening logistics (sections 7e and 7j) | shortened, or a bracketed note in place |
| analysis plan: attribution of who proposed a step | removed; the author's approval records and all dates kept |
| notebooks: messages addressed to the analyst, in the code and in the printed outputs of the executed copies | neutral wording; no computed value changed |
| README: contributor statement | replaced by a statement of research responsibility |

Changed files:

- `figures/make_figures.py`: journal wording in comments: 5
- `notebooks/01_tcia_inventory.ipynb`: Colab account e-mail: 2
- `notebooks/02_labels_features_tasks.ipynb`: Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/03_metadata_audit.ipynb`: Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/04_posthoc_checks.ipynb`: Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/05_image_model_check.ipynb`: Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/06_image_posthoc_checks.ipynb`: Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/07_screen_validation.ipynb`: Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/08_transfer.ipynb`: Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/09_transfer_posthoc.ipynb`: Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/10_figures_tables.ipynb`: Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/11_external_lung.ipynb`: Colab account e-mail: 2
- `notebooks/executed/01_tcia_inventory.ipynb`: Colab user metadata removed from cells: 57; Colab account e-mail: 2
- `notebooks/executed/02_labels_features_tasks.ipynb`: Colab user metadata removed from cells: 42; Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/executed/03_metadata_audit.ipynb`: Colab user metadata removed from cells: 36; Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/executed/04_posthoc_checks.ipynb`: Colab user metadata removed from cells: 33; Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/executed/05_image_model_check.ipynb`: Colab user metadata removed from cells: 24; Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/executed/06_image_posthoc_checks.ipynb`: Colab user metadata removed from cells: 27; Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/executed/07_screen_validation.ipynb`: Colab user metadata removed from cells: 24; Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/executed/08_transfer.ipynb`: Colab user metadata removed from cells: 24; Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/executed/09_transfer_posthoc.ipynb`: Colab user metadata removed from cells: 9; Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/executed/10_figures_tables.ipynb`: Colab user metadata removed from cells: 51; Colab account e-mail: 2; author attribution (authors listed in README): 1
- `notebooks/executed/11_external_lung.ipynb`: Colab user metadata removed from cells: 24; Colab account e-mail: 2
- `notebooks/executed/12_image_experiments.ipynb`: Colab user metadata removed from cells: 4
- `notebooks/executed/13_cnn.ipynb`: Colab user metadata removed from cells: 6
- `results/11_external_lung/a7_metadata_screen/a7_run.log`: local Windows paths: 6
- `results/12_review_checks/r10_run.log`: local Windows paths: 46
- `results/12_review_checks/r11_run.log`: local Windows paths: 13
- `results/12_review_checks/r11b_run.log`: local Windows paths: 13
- `results/12_review_checks/r12_run.log`: local Windows paths: 13
- `results/12_review_checks/r13_run.log`: local Windows paths: 13
- `results/12_review_checks/r14_run.log`: local Windows paths: 7
- `results/12_review_checks/r15_run.log`: local Windows paths: 13
