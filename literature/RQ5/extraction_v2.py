"""Normalised extraction table (4 Nov 2026, after the recheck report, finding M2). POST HOC re-coding; extraction.csv,
the first extraction, is kept unchanged as the record of the first screening.

What changes:
- The actual outcome of each study, instead of the search label (all LIDC rows had been labelled "LIDC primary vs
  metastatic").
- The evidence basis, in its own column, instead of "(abstract)" appended to the level.
- Two items that the protocol's single A3 level mixed together, now recorded separately:
  - **External test on other institutions:** the model was tested on data from institutions not used for training. This
    is "site-held-out testing" in the protocol's literal wording.
  - **Within-data control:** a design that controls the class-site or class-scanner association in the data used, such
    as matching, stratification or testing within site or scanner, or harmonization by scanner.
- Whether the model was trained on the flagged collection(s), where a shortcut could be learned.
- The level under the clarified definition (A3 = within-data control) and under the protocol's literal wording
  (external tests on other institutions count as A3).

The design facts come from the first extraction. For Luo et al (record 245), they come from the article's Results and
Fig. 1, re-read on 4 Nov 2026: training and internal test set 1 from Institution 1; external test set 2 from
Institutions 2-4; external test set 3 from TCIA; macro AUC 0.850, 0.811 and 0.813.
Output: extraction_v2.csv.  Usage: python extraction_v2.py"""
import pandas as pd

e = pd.read_csv('extraction.csv', dtype=str)
OUTCOME = {
    '17': 'ccRCC vs pRCC vs chRCC (3-class)', '52': 'five renal tumour subtypes incl. benign',
    '222': 'renal tumour subtype (ccRCC, pRCC, chRCC)', '245': 'renal tumour subtype (4-class: ccRCC, pRCC, chRCC/RO, fpAML)',
    '313': 'LUAD vs LUSC', '527': 'LUAD vs LUSC',
    '346': 'benign vs primary vs metastatic (3-class) and benign vs malignant',
    '365': 'benign vs malignant (pathologic diagnosis)', '380': 'benign vs primary vs metastatic (3-class)',
    '543': 'benign vs malignant (confirmed diagnosis)', 'REF:Kumar2017': 'benign vs malignant (pathologically proven)',
    'REF:Shen2016': 'benign vs malignant (definite diagnosis)', 'REF:Wang2018': 'benign vs malignant (pathologic diagnosis)'}
TASK = {'KIRC vs KIRP': 'KIRC vs KIRP', 'LUAD vs LUSC': 'LUAD vs LUSC', 'LIDC primary vs metastatic': 'LIDC-IDRI diagnosis'}
EXTERNAL = {  # tested on data from institutions not used for training (from the first extraction; 245 re-read)
    '17': 'yes: TCGA kidney (abstract)', '52': 'yes: TCIA kidney', '222': 'yes: TCGA kidney',
    '245': 'yes: Institutions 2-4 and TCIA kidney', '313': 'no', '527': 'no: held-out collection from a training institution',
    '346': 'no', '365': 'no', '380': 'no', '543': 'no', 'REF:Kumar2017': 'no', 'REF:Shen2016': 'no', 'REF:Wang2018': 'no'}
TRAINED_ON_FLAGGED = {'17': 'no', '52': 'no', '222': 'no', '245': 'no', '313': 'yes', '527': 'yes (TCGA-LUAD)',
                      '346': 'yes', '365': 'yes', '380': 'yes', '543': 'yes', 'REF:Kumar2017': 'yes',
                      'REF:Shen2016': 'yes', 'REF:Wang2018': 'yes'}
rows = []
for r in e.itertuples():
    basis = r.extracted_from
    full = basis == 'full text'
    first = r.acquisition_level.split(' ')[0]                      # A0..A3 as coded first
    clarified = first if full else 'not ascertainable'
    literal = ('A3' if EXTERNAL[r.record_no].startswith('yes') else first) if full else 'not ascertainable'
    rows.append({'record_no': r.record_no, 'task': TASK[r.task], 'outcome': OUTCOME[r.record_no], 'year': r.year,
                 'venue': r.venue, 'collections': r.collections, 'best_reported_metric': r.best_reported_metric,
                 'evidence_basis': basis, 'trained_on_flagged_collections': TRAINED_ON_FLAGGED[r.record_no],
                 'external_test_other_institutions': EXTERNAL[r.record_no],
                 'within_data_acquisition_control': 'none found' if full else 'not ascertainable',
                 'acquisition_level_first_coding': r.acquisition_level,
                 'acquisition_level_clarified': clarified, 'acquisition_level_protocol_literal': literal,
                 'note': r.note if isinstance(r.note, str) else ''})
out = pd.DataFrame(rows)
out.to_csv('extraction_v2.csv', index=False, encoding='utf-8')
pd.set_option('display.width', 250)
print(out[['record_no', 'task', 'evidence_basis', 'trained_on_flagged_collections', 'external_test_other_institutions',
           'acquisition_level_clarified', 'acquisition_level_protocol_literal']].to_string(index=False))
full = out[out.evidence_basis == 'full text']
print('\nfull texts:', len(full), '| clarified A3:', (full.acquisition_level_clarified == 'A3').sum(),
      '| literal A3:', (full.acquisition_level_protocol_literal == 'A3').sum(),
      '| external test on other institutions:', full.external_test_other_institutions.str.startswith('yes').sum())
