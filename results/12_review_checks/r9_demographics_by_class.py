"""Check R9 (5 Nov 2026; POST HOC, descriptive): patient sex and age by class.
For the four flagged tasks and the external lung cohorts, from the DICOM study records of the TCIA interface
(results/01_inventory/studies.csv.gz: PatientSex, PatientAge; the first study with a value per patient). Body outline
and habitus differ by sex and age, so these describe a possible non-acquisition contributor to background and
silhouette signals. Cohorts without saved study records are listed as not available.
Output: r9_demographics_by_class.csv.  Usage: python r9_demographics_by_class.py (from this folder)"""
import re
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1]
st = pd.read_csv(R / '01_inventory' / 'studies.csv.gz', low_memory=False)
st['age'] = st.PatientAge.astype(str).str.extract(r'(\d+)Y', expand=False).astype(float)
st['sex'] = st.PatientSex.where(st.PatientSex.isin(['M', 'F']))
demo = (st.sort_values('StudyDate').groupby(['Collection', 'PatientID'])
          .agg(sex=('sex', lambda s: s.dropna().iloc[0] if s.notna().any() else None),
               age=('age', lambda s: s.dropna().iloc[0] if s.notna().any() else float('nan'))).reset_index())

labels = pd.read_csv(R / '02_tasks_features' / 'labels_long.csv')
groups = []
for tid in ['renal subtype (KIRC vs KIRP)', 'CPTAC-PDA | stage', 'lung histology (LUAD vs LUSC)',
            'LIDC-IDRI | primary vs metastatic']:
    d = labels[labels.task_id == tid][['collection', 'PatientID', 'label']]
    groups.append((tid, d))
# external lung cohorts: patients and labels from the saved external scores (one row per patient)
s08 = pd.concat([pd.read_csv(f) for f in sorted((R / '08_transfer' / 'real' / 'per_pair').glob('lung*scores.csv.gz'))])
s11 = pd.read_csv(R / '11_external_lung' / 'real' / 'transfer_scores.csv.gz')
ext = pd.concat([s08[s08.direction == 'external'], s11.assign(direction='external')])
for pair, g in ext.groupby('pair'):
    g = g.drop_duplicates('key')
    d = pd.DataFrame({'collection': g.key.str.split('__').str[0], 'PatientID': g.key.str.split('__').str[1],
                      'label': g.label.map({0: 'adenocarcinoma', 1: 'squamous cell carcinoma'})})
    groups.append((pair.replace('lung: TCGA -> ', 'external lung cohort: '), d))

rows = []
for name, d in groups:
    m = d.merge(demo, left_on=['collection', 'PatientID'], right_on=['Collection', 'PatientID'], how='left')
    for cls, g in m.groupby('label'):
        n_sex, n_age = int(g.sex.notna().sum()), int(g.age.notna().sum())
        rows.append({'group': name, 'class': cls, 'n_patients': len(g), 'n_sex_known': n_sex,
                     'male_share': round((g.sex == 'M').sum() / n_sex, 3) if n_sex else None, 'n_age_known': n_age,
                     'median_age': g.age.median() if n_age else None,
                     'age_q1': g.age.quantile(0.25) if n_age else None, 'age_q3': g.age.quantile(0.75) if n_age else None})
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name('r9_demographics_by_class.csv'), index=False)
pd.set_option('display.width', 250)
print(out.to_string(index=False))
