"""Check R11b (6 Nov 2026; POST HOC, descriptive; added AFTER the R11 result was seen, not in plan 7k).
R11's sex-and-age baseline imputes missing values with an indicator. In LIDC-IDRI primary versus metastatic it
predicted the class (AUC 0.81), and age or sex was missing far more often for one class. This splits that baseline,
for the four flagged tasks, into (i) the two missingness indicators alone (which patients have sex or age recorded in
the DICOM study records) and (ii) sex and age among the patients with both recorded, with the same model and
statistics as R11 (notebook 05's code: 5 x 5 CV, bootstrap CI, permutation test).
Output: r11b_demographics_detail.csv. Usage: python r11b_demographics_detail.py (from this folder)"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import review_lib2 as L

HERE = Path(__file__).resolve().parent
FLAGGED = ['renal subtype (KIRC vs KIRP)', 'CPTAC-PDA | stage', 'lung histology (LUAD vs LUSC)',
           'LIDC-IDRI | primary vs metastatic']
t0 = time.time()
labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
feat = L.patient_features()
demo = L.demographics()
rows = []
for tid in FLAGGED:
    df0, y0, classes = L.task_frame(tid, labels, feat)
    df0 = df0.merge(demo, on=['collection', 'PatientID'], how='left', validate='many_to_one')
    df0['sex_missing'] = df0.male.isna().astype(float)
    df0['age_missing'] = df0.age.isna().astype(float)
    base = {'task_id': tid}
    for j, c in enumerate(classes):
        m = y0 == j
        base[f'class{j}'] = c
        base[f'class{j}_n'] = int(m.sum())
        base[f'class{j}_sex_missing'] = int(df0.male[m].isna().sum())
        base[f'class{j}_age_missing'] = int(df0.age[m].isna().sum())
    for name, cols, keep in (('missingness indicators only', ['sex_missing', 'age_missing'], np.ones(len(y0), bool)),
                             ('sex + age, patients with both recorded', ['male', 'age'],
                              (df0.male.notna() & df0.age.notna()).to_numpy())):
        df, y = df0[keep].reset_index(drop=True), y0[keep]
        r = {**base, 'model': name, 'n_patients': len(y), 'n_per_class': ' | '.join(str(int((y == j).sum()))
                                                                                    for j in range(len(classes)))}
        if min(np.bincount(y, minlength=len(classes))) < 5 or df[cols].nunique().max() < 2:
            r['note'] = 'too few patients or no variation'
        else:
            P = L.cv_oof_df(df, y, cols, L.splits_for(y, L.SEED))
            lo, hi = L.boot_ci(y, P)
            obs, p, null = L.permutation_test(L.encoded_folds_df(df, cols), y)
            r.update({'auc': L.mean_auc(y, P), 'ci_low': lo, 'ci_high': hi, 'perm_p': p})
        rows.append(r)
        print(f'{tid:<36} {name:<40} {r.get("auc", float("nan")):.3f} n {len(y)} [{time.time() - t0:.0f} s]', flush=True)
out = pd.DataFrame(rows)
out.to_csv(HERE / 'r11b_demographics_detail.csv', index=False)
pd.set_option('display.width', 250)
print(out.round(3).to_string(index=False))
