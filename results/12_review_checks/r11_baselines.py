"""Check R11 (6 Nov 2026, plan 7k.2; POST HOC).
Can one patient- or scan-level property alone predict the label? Notebook 05's logistic regression (5 x 5 CV,
patient-bootstrap CI, permutation test with 999 permutations on the fixed split) on:
  (a) scan length along the body axis (DICOM header, hdr_coverage_mm);
  (b) reconstruction diameter (hdr_recon_diameter);
  (c) both;      (a)-(c) for the 19 CT tasks, patients with an embedded scan (headers of notebooks 05 and 07);
  (d) sex and age (study records, as R9), for all 26 tasks.
Missing values are imputed inside the training folds with an indicator (notebook 05's numeric pipeline). BH within each
baseline over its tasks; flagged as the screen (lower CI >= 0.60 and q < 0.05). For the four flagged tasks, the
background-only probe (refitted with notebook 05's code and splits) is compared with each baseline by notebook 05's
paired bootstrap (boot_diff).
Outputs: r11_baselines.csv, r11_background_vs_baselines.csv. Usage: python r11_baselines.py (from this folder)"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import review_lib2 as L

HERE = Path(__file__).resolve().parent
FLAGGED = ['renal subtype (KIRC vs KIRP)', 'CPTAC-PDA | stage', 'lung histology (LUAD vs LUSC)',
           'LIDC-IDRI | primary vs metastatic']
BASELINES = {'scan length': ['hdr_coverage_mm'], 'reconstruction diameter': ['hdr_recon_diameter'],
             'scan length + reconstruction diameter': ['hdr_coverage_mm', 'hdr_recon_diameter'],
             'sex + age': ['male', 'age']}
t0 = time.time()
labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
feat = L.patient_features()
a03 = pd.read_csv(L.R / '03_audit' / 'audit_primary.csv')
u07 = pd.read_csv(L.R / '07_screen_validation' / 'real' / 'image_probe_unflagged_tasks.csv')
ct_image_tasks = FLAGGED + sorted(u07.task_id.unique())
assert len(ct_image_tasks) == 19, len(ct_image_tasks)
demo = L.demographics()
embedded = L.KEYS05 | L.KEYS07
rows, diffs = [], []
for tid in a03.task_id:
    df0, y0, classes = L.task_frame(tid, labels, feat)
    df0 = df0.merge(demo, on=['collection', 'PatientID'], how='left', validate='many_to_one')
    for name, cols in BASELINES.items():
        if name != 'sex + age' and tid not in ct_image_tasks:
            continue
        if name == 'sex + age':
            keep = np.ones(len(df0), bool)
        else:
            keep = df0.key.isin(embedded).to_numpy()
        df, y = df0[keep].reset_index(drop=True), y0[keep]
        if name != 'sex + age':
            h = pd.DataFrame([L.header(k) for k in df.key])
            for c in ('hdr_coverage_mm', 'hdr_recon_diameter'):
                df[c] = pd.to_numeric(h[c], errors='coerce') if c in h else np.nan
        splits = L.splits_for(y, L.SEED)
        P = L.cv_oof_df(df, y, cols, splits)
        lo, hi = L.boot_ci(y, P)
        obs, p, null = L.permutation_test(L.encoded_folds_df(df, cols), y)
        rows.append({'task_id': tid, 'baseline': name, 'n_patients': len(y), 'n_classes': len(classes),
                     'share_missing': float(df[cols].isna().any(axis=1).mean()), 'auc': L.mean_auc(y, P),
                     'ci_low': lo, 'ci_high': hi, 'perm_p': p, 'null_mean_auc': float(null.mean())})
        if tid in FLAGGED:
            for enc in ('ctfm', 'merlin'):
                Xb = L.load_input(df.key, enc, 'background')
                Pb, _ = L.cv_oof_matrix(Xb, y, splits)
                d, dlo, dhi = L.boot_diff(y, Pb, P)
                diffs.append({'task_id': tid, 'encoder': enc, 'baseline': name, 'background_auc': L.mean_auc(y, Pb),
                              'baseline_auc': L.mean_auc(y, P), 'background_minus_baseline': d, 'ci_low': dlo,
                              'ci_high': dhi})
        print(f'{tid:<44} {name:<38} AUC {rows[-1]["auc"]:.3f} ({lo:.3f}-{hi:.3f}) p {p:.3f} '
              f'[{time.time() - t0:.0f} s]', flush=True)
res = pd.DataFrame(rows)
res['q_bh'] = np.nan
for name, g in res.groupby('baseline'):
    res.loc[g.index, 'q_bh'] = L.bh(g.perm_p.to_numpy())
res['flagged'] = (res.ci_low >= 0.60) & (res.q_bh < 0.05)
res.to_csv(HERE / 'r11_baselines.csv', index=False)
pd.DataFrame(diffs).to_csv(HERE / 'r11_background_vs_baselines.csv', index=False)
pd.set_option('display.width', 250)
print(res.round(3).to_string(index=False))
print(pd.DataFrame(diffs).round(3).to_string(index=False))
print(f'finished in {time.time() - t0:.0f} s')
