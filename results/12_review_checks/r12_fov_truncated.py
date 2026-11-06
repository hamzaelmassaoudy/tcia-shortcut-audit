"""Check R12 (6 Nov 2026, plan 7k.3; POST HOC).
Scans with the body cut off by the reconstruction field of view: their body mask is the whole field-of-view circle
(plan 7f, item 4), so the background-only input is nearly empty and the silhouette shows the field-of-view circle.
They are the 9 of notebook 05's 525 scans whose dilated body mask covers >= 75% of the image. Here they are counted by
class, and for each task that contains one, the image, background-only and silhouette probes (both encoders) are
refitted without them and compared with a refit on all scans, with notebook 05's code (5 x 5 CV, patient-bootstrap CI,
permutation test on the fixed split). BH over the four flagged tasks per encoder and input, with the stored P of the
task that has no such scan (notebook 05 for image and background, notebook 06 for silhouette).
Outputs: r12_fov_truncated_scans.csv, r12_fov_truncated.csv. Usage: python r12_fov_truncated.py (from this folder)"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import review_lib2 as L

HERE = Path(__file__).resolve().parent
FLAGGED = ['renal subtype (KIRC vs KIRP)', 'CPTAC-PDA | stage', 'lung histology (LUAD vs LUSC)',
           'LIDC-IDRI | primary vs metastatic']
INPUTS = {'image': 'original', 'background': 'background', 'silhouette': 'silhouette'}
t0 = time.time()
labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
feat = L.patient_features()
inv = pd.read_csv(L.C05 / 'image_inventory.csv')
cut = set(inv.key[inv.body_fraction_dilated >= 0.75])
assert len(cut) == 9, len(cut)
p05 = pd.read_csv(L.R / '05_image_check' / 'real' / 'image_probe.csv')
p06 = pd.read_csv(L.R / '06_image_posthoc' / 'real' / 'silhouette_probe.csv')


def stored(tid, enc, name):
    if name == 'silhouette':
        r = p06[(p06.task_id == tid) & (p06.encoder == enc)].iloc[0]
        return r.silhouette_auc, r.silhouette_ci_low, r.silhouette_ci_high, r.silhouette_perm_p, bool(r.silhouette_flagged)
    r = p05[(p05.task_id == tid) & (p05.encoder == enc)].iloc[0]
    return (r[f'{name}_auc'], r[f'{name}_ci_low'], r[f'{name}_ci_high'], r[f'{name}_perm_p'], bool(r[f'{name}_flagged']))


scans, rows = [], []
for tid in FLAGGED:
    df0, y0, classes = L.task_frame(tid, labels, feat)
    inside = df0.key.isin(cut).to_numpy()
    for k, yy in zip(df0.key[inside], y0[inside]):
        r = inv.set_index('key').loc[k]
        scans.append({'key': k, 'task_id': tid, 'class': classes[yy], 'body_fraction_dilated': r.body_fraction_dilated,
                      'recon_diameter_mm': r.hdr_recon_diameter})
    if not inside.any():
        for enc in ('ctfm', 'merlin'):
            for name in INPUTS:
                a, lo, hi, p, f = stored(tid, enc, name)
                for subset in ('all scans', 'without cut-off scans'):
                    rows.append({'task_id': tid, 'encoder': enc, 'input': name, 'subset': subset,
                                 'n_patients': len(y0), 'n_removed': 0, 'auc': a, 'ci_low': lo, 'ci_high': hi,
                                 'perm_p': p, 'source': 'stored (no cut-off scan in this task)', 'stored_auc': a,
                                 'stored_flagged': f})
        continue
    for enc in ('ctfm', 'merlin'):
        for name, variant in INPUTS.items():
            X0 = L.load_input(df0.key, enc, variant)
            a_s, _, _, _, f_s = stored(tid, enc, name)
            for subset, keep in (('all scans', np.ones(len(y0), bool)), ('without cut-off scans', ~inside)):
                ok = keep & ~np.isnan(X0).any(axis=1)
                X, y = X0[ok], y0[ok]
                P, _ = L.cv_oof_matrix(X, y, L.splits_for(y, L.SEED))
                lo, hi = L.boot_ci(y, P)
                obs, p, null = L.permutation_test(L.scaled_folds(X), y)
                rows.append({'task_id': tid, 'encoder': enc, 'input': name, 'subset': subset, 'n_patients': len(y),
                             'n_removed': int((~keep).sum()), 'auc': L.mean_auc(y, P), 'ci_low': lo, 'ci_high': hi,
                             'perm_p': p, 'source': 'refit', 'stored_auc': a_s, 'stored_flagged': f_s})
                print(f'{tid:<36} {enc:<6} {name:<10} {subset:<22} n {len(y)} AUC {rows[-1]["auc"]:.4f} '
                      f'({lo:.4f}-{hi:.4f}) p {p:.4f} [{time.time() - t0:.0f} s]', flush=True)
res = pd.DataFrame(rows)
res['q_bh_4tasks'] = np.nan
for _, g in res.groupby(['encoder', 'input', 'subset']):
    assert len(g) == 4
    res.loc[g.index, 'q_bh_4tasks'] = L.bh(g.perm_p.to_numpy())
res['flagged'] = (res.ci_low >= 0.60) & (res.q_bh_4tasks < 0.05)
res['auc_minus_stored'] = res.auc - res.stored_auc
res.to_csv(HERE / 'r12_fov_truncated.csv', index=False)
pd.DataFrame(scans).to_csv(HERE / 'r12_fov_truncated_scans.csv', index=False)
pd.set_option('display.width', 250)
print(pd.DataFrame(scans).to_string(index=False))
print(res.drop(columns='source').round(4).to_string(index=False))
print(f'finished in {time.time() - t0:.0f} s')
