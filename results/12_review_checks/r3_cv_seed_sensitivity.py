"""Check R3 (4 Nov 2026; POST HOC sensitivity, no new data).
How much do the cross-validated AUCs and their lower CI bounds depend on the random split? The paper's estimates use
splits seeded 2026..2030 (5 x 5 CV). Here the same models (notebook 05's verbatim code, review_lib.py) are refitted
with 20 further seeds (5 x 5 CV each); the bootstrap CI is recomputed per seed for the tasks whose flag decision could
be close. The permutation P values use one fixed label-independent split and do not depend on the CV seed, so the flag
decision per seed changes only through the lower CI bound (q taken from notebooks 03 and 05).
Outputs: r3_metadata_seeds.csv, r3_image_seeds.csv, r3_summary.csv. Usage: python r3_cv_seed_sensitivity.py"""
import time
import numpy as np
import pandas as pd
import review_lib as L

SEEDS = [L.SEED] + [3000 + 10 * k for k in range(20)]      # seed + r for r = 0..4, so these never overlap
CI_TASKS = ['renal subtype (KIRC vs KIRP)', 'CPTAC-PDA | stage', 'lung histology (LUAD vs LUSC)',
            'LIDC-IDRI | primary vs metastatic', 'LIDC-IDRI | benign vs malignant', 'TCGA-LIHC | grade',
            'TCGA-KIRC | stage']
labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
feat = L.patient_features()
a03 = pd.read_csv(L.R / '03_audit' / 'audit_primary.csv').set_index('task_id')
p05 = pd.read_csv(L.R / '05_image_check' / 'real' / 'image_probe.csv')
t0 = time.time()

rows = []
for tid in a03.index:
    df, y, _ = L.task_frame(tid, labels, feat)
    for s in SEEDS:
        P = L.cv_oof_df(df, y, L.SCANNER_COLS, L.splits_for(y, s))
        lo, hi = L.boot_ci(y, P) if tid in CI_TASKS else (np.nan, np.nan)
        rows.append({'task_id': tid, 'seed': s, 'auc': L.mean_auc(y, P), 'ci_low': lo, 'ci_high': hi,
                     'q': a03.perm_q_bh[tid], 'flag': bool(lo >= 0.60 and a03.perm_q_bh[tid] < 0.05) if tid in CI_TASKS else np.nan})
    print(f'metadata {tid}: done ({time.time() - t0:.0f} s)', flush=True)
meta = pd.DataFrame(rows)
meta.to_csv('r3_metadata_seeds.csv', index=False)

rows = []
for tid in p05.task_id.unique():
    df, y, _ = L.task_frame(tid, labels, feat)
    for enc in ('ctfm', 'merlin'):
        ref = p05[(p05.task_id == tid) & (p05.encoder == enc)].iloc[0]
        for v, name in (('original', 'image'), ('background', 'background')):
            X = L.load_emb(df.key, enc, v, L.R / '05_image_check' / 'cache' / 'embeddings')
            for s in SEEDS:
                P, _ = L.cv_oof_matrix(X, y, L.splits_for(y, s))
                lo, hi = L.boot_ci(y, P)
                q = ref[f'{name}_perm_q_bh']
                rows.append({'task_id': tid, 'encoder': enc, 'input': name, 'seed': s, 'auc': L.mean_auc(y, P),
                             'ci_low': lo, 'ci_high': hi, 'q': q, 'flag': bool(lo >= 0.60 and q < 0.05),
                             'paper_auc': ref[f'{name}_auc'], 'paper_flag': bool(ref[f'{name}_flagged'])})
        print(f'image {tid} {enc}: done ({time.time() - t0:.0f} s)', flush=True)
img = pd.DataFrame(rows)
img.to_csv('r3_image_seeds.csv', index=False)

def summ(d, keys):
    other = d[d.seed != L.SEED]
    g = other.groupby(keys)
    s = pd.DataFrame({'auc_seed2026': d[d.seed == L.SEED].set_index(keys).auc,
                      'auc_min': g.auc.min(), 'auc_median': g.auc.median(), 'auc_max': g.auc.max(),
                      'ci_low_seed2026': d[d.seed == L.SEED].set_index(keys).ci_low,
                      'ci_low_min': g.ci_low.min(), 'ci_low_max': g.ci_low.max(),
                      'flag_share_other_seeds': g.flag.apply(lambda f: np.nan if f.isna().all() else float(f.astype(bool).mean()))})
    return s.reset_index()
out = pd.concat([summ(meta, ['task_id']).assign(kind='metadata', encoder='', input='metadata'),
                 summ(img, ['task_id', 'encoder', 'input']).assign(kind='image probe')], ignore_index=True)
out.to_csv('r3_summary.csv', index=False)
pd.set_option('display.width', 250)
print(out.round(3).to_string(index=False))
print(f'finished in {time.time() - t0:.0f} s')
