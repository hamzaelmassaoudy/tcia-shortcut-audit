"""Check R13 (6 Nov 2026, plan 7k.4; POST HOC, label-free).
Kernel names are vendor-specific, so notebook 07's pooled kernel decoding (experiment C) partly decodes the vendor.
Here the kernel is decoded within each manufacturer that has at least two kernels with >= 20 scans (notebook 07's
minimum), with notebook 07's code: logistic probe on the embeddings, 5 x 5 CV, macro one-vs-rest AUC with a patient-
bootstrap CI, the within-collection AUC, and shuffled targets (seed 999); both encoders, three inputs, the same 1,249
scans. First, the pooled kernel decoding is recomputed; it must agree with notebook 07's within 0.001.
Output: r13_kernel_within_vendor.csv. Usage: python r13_kernel_within_vendor.py (from this folder)"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import review_lib2 as L

HERE = Path(__file__).resolve().parent
DEC_MIN_LEVEL = 20                                     # notebook 07, cell 3
VARIANTS = ('original', 'background', 'silhouette')
INPUT_NAME = {'original': 'image', 'background': 'background', 'silhouette': 'silhouette'}
# notebook 07's within_collection_auc, executed verbatim from the project notebook (cell "15. Experiment C")
nb07 = json.loads((L.PROJECT / 'notebooks' / '07_screen_validation.ipynb').read_text(encoding='utf-8'))
c19 = next(''.join(c['source']) for c in nb07['cells'] if c['cell_type'] == 'code'
           and 'def within_collection_auc' in ''.join(c['source']))
exec(compile(c19[c19.index('def within_collection_auc'):c19.index('def decoding_targets')], 'nb07_c15', 'exec'),
     vars(L))
within_collection_auc = L.within_collection_auc

t0 = time.time()
keys = sorted(k for k in L.KEYS05 | L.KEYS07 if all(L.emb_file(k, v).exists() for v in VARIANTS))
h = pd.DataFrame([L.header(k) for k in keys])
text = lambda v: str(v).strip() if pd.notna(v) and str(v).strip() else None
vendor = h.hdr_manufacturer.map(lambda v: text(v).split()[0] if text(v) else None)
kernel = h.hdr_kernel.map(text)
coll = np.array([k.split('__')[0] for k in keys], dtype=object)
emb = {(e, v): L.load_input(keys, e, v) for e in ('ctfm', 'merlin') for v in VARIANTS}
ref = pd.read_csv(L.R / '07_screen_validation' / 'decoding' / 'acquisition_decoding.csv')
ref = ref[ref.target == 'kernel'].set_index(['encoder', 'input'])
print(f'{len(keys)} scans with all three inputs ({time.time() - t0:.0f} s)', flush=True)

scopes = [('pooled (reproduction of notebook 07)', np.ones(len(keys), bool))]
for v in sorted(vendor.dropna().unique()):
    scopes.append((f'within {v}', (vendor == v).to_numpy()))
rows = []
for scope, m in scopes:
    vals = kernel.where(m)
    counts = vals.value_counts()
    levels = sorted(counts[counts >= DEC_MIN_LEVEL].index)
    if len(levels) < 2:
        rows.append({'scope': scope, 'n_scans_vendor': int(m.sum()), 'note': 'fewer than two kernels with >= 20 scans',
                     'values': ' | '.join(f'{l}: {int(counts[l])}' for l in counts.index)})
        continue
    keep = vals.isin(levels).to_numpy()
    y = vals[keep].map({l: i for i, l in enumerate(levels)}).to_numpy()
    splits = L.splits_for(y, L.SEED)
    y_shuf = np.random.default_rng(999).permutation(y)
    base = {'scope': scope, 'n_scans_vendor': int(m.sum()), 'n_scans': int(keep.sum()),
            'n_collections': len(set(coll[keep])), 'n_kernels': len(levels),
            'values': ' | '.join(f'{l}: {int(counts[l])}' for l in levels)}
    for enc in ('ctfm', 'merlin'):
        for v in VARIANTS:
            X = emb[(enc, v)][keep]
            P, _ = L.cv_oof_matrix(X, y, splits)
            lo, hi = L.boot_ci(y, P)
            Ps, _ = L.cv_oof_matrix(X, y_shuf, L.splits_for(y_shuf, L.SEED))
            r = {**base, 'encoder': enc, 'input': INPUT_NAME[v], 'auc': L.mean_auc(y, P), 'ci_low': lo, 'ci_high': hi,
                 'within_collection_auc': within_collection_auc(y, P, coll[keep]),
                 'shuffled_target_auc': L.mean_auc(y_shuf, Ps)}
            if scope.startswith('pooled'):
                q = ref.loc[(enc, INPUT_NAME[v])]
                r.update({'nb07_auc': q.pooled_auc, 'nb07_within_collection_auc': q.within_collection_auc,
                          'nb07_largest_difference': max(abs(r['auc'] - q.pooled_auc), abs(r['ci_low'] - q.pooled_ci_low),
                                                         abs(r['ci_high'] - q.pooled_ci_high),
                                                         abs(r['within_collection_auc'] - q.within_collection_auc))})
                r['nb07_reproduced'] = r['nb07_largest_difference'] < 0.001
            rows.append(r)
            print(f'{scope:<38} {enc:<6} {INPUT_NAME[v]:<10} n {int(keep.sum())} k {len(levels)} AUC {r["auc"]:.3f} '
                  f'({lo:.3f}-{hi:.3f}) within-coll {r["within_collection_auc"]:.3f} shuffled '
                  f'{r["shuffled_target_auc"]:.3f} [{time.time() - t0:.0f} s]', flush=True)
res = pd.DataFrame(rows)
res.to_csv(HERE / 'r13_kernel_within_vendor.csv', index=False)
pd.set_option('display.width', 250)
print(res.round(4).to_string(index=False))
print(f'finished in {time.time() - t0:.0f} s')
