"""POST HOC sensitivity check of notebook 11 (plan 7j A.9), written on 4 Nov 2026 after the real-label results were read.

Question: does plan 7j's reading change if the 17 Lung-PET-CT-Dx patients whose main series was replaced under the
amendment (cache/series_set_aside.csv) are left out?

Method: the TCGA source models are fixed (nothing is refitted), so each external AUC is recomputed from notebook 11's
saved per-patient scores (real/transfer_scores.csv.gz) on the remaining patients, with notebook 11's own AUC and
bootstrap functions (2,000 patient resamples, seed 2026) and notebook 08's transfer rule (95% CI wholly above 0.5).
No permutation p is given: it needs refits on permuted source labels. Step 1 checks that the same code reproduces
notebook 11's numbers on all patients. Usage: python sensitivity_without_swapped.py (from this folder)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
SEED, N_BOOT, TOL = 2026, 2000, 1e-9


def auc_binary(y, s):  # notebook 11, unchanged
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(y, s):  # notebook 11's boot_ci on as_P(s): one repeat, two classes
    rng = np.random.default_rng(SEED)
    vals = []
    while len(vals) < N_BOOT:
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auc_binary(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


scores = pd.read_csv(RUN / 'real' / 'transfer_scores.csv.gz')
swapped = set(pd.read_csv(RUN / 'cache' / 'series_set_aside.csv').key)
transfer = pd.read_csv(RUN / 'real' / 'transfer.csv')
silhouette = pd.read_csv(RUN / 'real' / 'silhouette_transfer.csv')
recorded = {(r.pair, r.encoder, r.input): (r.external_auc, r.external_ci_low, r.external_ci_high)
            for r in transfer.itertuples()}
recorded.update({(r.pair, r.encoder, 'silhouette'): (r.silhouette_external_auc, r.silhouette_ci_low, r.silhouette_ci_high)
                 for r in silhouette.itertuples()})
assert len(swapped) == 17 and all(k.startswith('Lung-PET-CT-Dx__') for k in swapped)

rows = []
for (pair, enc, inp), g in scores.groupby(['pair', 'encoder', 'input'], sort=False):
    y, s = g.label.to_numpy(int), g.score.to_numpy(float)
    a_all = auc_binary(y, s)
    lo_all, hi_all = boot_ci(y, s)
    ref = recorded[(pair, enc, inp)]
    assert max(abs(a_all - ref[0]), abs(lo_all - ref[1]), abs(hi_all - ref[2])) < TOL, (pair, enc, inp, ref)
    keep = ~g.key.isin(swapped).to_numpy()
    y2, s2 = y[keep], s[keep]
    a, (lo, hi) = auc_binary(y2, s2), boot_ci(y2, s2)
    rows.append({'pair': pair, 'encoder': enc, 'input': inp, 'n_all': len(y), 'n_without_swapped': int(keep.sum()),
                 'class1_without_swapped': int(y2.sum()), 'external_auc_all': a_all, 'ci_low_all': lo_all,
                 'ci_high_all': hi_all, 'transfers_all': bool(lo_all > 0.5), 'external_auc_without_swapped': a,
                 'ci_low_without_swapped': lo, 'ci_high_without_swapped': hi, 'transfers_without_swapped': bool(lo > 0.5)})
out = pd.DataFrame(rows)
out.to_csv(HERE / 'transfer_without_swapped.csv', index=False)

new = ['NSCLC-Radiomics', 'Lung-PET-CT-Dx']
bg = out[out.input == 'background']
transferred = {t: bool(bg[bg.pair == f'lung: TCGA -> {t}'].transfers_without_swapped.any()) for t in new}
n = sum(transferred.values())
reading = ('consistent across independent cohorts' if n == 0 else 'partly consistent' if n < len(new) else 'not confirmed')
summary = {'label': 'POST HOC sensitivity check (not in plan 7j); written 2026-11-04 after the real results were read',
           'patients_left_out': sorted(swapped), 'all_patients_reproduce_notebook11': True,
           'background_transferred_without_swapped': transferred, 'reading_7j_without_swapped': reading,
           'any_change_in_transfer_decision': bool((out.transfers_all != out.transfers_without_swapped).any())}
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
pd.set_option('display.width', 250)
print(out[out.pair.str.contains('Lung-PET-CT-Dx')].drop(columns='pair').round(3).to_string(index=False))
print(json.dumps({k: v for k, v in summary.items() if k != 'patients_left_out'}, indent=1))
