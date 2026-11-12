"""Post hoc, descriptive (12 Nov 2026; not in plan 7o): the network's cross-validated AUC on the background-only input
minus that on the full image in TCGA lung, with a paired patient bootstrap (2000 resamples, seed 2026), from the saved
out-of-fold predictions of notebook 13. Written after the results of notebook 13 had been seen.
Usage (from this folder): python posthoc_background_minus_full.py"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
R = HERE.parent
SEED, N_BOOT, SOURCE = 2026, 2000, 'lung histology (LUAD vs LUSC)'


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rows = []
for mode, seeds in (('rehearsal', (0,)), ('real', (0, 1, 2))):
    lab = pd.read_csv(R / '12_image_experiments' / mode / 'dataset_labels.csv')
    src = lab[lab.dataset == SOURCE].reset_index(drop=True)
    ks, ys = src.key.to_numpy(), src.y.to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(ks, ys))
    s = {}
    for inp in ('full', 'background'):
        oof = np.zeros((len(seeds), len(ks)))
        for f, (tr, te) in enumerate(folds):
            for si, sd in enumerate(seeds):
                with np.load(glob.glob(str(HERE / 'checkpoints' / f'{mode}__cv__{inp}__fold{f}__seed{sd}__*.npz'))[0]) as z:
                    oof[si, te] = z['p']
        s[inp] = oof.mean(0)
    rec = pd.read_csv(HERE / mode / 'cnn_internal.csv').set_index('input').auc
    assert abs(auc(ys, s['full']) - rec['full']) < 1e-12 and abs(auc(ys, s['background']) - rec['background']) < 1e-12
    rng = np.random.default_rng(SEED)
    vals = []
    while len(vals) < N_BOOT:
        i = rng.choice(len(ys), len(ys), replace=True)
        if len(np.unique(ys[i])) == 2:
            vals.append(auc(ys[i], s['background'][i]) - auc(ys[i], s['full'][i]))
    rows.append({'mode': mode, 'n': len(ys), 'seeds': len(seeds), 'auc_full': auc(ys, s['full']),
                 'auc_background': auc(ys, s['background']),
                 'background_minus_full': auc(ys, s['background']) - auc(ys, s['full']),
                 'ci_low': float(np.percentile(vals, 2.5)), 'ci_high': float(np.percentile(vals, 97.5))})
out = pd.DataFrame(rows)
out.to_csv(HERE / 'posthoc_background_minus_full.csv', index=False)
print(out.round(3).to_string())
