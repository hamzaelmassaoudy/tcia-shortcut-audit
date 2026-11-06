"""Check R8b (4 Nov 2026; POST HOC, descriptive).
Describes the refitting sensitivity analysis R8 more completely, from its saved replicates and from its resampling
and split functions (no model is fitted here):
- per model: original AUC, recentred range (AUC +/- 1.96 SD), resample mean, mean shift, resample 2.5th and 97.5th
  percentiles, and whether 0.60 is cleared under each summary;
- width ratios against the conditional CIs, for the originally flagged models and for all 20;
- effective training size: share of the original distinct patients in a bootstrap resample and in each of its
  cross-validation training folds (200 resamples per task, R8's own functions and seeds), against the original 5-fold
  analysis (about 80%).
Outputs: r8b_model_summary.csv, r8b_training_size.csv.  Usage: python r8b_refit_summary.py"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import review_lib as L

rep = pd.read_csv('r8_refit_bootstrap_replicates.csv')
summ = pd.read_csv('r8_refit_bootstrap.csv')
rows = []
for r in summ.itertuples():
    g = rep[(rep.task_id == r.task_id) & (rep.model == r.model)].auc
    p25, p975 = g.quantile(0.025), g.quantile(0.975)
    rows.append({'task_id': r.task_id, 'model': r.model, 'paper_auc': r.paper_auc, 'q': r.q, 'paper_flag': r.paper_flag,
                 'paper_ci_low': r.paper_ci_low, 'paper_ci_high': r.paper_ci_high, 'refit_sd': r.refit_sd,
                 'recentred_low': r.refit_normal_low, 'recentred_high': r.refit_normal_high,
                 'resample_mean': g.mean(), 'mean_shift': g.mean() - r.paper_auc, 'resample_p2.5': p25,
                 'resample_p97.5': p975, 'clears_0.60_recentred': r.refit_normal_low >= 0.60,
                 'clears_0.60_percentile': p25 >= 0.60,
                 'width_ratio': (r.refit_normal_high - r.refit_normal_low) / (r.paper_ci_high - r.paper_ci_low)})
m = pd.DataFrame(rows)
m.to_csv('r8b_model_summary.csv', index=False)

# effective training size, with R8's resampling and split functions and seeds
labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
feat = L.patient_features()
TASKS = ['renal subtype (KIRC vs KIRP)', 'CPTAC-PDA | stage', 'lung histology (LUAD vs LUSC)',
         'LIDC-IDRI | primary vs metastatic']


def resample(y, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum(), replace=True) for c in np.unique(y)])


ts = []
for tid in TASKS:
    _, y, _ = L.task_frame(tid, labels, feat)
    n = len(y)
    for b in range(200):
        rng = np.random.default_rng([L.SEED, b])
        for t in TASKS[:TASKS.index(tid)]:               # R8 drew the tasks in this order within one generator
            _, yt, _ = L.task_frame(t, labels, feat)
            resample(yt, rng)
        idx = resample(y, rng)
        yb = y[idx]
        for r in range(L.N_REPEATS):
            for tr, _ in StratifiedGroupKFold(L.N_SPLITS, shuffle=True, random_state=100_000 + 10 * b + r).split(
                    np.zeros(len(yb)), yb, idx):
                ts.append({'task_id': tid, 'b': b, 'resample_distinct_share': len(np.unique(idx)) / n,
                           'train_fold_distinct_share': len(np.unique(idx[tr])) / n})
ts = pd.DataFrame(ts)
ts.to_csv('r8b_training_size.csv', index=False)

pd.set_option('display.width', 250)
print(m.round(3).to_string(index=False))
f = m[m.paper_flag]
print('\nwidth ratio, originally flagged:', round(f.width_ratio.min(), 4), '-', round(f.width_ratio.max(), 4),
      '| all 20:', round(m.width_ratio.min(), 4), '-', round(m.width_ratio.max(), 4))
print('mean shift, all 20:', round(m.mean_shift.min(), 4), '-', round(m.mean_shift.max(), 4),
      '| negative in', int((m.mean_shift < 0).sum()), 'of 20 | flagged:', round(f.mean_shift.min(), 4), '-', round(f.mean_shift.max(), 4))
print('resample distinct share:', ts.resample_distinct_share.describe()[['mean', 'min', 'max']].round(3).to_dict())
print('training-fold distinct share:', ts.train_fold_distinct_share.describe()[['mean', 'min', 'max']].round(3).to_dict())
