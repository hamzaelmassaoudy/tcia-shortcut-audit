"""Check R8 (4 Nov 2026; POST HOC sensitivity analysis).

The paper's 95% CIs bootstrap patients over fixed out-of-fold predictions, so they are conditional on the fitted models
and ignore training variability. R8 estimates the variability of the whole procedure: each bootstrap replicate draws
patients with replacement within each class, then reruns the complete 5 x 5 cross-validation, fitting every model anew.
Copies of the same patient are kept in the same fold (StratifiedGroupKFold, groups = original patient), so a patient is
never in both training and test data. The splits are redrawn in every replicate.

Models (the paper's own code, review_lib.py): the metadata model on the primary fields, and the linear probes on the
frozen CT-FM and Merlin embeddings (full image, background only), for the four flagged tasks.

Reported per model:
- the paper's AUC and its conditional CI;
- the refit-bootstrap SD;
- a normal interval, paper AUC +/- 1.96 SD;
- the replicate percentiles. These are biased downwards, because each replicate trains on about 63% unique patients.

Outputs: r8_refit_bootstrap_replicates.csv, r8_refit_bootstrap.csv.
Usage: python r8_refit_bootstrap.py [B]  (default B = 1000)"""
import sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
import review_lib as L

L.N_JOBS = 1                       # parallelise over replicates instead of folds
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
TASKS = ['renal subtype (KIRC vs KIRP)', 'CPTAC-PDA | stage', 'lung histology (LUAD vs LUSC)',
         'LIDC-IDRI | primary vs metastatic']
labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
feat = L.patient_features()
a03 = pd.read_csv(L.R / '03_audit' / 'audit_primary.csv').set_index('task_id')
p05 = pd.read_csv(L.R / '05_image_check' / 'real' / 'image_probe.csv')
EMB = L.R / '05_image_check' / 'cache' / 'embeddings'


def resample(y, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum(), replace=True) for c in np.unique(y)])


def grouped_splits(yb, groups, seed):
    return [list(StratifiedGroupKFold(L.N_SPLITS, shuffle=True, random_state=seed + r).split(np.zeros(len(yb)), yb, groups))
            for r in range(L.N_REPEATS)]


def replicate(b, task_data):
    rng = np.random.default_rng([L.SEED, b])
    out = []
    for tid, (df, y, embs) in task_data.items():
        idx = resample(y, rng)
        yb = y[idx]
        sp = grouped_splits(yb, idx, 100_000 + 10 * b)
        out.append({'b': b, 'task_id': tid, 'model': 'metadata',
                    'auc': L.mean_auc(yb, L.cv_oof_df(df.iloc[idx].reset_index(drop=True), yb, L.SCANNER_COLS, sp)),
                    'n_unique': len(np.unique(idx))})
        for (enc, inp), X in embs.items():
            P, _ = L.cv_oof_matrix(X[idx], yb, sp)
            out.append({'b': b, 'task_id': tid, 'model': f'{enc} {inp}', 'auc': L.mean_auc(yb, P),
                        'n_unique': len(np.unique(idx))})
    return out


task_data = {}
for tid in TASKS:
    df, y, _ = L.task_frame(tid, labels, feat)
    embs = {(enc, inp): L.load_emb(df.key, enc, v, EMB)
            for enc in ('ctfm', 'merlin') for v, inp in (('original', 'image'), ('background', 'background'))}
    task_data[tid] = (df, y, embs)

t0 = time.time()
reps = Parallel(n_jobs=-1, verbose=5)(delayed(replicate)(b, task_data) for b in range(B))
reps = pd.DataFrame([r for rr in reps for r in rr])
reps.to_csv('r8_refit_bootstrap_replicates.csv', index=False)

rows = []
for (tid, model), g in reps.groupby(['task_id', 'model'], sort=False):
    if model == 'metadata':
        a = a03.loc[tid]
        auc, lo, hi, q = a.auc, a.ci_low, a.ci_high, a.perm_q_bh
    else:
        enc, inp = model.split()
        r = p05[(p05.task_id == tid) & (p05.encoder == enc)].iloc[0]
        auc, lo, hi, q = r[f'{inp}_auc'], r[f'{inp}_ci_low'], r[f'{inp}_ci_high'], r[f'{inp}_perm_q_bh']
    sd = g.auc.std(ddof=1)
    rows.append({'task_id': tid, 'model': model, 'paper_auc': auc, 'paper_ci_low': lo, 'paper_ci_high': hi, 'q': q,
                 'paper_flag': bool(lo >= 0.60 and q < 0.05), 'refit_sd': sd,
                 'refit_normal_low': auc - 1.96 * sd, 'refit_normal_high': auc + 1.96 * sd,
                 'refit_flag': bool(auc - 1.96 * sd >= 0.60 and q < 0.05),
                 'replicate_mean': g.auc.mean(), 'replicate_p2.5': g.auc.quantile(0.025),
                 'replicate_p97.5': g.auc.quantile(0.975), 'conditional_ci_width': hi - lo,
                 'refit_interval_width': 3.92 * sd, 'n_replicates': len(g)})
res = pd.DataFrame(rows)
res.to_csv('r8_refit_bootstrap.csv', index=False)
pd.set_option('display.width', 250)
print(res.round(3).to_string(index=False))
print(f'B = {B}; finished in {time.time() - t0:.0f} s')
