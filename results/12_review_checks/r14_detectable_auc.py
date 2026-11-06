"""Check R14 (6 Nov 2026, plan 7k.5; POST HOC, simulation, for CLAIM item 21).
How large must a task's true AUC be for the flag's CI criterion (lower bound of the 95% percentile patient-bootstrap
CI >= 0.60) to be met with 80% probability, at that task's class sizes?
Model: binormal scores with equal variances (negatives N(0, 1), positives N(d, 1), true AUC = Phi(d / sqrt 2)).
1,000 simulated data sets per AUC value, each with 1,000 bootstrap resamples of all patients (resamples missing a
class are dropped, as in the paper); the AUC is found by bisection to 0.001. Every AUC value uses the same random
numbers (seed 2026; data set d uses default_rng([2026, d])), so the probability rises monotonically with the AUC.
Not modelled: the permutation q criterion (it can only raise the AUC needed) and the averaging over CV repeats.
Class sizes: the 26 metadata tasks (the four-class task is not simulated) and the 19 CT image tasks (patients with an
embedded scan). Output: r14_detectable_auc.csv. Usage: python r14_detectable_auc.py (from this folder)"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
import review_lib2 as L

HERE = Path(__file__).resolve().parent
N_SIM, N_BOOT_SIM, POWER, CUT = 1000, 1000, 0.80, 0.60


def lower_bounds(auc_true, n0, n1):
    d = np.sqrt(2) * norm.ppf(auc_true)
    n = n0 + n1
    lab = np.r_[np.zeros(n0, bool), np.ones(n1, bool)]
    out = np.empty(N_SIM)
    for s in range(N_SIM):
        rng = np.random.default_rng([2026, s])
        z = rng.standard_normal(n)
        idx = rng.integers(0, n, size=(N_BOOT_SIM, n))
        W = np.bincount((idx + np.arange(N_BOOT_SIM)[:, None] * n).ravel(), minlength=N_BOOT_SIM * n
                        ).reshape(N_BOOT_SIM, n).astype(float)
        score = z + d * lab
        o = np.argsort(score, kind='stable')
        Wo, pos = W[:, o], lab[o]
        negw = Wo * ~pos
        below = np.cumsum(negw, axis=1) - negw
        wp, wn = Wo[:, pos].sum(1), Wo[:, ~pos].sum(1)
        ok = (wp > 0) & (wn > 0)
        a = (Wo * pos * below).sum(1)[ok] / (wp[ok] * wn[ok])
        out[s] = np.percentile(a, 2.5)
    return out


def detectable(n0, n1):
    lo, hi = CUT, 0.999
    while hi - lo > 0.001:
        mid = (lo + hi) / 2
        if (lower_bounds(mid, n0, n1) >= CUT).mean() >= POWER:
            hi = mid
        else:
            lo = mid
    return hi, float((lower_bounds(hi, n0, n1) >= CUT).mean())


t0 = time.time()
labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
feat = L.patient_features()
a03 = pd.read_csv(L.R / '03_audit' / 'audit_primary.csv').set_index('task_id')
u07 = pd.read_csv(L.R / '07_screen_validation' / 'real' / 'image_probe_unflagged_tasks.csv')
p05 = pd.read_csv(L.R / '05_image_check' / 'real' / 'image_probe.csv')
FLAGGED = list(p05.task_id.unique())
ct_image = FLAGGED + sorted(u07.task_id.unique())
cfg = []
for tid in a03.index:
    df, y, classes = L.task_frame(tid, labels, feat)
    cfg.append({'kind': 'metadata', 'task_id': tid, 'n_classes': len(classes), 'counts': np.bincount(y)})
    if tid in ct_image:
        keep = df.key.isin(L.KEYS05 | L.KEYS07).to_numpy()
        cfg.append({'kind': 'image probe', 'task_id': tid, 'n_classes': len(classes), 'counts': np.bincount(y[keep])})
jobs = [i for i, c in enumerate(cfg) if c['n_classes'] == 2]
res = dict(zip(jobs, Parallel(n_jobs=6)(delayed(detectable)(int(cfg[i]['counts'][0]), int(cfg[i]['counts'][1]))
                                        for i in jobs)))
rows = []
for i, c in enumerate(cfg):
    r = {'kind': c['kind'], 'task_id': c['task_id'], 'n_patients': int(c['counts'].sum()),
         'class_sizes': ' | '.join(str(int(v)) for v in c['counts'])}
    if c['n_classes'] == 2:
        a, pw = res[i]
        r.update({'detectable_auc_80': round(a, 3), 'probability_at_detectable_auc': pw})
    else:
        r['note'] = 'four classes: not simulated'
    if c['kind'] == 'metadata':
        r['observed_metadata_auc'] = a03.auc[c['task_id']]
    else:
        if c['task_id'] in FLAGGED:
            g = p05[p05.task_id == c['task_id']]
        else:
            g = u07[u07.task_id == c['task_id']]
        r['observed_background_auc_max'] = float(g.background_auc.max())
    rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(HERE / 'r14_detectable_auc.csv', index=False)
pd.set_option('display.width', 250)
print(out.to_string(index=False))
print(f'finished in {time.time() - t0:.0f} s')
