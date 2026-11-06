"""Check R21 (plan 7n, written on 7 Nov 2026; POST HOC, descriptive): mask coverage and body area by class in the four
flagged tasks, from the saved per-scan preparation records of notebook 05. Usage (from this folder): python r21_mask_coverage.py
"""
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
R = HERE.parent
OUT = HERE / 'real'
OUT.mkdir(exist_ok=True)
SEED, N_BOOT = 2026, 2000
TASKS = ['renal subtype (KIRC vs KIRP)', 'CPTAC-PDA | stage', 'lung histology (LUAD vs LUSC)',
         'LIDC-IDRI | primary vs metastatic']
QTY = {'body_fraction_dilated': 'share of the volume inside the dilated body mask', 'body_area_cm2': 'mean body area, cm2'}


def auc(y, s):
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(y, s):
    rng = np.random.default_rng(SEED); vals = []
    while len(vals) < N_BOOT:
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) == 2:
            vals.append(auc(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), np.array(vals)


labels = pd.read_csv(R / '02_tasks_features' / 'labels_long.csv')
hdr = {}
for f in (R / '05_image_check' / 'cache' / 'headers').glob('*.json'):
    d = json.loads(f.read_text(encoding='utf-8'))
    hdr[d['key']] = d
rows = []
for t in TASKS:
    d = labels[labels.task_id == t].copy()
    d['key'] = d.collection + '__' + d.PatientID
    classes = sorted(d.label.unique())
    y = d.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    for q in QTY:
        x = d.key.map(lambda k: (hdr.get(k) or {}).get(q)).astype(float).to_numpy()
        a = auc(y, x); lo, hi, vals = boot(y, x)
        p = float(min(1, 2 * min((vals <= 0.5).mean(), (vals >= 0.5).mean())))   # bootstrap two-sided P for AUC = 0.5
        med = lambda m: f'{np.nanmedian(x[m]):.3g} ({np.nanpercentile(x[m], 25):.3g}–{np.nanpercentile(x[m], 75):.3g})'
        rows.append({'task_id': t, 'quantity': q, 'description': QTY[q], 'n_scans': int(np.isfinite(x).sum()),
                     'class0': classes[0], 'class1': classes[1], 'median_iqr_class0': med(y == 0),
                     'median_iqr_class1': med(y == 1), 'auc_class1_higher': a, 'ci_low': lo, 'ci_high': hi,
                     'boot_p': p})
res = pd.DataFrame(rows)
p = res.boot_p.to_numpy(); o = np.argsort(p); m = len(p)
q = np.empty(m); q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
res['q_bh_8'] = np.minimum(q, 1)
res['reading'] = np.where((res.ci_low > 0.5) | (res.ci_high < 0.5), np.where(res.q_bh_8 < 0.05, 'differs by class', 'no detectable difference'), 'no detectable difference')
res.to_csv(OUT / 'r21_mask_coverage_by_class.csv', index=False)
manifest = {'check': 'R21 (plan 7n)', 'run_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'plan_sha256': hashlib.sha256((R.parent / 'protocol' / 'analysis_plan_7n.md').read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]},
            'outputs': {'r21_mask_coverage_by_class.csv': hashlib.sha256((OUT / 'r21_mask_coverage_by_class.csv').read_bytes()).hexdigest()}}
(OUT / 'r21_run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(res.drop(columns=['description']).round(3).to_string())
