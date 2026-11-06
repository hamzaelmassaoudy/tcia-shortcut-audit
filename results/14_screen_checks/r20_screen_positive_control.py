"""Check R20 (plan 7m, written on 7 Nov 2026; POST HOC): sensitivity and false-flag rate of the metadata screen.

Synthetic labels are tied, at controlled strength, to one acquisition group G of real label cards (LIDC-IDRI CT and
Duke-Breast-Cancer-MRI), and the study's screen (tool v1.0.1, unchanged) is run on each synthetic task.
Usage (from this folder):  python r20_screen_positive_control.py rehearsal | real
"""
import hashlib, json, sys, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
MODE = sys.argv[1]
assert MODE in ('rehearsal', 'real')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'tool'))
import tcia_shortcut_audit as T            # noqa: E402  the study's screen, unchanged

OUT = HERE / MODE
OUT.mkdir(exist_ok=True)
BASES = [('LIDC-IDRI', 'CT'), ('Duke-Breast-Cancer-MRI', 'MR')]
SOURCES = ['vendor', 'image count', 'description word']
ORS = [1, 15] if MODE == 'rehearsal' else [1, 1.5, 2, 3, 5, 8, 15]
SIZES = [100, 250]
REPS = 3 if MODE == 'rehearsal' else 20
PREV = 0.35

primary = pd.read_csv(REPO / 'results' / '03_audit' / 'audit_primary.csv')
assert len(primary) == 26
REAL_P = primary.perm_p.to_numpy(float)


def bh_q_added(p_new):
    """BH q of a new P value added to the 26 real P values (family of 27)."""
    p = np.append(REAL_P, p_new)
    order = np.argsort(p, kind='mergesort')
    m = len(p)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return float(min(q[-1], 1.0))


def groups_for(card):
    g = {}
    top = card.vendor.value_counts().idxmax()
    g['vendor'] = (card.vendor == top).to_numpy(int), f'vendor = {top}'
    ic = card.image_count.astype(float)
    g['image count'] = (ic > ic.median()).to_numpy(int), f'image count > {ic.median():g}'
    docs = card.protocol_text.fillna('').str.split()
    df = pd.Series([w for d in docs for w in set(d)]).value_counts() / len(card)
    word = next(w for w, s in df.items() if 0.25 <= s <= 0.75)
    g['description word'] = docs.map(lambda d: int(word in d)).to_numpy(int), f'description word "{word}"'
    return g


def intercept(G, b):
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + b * G)))).mean() < PREV:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def one_task(card, G, desc, base, source, odds, n, rep):
    seed = int(hashlib.sha256(f'{base}|{source}|{odds}|{n}|{rep}'.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    b = np.log(odds)
    b0 = intercept(G, b)
    for _ in range(100):
        idx = np.sort(rng.choice(len(card), n, replace=False))
        y = (rng.random(n) < 1 / (1 + np.exp(-(b0 + b * G[idx])))).astype(int)
        if min(y.sum(), n - y.sum()) >= 10:
            break
    df = card.iloc[idx].reset_index(drop=True)
    A = T.Audit(seed=2026, n_jobs=1)
    splits = A.splits(y)
    P = A.oof(df, y, T.FEATURE_COLS, splits)
    a = A.mean_auc(y, P)
    lo, hi = A.boot_ci(y, P)
    p = q = np.nan
    flagged = False
    if lo >= T.FLAG_LOWER_CI:
        _, p, _ = A.permutation_test(df, y, T.FEATURE_COLS)
        q = bh_q_added(p)
        flagged = bool(q < 0.05)
    return {'base': base, 'source': source, 'group': desc, 'odds_ratio': odds, 'n': n, 'rep': rep, 'seed': seed,
            'n_positive': int(y.sum()), 'oracle_auc': T.Audit.auc_binary(y, G[idx]), 'screen_auc': a,
            'ci_low': lo, 'ci_high': hi, 'perm_p': p, 'q_family27': q, 'flagged': flagged}


series = pd.read_csv(REPO / 'results' / '01_inventory' / 'series_metadata.csv.gz', low_memory=False)
jobs, gdesc = [], []
for base, mod in BASES:
    ms = T.main_series(series[series.Collection == base])
    ms = ms[ms.modality == mod].reset_index(drop=True)
    card = T.label_card(ms, set(T.tokenize(base)))
    for source, (G, desc) in groups_for(card).items():
        gdesc.append({'base': base, 'modality': mod, 'patients': len(card), 'source': source, 'group': desc,
                      'share_in_group': float(G.mean())})
        for odds in ORS:
            for n in SIZES:
                for rep in range(REPS):
                    jobs.append((card, G, desc, base, source, odds, n, rep))
print(len(jobs), 'tasks', flush=True)
rows = Parallel(n_jobs=-1, verbose=5)(delayed(one_task)(*j) for j in jobs)
tasks = pd.DataFrame(rows)
tasks.to_csv(OUT / 'r20_tasks.csv', index=False)
pd.DataFrame(gdesc).to_csv(OUT / 'r20_groups.csv', index=False)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


bins = [0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
labels = ['<0.55', '0.55-0.60', '0.60-0.65', '0.65-0.70', '0.70-0.75', '0.75-0.80', '>=0.80']
tasks['oracle_bin'] = pd.cut(tasks.oracle_auc, bins, labels=labels, right=False)
det = []
for (n, b_), g in tasks.groupby(['n', 'oracle_bin'], observed=True):
    lo, hi = wilson(int(g.flagged.sum()), len(g))
    det.append({'n': n, 'oracle_auc_bin': b_, 'tasks': len(g), 'flagged': int(g.flagged.sum()),
                'detection': g.flagged.mean(), 'ci_low': lo, 'ci_high': hi})
pd.DataFrame(det).to_csv(OUT / 'r20_detection_by_oracle_auc.csv', index=False)
null = []
for (base, src), g in tasks[tasks.odds_ratio == 1].groupby(['base', 'source']):
    lo, hi = wilson(int(g.flagged.sum()), len(g))
    null.append({'base': base, 'source': src, 'tasks': len(g), 'flagged': int(g.flagged.sum()), 'ci_high': hi})
g = tasks[tasks.odds_ratio == 1]
lo, hi = wilson(int(g.flagged.sum()), len(g))
null.append({'base': 'all', 'source': 'all', 'tasks': len(g), 'flagged': int(g.flagged.sum()), 'ci_high': hi})
pd.DataFrame(null).to_csv(OUT / 'r20_false_flags.csv', index=False)
d80 = []
if MODE == 'real':
    from sklearn.linear_model import LogisticRegression
    for n, g in tasks.groupby('n'):
        m = LogisticRegression(C=1e6, max_iter=5000).fit(g[['oracle_auc']], g.flagged.astype(int))
        x80 = (np.log(0.8 / 0.2) - m.intercept_[0]) / m.coef_[0][0]
        d80.append({'n': n, 'oracle_auc_at_80pct_detection': float(x80), 'tasks': len(g)})
    pd.DataFrame(d80).to_csv(OUT / 'r20_detection_80.csv', index=False)
gate = None
if MODE == 'rehearsal':
    nul = pd.DataFrame(null[:-1])
    strong = tasks[(tasks.odds_ratio == 15) & (tasks.n == 250)]
    gate = {'false_flags_at_most_1_per_cohort_and_source': bool((nul.flagged <= 1).all()),
            'detection_or15_n250_above_0.5': bool(strong.flagged.mean() > 0.5)}
files = sorted(OUT.glob('r20_*.csv'))
manifest = {'check': 'R20 (plan 7m)', 'mode': MODE, 'run_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'tool_version': T.__version__, 'rehearsal_gate': gate,
            'plan_sha256': hashlib.sha256((REPO / 'protocol' / 'analysis_plan_7m.md').read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]},
            'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(OUT / 'r20_run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
pd.set_option('display.width', 250)
print(pd.DataFrame(gdesc).to_string()); print(pd.DataFrame(det).round(3).to_string()); print(pd.DataFrame(null).to_string())
print(pd.DataFrame(d80).round(3).to_string() if d80 else ''); print('gate', gate)
