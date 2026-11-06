"""Check R19 (plan 7m, written on 7 Nov 2026; POST HOC): contributing country and the TCGA -> CPTAC lung transfer.

Uses the saved scores of the fixed TCGA lung probes (notebook 11) and GDC demographic records, fetched once and saved
unchanged in gdc_demographics.json (a rerun reads the saved file). No model is refitted and no image is read.
Usage (from this folder): python r19_country_cptac.py
"""
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
R = HERE.parent
OUT = HERE / 'real'
OUT.mkdir(exist_ok=True)
SEED, N_BOOT = 2026, 2000
LUNG = 'lung histology (LUAD vs LUSC)'


def auc(y, s):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_group_auc(y, s, g):
    num = den = 0.0
    for gv in np.unique(g):
        m = g == gv
        pos, neg = s[m & (y == 1)], s[m & (y == 0)]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
    return (num / den if den else np.nan), int(den)


def boot(y, s, stat, g=None):
    rng = np.random.default_rng(SEED)
    vals = []
    while len(vals) < N_BOOT:
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        v = stat(y[i], s[i]) if g is None else stat(y[i], s[i], g[i])[0]
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def loo_share(y, g):
    """Leave-one-patient-out share of squamous among the other patients of the same country (the country-only score)."""
    df = pd.DataFrame({'y': y, 'g': g})
    tot = df.groupby('g').y.transform('sum'); n = df.groupby('g').y.transform('size')
    out = (tot - df.y) / (n - 1)
    return out.fillna(df.y.mean()).to_numpy()


# ---- patients
sc = pd.read_csv(R / '11_external_lung' / 'real' / 'transfer_scores.csv.gz')
cp = sc[sc.pair == 'lung: TCGA -> CPTAC'].copy()
cp['PatientID'] = cp.key.str.split('__').str[1]
labels = pd.read_csv(R / '02_tasks_features' / 'labels_long.csv')
tg = labels[labels.task_id == LUNG][['collection', 'PatientID', 'label']].copy()
ids = sorted(set(cp.PatientID) | set(tg.PatientID))

# ---- GDC demographic records (fetched once, saved unchanged)
cache = HERE / 'gdc_demographics.json'
if not cache.exists():
    hits = []
    for i in range(0, len(ids), 50):
        flt = {'op': 'in', 'content': {'field': 'submitter_id', 'value': ids[i:i + 50]}}
        body = {'filters': flt, 'format': 'json', 'expand': 'demographic,tissue_source_site,project', 'size': 200}
        for attempt in range(5):                      # POST: the filter is too long for a URL
            try:
                r = requests.post('https://api.gdc.cancer.gov/cases', json=body, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
        hits += r.json()['data']['hits']
    cache.write_text(json.dumps({'fetched_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                 'endpoint': 'https://api.gdc.cancer.gov/cases', 'hits': hits}, indent=1), encoding='utf-8')
hits = json.loads(cache.read_text(encoding='utf-8'))['hits']
bad = {None, '', 'not reported', 'Not Reported', 'unknown', 'Unknown'}
rec = {}
for h in hits:
    d = h.get('demographic') or {}
    rec[h['submitter_id']] = {'country_of_birth': d.get('country_of_birth') if d.get('country_of_birth') not in bad else None,
                              'country_of_residence': d.get('country_of_residence_at_enrollment')
                              if d.get('country_of_residence_at_enrollment') not in bad else None,
                              'tss_name': (h.get('tissue_source_site') or {}).get('name')}

# ---- (a) class by source
cp['country'] = cp.PatientID.map(lambda p: (rec.get(p) or {}).get('country_of_birth'))
one = cp.drop_duplicates('PatientID')
cls = {0: 'adenocarcinoma', 1: 'squamous cell carcinoma'}
tab_cp = (one.assign(country=one.country.fillna('not recorded'), cls=one.label.map(cls))
             .groupby(['country', 'cls']).size().unstack(fill_value=0).reset_index())
tab_cp.insert(0, 'cohort', 'CPTAC lung (country of birth)')
site = pd.read_csv(R / '12_review_checks' / 'r4_site_mapping.csv', dtype=str)
site = site[site.collection.str.startswith('TCGA')]
tss2inst = dict(zip(site.tss, site.institution))
tg['tss'] = tg.PatientID.str.split('-').str[1]
tg['institution'] = tg.tss.map(tss2inst).fillna('unmapped')
tg['country'] = tg.PatientID.map(lambda p: (rec.get(p) or {}).get('country_of_residence')).fillna('not recorded')
tg['cls'] = tg.collection
tab_tg = tg.groupby(['institution', 'country', 'cls']).size().unstack(fill_value=0).reset_index()
tab_tg.insert(0, 'cohort', 'TCGA lung (institution; country of residence)')
pd.concat([tab_tg, tab_cp], ignore_index=True).to_csv(OUT / 'r19_class_by_source.csv', index=False)

# ---- (b) country alone and (c) within-country AUCs on CPTAC lung
rows = []
nr = cp.drop_duplicates('PatientID')
nr = nr[nr.country.isna()]
keep_nr = nr.label.nunique() == 2          # plan 7m: no recorded country is its own group if it holds both classes
cp['country_group'] = cp.country.fillna('not recorded') if keep_nr else cp.country
for (enc, inp), g in cp.groupby(['encoder', 'input']):
    g = g[g.country_group.notna()].assign(country=lambda d: d.country_group)
    y, s, c = g.label.to_numpy(), g.score.to_numpy(), g.country.to_numpy()
    cs = loo_share(y, c)
    w, npairs = within_group_auc(y, s, c)
    lo, hi = boot(y, s, within_group_auc, c) if npairs >= 20 else (np.nan, np.nan)
    co = auc(y, cs); co_lo, co_hi = boot(y, cs, auc)
    reading = None
    if (enc, inp) in {('ctfm', 'background'), ('merlin', 'background'), ('merlin', 'image')}:
        if npairs < 20:
            reading = 'not informative'
        elif lo > 0.5:
            reading = 'not explained by contributing country'
        elif w < 0.60 and co_lo > 0.5:
            reading = 'consistent with contributing source'
        else:
            reading = 'inconclusive'
    rows.append({'encoder': enc, 'input': inp, 'n_with_country': len(g), 'n_missing_country': int(cp[(cp.encoder == enc) & (cp.input == inp)].country.isna().sum()), 'missing_as_group': bool(keep_nr),
                 'n_countries': int(len(np.unique(c))), 'probe_auc_same_patients': auc(y, s),
                 'country_only_auc': co, 'country_only_ci_low': co_lo, 'country_only_ci_high': co_hi,
                 'within_country_auc': w, 'within_country_ci_low': lo, 'within_country_ci_high': hi,
                 'same_country_pairs': npairs, 'reading': reading})
res = pd.DataFrame(rows)
res.to_csv(OUT / 'r19_within_country.csv', index=False)

files = sorted(OUT.glob('r19_*.csv')) + [cache]
manifest = {'check': 'R19 (plan 7m)', 'run_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'plan_sha256': hashlib.sha256((R.parent / 'protocol' / 'analysis_plan_7m.md').read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]},
            'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(OUT / 'r19_run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(pd.concat([tab_tg, tab_cp]).to_string()); print(res.round(3).to_string())
