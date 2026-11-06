"""Checks R17 (crossed assembly) and R18 (sex and the CPTAC transfer); plan 7l, written on 6 Nov 2026. POST HOC.

Uses only saved per-patient scores of the fixed TCGA lung probes (notebook 11) and sex from the TCIA study records.
No model is refitted and no image is read.
Usage (from this folder):  python r17_r18_transfer_checks.py rehearsal   # labels permuted within each cohort
                           python r17_r18_transfer_checks.py real
"""
import hashlib, itertools, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MODE = sys.argv[1]
assert MODE in ('rehearsal', 'real')
HERE = Path(__file__).resolve().parent
R = HERE.parent
OUT = HERE / MODE
OUT.mkdir(exist_ok=True)
SEED, N_BOOT = 2026, 2000
COHORT = {'lung: TCGA -> CPTAC': 'CPTAC lung', 'lung: TCGA -> NSCLC Radiogenomics': 'NSCLC Radiogenomics',
          'lung: TCGA -> NSCLC-Radiomics': 'NSCLC-Radiomics', 'lung: TCGA -> Lung-PET-CT-Dx': 'Lung-PET-CT-Dx'}
INPUT_NAME = {'image': 'full image', 'background': 'background only', 'silhouette': 'silhouette'}


def auc(y, s):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(y, s, stat=auc, groups=None):
    rng = np.random.default_rng(SEED)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    while len(vals) < N_BOOT:
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        v = stat(y[i], s[i]) if groups is None else stat(y[i], s[i], groups[i])
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def within_group_auc(y, s, g):
    """Share of positive-negative pairs from the same group ordered correctly (ties one half)."""
    num = den = 0.0
    for gv in np.unique(g):
        m = g == gv
        pos, neg = s[m & (y == 1)], s[m & (y == 0)]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
    return num / den if den else np.nan


sc = pd.read_csv(R / '11_external_lung' / 'real' / 'transfer_scores.csv.gz')
sc['cohort'] = sc.pair.map(COHORT)
assert sc.cohort.notna().all()
if MODE == 'rehearsal':            # permute the labels within each cohort, once per patient (same for every model)
    rng = np.random.default_rng(SEED)
    lab = sc.drop_duplicates('key')[['key', 'cohort', 'label']].copy()
    lab['label'] = lab.groupby('cohort').label.transform(lambda v: rng.permutation(v.to_numpy()))
    sc = sc.drop(columns='label').merge(lab[['key', 'label']], on='key', how='left')

# ---------- R17 ----------
rows, offs, repro = [], [], []
rec = pd.read_csv(R / '11_external_lung' / 'real' / 'transfer.csv')
sil = pd.read_csv(R / '11_external_lung' / 'real' / 'silhouette_transfer.csv')
cohorts = list(COHORT.values())
for (enc, inp), g in sc.groupby(['encoder', 'input']):
    for c in cohorts:
        d = g[g.cohort == c]
        a = auc(d.label, d.score); lo, hi = boot(d.label.to_numpy(), d.score.to_numpy())
        rows.append({'encoder': enc, 'input': INPUT_NAME[inp], 'set': 'natural', 'adenocarcinoma_from': c,
                     'squamous_from': c, 'n_adeno': int((d.label == 0).sum()), 'n_squamous': int((d.label == 1).sum()),
                     'auc': a, 'ci_low': lo, 'ci_high': hi})
        for cls in (0, 1):
            x = d.score[d.label == cls].to_numpy()
            offs.append({'encoder': enc, 'input': INPUT_NAME[inp], 'cohort': c,
                         'class': 'squamous' if cls else 'adenocarcinoma', 'n': len(x), 'mean_score': x.mean(),
                         'median_score': np.median(x)})
        if MODE == 'real':
            pair = [p for p, v in COHORT.items() if v == c][0]
            if inp == 'silhouette':
                t = sil[(sil.pair == pair) & (sil.encoder == enc)]
                col = 'silhouette_external_auc'
            else:
                t = rec[(rec.pair == pair) & (rec.encoder == enc) & (rec.input == inp)]
                col = 'external_auc'
            ref = float(t[col].iloc[0]) if len(t) == 1 else np.nan
            repro.append({'encoder': enc, 'input': inp, 'cohort': c, 'auc': a, 'recorded': ref,
                          'reproduced': bool(abs(a - ref) < 1e-9)})
    for x, y in itertools.permutations(cohorts, 2):
        d = pd.concat([g[(g.cohort == x) & (g.label == 0)], g[(g.cohort == y) & (g.label == 1)]])
        a = auc(d.label, d.score); lo, hi = boot(d.label.to_numpy(), d.score.to_numpy())
        rows.append({'encoder': enc, 'input': INPUT_NAME[inp], 'set': 'crossed', 'adenocarcinoma_from': x,
                     'squamous_from': y, 'n_adeno': int((d.label == 0).sum()), 'n_squamous': int((d.label == 1).sum()),
                     'auc': a, 'ci_low': lo, 'ci_high': hi})
r17 = pd.DataFrame(rows)
nat = r17[r17.set == 'natural']
gate17 = bool(nat.auc.between(0.35, 0.65).all()) if MODE == 'rehearsal' else None
if MODE == 'real':
    rp = pd.DataFrame(repro); rp.to_csv(OUT / 'r17_reproduction.csv', index=False)
    assert rp.reproduced.all(), 'reproduction gate failed:\n' + rp.to_string()
r17.to_csv(OUT / 'r17_crossed_assembly.csv', index=False)
pd.DataFrame(offs).to_csv(OUT / 'r17_cohort_offsets.csv', index=False)
summ = (r17.groupby(['encoder', 'input', 'set']).auc.agg(['min', 'max', 'count']).reset_index())
summ.to_csv(OUT / 'r17_summary.csv', index=False)

# ---------- R18 ----------
st = pd.read_csv(R / '01_inventory' / 'studies.csv.gz', low_memory=False)
st['sex'] = st.PatientSex.where(st.PatientSex.isin(['M', 'F']))
demo = (st.sort_values('StudyDate').groupby(['Collection', 'PatientID'])
          .agg(sex=('sex', lambda s: s.dropna().iloc[0] if s.notna().any() else None)).reset_index())
demo['key'] = demo.Collection + '__' + demo.PatientID
rows18 = []
for c in ['CPTAC lung', 'NSCLC Radiogenomics']:
    for (enc, inp), g in sc[sc.cohort == c].groupby(['encoder', 'input']):
        d = g.merge(demo[['key', 'sex']], on='key', how='left')
        d = d[d.sex.notna()]
        y, s, sex = d.label.to_numpy(), d.score.to_numpy(), d.sex.to_numpy()
        male = (sex == 'M').astype(float)
        row = {'cohort': c, 'encoder': enc, 'input': INPUT_NAME[inp], 'n_with_sex': len(d), 'n_missing_sex': len(g) - len(d),
               'male_share_adeno': male[y == 0].mean(), 'male_share_squamous': male[y == 1].mean(),
               'sex_only_auc': auc(y, male), 'probe_auc_same_patients': auc(y, s)}
        row['sex_only_ci_low'], row['sex_only_ci_high'] = boot(y, male)
        for sv, name in (('M', 'men'), ('F', 'women')):
            m = sex == sv
            row[f'n_{name}_adeno'], row[f'n_{name}_squamous'] = int(((y == 0) & m).sum()), int(((y == 1) & m).sum())
            row[f'auc_{name}'] = auc(y[m], s[m])
            row[f'auc_{name}_ci_low'], row[f'auc_{name}_ci_high'] = boot(y[m], s[m])
        w = within_group_auc(y, s, sex)
        lo, hi = boot(y, s, within_group_auc, sex)
        row.update(within_sex_auc=w, within_sex_ci_low=lo, within_sex_ci_high=hi)
        if c == 'CPTAC lung' and (enc, inp) in {('ctfm', 'background'), ('merlin', 'background'), ('merlin', 'image')}:
            row['reading'] = ('sex not a sufficient explanation' if lo > 0.5 else
                              'inconclusive at this sample size' if w >= 0.60 else 'sex cannot be excluded')
        rows18.append(row)
r18 = pd.DataFrame(rows18)
r18.to_csv(OUT / 'r18_within_sex.csv', index=False)
gate18 = bool(r18.within_sex_auc.between(0.35, 0.65).all()) if MODE == 'rehearsal' else None

files = sorted(p for p in OUT.glob('*.csv'))
manifest = {'check': 'R17, R18 (plan 7l)', 'mode': MODE, 'run_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'plan_sha256': hashlib.sha256((R.parent / 'protocol' / 'analysis_plan_7l.md').read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rehearsal_gates': {'r17_natural_auc_0.35_0.65': gate17, 'r18_within_sex_0.35_0.65': gate18},
            'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]},
            'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest['rehearsal_gates']))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(summ.round(3).to_string())
