"""Checks R22 (differences with confidence intervals), R23 (the experiments of plan 7o within site) and R24 (specificity
of the habitus measures); plan 7p, written on 13 Nov 2026. POST HOC and descriptive.

Uses only saved embeddings, saved per-patient scores, the scan records of notebook 12 and the predictions of notebook
13. No image is read and no network is trained. The probes are refitted with the settings and splits of notebooks 05
and 12, and every refitted AUC is compared with its recorded value before any new quantity is written.

Usage (from this folder):  python r22_r24_source_checks.py rehearsal   # labels permuted within each cohort
                           python r22_r24_source_checks.py real
"""
import glob, hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODE = sys.argv[1]
assert MODE in ('rehearsal', 'real')
HERE = Path(__file__).resolve().parent
R = HERE.parent
OUT = HERE / MODE
OUT.mkdir(exist_ok=True)
STARTED = datetime.now(timezone.utc).isoformat(timespec='seconds')
A12, B13 = R / '12_image_experiments', R / '13_cnn'
SEED, N_BOOT, N_SPLITS, N_REPEATS, N_PLACEBO, TOL = 2026, 2000, 5, 5, 200, 0.002
LR_PARAMS = dict(C=1.0, class_weight='balanced', max_iter=5000)
ENCODERS = ('ctfm', 'merlin')
SOURCE, CPTAC, RADIOGENOMICS = 'lung histology (LUAD vs LUSC)', 'CPTAC lung', 'NSCLC Radiogenomics'
INDEPENDENT = ('NSCLC Radiogenomics', 'NSCLC-Radiomics', 'Lung-PET-CT-Dx')
PAIR = {CPTAC: 'lung: TCGA -> CPTAC', 'NSCLC Radiogenomics': 'lung: TCGA -> NSCLC Radiogenomics',
        'NSCLC-Radiomics': 'lung: TCGA -> NSCLC-Radiomics', 'Lung-PET-CT-Dx': 'lung: TCGA -> Lung-PET-CT-Dx'}
HABITUS = ['ap_lr_ratio', 'subcut_fat_cm2', 'arms_beside_trunk', 'table_height_mm', 'centroid_offset_mm',
           'scan_length_mm', 'lung_volume_l']
CACHE = {'12': A12 / 'cache' / 'embeddings', '05': R / '05_image_check' / 'cache' / 'embeddings',
         '08': R / '08_transfer' / 'cache' / 'embeddings'}
BALANCED_SITE = 'university of pittsburgh'


# ---------------------------------------------------------------- the study's functions (notebooks 05, 11 and 12)
def auc_binary(y, s):
    """AUC by ranks (ties count one half); y in {0,1}."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rep_auc(y, P):
    """Repeat-averaged AUC of predictions P (repeats, patients, 2)."""
    return float(np.mean([auc_binary(y, P[r][:, 1]) for r in range(P.shape[0])]))


def as_P(s):
    return np.stack([-np.asarray(s, float), np.asarray(s, float)], axis=1)[None]


def repeated_splits(y):
    return [list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED + r).split(np.zeros(len(y)), y))
            for r in range(N_REPEATS)]


def probe():
    return Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(**LR_PARAMS))])


def habitus_model():
    return Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                     ('model', LogisticRegression(**LR_PARAMS))])


def cv_oof(X, y, make=probe):
    P = np.zeros((N_REPEATS, len(y), 2))
    for r, folds in enumerate(repeated_splits(y)):
        for tr, te in folds:
            P[r, te] = make().fit(X[tr], y[tr]).predict_proba(X[te])
    return P


def residual_oof(X, H, y):
    """Embeddings with the habitus measures regressed out inside each training fold (notebook 12, G2)."""
    P = np.zeros((N_REPEATS, len(y), 2))
    for r, folds in enumerate(repeated_splits(y)):
        for tr, te in folds:
            hp = Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]).fit(H[tr])
            reg = LinearRegression().fit(hp.transform(H[tr]), X[tr])
            Rtr, Rte = X[tr] - reg.predict(hp.transform(H[tr])), X[te] - reg.predict(hp.transform(H[te]))
            P[r, te] = probe().fit(Rtr, y[tr]).predict_proba(Rte)
    return P


def fit_apply(Xa, ya, Xb):
    return probe().fit(Xa, ya).decision_function(Xb)


def residual_fit_apply(Xs, Hs, ys, Xt, Ht):
    hp = Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]).fit(Hs)
    reg = LinearRegression().fit(hp.transform(Hs), Xs)
    return fit_apply(Xs - reg.predict(hp.transform(Hs)), ys, Xt - reg.predict(hp.transform(Ht)))


# ---------------------------------------------------------------- bootstrap
def resamples(y_list):
    """Index sets (one per cohort in y_list), each with both classes; the generator restarts from the seed."""
    rng = np.random.default_rng(SEED)
    done = 0
    while done < N_BOOT:
        idx = [rng.choice(len(y), len(y), replace=True) for y in y_list]
        if all(len(np.unique(y[i])) == 2 for y, i in zip(y_list, idx)):
            done += 1
            yield idx


def ci(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired(y, Pa, Pb):
    d = rep_auc(y, Pa) - rep_auc(y, Pb)
    lo, hi = ci([rep_auc(y[i], Pa[:, i]) - rep_auc(y[i], Pb[:, i]) for (i,) in resamples([y])])
    return rep_auc(y, Pa), rep_auc(y, Pb), d, lo, hi


def unpaired(ya, sa, yb, sb):
    d = auc_binary(ya, sa) - auc_binary(yb, sb)
    lo, hi = ci([auc_binary(ya[i], sa[i]) - auc_binary(yb[j], sb[j]) for i, j in resamples([ya, yb])])
    return auc_binary(ya, sa), auc_binary(yb, sb), d, lo, hi


def within_auc(y, s, g):
    """Share of positive-negative pairs from the same group that are ordered correctly (ties one half)."""
    num = den = 0.0
    for gv in np.unique(g):
        m = g == gv
        pos, neg = s[m & (y == 1)], s[m & (y == 0)]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
    return num / den if den else np.nan


def rep_within(y, P, g):
    return float(np.mean([within_auc(y, P[r][:, 1], g) for r in range(P.shape[0])]))


def boot_stat(y, P, stat):
    return ci([stat(y[i], P[:, i], i) for (i,) in resamples([y])])


# ---------------------------------------------------------------- data
lab = pd.read_csv(A12 / MODE / 'dataset_labels.csv')
frames = {d: g.reset_index(drop=True) for d, g in lab.groupby('dataset', sort=False)}
rec = pd.read_csv(A12 / 'cache' / 'scan_records.csv').set_index('key')


def emb(keys, enc, variant, cache):
    rows = []
    for k in keys:
        with np.load(CACHE[cache] / f'{k}.npz') as z:
            n = f'{enc}__{variant}'
            rows.append(z[n].astype(float) if n in z.files else None)
    d = next(len(a) for a in rows if a is not None)
    return np.vstack([a if a is not None else np.full(d, np.nan) for a in rows])


def habitus(keys):
    return rec.loc[list(keys), HABITUS].astype(float).to_numpy()


src = frames[SOURCE]
ks, ys = src.key.to_numpy(), src.y.to_numpy()
cp = frames[CPTAC]
kc, yc = cp.key.to_numpy(), cp.y.to_numpy()
Hs, Hc = habitus(ks), habitus(kc)
sites = pd.read_csv(R / '12_review_checks' / 'r4_site_mapping.csv')
site_of = {(r.collection, str(r.tss)): r.institution for r in sites.itertuples()}
site = np.array([site_of[(k.split('__')[0], k.split('__')[1].split('-')[1])] for k in ks])
mixed = [g for g in np.unique(site) if len(np.unique(src.y[site == g])) == 2] if MODE == 'real' else None
print(f'{MODE}: TCGA lung {len(ys)} patients ({int(ys.sum())} squamous), CPTAC lung {len(yc)}; sites', dict(pd.Series(site).value_counts()))

# ---------------------------------------------------------------- refits and the reproduction gate
recorded, refit = {}, {}
OOF, EXT = {}, {}                         # out-of-fold predictions in TCGA; scores of the fixed TCGA models on CPTAC
ok_t = {}
for enc in ENCODERS:
    X = {'full image': emb(ks, enc, 'original', '05'), 'background only': emb(ks, enc, 'background', '05'),
         'lungs only': emb(ks, enc, 'lungs', '12'), 'tumour only': emb(ks, enc, 'tumour', '12'),
         'full image, harmonized': emb(ks, enc, 'harm_full', '12'),
         'background only, harmonized': emb(ks, enc, 'harm_background', '12')}
    Xc = {'full image': emb(kc, enc, 'original', '08'), 'background only': emb(kc, enc, 'background', '08'),
          'full image, harmonized': emb(kc, enc, 'harm_full', '12'),
          'background only, harmonized': emb(kc, enc, 'harm_background', '12')}
    for name, M in X.items():
        ok = ~np.isnan(M).any(axis=1)
        ok_t[(enc, name)] = ok
        P = np.full((N_REPEATS, len(ys), 2), np.nan)
        P[:, ok] = cv_oof(M[ok], ys[ok])
        OOF[(enc, name)] = P
        refit[('internal', enc, name)] = rep_auc(ys[ok], P[:, ok])
    OOF[(enc, 'background only, habitus removed')] = residual_oof(X['background only'], Hs, ys)
    refit[('internal', enc, 'background only, habitus removed')] = rep_auc(ys, OOF[(enc, 'background only, habitus removed')])
    for name in Xc:
        EXT[(enc, name)] = fit_apply(X[name], ys, Xc[name])
        refit[('CPTAC', enc, name)] = auc_binary(yc, EXT[(enc, name)])
    EXT[(enc, 'background only, habitus removed')] = residual_fit_apply(X['background only'], Hs, ys, Xc['background only'], Hc)
    refit[('CPTAC', enc, 'background only, habitus removed')] = auc_binary(yc, EXT[(enc, 'background only, habitus removed')])
    if enc == 'merlin':
        X_bg_m, Xc_bg_m = X['background only'], Xc['background only']
    else:
        X_bg_c, Xc_bg_c = X['background only'], Xc['background only']
OOF[('-', 'habitus model')] = cv_oof(Hs, ys, habitus_model)
refit[('internal', '-', 'habitus model')] = rep_auc(ys, OOF[('-', 'habitus model')])
hm = habitus_model().fit(Hs, ys)
EXT[('-', 'habitus model')] = hm.decision_function(Hc)
refit[('CPTAC', '-', 'habitus model')] = auc_binary(yc, EXT[('-', 'habitus model')])

g1 = pd.read_csv(A12 / MODE / 'g1_internal.csv'); g1 = g1[g1.dataset == SOURCE].set_index(['input', 'encoder'])
g2 = pd.read_csv(A12 / MODE / 'g2_internal.csv').set_index(['model', 'encoder'])
g3 = pd.read_csv(A12 / MODE / 'g3_internal.csv').set_index(['input', 'encoder'])
g2t = pd.read_csv(A12 / MODE / 'g2_transfer.csv').set_index(['input', 'encoder', 'target'])
g3t = pd.read_csv(A12 / MODE / 'g3_transfer.csv').set_index(['input', 'encoder', 'target'])
for enc in ENCODERS:
    recorded[('internal', enc, 'lungs only')] = g1.loc[('lungs', enc), 'auc']
    recorded[('internal', enc, 'tumour only')] = g1.loc[('tumour', enc), 'auc']
    recorded[('internal', enc, 'full image, harmonized')] = g3.loc[('harm_full', enc), 'auc']
    recorded[('internal', enc, 'background only, harmonized')] = g3.loc[('harm_background', enc), 'auc']
    recorded[('internal', enc, 'background only, habitus removed')] = g2.loc[('background, habitus removed', enc), 'auc']
    recorded[('CPTAC', enc, 'background only, habitus removed')] = g2t.loc[('background, habitus removed', enc, CPTAC), 'external_auc']
    recorded[('CPTAC', enc, 'full image, harmonized')] = g3t.loc[('harm_full', enc, CPTAC), 'external_auc']
    recorded[('CPTAC', enc, 'background only, harmonized')] = g3t.loc[('harm_background', enc, CPTAC), 'external_auc']
recorded[('internal', '-', 'habitus model')] = g2.loc[('habitus', '-'), 'auc']
recorded[('CPTAC', '-', 'habitus model')] = g2t.loc[('habitus model', '-', CPTAC), 'external_auc']
if MODE == 'real':                                   # the native probes were recorded by notebooks 05 and 11
    p05 = pd.read_csv(R / '05_image_check' / 'real' / 'image_probe.csv')
    p05 = p05[p05.task_id == SOURCE].set_index('encoder')
    t11 = pd.read_csv(R / '11_external_lung' / 'real' / 'transfer.csv')
    t11 = t11[t11.pair == PAIR[CPTAC]].set_index(['encoder', 'input'])
    for enc in ENCODERS:
        recorded[('internal', enc, 'full image')] = p05.loc[enc, 'image_auc']
        recorded[('internal', enc, 'background only')] = p05.loc[enc, 'background_auc']
        recorded[('CPTAC', enc, 'full image')] = t11.loc[(enc, 'image'), 'external_auc']
        recorded[('CPTAC', enc, 'background only')] = t11.loc[(enc, 'background'), 'external_auc']
rep = pd.DataFrame([{'where': k[0], 'encoder': k[1], 'model': k[2], 'recorded_auc': v, 'refitted_auc': refit[k],
                     'difference': refit[k] - v, 'within_tolerance': bool(abs(refit[k] - v) <= TOL)}
                    for k, v in recorded.items()])
rep.to_csv(OUT / 'r22_reproduction.csv', index=False)
print(rep.round(4).to_string())
assert rep.within_tolerance.all(), 'reproduction gate not passed: no new quantity was written'
print(f'reproduction gate passed: {len(rep)} refitted AUCs within {TOL} (largest difference {rep.difference.abs().max():.5f})')

# ---------------------------------------------------------------- R22: differences
rows = []


def add(check, comparison, enc, cohort, n, r):
    a, b, d, lo, hi = r
    rows.append({'check': check, 'comparison': comparison, 'encoder': enc, 'cohort': cohort, 'n': n, 'auc_first': a,
                 'auc_second': b, 'difference': d, 'ci_low': lo, 'ci_high': hi,
                 'reading': 'shown' if (lo > 0 or hi < 0) else 'not shown'})


for enc in ENCODERS:
    add('a', 'background only minus background only with habitus removed', enc, CPTAC, len(yc),
        paired(yc, as_P(EXT[(enc, 'background only')]), as_P(EXT[(enc, 'background only, habitus removed')])))
    add('b', 'background only: native minus harmonized geometry', enc, CPTAC, len(yc),
        paired(yc, as_P(EXT[(enc, 'background only')]), as_P(EXT[(enc, 'background only, harmonized')])))
    add('b', 'full image: native minus harmonized geometry', enc, CPTAC, len(yc),
        paired(yc, as_P(EXT[(enc, 'full image')]), as_P(EXT[(enc, 'full image, harmonized')])))
    ok = ok_t[(enc, 'tumour only')]
    add('c', 'background only minus tumour only', enc, 'TCGA lung (tumour detected)', int(ok.sum()),
        paired(ys[ok], OOF[(enc, 'background only')][:, ok], OOF[(enc, 'tumour only')][:, ok]))
    add('c', 'background only minus lungs only', enc, 'TCGA lung', len(ys),
        paired(ys, OOF[(enc, 'background only')], OOF[(enc, 'lungs only')]))
    add('d', 'background only minus full image', enc, 'TCGA lung', len(ys),
        paired(ys, OOF[(enc, 'background only')], OOF[(enc, 'full image')]))
sc = pd.read_csv(R / '11_external_lung' / MODE / 'transfer_scores.csv.gz')   # scores of the fixed TCGA probes (notebook 11)
ylab = lab.set_index('key').y
for enc in ENCODERS:
    for inp, name in (('image', 'full image'), ('background', 'background only')):
        a = sc[(sc.pair == PAIR[CPTAC]) & (sc.encoder == enc) & (sc.input == inp)]
        ya, sa = a.label.to_numpy(), a.score.to_numpy()          # labels as notebook 11 scored them (permuted in rehearsal)
        if MODE == 'real':
            assert (ya == ylab.loc[a.key].to_numpy()).all()
        for t in INDEPENDENT:
            b = sc[(sc.pair == PAIR[t]) & (sc.encoder == enc) & (sc.input == inp)]
            yb, sb = b.label.to_numpy(), b.score.to_numpy()
            if MODE == 'real':
                assert (yb == ylab.loc[b.key].to_numpy()).all()
            add('e', f'{name}: CPTAC lung minus {t}', enc, f'CPTAC lung and {t}', len(ya) + len(yb), unpaired(ya, sa, yb, sb))
r22 = pd.DataFrame(rows)
r22.to_csv(OUT / 'r22_differences.csv', index=False)
print(r22.round(3).to_string())

# ---------------------------------------------------------------- R23: within site (TCGA lung)
def cnn_oof(inp):
    seeds = (0,) if MODE == 'rehearsal' else (0, 1, 2)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(ks, ys))
    oof = np.zeros((len(seeds), len(ks)))
    for f, (tr, te) in enumerate(folds):
        for si, sd in enumerate(seeds):
            with np.load(glob.glob(str(B13 / 'checkpoints' / f'{MODE}__cv__{inp}__fold{f}__seed{sd}__*.npz'))[0]) as z:
                oof[si, te] = z['p']
    s = oof.mean(0)
    rc = pd.read_csv(B13 / MODE / 'cnn_internal.csv').set_index('input').auc[inp]
    assert abs(auc_binary(ys, s) - rc) < 1e-12, inp
    return as_P(s)


models = [(enc, name) for name in ('full image', 'background only', 'lungs only', 'tumour only',
                                   'background only, habitus removed', 'full image, harmonized',
                                   'background only, harmonized') for enc in ENCODERS]
models.insert(8, ('-', 'habitus model'))
P23 = {m: OOF[m] for m in models}
P23[('-', '3D network, full image')] = cnn_oof('full')
P23[('-', '3D network, background only')] = cnn_oof('background')
pit = site == BALANCED_SITE
rows = []
for (enc, name), P in P23.items():
    ok = ~np.isnan(P[0, :, 1])
    y_, P_, g_ = ys[ok], P[:, ok], site[ok]
    lo, hi = boot_stat(y_, P_, lambda yy, PP, i: rep_within(yy, PP, g_[i]))
    m = g_ == BALANCED_SITE
    plo, phi = boot_stat(y_[m], P_[:, m], lambda yy, PP, i: rep_auc(yy, PP))
    w = rep_within(y_, P_, g_)
    rows.append({'model': name, 'encoder': enc, 'n': int(ok.sum()), 'auc_all_patients': rep_auc(y_, P_),
                 'within_site_auc': w, 'within_ci_low': lo, 'within_ci_high': hi,
                 'balanced_site_n': int(m.sum()), 'balanced_site_squamous': int(y_[m].sum()),
                 'balanced_site_auc': rep_auc(y_[m], P_[:, m]), 'balanced_ci_low': plo, 'balanced_ci_high': phi,
                 'reading': 'holds within site' if lo > 0.5 else 'not shown within site'})
r23 = pd.DataFrame(rows)
if MODE == 'real':
    ref = r23[(r23.model == 'background only') & (r23.encoder == 'merlin')].within_site_auc.iloc[0]
    assert abs(ref - 0.7874) <= TOL, ('within-site reference not reproduced', ref)
r23.to_csv(OUT / 'r23_within_site.csv', index=False)
print(r23.round(3).to_string())
rows = []
for j, mname in enumerate(HABITUS):
    x = Hs[:, j]; ok = np.isfinite(x)
    y_, P_, g_ = ys[ok], as_P(x[ok]), site[ok]
    lo, hi = boot_stat(y_, P_, lambda yy, PP, i: rep_within(yy, PP, g_[i]))
    rows.append({'measure': mname, 'n': int(ok.sum()), 'auc_all_patients': rep_auc(y_, P_),
                 'within_site_auc': rep_within(y_, P_, g_), 'within_ci_low': lo, 'within_ci_high': hi})
r23h = pd.DataFrame(rows)
r23h.to_csv(OUT / 'r23_habitus_within_site.csv', index=False)
print(r23h.round(3).to_string())

# ---------------------------------------------------------------- R24: specificity of the habitus measures
rows = []
for enc, Xs_, Xc_ in (('ctfm', X_bg_c, Xc_bg_c), ('merlin', X_bg_m, Xc_bg_m)):
    rng = np.random.default_rng(SEED)
    real_auc = auc_binary(yc, EXT[(enc, 'background only, habitus removed')])
    plc = np.array([auc_binary(yc, residual_fit_apply(Xs_, Hs[rng.permutation(len(ys))], ys, Xc_, Hc[rng.permutation(len(yc))]))
                    for _ in range(N_PLACEBO)])
    p = (1 + int((plc <= real_auc + 1e-12).sum())) / (1 + N_PLACEBO)
    rows.append({'encoder': enc, 'native_background_auc': auc_binary(yc, EXT[(enc, 'background only')]),
                 'habitus_removed_auc': real_auc, 'placebo_median_auc': float(np.median(plc)),
                 'placebo_p2_5': float(np.percentile(plc, 2.5)), 'placebo_p97_5': float(np.percentile(plc, 97.5)),
                 'placebo_at_or_below': int((plc <= real_auc + 1e-12).sum()), 'n_placebo': N_PLACEBO, 'p_one_sided': p,
                 'reading': 'specific to the measures' if p < 0.05 else 'not shown to be specific'})
r24 = pd.DataFrame(rows)
r24.to_csv(OUT / 'r24_placebo.csv', index=False)
print(r24.round(3).to_string())
rg = frames[RADIOGENOMICS]
Hr, yr = habitus(rg.key.to_numpy()), rg.y.to_numpy()
rows = []
for j, mname in [(None, 'none (all seven measures)')] + list(enumerate(HABITUS)):
    cols = [c for c in range(len(HABITUS)) if c != j]
    m = habitus_model().fit(Hs[:, cols], ys)
    sc_, sr_ = m.decision_function(Hc[:, cols]), m.decision_function(Hr[:, cols])
    lo, hi = ci([auc_binary(yc[i], sc_[i]) for (i,) in resamples([yc])])
    lo2, hi2 = ci([auc_binary(yr[i], sr_[i]) for (i,) in resamples([yr])])
    rows.append({'measure_left_out': mname, 'cptac_auc': auc_binary(yc, sc_), 'cptac_ci_low': lo, 'cptac_ci_high': hi,
                 'radiogenomics_auc': auc_binary(yr, sr_), 'radiogenomics_ci_low': lo2, 'radiogenomics_ci_high': hi2})
r24b = pd.DataFrame(rows)
r24b.to_csv(OUT / 'r24_leave_one_measure_out.csv', index=False)
coef = pd.DataFrame({'measure': HABITUS, 'coefficient_standardized': hm.named_steps['model'].coef_[0]})
coef.to_csv(OUT / 'r24_habitus_coefficients.csv', index=False)
print(r24b.round(3).to_string()); print(coef.round(3).to_string())


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


man = {'script': 'r22_r24_source_checks.py', 'plan': '7p (R22-R24)', 'mode': MODE,
       'labels': 'real' if MODE == 'real' else 'permuted within each cohort (rehearsal labels of notebook 12)',
       'labels_md5': md5(A12 / MODE / 'dataset_labels.csv'), 'run_started_utc': STARTED,
       'run_finished_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
       'settings': {'SEED': SEED, 'N_BOOT': N_BOOT, 'N_SPLITS': N_SPLITS, 'N_REPEATS': N_REPEATS, 'N_PLACEBO': N_PLACEBO,
                    'TOL': TOL, 'LR_PARAMS': LR_PARAMS, 'HABITUS': HABITUS, 'BALANCED_SITE': BALANCED_SITE},
       'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                    'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__},
       'reproduction': {'n': int(len(rep)), 'largest_abs_difference': float(rep.difference.abs().max())},
       'outputs': [{'file': p.name, 'md5': md5(p)} for p in sorted(OUT.glob('*.csv'))]}
(OUT / 'run_manifest.json').write_text(json.dumps(man, indent=2))
print('done:', MODE)
