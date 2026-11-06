#!/usr/bin/env python3
"""TCIA acquisition-shortcut audit: can a benchmark's labels be predicted from scan metadata alone?

For a classification task built on The Cancer Imaging Archive (TCIA), the tool
  1. downloads the series descriptions of the task's collections from the public TCIA API (no images),
  2. picks one main series per patient with a fixed rule,
  3. builds a "label card" per patient from that series: scanner vendor and model, software version, image count,
     slice thickness and reconstruction kernel parsed from the text, body part, and the words of the descriptions
     (words that name a diagnosis, a class or a hospital are blocked),
  4. trains a simple logistic regression on the label card only, with repeated cross-validation, a bootstrap 95% CI
     and a permutation test against chance,
  5. reports which fields carry the signal, how each field is distributed by class, and, if contributing sites are
     given, whether the signal survives within one site.

A task is flagged when the lower 95% CI bound of the AUC is >= 0.60 and the permutation p < 0.05 (for one task; for a
family of tasks apply Benjamini-Hochberg to the p-values first). A flag means: before training an image model on this
task, check that the model does not just learn how the scans were acquired. A task that is not flagged is not shown to
be free of such shortcuts.

Version 1.0.1 (4 Nov 2026) changes only the wording of the report and the log; every computation is that of 1.0.0.

The rules are those of notebooks 01-03 of "Screening public cancer CT benchmarks for acquisition shortcuts: tissue-free
prediction and the limits of external validation" (H. El Massaoudy). Run `python tcia_shortcut_audit.py --help` for the options.
"""
import argparse, datetime, gzip, hashlib, json, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

__version__ = '1.0.1'
TCIA_API = 'https://services.cancerimagingarchive.net/nbia-api/services/v1'

# ----------------------------------------------------------------------------------------------------------------------
# Fixed rules (notebook 01, rule 7a.7: main series)
IMAGING_MODALITIES = ('CT', 'MR')
MIN_IMAGES = 10
EXCLUDE_SERIES_PATTERN = (r'scout|locali[sz]er|\bloc\b|topogram|survey|calibration|\bcal\b|'
                          r'bolus|smart\s*prep|monitor|dose\s*(?:report|info)|screen\s*save|'
                          r'secondary|derived|\bmpr\b|\bmip\b|reformat|fusion')
CT_REFORMAT_PATTERN = r'\bsag|\bcor\b|coronal'

# Fixed rules (notebook 02, rules 7a.3 and 7a.4: features and words)
BLOCKED_PHRASES = r'(?:high|low)[\s_-]*grade'
LABEL_WORDS = {
    'tumor', 'tumour', 'tumors', 'braintumor', 'cancer', 'ca', 'carcinoma', 'kanker', 'malignant', 'malignancy',
    'benign', 'neoplasm', 'mass', 'masses', 'lesion', 'lesions', 'nodule', 'nodules', 'mets', 'met', 'metastatic',
    'metastasis', 'metastases', 'grade', 'stage', 'gbm', 'lgg', 'glioma', 'glioblastoma', 'hcc', 'rcc', 'ccrcc',
    'kirc', 'kirp', 'kich', 'luad', 'lusc', 'adenocarcinoma', 'adeno', 'squamous', 'chromophobe', 'papillary',
    'her2', 'triple', 'tnbc', 'luminal', 'mgmt', 'idh', 'egfr', 'kras', 'onco', 'oncology', 'onc', 'tcga', 'cptac',
    'upenn', 'lidc', 'prostatex', 'endometrial', 'endometrio', 'lymphoma', 'melanoma', 'sarcoma', 'vhl'}
SITE_WORDS = {'roswell', 'rpci', 'bjh', 'bwh', 'va', 'uob', 'fmc', 'osf', 'mskcc', 'mdacc', 'mayo', 'emory',
              'stanford', 'duke', 'penn', 'radboud', 'amc'}
CARE_PATHWAY_WORDS = {'biopsy', 'bx', 'guided', 'screen', 'screening', 'nlst', 'acrin', 'ltrc', 'followup',
                      'follow', 'fu', 'restaging', 'staging', 'surveillance', 'preop', 'postop', 'stealth',
                      'outside', 'external', 'foreign', 'comparison', 'read', 'second', 'opinion', 'research',
                      'trial', 'hematuria', 'pain', 'jaundice', 'trauma', 'claustro', 'detectie', 'detection',
                      'kliniek', 'stone', 'workup', 'hx'}
PROCESSING_WORDS = {'captk', 'processed'}
KERNEL_RE = re.compile(r'\b((?:[bhiq]r|[bcdhiq]|fc)\d{2}[fsh]?)\b', re.I)
GE_KERNEL_WORDS = {'standard', 'std', 'stnd', 'soft', 'lung', 'bone', 'detail', 'edge'}
THICK_MM_RE = re.compile(r'(\d+(?:\.\d+)?)\s*mm\b', re.I)
THICK_BEFORE_KERNEL_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s+(?:(?:[bhiq]r|[bcdhiq]|fc)\d{2}[fsh]?|standard|std|stnd|soft|lung|bone|detail)\b', re.I)
USUAL_THICKNESSES = {0.5, 0.6, 0.625, 0.75, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.75, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 10.0}

# Fixed rules (notebook 03, rules 7a.2 and 7b: model and statistics)
FIELD_GROUPS = {
    'scanner (vendor, model)': ['vendor', 'model'],
    'software version': ['software_version'],
    'image count': ['image_count'],
    'slice thickness and kernel': ['slice_thickness_mm', 'kernel'],
    'body part': ['body_part'],
    'description words': ['protocol_text'],
}
FEATURE_COLS = [c for cols in FIELD_GROUPS.values() for c in cols]
CATEGORICAL = {'vendor', 'model', 'software_version', 'kernel', 'body_part', 'site'}
LOG_NUMERIC = {'image_count'}
TEXT = {'protocol_text'}
LR_PARAMS = dict(C=1.0, class_weight='balanced', max_iter=5000)
FLAG_LOWER_CI, FLAG_P = 0.60, 0.05


# ----------------------------------------------------------------------------------------------------------------------
# 1. Series metadata from the TCIA API (replies cached on disk, so a rerun gives the same numbers)
def fetch_series(collection, cache_dir, log):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (hashlib.sha1(f'getSeries|{collection}'.encode()).hexdigest() + '.json.gz')
    if path.exists():
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            rec = json.load(f)
        log(f'  {collection}: {len(rec["data"])} series (cached reply of {rec["fetched_utc"]})')
        return pd.DataFrame(rec['data'])
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1.5,
                                                      status_forcelist=(429, 500, 502, 503, 504))))
    r = s.get(f'{TCIA_API}/getSeries', params={'Collection': collection}, timeout=300)
    r.raise_for_status()
    data = json.loads(r.text) if r.text.strip() else []
    if not data:
        raise SystemExit(f'TCIA returned no series for collection "{collection}". Check the exact name '
                         f'(e.g. on https://www.cancerimagingarchive.net/browse-collections/).')
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump({'collection': collection, 'fetched_utc': now(), 'data': data}, f)
    log(f'  {collection}: {len(data)} series (downloaded)')
    return pd.DataFrame(data)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


# ----------------------------------------------------------------------------------------------------------------------
# 2. One main series per patient and modality (rule 7a.7)
def main_series(series):
    series = series.replace(r'^\s*$', np.nan, regex=True).copy()
    for c in ('ProtocolName', 'SeriesDescription', 'StudyDesc', 'BodyPartExamined', 'Manufacturer',
              'ManufacturerModelName', 'SoftwareVersions', 'StudyDate', 'SeriesNumber', 'FileSize'):
        if c not in series:
            series[c] = np.nan
    series['ImageCount'] = pd.to_numeric(series['ImageCount'], errors='coerce')
    series['StudyDate_dt'] = pd.to_datetime(series['StudyDate'], errors='coerce')
    img = series[series.Modality.isin(IMAGING_MODALITIES)].copy()
    desc = img.SeriesDescription.where(img.SeriesDescription.notna(), img.ProtocolName).fillna('')
    excluded = (desc.str.contains(EXCLUDE_SERIES_PATTERN, flags=re.I, regex=True)
                | ((img.Modality == 'CT') & desc.str.contains(CT_REFORMAT_PATTERN, flags=re.I, regex=True))
                | (img.ImageCount.fillna(0) < MIN_IMAGES))
    if 'ThirdPartyAnalysis' in img:
        excluded |= img.ThirdPartyAnalysis.astype(str).str.strip().str.lower() == 'yes'
    img['eligible'] = ~excluded
    rows = []
    for (col, mod, pid), g in img.groupby(['Collection', 'Modality', 'PatientID'], sort=True):
        order = (g.groupby('StudyInstanceUID').StudyDate_dt.min().reset_index()
                  .sort_values(['StudyDate_dt', 'StudyInstanceUID'], na_position='last'))
        for rank, suid in enumerate(order.StudyInstanceUID, 1):
            e = g[(g.StudyInstanceUID == suid) & g.eligible]
            if len(e):
                b = e.sort_values(['ImageCount', 'SeriesNumber', 'SeriesInstanceUID'], ascending=[False, True, True]).iloc[0]
                rows.append({'collection': col, 'PatientID': pid, 'modality': mod, 'study_rank_used': rank,
                             'SeriesInstanceUID': b.SeriesInstanceUID, 'SeriesDescription': b.SeriesDescription,
                             'ProtocolName': b.ProtocolName, 'StudyDesc': b.StudyDesc, 'BodyPartExamined': b.BodyPartExamined,
                             'Manufacturer': b.Manufacturer, 'ManufacturerModelName': b.ManufacturerModelName,
                             'SoftwareVersions': b.SoftwareVersions, 'ImageCount': b.ImageCount, 'FileSize': b.FileSize})
                break
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# 3. The label card (rules 7a.3 and 7a.4)
def parse_thickness(t):
    m = THICK_MM_RE.search(t)
    if m:
        v = float(m.group(1))
        return v if 0.4 <= v <= 10 else np.nan
    m = THICK_BEFORE_KERNEL_RE.search(t)
    v = float(m.group(1)) if m else np.nan
    return v if v in USUAL_THICKNESSES else np.nan


def parse_kernel(t, modality, vendor):
    if modality != 'CT':
        return np.nan
    m = KERNEL_RE.search(t)
    if m:
        return m.group(1).upper()
    if vendor == 'GE':
        words = set(re.split(r'[^a-z]+', t.lower())) & GE_KERNEL_WORDS
        return '/'.join(sorted(words)).upper() if words else np.nan
    return np.nan


def tokenize(t):
    if not isinstance(t, str):
        return set()
    t = re.sub(BLOCKED_PHRASES, ' grade ', t.lower())
    return {w.strip('.') for w in re.split(r'[^a-z0-9.]+', t) if w.strip('.')}


def label_card(ms, blocked_extra):
    text = ms[['SeriesDescription', 'ProtocolName']].fillna('').astype(str).agg(' | '.join, axis=1)
    as_text = lambda s: s.map(lambda v: v if isinstance(v, str) else np.nan).astype(object)   # missing stays NaN
    vendor = as_text(ms.Manufacturer).str.upper().str.extract(r'^\s*([A-Z]+)', expand=False)
    card = pd.DataFrame({
        'collection': ms.collection, 'PatientID': ms.PatientID, 'modality': ms.modality,
        'vendor': vendor, 'model': as_text(ms.ManufacturerModelName).str.strip().str.upper(),
        'software_version': ms.SoftwareVersions.astype('string').str.strip(),
        'image_count': pd.to_numeric(ms.ImageCount, errors='coerce'), 'body_part': ms.BodyPartExamined,
        'slice_thickness_mm': text.map(parse_thickness),
    })
    card['kernel'] = [parse_kernel(t, m, v) for t, m, v in zip(text, card.modality, card.vendor.astype(object))]
    blocked = LABEL_WORDS | SITE_WORDS | PROCESSING_WORDS | blocked_extra
    kept, care, removed = [], [], []
    for r in ms.itertuples():
        words = set()
        for field in ('SeriesDescription', 'ProtocolName', 'StudyDesc'):
            words |= tokenize(getattr(r, field))
        kept.append(' '.join(sorted(w for w in words if w not in blocked and w not in CARE_PATHWAY_WORDS)))
        care.append(' '.join(sorted(w for w in words if w in CARE_PATHWAY_WORDS)))
        removed.append(' '.join(sorted(w for w in words if w in blocked)))
    card['protocol_text'], card['care_pathway_words'], card['blocked_words'] = kept, care, removed
    for c in ('vendor', 'model', 'software_version', 'kernel', 'body_part'):
        card[c] = card[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    return card


# ----------------------------------------------------------------------------------------------------------------------
# 4. Model and statistics (notebook 03)
class TokenBinarizer(BaseEstimator, TransformerMixin):
    """One 0/1 column per word used by at least `min_count` training patients."""
    def __init__(self, min_count=3):
        self.min_count = min_count
    def _docs(self, X):
        X = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X).ravel())
        return X.fillna('').astype(str).tolist()
    def fit(self, X, y=None):
        counts = {}
        for d in self._docs(X):
            for t in set(d.split()):
                counts[t] = counts.get(t, 0) + 1
        self.vocab_ = {t: i for i, t in enumerate(sorted(t for t, c in counts.items() if c >= self.min_count))}
        return self
    def transform(self, X):
        rows, cols = [], []
        docs = self._docs(X)
        for i, d in enumerate(docs):
            for t in set(d.split()):
                j = self.vocab_.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), max(len(self.vocab_), 1)))


class Audit:
    def __init__(self, seed=2026, n_splits=5, n_repeats=5, n_boot=2000, n_perm=999, min_count=3, n_jobs=-1):
        self.seed, self.n_splits, self.n_repeats = seed, n_splits, n_repeats
        self.n_boot, self.n_perm, self.min_count, self.n_jobs = n_boot, n_perm, min_count, n_jobs

    def column_transformer(self, cols):
        parts = []
        for c in cols:
            if c in CATEGORICAL:
                parts.append((c, Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='missing',
                                                                  keep_empty_features=True)),
                                           ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                                                    min_frequency=self.min_count, sparse_output=True))]), [c]))
            elif c in TEXT:
                parts.append((c, TokenBinarizer(self.min_count), [c]))
            else:
                steps = [('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'))] if c in LOG_NUMERIC else []
                steps += [('fill', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                          ('scale', StandardScaler())]
                parts.append((c, Pipeline(steps), [c]))
        return ColumnTransformer(parts, remainder='drop', sparse_threshold=0.3)

    def model(self, cols):
        return Pipeline([('prep', self.column_transformer(cols)), ('model', LogisticRegression(**LR_PARAMS))])

    @staticmethod
    def auc_binary(y, s):
        n1 = int(y.sum()); n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            return np.nan
        r = rankdata(s)
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    def auc(self, y, P):
        """Binary AUC, or macro one-vs-rest AUC for more than two classes."""
        if P.shape[1] == 2:
            return self.auc_binary(y, P[:, 1])
        return float(np.nanmean([self.auc_binary((y == j).astype(int), P[:, j]) for j in range(P.shape[1])]))

    def splits(self, y):
        return [list(StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed + r).split(np.zeros(len(y)), y))
                for r in range(self.n_repeats)]

    def _fit_fold(self, df, y, cols, train, test, k):
        warnings.filterwarnings('ignore')
        m = self.model(cols).fit(df.iloc[train], y[train])
        proba = np.zeros((len(test), k))
        proba[:, m.classes_] = m.predict_proba(df.iloc[test])
        return test, proba

    def oof(self, df, y, cols, splits):
        k = int(y.max()) + 1
        jobs = [(r, tr, te) for r, folds in enumerate(splits) for tr, te in folds]
        res = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_fold)(df, y, cols, tr, te, k) for _, tr, te in jobs)
        P = np.zeros((len(splits), len(y), k))
        for (r, _, _), (te, proba) in zip(jobs, res):
            P[r, te] = proba
        return P

    def mean_auc(self, y, P):
        return float(np.mean([self.auc(y, P[r]) for r in range(P.shape[0])]))

    def boot_ci(self, y, P):
        rng = np.random.default_rng(self.seed)
        k, vals = P.shape[2], []
        while len(vals) < self.n_boot:
            idx = rng.choice(len(y), len(y), replace=True)
            if len(np.unique(y[idx])) < k:
                continue
            vals.append(np.mean([self.auc(y[idx], P[r][idx]) for r in range(P.shape[0])]))
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    def estimate(self, df, y, cols, splits):
        P = self.oof(df, y, cols, splits)
        lo, hi = self.boot_ci(y, P)
        return {'auc': self.mean_auc(y, P), 'ci_low': lo, 'ci_high': hi}, P

    def _single_cv_auc(self, folds, y):
        P = np.zeros((len(y), int(y.max()) + 1))
        for tr, te, Xtr, Xte in folds:
            m = LogisticRegression(**LR_PARAMS).fit(Xtr, y[tr])
            P[np.ix_(te, m.classes_)] = m.predict_proba(Xte)
        return self.auc(y, P)

    def permutation_test(self, df, y, cols):
        """One fixed 5-fold split that does not depend on the labels; the encoding uses no labels, so it is fitted
        once per training fold and each permutation refits only the model."""
        folds = []
        for tr, te in KFold(self.n_splits, shuffle=True, random_state=self.seed).split(np.zeros(len(df))):
            ct = self.column_transformer(cols).fit(df.iloc[tr])
            folds.append((tr, te, ct.transform(df.iloc[tr]), ct.transform(df.iloc[te])))
        observed = self._single_cv_auc(folds, y)
        rng = np.random.default_rng(self.seed)
        perms = [rng.permutation(y) for _ in range(self.n_perm)]
        chunks = [perms[i:i + 50] for i in range(0, len(perms), 50)]
        null = np.concatenate(Parallel(n_jobs=self.n_jobs)(
            delayed(lambda c: [self._single_cv_auc(folds, yp) for yp in c])(c) for c in chunks))
        return observed, (1 + int((null >= observed).sum())) / (1 + self.n_perm), null

    def within_site(self, y, P, site):
        def wauc(yy, ss, st):
            num, den, n = 0.0, 0, 0
            for s in np.unique(st):
                m = st == s
                a, b = yy[m], ss[m]
                n1 = int(a.sum()); n0 = len(a) - n1
                if n1 and n0:
                    r = rankdata(b)
                    num += r[a == 1].sum() - n1 * (n1 + 1) / 2
                    den += n1 * n0
                    n += 1
            return (num / den if den else np.nan), den, n
        vals = [wauc(y, P[r][:, 1], site)[0] for r in range(P.shape[0])]
        _, pairs, n_sites = wauc(y, P[0][:, 1], site)
        if pairs == 0:
            return {'auc': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n_pairs': 0, 'n_sites_with_pairs': 0}
        rng = np.random.default_rng(self.seed)
        groups = [np.where(site == s)[0] for s in np.unique(site)]
        boots = []
        while len(boots) < self.n_boot:
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            v = np.mean([wauc(y[idx], P[r][idx, 1], site[idx])[0] for r in range(P.shape[0])])
            if not np.isnan(v):
                boots.append(v)
        return {'auc': float(np.mean(vals)), 'ci_low': float(np.percentile(boots, 2.5)),
                'ci_high': float(np.percentile(boots, 97.5)), 'n_pairs': int(pairs), 'n_sites_with_pairs': n_sites}


# ----------------------------------------------------------------------------------------------------------------------
# 5. Putting it together
def load_task(args, log):
    """Patients, labels (and sites) of the task, with their series metadata."""
    if args.labels:
        lab = pd.read_csv(args.labels, dtype=str)
        need = {'PatientID', 'label'}
        assert need <= set(lab.columns), f'the labels file needs the columns {sorted(need)} (optional: collection, site)'
        if 'collection' not in lab:
            assert args.collections, 'give --collections, or a "collection" column in the labels file'
            if len(args.collections) > 1:
                raise SystemExit('with several collections, the labels file needs a "collection" column')
            lab['collection'] = args.collections[0]
        collections = sorted(lab.collection.unique())
    else:
        assert args.collections and len(args.collections) >= 2, \
            '--label-by-collection needs two or more --collections (the class is the collection)'
        collections, lab = list(args.collections), None
    if args.series_csv:
        series = pd.read_csv(args.series_csv, low_memory=False)
        series = series[series.Collection.isin(collections)]
        log(f'Series metadata from {args.series_csv}: {len(series)} series of {collections}')
    else:
        log('Series metadata from the TCIA API (no images are downloaded):')
        series = pd.concat([fetch_series(c, Path(args.cache_dir), log).assign(Collection=c) for c in collections],
                           ignore_index=True)
    ms = main_series(series)
    modality = args.modality or ms.groupby('modality').PatientID.nunique().idxmax()
    ms = ms[ms.modality == modality].reset_index(drop=True)
    if lab is None:
        lab = ms[['collection', 'PatientID']].assign(label=ms.collection)
    task = lab.merge(ms, on=['collection', 'PatientID'], how='inner', validate='one_to_one')
    missing = len(lab) - len(task)
    log(f'Modality {modality}: {len(task)} labelled patients with a main series'
        + (f' ({missing} labelled patients have none and are left out)' if missing else ''))
    return task.reset_index(drop=True), modality, series


def fields_by_class(card, y, classes):
    rows = []
    for c in ('vendor', 'model', 'software_version', 'kernel', 'body_part'):
        v = card[c].fillna('missing')
        for val in v.value_counts().index[:15]:
            rows.append({'field': c, 'value': val, **{f'share_{cl}': round(float((v[y == j] == val).mean()), 3)
                                                      for j, cl in enumerate(classes)}})
    for c in ('image_count', 'slice_thickness_mm'):
        rows.append({'field': c, 'value': 'median', **{f'share_{cl}': float(np.nanmedian(card[c][y == j].astype(float)))
                                                       if card[c][y == j].notna().any() else np.nan
                                                       for j, cl in enumerate(classes)}})
    return pd.DataFrame(rows)


def run(args):
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    def log(msg):
        print(msg, flush=True)
        lines.append(msg)
    log(f'TCIA acquisition-shortcut audit {__version__}, started {now()}')
    task, modality, series = load_task(args, log)
    classes = sorted(task.label.unique())
    if len(classes) < 2:
        raise SystemExit('the task needs at least two classes')
    counts = task.label.value_counts()
    if counts.min() < args.min_per_class:
        log(f'WARNING: smallest class has {counts.min()} patients (< {args.min_per_class}); results will be unstable')
    y = task.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    blocked_extra = set(w.lower() for w in args.block_words) | {w for c in list(classes) + list(task.collection.unique())
                                                               for w in tokenize(str(c))}
    card = label_card(task, blocked_extra)
    df = card.copy()
    if 'site' in task:
        df['site'] = task['site'].values
    A = Audit(args.seed, n_boot=args.n_boot, n_perm=args.n_perm, min_count=args.min_count, n_jobs=args.jobs)
    splits = A.splits(y)

    log(f'Classes: ' + ', '.join(f'{c} ({int((y == j).sum())})' for j, c in enumerate(classes)))
    log('Fitting the metadata-only model (5 x 5 cross-validation) ...')
    P = A.oof(df, y, FEATURE_COLS, splits)
    a = A.mean_auc(y, P)
    lo, hi = A.boot_ci(y, P)
    obs, p, null = A.permutation_test(df, y, FEATURE_COLS)
    flagged = bool(lo >= FLAG_LOWER_CI and p < FLAG_P)
    summary = {'tool_version': __version__, 'collections': sorted(task.collection.unique()), 'modality': modality,
               'classes': {c: int((y == j).sum()) for j, c in enumerate(classes)}, 'n_patients': int(len(y)),
               'auc': a, 'ci_low': lo, 'ci_high': hi, 'perm_p': p, 'perm_observed_single_cv': obs,
               'null_mean_auc': float(null.mean()), 'flagged': flagged,
               'flag_rule': f'lower CI >= {FLAG_LOWER_CI} and permutation p < {FLAG_P} (apply BH first for a family of tasks)',
               'settings': {'seed': args.seed, 'n_boot': args.n_boot, 'n_perm': args.n_perm, 'min_count': args.min_count,
                            'cv': '5-fold stratified, 5 repeats', 'model': 'logistic regression, C=1, balanced'},
               'started_utc': lines[0].split('started ')[-1]}
    log(f'Metadata-only AUC {a:.3f} (95% CI {lo:.3f}-{hi:.3f}), permutation p {p:.4f} -> '
        + ('FLAGGED: acquisition metadata predicts the label' if flagged else 'not flagged'))

    log('Field groups: alone and left out ...')
    groups = []
    for g, cols in FIELD_GROUPS.items():
        alone, _ = A.estimate(df, y, cols, splits)
        left = A.mean_auc(y, A.oof(df, y, [c for c in FEATURE_COLS if c not in cols], splits))
        groups.append({'group': g, 'filled_fraction': float(df[cols].replace('', np.nan).notna().any(axis=1).mean()),
                       'alone_auc': alone['auc'], 'alone_ci_low': alone['ci_low'], 'alone_ci_high': alone['ci_high'],
                       'drop_when_left_out': a - left})
    groups = pd.DataFrame(groups)

    site_res = None
    if 'site' in df and df.site.notna().any() and len(classes) == 2:
        log('Site analysis (site column given) ...')
        has = df.site.notna().to_numpy()
        ds, ys = df[has].reset_index(drop=True), y[has]
        sarr = ds.site.astype(str).to_numpy()
        site_res = {'n_patients_with_site': int(has.sum()), 'n_sites': int(len(np.unique(sarr)))}
        est, _ = A.estimate(ds, ys, ['site'], A.splits(ys))
        site_res.update({f'site_only_{k}': v for k, v in est.items()})
        if len(np.unique(sarr)) >= 2:
            gkf = list(GroupKFold(min(5, len(np.unique(sarr)))).split(np.zeros(len(ys)), ys, sarr))
            if all(len(np.unique(ys[tr])) == 2 for tr, _ in gkf):
                Pg = A.oof(ds, ys, FEATURE_COLS, [gkf])
                lo_g, hi_g = A.boot_ci(ys, Pg)
                site_res.update({'sites_held_out_auc': A.mean_auc(ys, Pg), 'sites_held_out_ci_low': lo_g,
                                 'sites_held_out_ci_high': hi_g})
        site_res.update({f'within_site_{k}': v for k, v in A.within_site(ys, P[:, has], sarr).items()})
        summary['site_analysis'] = site_res

    fb = fields_by_class(card, y, classes)
    keep = ['collection', 'PatientID', 'modality', 'label', 'SeriesInstanceUID', 'study_rank_used']
    task[keep].to_csv(out / 'main_series.csv', index=False)
    card.assign(label=task.label.values).to_csv(out / 'label_card.csv', index=False)
    groups.to_csv(out / 'field_groups.csv', index=False)
    fb.to_csv(out / 'fields_by_class.csv', index=False)
    pd.DataFrame({'PatientID': task.PatientID, 'label': task.label,
                  **{f'p_{c}': P.mean(0)[:, j] for j, c in enumerate(classes)}}).to_csv(out / 'oof_scores.csv', index=False)
    summary['finished_utc'] = now()
    summary['runtime_s'] = round(time.time() - t0, 1)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    (out / 'report.md').write_text(report(summary, groups, fb, site_res, classes), encoding='utf-8')
    log(f'Done in {summary["runtime_s"]} s. Report: {out / "report.md"}')


def report(s, groups, fb, site_res, classes):
    verdict = ('**FLAGGED.** The label can be predicted from how the scans were acquired, without looking at any image. '
               'An image model trained on this task may learn the acquisition instead of the disease. Before trusting such '
               'a model, test it on images with the segmented tissue removed (background only), within one site or scanner, '
               'and on an independent dataset that was not assembled the same way.') if s['flagged'] else \
              ('**Not flagged.** At the thresholds used, acquisition metadata did not predict the label. This does not show '
               'that the task is free of acquisition shortcuts: fields missing from the metadata, such as reconstruction '
               'details stored only in the image headers, are not tested, and the sensitivity of this screen has not been '
               'established.')
    g = groups.sort_values('alone_auc', ascending=False)
    rows = '\n'.join(f"| {r.group} | {r.filled_fraction:.0%} | {r.alone_auc:.3f} ({r.alone_ci_low:.3f}-{r.alone_ci_high:.3f}) | "
                     f"{r.drop_when_left_out:+.3f} |" for r in g.itertuples())
    cls_cols = [c for c in fb.columns if c.startswith('share_')]
    fbt = '\n'.join('| ' + ' | '.join([str(r['field']), str(r['value'])] + [f'{r[c]:.2f}' if pd.notna(r[c]) else '' for c in cls_cols]) + ' |'
                    for _, r in fb.iterrows())
    site_txt = ''
    if site_res:
        site_txt = ('\n## Site or protocol?\n\n'
                    f"- {site_res['n_patients_with_site']} patients from {site_res['n_sites']} sites (as supplied; a "
                    "site may not be where the images were acquired).\n"
                    f"- Site alone: AUC {site_res.get('site_only_auc', float('nan')):.3f}.\n"
                    + (f"- Whole sites held out: AUC {site_res['sites_held_out_auc']:.3f} "
                       f"({site_res['sites_held_out_ci_low']:.3f}-{site_res['sites_held_out_ci_high']:.3f}).\n" if 'sites_held_out_auc' in site_res else '')
                    + f"- Within one site (pairs from the same site): AUC {site_res.get('within_site_auc', float('nan')):.3f} "
                      f"({site_res.get('within_site_ci_low', float('nan')):.3f}-{site_res.get('within_site_ci_high', float('nan')):.3f}). "
                    'If this stays high, the signal is protocol linked to the label, not only the site.\n')
    return f"""# Acquisition-shortcut audit

Collections: {', '.join(s['collections'])} · modality {s['modality']} · {s['n_patients']} patients · classes: {', '.join(f'{k} ({v})' for k, v in s['classes'].items())}

## Result

Metadata-only AUC **{s['auc']:.3f}** (95% CI {s['ci_low']:.3f}-{s['ci_high']:.3f}); permutation p = {s['perm_p']:.4f} (null mean {s['null_mean_auc']:.3f}).

{verdict}

Rule: {s['flag_rule']}.

## Which fields carry the signal

| Field group | Filled | AUC alone (95% CI) | Change in AUC when left out |
|---|---|---|---|
{rows}

## Fields by class (share of patients, or median)

| Field | Value | {' | '.join(classes)} |
|---|---|{'---|' * len(classes)}
{fbt}
{site_txt}
## How this was computed

One main series per patient (earliest study; the eligible series with the most images; scouts, localisers, reformats and
analysis results excluded). Label card: scanner vendor and model, software version, image count, slice thickness and
kernel parsed from the descriptions, body part, and description words (words naming a diagnosis, class or hospital are
removed; words about why or when the scan was taken are kept apart and not used). Model: logistic regression,
5-fold stratified cross-validation repeated 5 times; 95% CI from {s['settings']['n_boot']} patient bootstrap resamples;
permutation test with {s['settings']['n_perm']} label permutations. Tool version {s['tool_version']}.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='Examples:\n'
                                        '  python tcia_shortcut_audit.py --collections TCGA-LUAD TCGA-LUSC --label-by-collection --out lung\n'
                                        '  python tcia_shortcut_audit.py --labels my_labels.csv --collections NSCLC-Radiomics --out nsclc\n')
    ap.add_argument('--collections', nargs='+', help='TCIA collection name(s), exactly as on the TCIA API')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--labels', help='CSV with PatientID and label (optional: collection, site)')
    src.add_argument('--label-by-collection', action='store_true', help='the class is the collection (e.g. TCGA-LUAD vs TCGA-LUSC)')
    ap.add_argument('--modality', choices=['CT', 'MR'], help='default: the modality with more patients')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--series-csv', help='use a saved series-metadata table instead of the API (columns as TCIA getSeries)')
    ap.add_argument('--cache-dir', default='.tcia_cache', help='where API replies are cached (default: .tcia_cache)')
    ap.add_argument('--block-words', nargs='*', default=[], help='extra words to remove from the descriptions')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--n-perm', type=int, default=999)
    ap.add_argument('--min-count', type=int, default=3, help='a category or word is used only if >= this many training patients have it')
    ap.add_argument('--min-per-class', type=int, default=20, help='warn below this class size')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--jobs', type=int, default=-1, help='parallel jobs (default: all CPUs)')
    ap.add_argument('--version', action='version', version=__version__)
    run(ap.parse_args(argv))


if __name__ == '__main__':
    main()
