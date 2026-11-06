"""Shared helpers for the post hoc checks R10-R16 of 6 Nov 2026 (plan 7k).

Like review_lib.py (which it imports, unchanged), but it executes the whole of notebook 05's cell 11 "Model building
blocks", so the permutation tests, within-group functions and BH are the paper's own code too. It adds:
- boot_ci_levels / boot_mean_levels: notebook 05's boot_ci / boot_mean with the same random resamples, returning the
  percentile bounds for several CI levels (the 95% bounds equal boot_ci's / boot_mean's);
- loaders for the cached DICOM headers and embeddings of notebooks 05, 06 and 07.
"""
import json
import numpy as np
import pandas as pd
import review_lib as _L
from review_lib import *  # noqa: F401,F403  (settings and helpers of review_lib)

_code = _L.cell[_L.cell.index('class TokenBinarizer'):]
exec(compile(_code, 'nb05_cell11_full', 'exec'), globals())

C05 = _L.R / '05_image_check' / 'cache'
C06 = _L.R / '06_image_posthoc' / 'cache'
C07 = _L.R / '07_screen_validation' / 'cache'
KEYS05 = {p.stem for p in (C05 / 'embeddings').glob('*.npz')}
KEYS07 = {p.stem for p in (C07 / 'embeddings').glob('*.npz')}


def boot_ci_levels(y, P, levels=(0.95,)):
    """notebook 05 boot_ci (same seed, same resamples), percentile bounds at each level."""
    rng = np.random.default_rng(SEED)
    k = P.shape[2]
    vals = []
    while len(vals) < N_BOOT:
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < k:
            continue
        vals.append(np.mean([auc(y[idx], P[r][idx]) for r in range(P.shape[0])]))
    return {lv: (float(np.percentile(vals, 50 * (1 - lv))), float(np.percentile(vals, 100 - 50 * (1 - lv))))
            for lv in levels}


def boot_mean_levels(x, levels=(0.95,)):
    """notebook 05 boot_mean (same seed, same resamples), percentile bounds at each level."""
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    vals = [x[rng.choice(len(x), len(x), replace=True)].mean() for _ in range(N_BOOT)]
    return float(x.mean()), {lv: (float(np.percentile(vals, 50 * (1 - lv))),
                                  float(np.percentile(vals, 100 - 50 * (1 - lv)))) for lv in levels}


def header(key):
    """The cached DICOM header of a scan (notebook 05's cache first, then notebook 07's)."""
    p = (C05 if key in KEYS05 else C07) / 'headers' / f'{key}.json'
    return json.loads(p.read_text()) if p.exists() else {'key': key}


def emb_file(key, variant):
    """Where a scan's embedding of one input lives: notebook 05 (image, background), notebook 06 (silhouettes of the
    notebook 05 scans) or notebook 07 (all three inputs of its scans)."""
    if key in KEYS05:
        return (C06 if variant == 'silhouette' else C05) / 'embeddings' / f'{key}.npz'
    return C07 / 'embeddings' / f'{key}.npz'


def load_input(keys, enc, variant):
    rows = []
    for k in keys:
        with np.load(emb_file(k, variant), allow_pickle=True) as z:
            name = f'{enc}__{variant}'
            rows.append(z[name] if name in z.files else np.full(z[f'{enc}__original'].shape, np.nan))
    return np.vstack(rows)


def demographics():
    """R9's per-patient sex and age (the first study with a value), from the study records."""
    st = pd.read_csv(_L.R / '01_inventory' / 'studies.csv.gz', low_memory=False)
    st['age'] = st.PatientAge.astype(str).str.extract(r'(\d+)Y', expand=False).astype(float)
    st['sex'] = st.PatientSex.where(st.PatientSex.isin(['M', 'F']))
    d = (st.sort_values('StudyDate').groupby(['Collection', 'PatientID'])
           .agg(sex=('sex', lambda s: s.dropna().iloc[0] if s.notna().any() else None),
                age=('age', lambda s: s.dropna().iloc[0] if s.notna().any() else float('nan'))).reset_index())
    d['male'] = d.sex.map({'M': 1.0, 'F': 0.0})
    return d.rename(columns={'Collection': 'collection'})[['collection', 'PatientID', 'male', 'age']]
