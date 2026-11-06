"""Shared helpers for the post hoc checks R1-R9 of 4 Nov 2026. The model code is taken verbatim from notebook 05
(notebooks/05_image_model_check.ipynb, cell 11 "Model building blocks"), so the checks use the paper's own functions;
the settings below are notebook 05's (cell 3) and its patient-feature construction (cell 4)."""
import json, re
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
R = PROJECT / 'results'

# notebook 05, cell 3 (settings)
SEED = 2026
N_SPLITS, N_REPEATS = 5, 5
N_BOOT = 2000
N_PERM = 999
MIN_COUNT = 3
LR_PARAMS = dict(C=1.0, class_weight='balanced', max_iter=5000)
SCANNER_COLS = ['vendor', 'model', 'software_version', 'image_count', 'slice_thickness_mm', 'kernel', 'body_part',
                'protocol_text']
CATEGORICAL = {'vendor', 'model', 'software_version', 'kernel', 'body_part',
               'hdr_kernel', 'hdr_filter_type', 'hdr_patient_position'}
LOG_NUMERIC = {'image_count', 'body_area_cm2'}
TEXT = {'protocol_text', 'care_text'}
N_JOBS = -1

# notebook 05, cell 11 (model building blocks), executed verbatim from the project notebook
nb = json.loads((PROJECT / 'notebooks' / '05_image_model_check.ipynb').read_text(encoding='utf-8'))
cell = next(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' and 'class TokenBinarizer' in ''.join(c['source']))
code = cell[cell.index('class TokenBinarizer'):cell.index('def _fit_fold_combined')]
exec(compile(code, 'nb05_cell11', 'exec'), globals())


def patient_features():
    """notebook 05, cell 4: notebook 03's patient features with the original word list."""
    IN02 = R / '02_tasks_features'
    feat0 = pd.read_csv(IN02 / 'patient_features.csv', low_memory=False)
    toks0 = pd.read_csv(IN02 / 'patient_tokens.csv.gz')
    KEY = ['collection', 'PatientID', 'modality']
    text = (toks0.assign(group=np.where(toks0.category == 'care pathway', 'care_text', 'protocol_text'))
                 .groupby(KEY + ['group']).token.agg(lambda s: ' '.join(sorted(set(s)))).unstack('group').reset_index())
    for c in ('protocol_text', 'care_text'):
        if c not in text:
            text[c] = ''
    feat = feat0.merge(text, on=KEY, how='left')
    for c in ('protocol_text', 'care_text'):
        feat[c] = feat[c].fillna('')
    for c in ('vendor', 'model', 'software_version', 'kernel', 'body_part'):
        feat[c] = feat[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    return feat


def task_frame(tid, labels, feat):
    """notebook 05 task_frame (without the extras): patients in labels_long order, classes sorted."""
    df = labels[labels.task_id == tid].merge(feat, on=['collection', 'PatientID', 'modality'], how='left',
                                             validate='one_to_one')
    df['key'] = df.collection + '__' + df.PatientID
    classes = sorted(df.label.unique())
    y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    return df.reset_index(drop=True), y, classes


def splits_for(y, seed):
    return [list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed + r).split(np.zeros(len(y)), y))
            for r in range(N_REPEATS)]


def load_emb(keys, enc, variant, folder):
    rows = []
    for k in keys:
        z = np.load(folder / f'{k}.npz', allow_pickle=True)
        name = f'{enc}__{variant}'
        rows.append(z[name] if name in z.files else np.full(z[f'{enc}__original'].shape, np.nan))
    return np.vstack(rows)
