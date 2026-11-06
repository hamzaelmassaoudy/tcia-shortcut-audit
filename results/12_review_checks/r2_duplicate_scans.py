"""Check R2 (4 Nov 2026; label-free): could the same scan sit in a source and an
external cohort? Patients have different identifiers in every collection, so overlap is checked on the scans:
(a) header fingerprints (rows, columns, slices, pixel and slice spacing, kVp, tube current, HU percentiles) that match
exactly across cohorts; (b) the highest cosine similarity of CT-FM full-image embeddings between each source scan and
every scan of each external cohort, against the same statistic within the source cohort.
Pairs: TCGA lung -> NSCLC Radiogenomics, CPTAC lung, NSCLC-Radiomics, Lung-PET-CT-Dx; TCGA kidney -> C4KC-KiTS.
Outputs: r2_duplicate_scans.csv (one row per pair), r2_top_pairs.csv. Usage: python r2_duplicate_scans.py"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
EMB = {'05': R / '05_image_check/cache/embeddings', '07': R / '07_screen_validation/cache/embeddings',
       '08': R / '08_transfer/cache/embeddings',
       '11': Path('G:/My Drive/Acquisition-Shortcut-Audit/results/11_external_lung/cache/embeddings')}
INV = {'05': R / '05_image_check/cache/image_inventory.csv', '07': R / '07_screen_validation/cache/image_inventory.csv',
       '08': R / '08_transfer/cache/image_inventory.csv', '11': R / '11_external_lung/cache/image_inventory.csv'}
COHORTS = {'TCGA lung': ('05', ['TCGA-LUAD', 'TCGA-LUSC']), 'TCGA kidney': ('05', ['TCGA-KIRC', 'TCGA-KIRP']),
           'NSCLC Radiogenomics': ('07', ['NSCLC Radiogenomics']), 'CPTAC lung': ('08', ['CPTAC-LUAD', 'CPTAC-LSCC']),
           'C4KC-KiTS': ('08', ['C4KC-KiTS']), 'NSCLC-Radiomics': ('11', ['NSCLC-Radiomics']),
           'Lung-PET-CT-Dx': ('11', ['Lung-PET-CT-Dx'])}
PAIRS = [('TCGA lung', t) for t in ('NSCLC Radiogenomics', 'CPTAC lung', 'NSCLC-Radiomics', 'Lung-PET-CT-Dx')] + \
        [('TCGA kidney', 'C4KC-KiTS')]
FP = ['hdr_rows', 'hdr_columns', 'hdr_n_slices', 'hdr_pixel_spacing', 'hdr_z_spacing', 'hdr_kvp', 'hdr_tube_current',
      'hu_p01', 'hu_p50', 'hu_p99']

def cohort(name):
    nbk, colls = COHORTS[name]
    inv = pd.read_csv(INV[nbk], low_memory=False)
    inv = inv[inv.collection.isin(colls) & inv.embedded.astype(bool)].copy() if 'embedded' in inv else inv[inv.collection.isin(colls)].copy()
    inv['key'] = inv.collection + '__' + inv.PatientID
    X = np.vstack([np.load(EMB[nbk] / f'{k}.npz')['ctfm__original'] for k in inv.key])
    fp = inv[FP].round(3).astype(str).agg('|'.join, axis=1)
    return inv.reset_index(drop=True), X, fp.to_numpy()

def standardize(A, B):
    """Raw CT-FM embeddings share a large common component (raw cosine about 0.995 for any two scans), so features are
    z-scored over the two cohorts together before the cosine; identical scans still give a cosine of about 1."""
    Z = np.vstack([A, B])
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    A, B = (A - mu) / sd, (B - mu) / sd
    return A / np.linalg.norm(A, axis=1, keepdims=True), B / np.linalg.norm(B, axis=1, keepdims=True)

cache = {}
rows, tops = [], []
for src, tgt in PAIRS:
    for n in (src, tgt):
        if n not in cache:
            cache[n] = cohort(n)
    (si, SX, sfp), (ti, TX, tfp) = cache[src], cache[tgt]
    SX, TX = standardize(SX, TX)
    cross = SX @ TX.T
    within = SX @ SX.T
    np.fill_diagonal(within, -1)
    best = cross.max(axis=1)
    fp_matches = int(np.isin(sfp, tfp).sum())
    i = int(best.argmax()); j = int(cross[i].argmax())
    rows.append({'source': src, 'target': tgt, 'n_source': len(si), 'n_target': len(ti),
                 'header_fingerprint_matches': fp_matches,
                 'max_cross_cosine': float(best.max()), 'median_cross_nn_cosine': float(np.median(best)),
                 'max_within_source_cosine': float(within.max()),
                 'median_within_source_nn_cosine': float(np.median(within.max(axis=1))),
                 'cross_pairs_above_0.95': int((cross > 0.95).sum()),
                 'closest_pair': f'{si.key[i]} ~ {ti.key[j]}'})
    order = np.argsort(-best)[:5]
    tops += [{'source': src, 'target': tgt, 'source_key': si.key[k], 'target_key': ti.key[int(cross[k].argmax())],
              'cosine': float(best[k])} for k in order]
out = pd.DataFrame(rows)
out.to_csv(HERE / 'r2_duplicate_scans.csv', index=False)
pd.DataFrame(tops).to_csv(HERE / 'r2_top_pairs.csv', index=False)
pd.set_option('display.width', 250)
print(out.round(4).to_string(index=False))
