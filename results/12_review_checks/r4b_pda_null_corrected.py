"""Check R4b (4 Nov 2026; POST HOC, descriptive diagnostic).

R4 compared the CPTAC-PDA stage Merlin silhouette AUC (0.313, below 0.5) with a label-permutation null, but it reused
the stratified folds drawn from the observed labels for every permuted label vector. R4b corrects this: for every
permutation the stratified 5 x 5 folds are drawn again from the permuted labels (seeds 2026-2030, the same procedure as
for the observed labels), as in scikit-learn's permutation_test_score. Lower-tail P = (1 + #null <= observed) / (1 + B).

It also records how closely this machine reproduces the published image-probe AUCs (all four flagged tasks, both
encoders, full image / background / silhouette), on the same patients, labels and folds.

Outputs: r4b_pda_null.csv, r4b_reproduction.csv.  Usage: python r4b_pda_null_corrected.py  (B = 2000)"""
import sys
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
import review_lib as L

B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
feat = L.patient_features()
EMB05 = L.R / '05_image_check' / 'cache' / 'embeddings'
EMB06 = L.R / '06_image_posthoc' / 'cache' / 'embeddings'

# reproduction of the published image-probe AUCs on this machine
p05 = pd.read_csv(L.R / '05_image_check' / 'real' / 'image_probe.csv')
s06 = pd.read_csv(L.R / '06_image_posthoc' / 'real' / 'silhouette_probe.csv')
rep = []
for tid in p05.task_id.unique():
    df, y, _ = L.task_frame(tid, labels, feat)
    sp = L.splits_for(y, L.SEED)
    for enc in ['ctfm', 'merlin']:
        r = p05[(p05.task_id == tid) & (p05.encoder == enc)].iloc[0]
        s = s06[(s06.task_id == tid) & (s06.encoder == enc)].iloc[0]
        for inp, var, folder, ref in [('full image', 'original', EMB05, r.image_auc),
                                      ('background', 'background', EMB05, r.background_auc),
                                      ('silhouette', 'silhouette', EMB06, s.silhouette_auc)]:
            P, _ = L.cv_oof_matrix(L.load_emb(df.key, enc, var, folder), y, sp)
            a = L.mean_auc(y, P)
            rep.append({'task_id': tid, 'encoder': enc, 'input': inp, 'published_auc': ref, 'this_machine_auc': a,
                        'difference': a - ref, 'scikit_learn': sklearn.__version__})
rep = pd.DataFrame(rep)
rep.to_csv('r4b_reproduction.csv', index=False)

# corrected null for CPTAC-PDA stage, Merlin silhouette
tid = 'CPTAC-PDA | stage'
df, y, _ = L.task_frame(tid, labels, feat)
X = L.load_emb(df.key, 'merlin', 'silhouette', EMB06)
obs = L.mean_auc(y, L.cv_oof_matrix(X, y, L.splits_for(y, L.SEED))[0])
published = s06[(s06.task_id == tid) & (s06.encoder == 'merlin')].silhouette_auc.iloc[0]
rng = np.random.default_rng(L.SEED)
perms = [rng.permutation(y) for _ in range(B)]


def one(yp):
    return L.mean_auc(yp, L.cv_oof_matrix(X, yp, L.splits_for(yp, L.SEED))[0])


null = np.array(Parallel(n_jobs=-1)(delayed(one)(yp) for yp in perms))
k = int((null <= obs).sum())
out = pd.DataFrame([{'analysis': 'CPTAC-PDA stage, Merlin silhouette, 5x5 CV; folds redrawn per permutation',
                     'observed_auc_this_machine': obs, 'published_auc': published, 'n_permutations': B,
                     'n_null_at_or_below_observed': k, 'lower_tail_p': (1 + k) / (1 + B),
                     'null_mean': null.mean(), 'null_2.5pct': np.percentile(null, 2.5),
                     'null_97.5pct': np.percentile(null, 97.5), 'n_null_at_or_below_published': int((null <= published).sum()),
                     'seed': L.SEED, 'scikit_learn': sklearn.__version__}])
out.to_csv('r4b_pda_null.csv', index=False)
pd.set_option('display.width', 250)
print(out.T.to_string())
print('largest |difference| in reproduction:', rep.difference.abs().max())
print(rep.round(5).to_string(index=False))
