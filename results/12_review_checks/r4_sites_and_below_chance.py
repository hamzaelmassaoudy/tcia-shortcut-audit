"""Check R4 (4 Nov 2026; POST HOC, descriptive diagnostics).
(a) The site proxy: TCGA tissue-source-site codes and NSCLC Radiogenomics affiliations, mapped to institutions
    (notebook 02, patient_sites.csv), with per-site class counts for the 12 binary tasks with sites.
(b) Why the kidney metadata model gives AUC 0.247 with whole sites held out: per held-out fold, the sites, class
    counts and mean predicted probability of KIRP; and where 0.247 lies in a label-permutation null of the same
    site-held-out analysis (200 permutations; the GroupKFold split does not depend on the labels).
(c) Why the CPTAC-PDA stage Merlin silhouette probe gives 0.313: per-repeat AUCs, and where 0.313 lies in a
    label-permutation null of the repeated 5 x 5 CV AUC (200 permutations, same splits seeded 2026).
Outputs: r4_site_table.csv, r4_site_mapping.csv, r4_kidney_heldout_folds.csv, r4_below_chance.csv.
Usage: python r4_sites_and_below_chance.py"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import review_lib as L

labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
sites = pd.read_csv(L.R / '02_tasks_features' / 'patient_sites.csv')
feat = L.patient_features()
site_of = sites.drop_duplicates('PatientID').set_index('PatientID').institution
a03s = pd.read_csv(L.R / '03_audit' / 'audit_sites.csv').set_index('task_id')
rng = np.random.default_rng(L.SEED)

# (a) site table and mapping
sites[['collection', 'site_source', 'tss', 'institution_raw', 'institution']].drop_duplicates() \
    .sort_values(['collection', 'institution', 'tss']).to_csv('r4_site_mapping.csv', index=False)
rows = []
for tid in a03s.index:
    df, y, classes = L.task_frame(tid, labels, feat)
    df['site'] = df.PatientID.map(site_of)
    d = df[df.site.notna()]
    tab = pd.crosstab(d.site, d.label)
    mixed = tab[(tab > 0).all(axis=1)]
    for s, r in tab.iterrows():
        rows.append({'task_id': tid, 'site': s, **{f'n_{c}': int(r.get(c, 0)) for c in classes},
                     'mixed_class': bool((r > 0).all())})
    rows.append({'task_id': tid, 'site': 'ALL', 'n_sites': len(tab), 'n_mixed_class_sites': len(mixed),
                 'patients_in_mixed_class_sites': int(mixed.to_numpy().sum()), 'patients_with_site': len(d)})
pd.DataFrame(rows).to_csv('r4_site_table.csv', index=False)

# (b) kidney, sites held out
tid = 'renal subtype (KIRC vs KIRP)'
df, y, classes = L.task_frame(tid, labels, feat)
df['site'] = df.PatientID.map(site_of)
has = df.site.notna().to_numpy()
ds, ys = df[has].reset_index(drop=True), y[has]
sarr = ds.site.astype(str).to_numpy()
gkf = list(GroupKFold(min(L.N_SPLITS, len(np.unique(sarr)))).split(np.zeros(len(ys)), ys, sarr))
Pg = L.cv_oof_df(ds, ys, L.SCANNER_COLS, [gkf])
obs = L.mean_auc(ys, Pg)
assert abs(obs - a03s.site_heldout_auc[tid]) < 1e-6, (obs, a03s.site_heldout_auc[tid])
frows = []
for f, (tr, te) in enumerate(gkf):
    for s in np.unique(sarr[te]):
        m = te[sarr[te] == s]
        frows.append({'fold': f, 'site': s, **{f'n_{c}': int((ys[m] == j).sum()) for j, c in enumerate(classes)},
                      'train_share_KIRP': float(ys[tr].mean()),
                      'mean_p_KIRP_in_site': float(Pg[0][m, 1].mean())})
    both = len(np.unique(ys[te])) == 2
    frows.append({'fold': f, 'site': 'FOLD', 'fold_auc': L.auc_binary(ys[te], Pg[0][te, 1]) if both else np.nan})
pd.DataFrame(frows).to_csv('r4_kidney_heldout_folds.csv', index=False)
null_k = []
for _ in range(200):
    yp = rng.permutation(ys)
    if all(len(np.unique(yp[tr])) == 2 for tr, _ in gkf):
        null_k.append(L.mean_auc(yp, L.cv_oof_df(ds, yp, L.SCANNER_COLS, [gkf])))
null_k = np.array(null_k)

# (c) CPTAC-PDA stage, Merlin silhouette
tid2 = 'CPTAC-PDA | stage'
df2, y2, _ = L.task_frame(tid2, labels, feat)
X2 = L.load_emb(df2.key, 'merlin', 'silhouette', L.R / '06_image_posthoc' / 'cache' / 'embeddings')
sp = L.splits_for(y2, L.SEED)
P2, _ = L.cv_oof_matrix(X2, y2, sp)
obs2 = L.mean_auc(y2, P2)
ref2 = pd.read_csv(L.R / '06_image_posthoc' / 'real' / 'silhouette_probe.csv')
ref2 = ref2[(ref2.task_id == tid2) & (ref2.encoder == 'merlin')].silhouette_auc.iloc[0]
per_rep = [L.auc(y2, P2[r]) for r in range(L.N_REPEATS)]
null_s = np.array([L.mean_auc(yp, L.cv_oof_matrix(X2, yp, sp)[0]) for yp in (rng.permutation(y2) for _ in range(200))])

out = pd.DataFrame([
    {'analysis': 'kidney metadata, sites held out', 'observed_auc': obs, 'paper_auc': a03s.site_heldout_auc[tid],
     'null_mean': null_k.mean(), 'null_2.5pct': np.percentile(null_k, 2.5), 'null_97.5pct': np.percentile(null_k, 97.5),
     'share_null_at_or_below_observed': float((null_k <= obs).mean()), 'n_null': len(null_k)},
    {'analysis': 'CPTAC-PDA stage, Merlin silhouette, 5x5 CV', 'observed_auc': obs2, 'paper_auc': ref2,
     'per_repeat_auc': ' '.join(f'{a:.3f}' for a in per_rep),
     'null_mean': null_s.mean(), 'null_2.5pct': np.percentile(null_s, 2.5), 'null_97.5pct': np.percentile(null_s, 97.5),
     'share_null_at_or_below_observed': float((null_s <= obs2).mean()), 'n_null': len(null_s)}])
out.to_csv('r4_below_chance.csv', index=False)
pd.set_option('display.width', 250)
print(out.round(3).to_string(index=False))
print(pd.DataFrame(frows).round(3).to_string(index=False))
print(pd.DataFrame(rows).query("site == 'ALL'").to_string(index=False))
