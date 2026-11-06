"""Check R5 (4 Nov 2026; POST HOC, descriptive).
(a) All 52 candidate tasks of notebook 02 with label source, modality, class sizes and the reason for exclusion.
(b) Field completeness by class (share of patients with each label-card field filled) for the 26 audited tasks.
(c) A missingness-only baseline: the same logistic regression and 5 x 5 CV (seed 2026) on 0/1 indicators of which
    label-card fields are filled, with a bootstrap CI; separates archive or curation patterns from acquisition values.
(d) For the four flagged CT tasks: body part, contrast given (DICOM header), scan coverage and slice spacing by class.
Outputs: r5_candidate_tasks.csv, r5_field_completeness.csv, r5_missingness_baseline.csv, r5_flagged_scan_profile.csv.
Usage: python r5_tasks_missingness_coverage.py"""
import numpy as np
import pandas as pd
import review_lib as L

tasks = pd.read_csv(L.R / '02_tasks_features' / 'tasks_final.csv')
tasks['exclusion_reason'] = np.where(tasks.eligible, '', np.where(tasks.n_classes < 2, 'one class only',
                                                                   'smallest class < 20 patients with imaging'))
tasks.loc[~tasks.eligible & (tasks.min_class_n == tasks.n_patients), 'exclusion_reason'] = 'one class only'
tasks.to_csv('r5_candidate_tasks.csv', index=False)

labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
feat = L.patient_features()
a03 = pd.read_csv(L.R / '03_audit' / 'audit_primary.csv')
FIELDS = ['vendor', 'model', 'software_version', 'image_count', 'slice_thickness_mm', 'kernel', 'body_part', 'protocol_text']
comp, base = [], []
for tid in a03.task_id:
    df, y, classes = L.task_frame(tid, labels, feat)
    filled = pd.DataFrame({f: (df[f].notna() & (df[f].astype(str).str.strip() != '')).astype(int) for f in FIELDS})
    for j, c in enumerate(classes):
        comp.append({'task_id': tid, 'class': c, 'n': int((y == j).sum()),
                     **{f'filled_{f}': float(filled[f][y == j].mean()) for f in FIELDS}})
    if filled.nunique().max() < 2:
        base.append({'task_id': tid, 'missingness_auc': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                     'note': 'every field equally filled in all patients'})
        continue
    X = filled.to_numpy(float)
    P, _ = L.cv_oof_matrix(X, y, L.splits_for(y, L.SEED))
    lo, hi = L.boot_ci(y, P)
    base.append({'task_id': tid, 'missingness_auc': L.mean_auc(y, P), 'ci_low': lo, 'ci_high': hi,
                 'metadata_auc': float(a03.set_index('task_id').auc[tid]), 'note': ''})
pd.DataFrame(comp).to_csv('r5_field_completeness.csv', index=False)
base = pd.DataFrame(base)
base.to_csv('r5_missingness_baseline.csv', index=False)

inv = pd.read_csv(L.R / '05_image_check' / 'cache' / 'image_inventory.csv', low_memory=False)
inv['key'] = inv.collection + '__' + inv.PatientID
prof = []
for tid in ['renal subtype (KIRC vs KIRP)', 'CPTAC-PDA | stage', 'lung histology (LUAD vs LUSC)', 'LIDC-IDRI | primary vs metastatic']:
    df, y, classes = L.task_frame(tid, labels, feat)
    d = df.merge(inv[['key', 'hdr_contrast_agent_given', 'hdr_coverage_mm', 'hdr_z_spacing']], on='key', how='left')
    for j, c in enumerate(classes):
        m = y == j
        bp = d.body_part[m].fillna('missing').value_counts(normalize=True).head(3)
        prof.append({'task_id': tid, 'class': c, 'n': int(m.sum()),
                     'body_part_top3': '; '.join(f'{k} {v:.0%}' for k, v in bp.items()),
                     'contrast_given_share': float(d.hdr_contrast_agent_given[m].eq('yes').mean()),
                     'contrast_missing_share': float(d.hdr_contrast_agent_given[m].isna().mean()),
                     'coverage_mm_median': float(d.hdr_coverage_mm[m].median()),
                     'z_spacing_mm_median': float(d.hdr_z_spacing[m].median())})
pd.DataFrame(prof).to_csv('r5_flagged_scan_profile.csv', index=False)
pd.set_option('display.width', 250)
print(tasks.exclusion_reason.value_counts().to_string())
print(base.round(3).to_string(index=False))
print(pd.DataFrame(prof).round(3).to_string(index=False))
