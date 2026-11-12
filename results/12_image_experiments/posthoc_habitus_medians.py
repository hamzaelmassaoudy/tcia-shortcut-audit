"""Post hoc, descriptive (12 Nov 2026; not in plan 7o): median, quartiles and mean of the seven habitus and positioning measures
of notebook 12 by class in the five lung cohorts, from the saved scan records and the real labels. Written after the
results of notebook 12 had been seen; it adds units to the class comparison of g2_habitus_by_class.csv and fits nothing.
Usage (from this folder): python posthoc_habitus_medians.py"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
MEASURES = ['ap_lr_ratio', 'subcut_fat_cm2', 'arms_beside_trunk', 'table_height_mm', 'centroid_offset_mm',
            'scan_length_mm', 'lung_volume_l']
rec = pd.read_csv(HERE / 'cache' / 'scan_records.csv')
lab = pd.read_csv(HERE / 'real' / 'dataset_labels.csv')
d = lab.merge(rec[['dataset', 'key'] + MEASURES], on=['dataset', 'key'], how='left', validate='one_to_one')
d = d[~d.dataset.str.startswith('LIDC-IDRI')]
rows = []
for (ds, y, cls), g in d.groupby(['dataset', 'y', 'class'], sort=False):
    for m in MEASURES:
        v = g[m].dropna()
        rows.append({'dataset': ds, 'y': int(y), 'class': cls, 'measure': m, 'n': len(v), 'median': v.median(),
                     'q1': v.quantile(0.25), 'q3': v.quantile(0.75), 'mean': v.mean()})
out = pd.DataFrame(rows)
ref = pd.read_csv(HERE / 'real' / 'g2_habitus_by_class.csv').set_index(['dataset', 'measure']).n
tot = out.groupby(['dataset', 'measure']).n.sum()
assert (tot == ref.reindex(tot.index)).all()          # the same scans as the class comparison of notebook 12
out.to_csv(HERE / 'posthoc_habitus_medians.csv', index=False)
print(out[out.measure.isin(['lung_volume_l', 'scan_length_mm'])].round(3).to_string())
