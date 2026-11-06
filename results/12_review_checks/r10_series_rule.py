"""Check R10 (6 Nov 2026, plan 7k.1; POST HOC).
Does the main-series rule create the four flags? The metadata screen is rerun on the four flagged tasks with two other
main-series rules; everything else is the tool's (= notebook 03's) code and settings:
  original      the eligible series with the most images in the first study with one (rule 7a.7);
  fewest        the eligible series with the fewest images (at least 10) in that study; ties by series number, UID;
  nearest_5mm   the eligible series whose thickness parsed from its description is closest to 5 mm; ties by more
                images, then series number; if no eligible series of the study has a parsed thickness, the original.
The original rule must first reproduce notebook 03 (AUC, CI, permutation p within 1e-9).
Outputs: r10_series_rule.csv (one row per task and rule), r10_series_by_class.csv (image count and parsed thickness
by class). Usage: python r10_series_rule.py (from this folder)"""
import json, re, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'tool'))
import tcia_shortcut_audit as tool  # noqa: E402

TASKS = ['renal subtype (KIRC vs KIRP)', 'CPTAC-PDA | stage', 'lung histology (LUAD vs LUSC)',
         'LIDC-IDRI | primary vs metastatic']
ORIGINAL_MAIN_SERIES = tool.main_series


def make_main_series(rule):
    """tool.main_series (notebook 01, rule 7a.7) with only the choice within the study replaced."""
    def pick(e):
        if rule == 'original':
            return e.sort_values(['ImageCount', 'SeriesNumber', 'SeriesInstanceUID'], ascending=[False, True, True]).iloc[0]
        if rule == 'fewest':
            return e.sort_values(['ImageCount', 'SeriesNumber', 'SeriesInstanceUID'], ascending=[True, True, True]).iloc[0]
        if rule == 'nearest_5mm':
            text = e[['SeriesDescription', 'ProtocolName']].fillna('').astype(str).agg(' | '.join, axis=1)
            t = text.map(tool.parse_thickness)
            if t.notna().any():
                e2 = e.assign(_d=(t - 5.0).abs())[t.notna()]
                return e2.sort_values(['_d', 'ImageCount', 'SeriesNumber', 'SeriesInstanceUID'],
                                      ascending=[True, False, True, True]).iloc[0]
            return pick_original(e)
        raise ValueError(rule)

    def pick_original(e):
        return e.sort_values(['ImageCount', 'SeriesNumber', 'SeriesInstanceUID'], ascending=[False, True, True]).iloc[0]

    def main_series(series):
        # verbatim from tool.main_series except the marked line
        series = series.replace(r'^\s*$', np.nan, regex=True).copy()
        for c in ('ProtocolName', 'SeriesDescription', 'StudyDesc', 'BodyPartExamined', 'Manufacturer',
                  'ManufacturerModelName', 'SoftwareVersions', 'StudyDate', 'SeriesNumber', 'FileSize'):
            if c not in series:
                series[c] = np.nan
        series['ImageCount'] = pd.to_numeric(series['ImageCount'], errors='coerce')
        series['StudyDate_dt'] = pd.to_datetime(series['StudyDate'], errors='coerce')
        img = series[series.Modality.isin(tool.IMAGING_MODALITIES)].copy()
        desc = img.SeriesDescription.where(img.SeriesDescription.notna(), img.ProtocolName).fillna('')
        excluded = (desc.str.contains(tool.EXCLUDE_SERIES_PATTERN, flags=re.I, regex=True)
                    | ((img.Modality == 'CT') & desc.str.contains(tool.CT_REFORMAT_PATTERN, flags=re.I, regex=True))
                    | (img.ImageCount.fillna(0) < tool.MIN_IMAGES))
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
                    b = pick(e)                                   # <- the only change
                    rows.append({'collection': col, 'PatientID': pid, 'modality': mod, 'study_rank_used': rank,
                                 'SeriesInstanceUID': b.SeriesInstanceUID, 'SeriesDescription': b.SeriesDescription,
                                 'ProtocolName': b.ProtocolName, 'StudyDesc': b.StudyDesc,
                                 'BodyPartExamined': b.BodyPartExamined, 'Manufacturer': b.Manufacturer,
                                 'ManufacturerModelName': b.ManufacturerModelName,
                                 'SoftwareVersions': b.SoftwareVersions, 'ImageCount': b.ImageCount,
                                 'FileSize': b.FileSize})
                    break
        return pd.DataFrame(rows)
    return main_series


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    prev = 1.0
    for rank, j in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[j] * len(p) / rank)
        q[j] = prev
    return q


t0 = time.time()
labels = pd.read_csv(PROJECT / 'results/02_tasks_features/labels_long.csv')
primary = pd.read_csv(PROJECT / 'results/03_audit/audit_primary.csv').set_index('task_id')
series_csv = PROJECT / 'results/01_inventory/series_metadata.csv.gz'
rows, by_class = [], []
WORK = HERE / '_r10_work'                      # the tool's run folders; removed at the end
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir()
if True:
    tmp = WORK
    for rule in ('original', 'fewest', 'nearest_5mm'):
        tool.main_series = make_main_series(rule)
        for tid in TASKS:
            lab = labels[labels.task_id == tid][['collection', 'PatientID', 'label']]
            lp = tmp / 'labels.csv'
            lab.to_csv(lp, index=False)
            out = tmp / f'{rule}_{len(rows)}'
            tool.main(['--labels', str(lp), '--collections', *sorted(lab.collection.unique()), '--modality', 'CT',
                       '--out', str(out), '--series-csv', str(series_csv)])
            s = json.loads((out / 'summary.json').read_text())
            ms = pd.read_csv(out / 'main_series.csv')
            card = pd.read_csv(out / 'label_card.csv')
            rows.append({'task_id': tid, 'rule': rule, 'n_patients': s['n_patients'], 'auc': s['auc'],
                         'ci_low': s['ci_low'], 'ci_high': s['ci_high'], 'perm_p': s['perm_p'],
                         'null_mean_auc': s['null_mean_auc'], 'series_map': dict(zip(ms.collection + '__' + ms.PatientID,
                                                                                     ms.SeriesInstanceUID))})
            for cls, g in card.groupby('label'):
                by_class.append({'task_id': tid, 'rule': rule, 'class': cls, 'n': len(g),
                                 'median_image_count': g.image_count.median(),
                                 'parsed_thickness_known': int(g.slice_thickness_mm.notna().sum()),
                                 'median_parsed_thickness_mm': g.slice_thickness_mm.median()})
            print(f'{rule:<12} {tid:<36} AUC {s["auc"]:.4f} ({s["ci_low"]:.4f}-{s["ci_high"]:.4f}) p {s["perm_p"]:.4f}'
                  f'  [{time.time() - t0:.0f} s]', flush=True)
tool.main_series = ORIGINAL_MAIN_SERIES
res = pd.DataFrame(rows)
orig = res[res.rule == 'original'].set_index('task_id')
res['n_changed'] = [sum(v != orig.loc[t, 'series_map'].get(k) for k, v in d.items())
                    for t, d in zip(res['task_id'], res['series_map'])]
res['share_changed'] = res.n_changed / res.n_patients
res = res.drop(columns='series_map')
res['q_bh_4tasks'] = np.nan
for rule, g in res.groupby('rule'):
    res.loc[g.index, 'q_bh_4tasks'] = bh(g.perm_p.to_numpy())
res['flag_holds'] = (res.ci_low >= 0.60) & (res.q_bh_4tasks < 0.05)
# reproduction of notebook 03 by the original rule
o = res[res.rule == 'original'].set_index('task_id')
diff = pd.concat([(o[c] - primary.loc[o.index, c]).abs() for c in ('auc', 'ci_low', 'ci_high', 'perm_p')], axis=1).max(axis=1)
res['nb03_largest_difference'] = res.task_id.map(diff).where(res.rule == 'original')
res['nb03_reproduced'] = res.nb03_largest_difference.lt(1e-9).where(res.rule == 'original')
res.to_csv(HERE / 'r10_series_rule.csv', index=False)
pd.DataFrame(by_class).to_csv(HERE / 'r10_series_by_class.csv', index=False)
shutil.rmtree(WORK, ignore_errors=True)
pd.set_option('display.width', 250)
print(res.round(4).to_string(index=False))
print(pd.DataFrame(by_class).round(3).to_string(index=False))
print(f'finished in {time.time() - t0:.0f} s')
