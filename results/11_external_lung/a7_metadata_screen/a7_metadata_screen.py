"""Plan 7j A.7: the open tool's metadata screen on the two new lung histology tasks (NSCLC-Radiomics, Lung-PET-CT-Dx).

Run details fixed in 00_RESEARCH_PLAN.md (7j A.7) on 4 Nov 2026 before this was run:
  - tool/tcia_shortcut_audit.py 1.0.0, unchanged, defaults, --modality CT;
  - metadata: notebook 11's saved getSeries replies (MD5s checked against its manifest), passed as --series-csv;
  - labels as in notebook 11; classes named "adenocarcinoma" and "squamous cell carcinoma" in both tasks;
  - flag: lower CI >= 0.60 and BH q < 0.05, BH over the 26 original tasks (notebook 03) plus these two;
  - primary run: the tool's own main-series rule; secondary run (Lung-PET-CT-Dx only): the series set aside under
    amendment A.9 removed, so the label cards come from the series notebook 11 embedded.
Usage: python a7_metadata_screen.py (from this folder; needs the project folder two levels up)."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                              # results/11_external_lung
PROJECT = RUN.parents[1]
sys.path.insert(0, str(PROJECT / 'tool'))
import tcia_shortcut_audit as tool

assert tool.__version__ == '1.0.0'
CLASS_NAMES = ('adenocarcinoma', 'squamous cell carcinoma')
EXPECTED = {'NSCLC-Radiomics': {'adenocarcinoma': 51, 'squamous cell carcinoma': 152},
            'Lung-PET-CT-Dx': {'adenocarcinoma': 251, 'squamous cell carcinoma': 61}}
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
manifest = json.loads((RUN / 'real' / 'run_manifest.json').read_text())

# 1. Metadata: notebook 11's saved replies, checked by MD5
frames = []
for coll in ('NSCLC-Radiomics', 'Lung-PET-CT-Dx'):
    path = RUN / 'cache' / 'api' / f'getSeries__Collection={coll}.json'
    assert md5(path) == manifest['series_reply_md5'][coll], f'{coll}: saved reply differs from notebook 11'
    frames.append(pd.DataFrame(json.loads(path.read_text())['data']).assign(Collection=coll))
series = pd.concat(frames, ignore_index=True)
series.to_csv(HERE / 'series_metadata.csv.gz', index=False)

# 2. Labels, as notebook 11 built them
lung1_path = RUN / 'labels' / 'NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv'
assert md5(lung1_path) == manifest['lung1_label_file']['md5']
lung1 = pd.read_csv(lung1_path)
lung1 = lung1[lung1.Histology.isin(CLASS_NAMES)][['PatientID', 'Histology']].rename(columns={'Histology': 'label'})
ids = pd.Series(series[series.Collection == 'Lung-PET-CT-Dx'].PatientID.unique())
assert ids.str.fullmatch(manifest['dx_id_rule']).all()
letter = ids.str.extract(manifest['dx_id_rule'], expand=False)
dx = pd.DataFrame({'PatientID': ids, 'label': letter.map({'A': CLASS_NAMES[0], 'G': CLASS_NAMES[1]})}).dropna()
labels = {'NSCLC-Radiomics': lung1, 'Lung-PET-CT-Dx': dx}
for coll, lab in labels.items():
    assert lab.label.value_counts().to_dict() == EXPECTED[coll], (coll, lab.label.value_counts().to_dict())
    lab.sort_values('PatientID').to_csv(HERE / f'labels_{coll}.csv', index=False)

# 3. The tool, unchanged (primary), and with the A.9 set-asides removed (secondary, Lung-PET-CT-Dx)
aside = pd.read_csv(RUN / 'cache' / 'series_set_aside.csv')
aside_uids = {u for s in aside.set_aside for u in s.split(';')}
series[~series.SeriesInstanceUID.isin(aside_uids)].to_csv(HERE / 'series_metadata_without_A9_set_aside.csv.gz',
                                                         index=False)
runs = {'NSCLC-Radiomics': ('NSCLC-Radiomics', 'series_metadata.csv.gz'),
        'Lung-PET-CT-Dx': ('Lung-PET-CT-Dx', 'series_metadata.csv.gz'),
        'Lung-PET-CT-Dx (A.9 set-asides removed)': ('Lung-PET-CT-Dx', 'series_metadata_without_A9_set_aside.csv.gz')}
out_dir = {name: HERE / ('primary_' + coll if 'A.9' not in name else 'secondary_' + coll) for name, (coll, _) in runs.items()}
for name, (coll, csv) in runs.items():
    print(f'\n######## {name}')
    tool.main(['--labels', str(HERE / f'labels_{coll}.csv'), '--collections', coll, '--modality', 'CT',
               '--series-csv', str(HERE / csv), '--out', str(out_dir[name])])

# 4. Same series as notebook 11? (primary: its rule's picks; secondary: the series it embedded)
inv = pd.read_csv(RUN / 'cache' / 'image_inventory.csv')
embedded = dict(zip(inv.collection + '__' + inv.PatientID, inv.SeriesInstanceUID))
rule_pick = dict(embedded)
rule_pick.update(dict(zip(aside.key, aside.rule_series)))
checks = {}
for name, d in out_dir.items():
    ms = pd.read_csv(d / 'main_series.csv')
    key = ms.collection + '__' + ms.PatientID
    ref = rule_pick if 'A.9' not in name else embedded
    checks[name] = {'patients': int(len(ms)), 'same_series_as_notebook11': int((key.map(ref) == ms.SeriesInstanceUID).sum())}
    assert checks[name]['same_series_as_notebook11'] == len(ms) == sum(EXPECTED[runs[name][0]].values()), (name, checks[name])

# 5. Flag with BH over the 26 original tasks plus the two new ones (7b.1); the secondary run replaces the primary
def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out

orig = pd.read_csv(PROJECT / 'results' / '03_audit' / 'audit_primary.csv')
assert np.allclose(bh(orig.perm_p), orig.perm_q_bh, atol=1e-12), 'BH here differs from notebook 03'
rows = []
for name, d in out_dir.items():
    s = json.loads((d / 'summary.json').read_text())
    rows.append({'task_id': f'{name} | histology', 'run': 'secondary' if 'A.9' in name else 'primary',
                 'n_patients': s['n_patients'], 'classes': json.dumps(s['classes']), 'auc': s['auc'],
                 'ci_low': s['ci_low'], 'ci_high': s['ci_high'], 'perm_p': s['perm_p'],
                 'null_mean_auc': s['null_mean_auc'], 'tool_flag_single_task': s['flagged']})
new = pd.DataFrame(rows)
results = []
for variant in ('primary', 'secondary'):
    picks = new[new.run == 'primary'] if variant == 'primary' else \
        pd.concat([new[(new.run == 'primary') & new.task_id.str.startswith('NSCLC')], new[new.run == 'secondary']])
    fam = pd.concat([orig[['task_id', 'auc', 'ci_low', 'ci_high', 'perm_p', 'flagged']].assign(origin='notebook 03'),
                     picks[['task_id', 'auc', 'ci_low', 'ci_high', 'perm_p']].assign(origin='plan 7j A.7')],
                    ignore_index=True)
    fam['q_bh_28'] = bh(fam.perm_p)
    fam['flag_28'] = (fam.ci_low >= tool.FLAG_LOWER_CI) & (fam.q_bh_28 < 0.05)
    fam['family'] = f'26 + 2 ({variant} runs of the new tasks)'
    results.append(fam)
fam = pd.concat(results, ignore_index=True)
fam.to_csv(HERE / 'a7_family_bh.csv', index=False)
new.to_csv(HERE / 'a7_new_tasks.csv', index=False)
changed = fam[(fam.origin == 'notebook 03') & (fam.flagged != fam.flag_28)]
summary = {'plan': '7j A.7', 'tool_version': tool.__version__, 'series_reply_md5': manifest['series_reply_md5'],
           'series_check': checks,
           'new_tasks': fam[fam.origin == 'plan 7j A.7'][['family', 'task_id', 'auc', 'ci_low', 'ci_high', 'perm_p',
                                                          'q_bh_28', 'flag_28']].to_dict('records'),
           'original_flags_changed_by_28_task_bh': changed[['family', 'task_id']].to_dict('records')}
(HERE / 'a7_summary.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')
pd.set_option('display.width', 250)
print('\n', json.dumps(summary['series_check'], indent=1))
print(fam[fam.origin == 'plan 7j A.7'].round(4).to_string(index=False))
print('original flags changed by the 28-task BH:', summary['original_flags_changed_by_28_task_bh'])
