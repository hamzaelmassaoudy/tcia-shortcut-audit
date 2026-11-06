"""Check that the tool reproduces the paper's metadata audit (notebook 03) for the two flagged collection-defined tasks.

Usage: python tests/reproduce_notebook03.py <project folder>
The project folder must hold results/01_inventory/series_metadata.csv.gz and results/03_audit/ from the paper's runs.
Passes if every AUC, CI bound, permutation p and field-group number equals notebook 03's within 1e-9.
"""
import json, sys, tempfile
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tcia_shortcut_audit as tool

PROJECT = Path(sys.argv[1])
TASKS = {'lung histology (LUAD vs LUSC)': ['TCGA-LUAD', 'TCGA-LUSC'],
         'renal subtype (KIRC vs KIRP)': ['TCGA-KIRC', 'TCGA-KIRP']}
primary = pd.read_csv(PROJECT / 'results/03_audit/audit_primary.csv').set_index('task_id')
groups = pd.read_csv(PROJECT / 'results/03_audit/audit_feature_groups.csv')
ok = True
with tempfile.TemporaryDirectory() as tmp:
    for tid, cols in TASKS.items():
        out = Path(tmp) / cols[0]
        tool.main(['--collections', *cols, '--label-by-collection', '--modality', 'CT', '--out', str(out),
                   '--series-csv', str(PROJECT / 'results/01_inventory/series_metadata.csv.gz')])
        s = json.loads((out / 'summary.json').read_text())
        r = primary.loc[tid]
        diffs = [abs(s[k] - r[k]) for k in ('auc', 'ci_low', 'ci_high', 'null_mean_auc')] + [abs(s['perm_p'] - r.perm_p)]
        g = pd.read_csv(out / 'field_groups.csv').set_index('group')
        f = groups[groups.task_id == tid].set_index('group')
        cols_g = ['alone_auc', 'alone_ci_low', 'alone_ci_high', 'drop_when_left_out', 'filled_fraction']
        diffs.append(float((g[cols_g] - f.loc[g.index, cols_g]).abs().max().max()))
        passed = max(diffs) < 1e-9
        ok &= passed
        print(f'{tid}: AUC {s["auc"]:.6f} vs {r.auc:.6f}; largest difference {max(diffs):.1e} -> {"PASS" if passed else "FAIL"}')
print('ALL PASS' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
