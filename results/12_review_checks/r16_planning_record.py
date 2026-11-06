"""Check R16 (6 Nov 2026, plan 7k.7; no computation).
For each analysis: its plan section and local date (from protocol/analysis_plan.md), the UTC start of its real-label
run (read here from the run records, results/*/run_manifest.json and logs), and which real-label results had been seen
when it was planned (from the plan's own status lines). Plan dates are local calendar dates; run records are in UTC,
so a run that started late in the UTC day falls on the next local date. The repository's git history starts on
5 Nov 2026; plan section 7k (checks R10-R16) was committed before R10 was started.
Output: r16_planning_record.csv. Usage: python r16_planning_record.py (from this folder)"""
import json, re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent


def started(rel):
    p = R / rel
    if p.suffix == '.json':
        return json.loads(p.read_text())['run_started_utc'].replace('+00:00', 'Z')
    m = re.search(r'20\d\d-\d\d-\d\dT[\d:]+(?:\+00:00|Z)', p.read_text(encoding='utf-8', errors='replace'))
    return m.group(0).replace('+00:00', 'Z') if m else ''


ROWS = [
    # analysis, plan section, planned (local date), run record, results seen when planned, label in the paper
    ('Notebook 03: metadata screen of the 26 tasks', '7a, 7b', '28 Oct 2026', '03_audit/run_manifest.json',
     'none (code tested on shuffled labels only)', 'primary'),
    ('Notebook 04: checks of the metadata flags (sites, language, words)', '7c', '29 Oct 2026',
     '04_posthoc/run_manifest.json', 'notebook 03', 'post hoc'),
    ('Notebook 05: image probes of the four flagged tasks, acquisition swap', '7d', '29 Oct 2026',
     '05_image_check/real/run_manifest.json', 'notebooks 03-04 (no image result)', 'prespecified'),
    ('Literature review (RQ5): search and screening', '7e', '29 Oct 2026', None,
     'notebooks 03-04 (searches run 29 Oct 2026; no run record)', 'prespecified protocol; rubric clarified after the first ratings'),
    ('Notebook 06: within-group, held-out, silhouette and header checks of the image probes', '7f', '29 Oct 2026',
     '06_image_posthoc/real/run_manifest.json', 'notebooks 03-05', 'post hoc'),
    ('Notebook 07: screen validation over 19 CT tasks (A); acquisition decoding (C)', '7g', '29 Oct 2026',
     '07_screen_validation/real/run_manifest.json', 'notebooks 03-05 (notebook 06 running)',
     'planned after the first results'),
    ('Notebook 08: external transfer (B)', '7g; details settled 30 Oct 2026', '29 Oct 2026',
     '08_transfer/real/run_manifest.json', 'notebooks 03-05 when planned; 03-07 when its details were settled',
     'planned after the first results'),
    ('Notebook 09: why the TCGA lung models transfer to CPTAC lung', '7h', '30 Oct 2026',
     '09_transfer_posthoc/real/run_manifest.json', 'notebooks 03-08', 'post hoc'),
    ('Notebook 10: robustness of the screen validation', '7i', '31 Oct 2026 (proposed 30 Oct)',
     '10_figures_tables/run_manifest.json', 'the per-task AUCs it reuses (notebooks 03-07)', 'post hoc sensitivity'),
    ('Notebook 11: two added independent lung cohorts', '7j (amendment A.9, 4 Nov 2026)', '3 Nov 2026',
     '11_external_lung/real/run_manifest.json', 'notebooks 03-10', 'planned after the first results; reading rule fixed'),
    ('Notebook 11, step A.7: metadata screen of the two new tasks', '7j, item 7', '4 Nov 2026',
     '11_external_lung/a7_metadata_screen/a7_run.log', 'notebooks 03-11 (stated in the plan)', 'post hoc'),
    ('Checks R1-R8b', 'script headers (not in the plan)', '4 Nov 2026', None,
     'all of notebooks 03-11 (no run record)', 'post hoc'),
    ('Check R9: sex and age by class', 'script header', '5 Nov 2026', None, 'all of notebooks 03-11 and R1-R8b',
     'post hoc'),
    ('Checks R10-R16', '7k (committed before R10 started)', '6 Nov 2026',
     '12_review_checks/r10_run.log', 'all of the above', 'post hoc'),
]
out = pd.DataFrame([{'analysis': a, 'plan_section': s, 'planned_local_date': d,
                     'real_label_run_started_utc': started(f) if f else '', 'run_record': f or '',
                     'results_seen_when_planned': seen, 'label_in_paper': lab} for a, s, d, f, seen, lab in ROWS])
for c in ('analysis', 'results_seen_when_planned'):                  # ranges with an en dash
    out[c] = out[c].str.replace(r'(\d|R\d)-(\d|R)', lambda m: m.group(1) + '–' + m.group(2), regex=True)
out.to_csv(HERE / 'r16_planning_record.csv', index=False)
pd.set_option('display.width', 250, 'display.max_colwidth', 70)
print(out.drop(columns='run_record').to_string(index=False))
