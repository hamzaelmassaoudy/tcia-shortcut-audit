"""RQ5 step 2 (plan 7e.3): merge duplicate records, by DOI first, then by normalized title.
Writes records_unique.csv (one row per study, with every source and query that found it) and adds a line to the log."""
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
raw = pd.read_csv(HERE / 'records_raw.csv', dtype=str).fillna('')

def norm_title(t):
    return re.sub(r'[^a-z0-9]+', ' ', t.lower()).strip()

raw['doi'] = raw.doi.str.lower().str.strip().str.replace(r'^https?://(dx\.)?doi\.org/', '', regex=True)
raw['ntitle'] = raw.title.map(norm_title)
raw = raw[raw.ntitle != ''].reset_index(drop=True)

# union-find over records that share a DOI or a normalized title
parent = list(range(len(raw)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
for col in ('doi', 'ntitle'):
    first = {}
    for i, v in enumerate(raw[col]):
        if not v:
            continue
        if v in first:
            parent[find(i)] = find(first[v])
        else:
            first[v] = i
raw['study'] = [find(i) for i in range(len(raw))]

def pick(g, col):
    vals = [v for v in g[col] if v]
    return max(vals, key=len) if vals else ''

rows = []
for _, g in raw.groupby('study', sort=False):
    rows.append({'title': pick(g, 'title'), 'year': pick(g, 'year'), 'venue': pick(g, 'venue'), 'doi': pick(g, 'doi'),
                 'abstract': pick(g, 'abstract'), 'url': g.url.iloc[0],
                 'tasks': ';'.join(sorted(set(g.task))), 'sources': ';'.join(sorted(set(g.source))),
                 'queries': ';'.join(sorted(set(g.query_id))), 'n_records': len(g)})
uniq = pd.DataFrame(rows).sort_values(['tasks', 'year', 'title']).reset_index(drop=True)
uniq.insert(0, 'record_no', range(1, len(uniq) + 1))
uniq.to_csv(HERE / 'records_unique.csv', index=False)
print(f'{len(raw)} records -> {len(uniq)} unique studies')
print(uniq.groupby('tasks').size().to_string())
print('with an abstract:', int((uniq.abstract != '').sum()))
