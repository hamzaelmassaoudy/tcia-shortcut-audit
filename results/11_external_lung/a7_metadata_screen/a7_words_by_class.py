"""Descriptive (no test): share of patients per class whose label-card description words include each word, for the
primary A.7 runs. Words used by fewer than 3 patients are left out, as in the tool. Written on 4 Nov 2026 after the A.7
results were seen, to show which words carry the Lung-PET-CT-Dx signal.
Usage: python a7_words_by_class.py (from this folder)."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
for coll in ('Lung-PET-CT-Dx', 'NSCLC-Radiomics'):
    card = pd.read_csv(HERE / f'primary_{coll}' / 'label_card.csv')
    words = card.protocol_text.fillna('').str.split()
    rows = []
    for w in sorted({x for ws in words for x in ws}):
        has = words.map(lambda ws: w in ws)
        if has.sum() >= 3:
            rows.append({'word': w, 'n_patients': int(has.sum()),
                         'share_adenocarcinoma': has[card.label == 'adenocarcinoma'].mean(),
                         'share_squamous': has[card.label == 'squamous cell carcinoma'].mean()})
    t = pd.DataFrame(rows)
    t['squamous_minus_adeno'] = t.share_squamous - t.share_adenocarcinoma
    t = t.reindex(t.squamous_minus_adeno.abs().sort_values(ascending=False).index)
    t.to_csv(HERE / f'words_by_class_{coll}.csv', index=False)
    print(coll); print(t.head(10).round(3).to_string(index=False))
