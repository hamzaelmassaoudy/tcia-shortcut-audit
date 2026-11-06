"""Check R6 (4 Nov 2026): one supplementary figure of the mask QC pictures that
notebooks 05/06 saved for two scans of each flagged task (middle axial slice: original with the body outline,
background only, silhouette), plus the distribution of the masked share of each volume over the 525 flagged-task scans.
No scan is re-downloaded. Outputs: r6_mask_qc.png, r6_mask_fraction_by_task.csv. Usage: python r6_mask_qc_figure.py"""
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
R = HERE.parent
qc = sorted((R / '06_image_posthoc' / 'cache' / 'qc').glob('*.png'))
ims = [Image.open(p).convert('RGB') for p in qc]
w = max(i.width for i in ims)
canvas = Image.new('RGB', (w, sum(i.height for i in ims)), 'white')
y = 0
for i in ims:
    canvas.paste(i, (0, y))
    y += i.height
canvas.save(HERE / 'r6_mask_qc.png')

inv = pd.read_csv(R / '05_image_check' / 'cache' / 'image_inventory.csv', low_memory=False)
lab = pd.read_csv(R / '02_tasks_features' / 'labels_long.csv')
task_of = {'TCGA-KIRC': 'renal subtype (KIRC vs KIRP)', 'TCGA-KIRP': 'renal subtype (KIRC vs KIRP)',
           'TCGA-LUAD': 'lung histology (LUAD vs LUSC)', 'TCGA-LUSC': 'lung histology (LUAD vs LUSC)',
           'CPTAC-PDA': 'CPTAC-PDA | stage', 'LIDC-IDRI': 'LIDC-IDRI | primary vs metastatic'}
inv['task_id'] = inv.collection.map(task_of)
s = inv.groupby('task_id').body_fraction_dilated.describe(percentiles=[0.05, 0.5, 0.95])
s.to_csv(HERE / 'r6_mask_fraction_by_task.csv')
print([p.name for p in qc])
print(s.round(3).to_string())
