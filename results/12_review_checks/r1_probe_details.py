"""Check R1 (4 Nov 2026; descriptive, from saved outputs, no refit).
Collects for the four flagged tasks: image and background probe AUC with CI, permutation P and BH q (4 tasks per
encoder, notebook 05), the full-image probe applied unchanged to the background-only inputs, and the acquisition swap
(which probe, n, rule). Output: r1_probe_details.csv. Usage: python r1_probe_details.py (from this folder)."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
p = pd.read_csv(R / '05_image_check' / 'real' / 'image_probe.csv')
s = pd.read_csv(R / '05_image_check' / 'real' / 'image_swap.csv')
cols = ['task_id', 'encoder', 'n_patients', 'image_auc', 'image_ci_low', 'image_ci_high', 'image_perm_p', 'image_perm_q_bh',
        'image_flagged', 'background_auc', 'background_ci_low', 'background_ci_high', 'background_perm_p',
        'background_perm_q_bh', 'background_flagged', 'image_probe_on_background_auc']
swap = s[['task_id', 'encoder', 'applicable', 'thin_class', 'thick_class', 'target_z_spacing', 'n_swapped',
          'shift_toward_thick', 'shift_ci_low', 'shift_ci_high', 'control_shift', 'swap_minus_control',
          'difference_ci_low', 'difference_ci_high', 'rule_met']].rename(columns=lambda c: c if c in ('task_id', 'encoder') else 'swap_' + c)
out = p[cols].merge(swap, on=['task_id', 'encoder'], how='left')
out['swap_scored_with'] = 'full-image probe, held-out folds (nb05 cv_oof_matrix extras)'
out.to_csv(HERE / 'r1_probe_details.csv', index=False)
pd.set_option('display.width', 250)
print(out.round(4).to_string(index=False))
