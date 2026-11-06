"""Check R15 (6 Nov 2026, plan 7k.6; POST HOC).
The shortcut rule (plan 7d.7: image probe flagged, and the background-only probe flagged or the acquisition swap met
its rule) counts a task if either encoder meets it, with BH over the four tasks per encoder and input. Here it is
reapplied with multiplicity control across encoders and inputs:
  (a) notebook 05's stored permutation P values, BH over all 16 image tests (4 tasks x 2 encoders x 2 inputs);
  (b) the CI criterion (lower bound >= 0.60) on a 98.75% CI (95% Bonferroni-adjusted for a task's four tests);
  (c) the swap rule (mean shift >= 0.05, shift CI and swap-minus-control CI above 0) with 97.5% CIs (two encoders).
The CIs come from a refit with notebook 05's code, splits and bootstrap resamples (the 95% bounds reproduce notebook
05). Outputs: r15_multiplicity_probes.csv, r15_multiplicity_swap.csv, r15_multiplicity_summary.csv.
Usage: python r15_multiplicity.py (from this folder)"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import review_lib2 as L

HERE = Path(__file__).resolve().parent
SWAP_MIN_DIFF_MM, SWAP_MIN_GAP_MM, SWAP_SHIFT_MIN = 0.5, 0.1, 0.05     # notebook 05, cell 3
SWAP_DIR = L.R / '05_image_check' / 'real' / 'swap_embeddings'
t0 = time.time()
labels = pd.read_csv(L.R / '02_tasks_features' / 'labels_long.csv')
feat = L.patient_features()
p05 = pd.read_csv(L.R / '05_image_check' / 'real' / 'image_probe.csv')
s05 = pd.read_csv(L.R / '05_image_check' / 'real' / 'image_swap.csv')
TASKS = list(p05.task_id.unique())


def swap_plan(y, z):
    """notebook 05 swap_plan (without its local-test branch)."""
    med = [float(np.nanmedian(z[y == j])) for j in range(2)]
    if abs(med[0] - med[1]) < SWAP_MIN_DIFF_MM:
        return {'applicable': False, 'medians': med}
    thin = int(np.argmin(med)); target = med[1 - thin]
    return {'applicable': True, 'medians': med, 'thin_class': thin, 'thick_class': 1 - thin, 'target_z': target,
            'rows': np.where((y == thin) & (z < target - SWAP_MIN_GAP_MM))[0]}


probes, swaps = [], []
for tid in TASKS:
    df, y, classes = L.task_frame(tid, labels, feat)
    z = pd.to_numeric(pd.Series([L.header(k).get('hdr_z_spacing') for k in df.key]), errors='coerce').to_numpy()
    plan = swap_plan(y, z)
    splits = L.splits_for(y, L.SEED)
    for enc in ('ctfm', 'merlin'):
        ref = p05[(p05.task_id == tid) & (p05.encoder == enc)].iloc[0]
        X = L.load_input(df.key, enc, 'original')
        Xb = L.load_input(df.key, enc, 'background')
        extras = {}
        if plan['applicable']:
            for v in ('swap', 'control'):
                M = np.full(X.shape, np.nan)
                for i in plan['rows']:
                    p = SWAP_DIR / f"{df.key[i]}__{plan['target_z']:.3f}.npz"
                    if p.exists():
                        with np.load(p, allow_pickle=True) as zz:
                            M[i] = zz[f'{enc}__{v}']
                extras[v] = M
        P, E = L.cv_oof_matrix(X, y, splits, extras)
        Pb, _ = L.cv_oof_matrix(Xb, y, splits)
        for name, PP in (('image', P), ('background', Pb)):
            lv = L.boot_ci_levels(y, PP, (0.95, 0.9875))
            probes.append({'task_id': tid, 'encoder': enc, 'input': name, 'auc': L.mean_auc(y, PP),
                           'ci95_low': lv[0.95][0], 'ci95_high': lv[0.95][1], 'ci9875_low': lv[0.9875][0],
                           'ci9875_high': lv[0.9875][1], 'perm_p_nb05': ref[f'{name}_perm_p'],
                           'q_bh_4tasks_nb05': ref[f'{name}_perm_q_bh'], 'flagged_nb05': bool(ref[f'{name}_flagged']),
                           'nb05_auc': ref[f'{name}_auc'], 'nb05_ci_low': ref[f'{name}_ci_low']})
        sr = {'task_id': tid, 'encoder': enc, 'applicable': plan['applicable']}
        if plan['applicable']:
            rows_ = plan['rows'][~np.isnan(E['swap'][0, plan['rows'], 0])]
            tc = plan['thick_class']
            sh = (E['swap'][:, rows_, tc] - P[:, rows_, tc]).mean(0)
            ch = (E['control'][:, rows_, tc] - P[:, rows_, tc]).mean(0)
            m_s, cs = L.boot_mean_levels(sh, (0.95, 0.975))
            m_c, cc = L.boot_mean_levels(ch, (0.95, 0.975))
            m_d, cd = L.boot_mean_levels(sh - ch, (0.95, 0.975))
            ref_s = s05[(s05.task_id == tid) & (s05.encoder == enc)].iloc[0]
            sr.update({'n_swapped': len(rows_), 'shift': m_s, 'shift_ci95_low': cs[0.95][0],
                       'shift_ci975_low': cs[0.975][0], 'swap_minus_control': m_d,
                       'difference_ci95_low': cd[0.95][0], 'difference_ci975_low': cd[0.975][0],
                       'rule_met_95': bool(m_s >= SWAP_SHIFT_MIN and cs[0.95][0] > 0 and cd[0.95][0] > 0),
                       'rule_met_975': bool(m_s >= SWAP_SHIFT_MIN and cs[0.975][0] > 0 and cd[0.975][0] > 0),
                       'nb05_shift': ref_s.shift_toward_thick, 'nb05_shift_ci_low': ref_s.shift_ci_low,
                       'nb05_difference_ci_low': ref_s.difference_ci_low, 'nb05_rule_met': bool(ref_s.rule_met)})
        else:
            sr.update({'rule_met_95': False, 'rule_met_975': False, 'nb05_rule_met': False})
        swaps.append(sr)
        print(f'{tid:<36} {enc:<6} done [{time.time() - t0:.0f} s]', flush=True)
pr = pd.DataFrame(probes)
pr['q_bh_16'] = L.bh(pr.perm_p_nb05.to_numpy())
pr['flagged_adjusted'] = (pr.ci9875_low >= 0.60) & (pr.q_bh_16 < 0.05)
pr['flagged_95_refit'] = (pr.ci95_low >= 0.60) & (pr.q_bh_4tasks_nb05 < 0.05)
pr['reproduces_nb05'] = (pr.auc - pr.nb05_auc).abs().lt(0.001) & (pr.ci95_low - pr.nb05_ci_low).abs().lt(0.001)
sw = pd.DataFrame(swaps)
summ = []
for tid in TASKS:
    for enc in ('ctfm', 'merlin'):
        g = pr[(pr.task_id == tid) & (pr.encoder == enc)].set_index('input')
        s = sw[(sw.task_id == tid) & (sw.encoder == enc)].iloc[0]
        summ.append({'task_id': tid, 'encoder': enc,
                     'shortcut_nb05': bool(g.flagged_nb05['image'] and (g.flagged_nb05['background'] or s.nb05_rule_met)),
                     'shortcut_adjusted': bool(g.flagged_adjusted['image'] and (g.flagged_adjusted['background']
                                                                                or s.rule_met_975)),
                     'image_flagged_adjusted': bool(g.flagged_adjusted['image']),
                     'background_flagged_adjusted': bool(g.flagged_adjusted['background']),
                     'swap_met_adjusted': bool(s.rule_met_975)})
summ = pd.DataFrame(summ)
pr.to_csv(HERE / 'r15_multiplicity_probes.csv', index=False)
sw.to_csv(HERE / 'r15_multiplicity_swap.csv', index=False)
summ.to_csv(HERE / 'r15_multiplicity_summary.csv', index=False)
pd.set_option('display.width', 250)
print(pr.round(4).to_string(index=False)); print(sw.round(4).to_string(index=False)); print(summ.to_string(index=False))
print(f'finished in {time.time() - t0:.0f} s')
