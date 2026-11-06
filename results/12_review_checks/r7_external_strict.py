"""Check R7 (4 Nov 2026; POST HOC, stricter sensitivity analysis).

The paper reads external transfer by a CI rule without multiplicity correction. R7 asks a stricter, different question
of the saved external scores: holding each source-trained model fixed, is its external AUC larger than expected if the
target labels were assigned at random? One-sided target-label permutation test of the rank-sum (ties as mid-ranks),
B = 99 999 Monte Carlo permutations (seed 2026), P = (1 + b) / (1 + B); exact P by enumeration (dynamic programming)
where the target has at most 200 patients. Holm correction over all 20 full-image and background comparisons of
Table 4. This conditions on the fitted source models, and it is not the paper's descriptive external permutation test
(which refits on permuted source labels).

Outputs: r7_external_strict.csv.  Usage: python r7_external_strict.py"""
import gzip
import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = L_R = __import__('review_lib').R
B, SEED = 99_999, 2026
s08 = pd.concat([pd.read_csv(f) for f in sorted((R / '08_transfer' / 'real' / 'per_pair').glob('*scores.csv.gz'))],
                ignore_index=True)
s08 = s08[s08.direction == 'external']
# notebook 11 re-scored the two notebook 08 lung targets identically; take only its two new targets from it
s11 = pd.read_csv(R / '11_external_lung' / 'real' / 'transfer_scores.csv.gz')
s11 = s11[s11.pair.isin(['lung: TCGA -> NSCLC-Radiomics', 'lung: TCGA -> Lung-PET-CT-Dx'])].assign(direction='external')
sc = pd.concat([s08, s11], ignore_index=True)
sc = sc[sc.input.isin(['image', 'background'])]
assert sc.groupby(['pair', 'encoder', 'input']).ngroups == 20
ref = pd.concat([pd.read_csv(R / '08_transfer' / 'real' / 'transfer.csv'),
                 pd.read_csv(R / '11_external_lung' / 'real' / 'transfer.csv')], ignore_index=True)


def exact_upper_p(ranks2, m, obs2):
    """P(sum of m of the doubled ranks >= obs2) over all C(n, m) subsets, by dynamic programming on integer sums."""
    n, tot = len(ranks2), int(ranks2.sum())
    dp = np.zeros((m + 1, tot + 1), dtype=object)  # exact integer counts
    dp[0, 0] = 1
    for r in ranks2.astype(int):
        dp[1:, r:] = dp[1:, r:] + dp[:-1, :tot + 1 - r]
    counts = dp[m]
    return float(sum(counts[int(obs2):]) / sum(counts))


rng = np.random.default_rng(SEED)
rows = []
for (pair, enc, inp), g in sc.groupby(['pair', 'encoder', 'input'], sort=True):
    y = g.label.to_numpy().astype(int)
    rk = rankdata(g.score.to_numpy())
    n, m = len(y), int(y.sum())
    obs = rk[y == 1].sum()
    auc = (obs - m * (m + 1) / 2) / (m * (n - m))
    b = 0
    for start in range(0, B, 5000):
        k = min(5000, B - start)
        idx = np.argsort(rng.random((k, n)), axis=1)[:, :m]
        b += int((rk[idx].sum(axis=1) >= obs - 1e-9).sum())
    p_mc = (1 + b) / (1 + B)
    p_exact = exact_upper_p(2 * rk, m, round(2 * obs)) if n <= 200 else np.nan
    r = ref[(ref.pair == pair) & (ref.encoder == enc) & (ref.input == inp)].iloc[0]
    rows.append({'pair': pair, 'encoder': enc, 'input': inp, 'n_target': n, 'n_positive': m, 'external_auc': auc,
                 'external_auc_table4': r.external_auc, 'table4_reading': 'transfers' if r.external_ci_low > 0.5
                 else 'does not transfer', 'p_monte_carlo': p_mc, 'p_exact': p_exact,
                 'p_used': p_exact if n <= 200 else p_mc})
out = pd.DataFrame(rows)
assert (out.external_auc - out.external_auc_table4).abs().max() < 1e-12
order = out.p_used.sort_values().index
adj, run = {}, 0.0
for i, ix in enumerate(order):
    run = max(run, min(1.0, out.p_used[ix] * (len(out) - i)))
    adj[ix] = run
out['p_holm_20'] = pd.Series(adj)
out['survives_holm_0.05'] = out.p_holm_20 < 0.05
out.to_csv('r7_external_strict.csv', index=False)
pd.set_option('display.width', 250)
print(out.drop(columns=['external_auc_table4']).round(5).to_string(index=False))
