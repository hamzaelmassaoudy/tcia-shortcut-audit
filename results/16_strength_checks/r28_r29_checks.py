"""Checks R28 and R29 (plan section 7r, written 26 November 2026 before they were computed; POST HOC).

R28  between-cohort AUC of the fixed TCGA lung probe scores for every ordered pair of the four lung targets (histology
     ignored), with a 95% patient-bootstrap CI; 1,000 within-cohort label permutations of each crossed test set; Spearman
     correlation between the crossed-set AUC with the recorded labels (R17) and the between-cohort AUC.
R29  flag counts at lower-bound thresholds 0.55, 0.60, 0.65 and 0.70 (q < 0.05 unchanged) for the metadata screen and
     the image probes of the 19 CT tasks.

Reads saved result files only; no model is refitted, no image is read. Usage: python r28_r29_checks.py (from this folder).
Outputs: r28_between_cohort_auc.csv, r28_summary.csv, r29_threshold_sensitivity.csv, r29_status_changes.csv,
run_manifest_r28_r29.json."""
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
R = HERE.parent
SEED, N_BOOT, N_PERM = 2026, 2000, 1000
THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
COHORT = {"lung: TCGA -> CPTAC": "CPTAC lung", "lung: TCGA -> NSCLC Radiogenomics": "NSCLC Radiogenomics",
          "lung: TCGA -> NSCLC-Radiomics": "NSCLC-Radiomics", "lung: TCGA -> Lung-PET-CT-Dx": "Lung-PET-CT-Dx"}
INPUT_NAME = {"image": "full image", "background": "background only", "silhouette": "silhouette"}
INPUTS = {"scores": R / "11_external_lung/real/transfer_scores.csv.gz",
          "r17": R / "13_transfer_checks/real/r17_crossed_assembly.csv",
          "audit": R / "03_audit/audit_primary.csv",
          "p05": R / "05_image_check/real/image_probe.csv",
          "u07": R / "07_screen_validation/real/image_probe_unflagged_tasks.csv"}


def auc(y, s):
    """AUC by ranks, ties one half (as in R17)."""
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(y, s):
    """95% percentile patient-bootstrap CI (as in R17)."""
    rng = np.random.default_rng(SEED)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    while len(vals) < N_BOOT:
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(auc(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def r28():
    sc = pd.read_csv(INPUTS["scores"])
    sc["cohort"] = sc.pair.map(COHORT)
    assert sc.cohort.notna().all()
    r17 = pd.read_csv(INPUTS["r17"])
    r17 = r17[r17.set == "crossed"].set_index(["encoder", "input", "adenocarcinoma_from", "squamous_from"])
    rows = []
    for (enc, inp), g in sc.groupby(["encoder", "input"]):
        for x, y in itertools.permutations(COHORT.values(), 2):
            gx, gy = g[g.cohort == x], g[g.cohort == y]
            yy = np.r_[np.zeros(len(gx)), np.ones(len(gy))]
            ss = np.r_[gx.score.to_numpy(), gy.score.to_numpy()]
            between = auc(yy, ss)
            lo, hi = boot(yy, ss)
            # crossed set: adenocarcinomas of x against squamous of y; permute labels within each cohort
            n_adeno_x, n_squam_y = int((gx.label == 0).sum()), int((gy.label == 1).sum())
            rng = np.random.default_rng([SEED, len(rows)])
            perm = np.empty(N_PERM)
            for k in range(N_PERM):
                a = rng.choice(gx.score.to_numpy(), n_adeno_x, replace=False)
                b = rng.choice(gy.score.to_numpy(), n_squam_y, replace=False)
                perm[k] = auc(np.r_[np.zeros(n_adeno_x), np.ones(n_squam_y)], np.r_[a, b])
            recorded = r17.loc[(enc, INPUT_NAME[inp], x, y)]
            rows.append({"encoder": enc, "input": INPUT_NAME[inp], "adenocarcinoma_from": x, "squamous_from": y,
                         "n_first_cohort": len(gx), "n_second_cohort": len(gy), "between_cohort_auc": between,
                         "between_ci_low": lo, "between_ci_high": hi, "crossed_auc_recorded_labels": recorded.auc,
                         "crossed_ci_low_recorded": recorded.ci_low, "permuted_mean_auc": perm.mean(),
                         "permuted_p2_5": np.percentile(perm, 2.5), "permuted_p97_5": np.percentile(perm, 97.5),
                         "share_permutations_auc_at_least_0.60": float((perm >= 0.60).mean()),
                         "share_permutations_auc_at_most_0.40": float((perm <= 0.40).mean())})
    d = pd.DataFrame(rows)
    summ = []
    for (enc, inp), g in d.groupby(["encoder", "input"]):
        rho, p = spearmanr(g.crossed_auc_recorded_labels, g.between_cohort_auc)
        summ.append({"encoder": enc, "input": inp, "crossed_sets": len(g),
                     "between_ci_above_0.50": int((g.between_ci_low > 0.5).sum()),
                     "between_ci_below_0.50": int((g.between_ci_high < 0.5).sum()),
                     "spearman_crossed_vs_between": rho, "spearman_p": p,
                     "max_abs_difference_permuted_mean_vs_between": float((g.permuted_mean_auc - g.between_cohort_auc).abs().max())})
    return d, pd.DataFrame(summ)


def r29():
    a = pd.read_csv(INPUTS["audit"])
    p05, u07 = pd.read_csv(INPUTS["p05"]), pd.read_csv(INPUTS["u07"])
    p05["metadata_flag"], u07["metadata_flag"] = True, False
    img = pd.concat([p05, u07], ignore_index=True)
    rows, changes = [], []
    for t in THRESHOLDS:
        m = a[(a.ci_low >= t) & (a.perm_q_bh < 0.05)]
        rows.append({"analysis": "metadata screen", "encoder": "", "input": "", "tasks": len(a), "threshold": t,
                     "flagged": len(m), "flagged_tasks": "; ".join(m.task_id)})
        for enc in ("ctfm", "merlin"):
            for inp, lo, q in (("full image", "image_ci_low", "image_perm_q_bh"),
                               ("background only", "background_ci_low", "background_perm_q_bh")):
                g = img[img.encoder == enc]
                f = g[(g[lo] >= t) & (g[q] < 0.05)]
                rows.append({"analysis": "image probe", "encoder": enc, "input": inp, "tasks": len(g), "threshold": t,
                             "flagged": len(f), "flagged_tasks": "; ".join(f.task_id)})
    out = pd.DataFrame(rows)
    base = out[out.threshold == 0.60].set_index(["analysis", "encoder", "input"]).flagged_tasks
    for _, r in out[out.threshold != 0.60].iterrows():
        b = set(filter(None, base[(r.analysis, r.encoder, r.input)].split("; ")))
        c = set(filter(None, r.flagged_tasks.split("; ")))
        for task in sorted(c - b):
            changes.append({"analysis": r.analysis, "encoder": r.encoder, "input": r.input, "threshold": r.threshold,
                            "task": task, "change": "flagged only at this threshold"})
        for task in sorted(b - c):
            changes.append({"analysis": r.analysis, "encoder": r.encoder, "input": r.input, "threshold": r.threshold,
                            "task": task, "change": "not flagged at this threshold"})
    return out, pd.DataFrame(changes)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    t0 = time.time()
    d28, s28 = r28()
    d29, c29 = r29()
    out = {"r28_between_cohort_auc.csv": d28, "r28_summary.csv": s28, "r29_threshold_sensitivity.csv": d29,
           "r29_status_changes.csv": c29}
    pd.set_option("display.width", 250)
    for name, frame in out.items():
        frame.to_csv(HERE / name, index=False)
        print(f"== {name}\n{frame.to_string(index=False, max_colwidth=60)}\n")
    manifest = {"check": "R28-R29", "plan": "protocol/analysis_plan_7r.md (26 November 2026)", "post_hoc": True,
                "settings": {"seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM, "thresholds": THRESHOLDS},
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__},
                "inputs": {k: {"path": v.relative_to(R.parent).as_posix(), "md5": md5(v)} for k, v in INPUTS.items()},
                "outputs": {name: md5(HERE / name) for name in out}, "seconds": round(time.time() - t0)}
    (HERE / "run_manifest_r28_r29.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"finished in {time.time() - t0:.0f} s")
