"""Checks R25-R27 (plan section 7q, written 25 November 2026 before they were computed; POST HOC).

R25  size of the external cohorts: probability that a fixed probe with true external AUC a meets the CI part of the
     transfer rule (lower bound of the 95% percentile patient-bootstrap CI above 0.50), for a = 0.65, 0.70, 0.739, and
     the true AUC at which this probability reaches 80%; binormal simulation exactly as in R14 (1,000 data sets x 1,000
     bootstrap resamples, data set s uses default_rng([2026, s])).
R26  agreement of the metadata screen with the background-only and full-image flags of the 19 CT tasks, with exact
     Clopper-Pearson 95% CIs.
R27  counts of test sets with CIs wholly above or below 0.50 in the crossed-assembly check R17 (real and rehearsal).

Reads saved result files only; no model is refitted, no image is read. Usage: python r25_r27_strength_checks.py
(from this folder). Outputs: r25_external_cohort_power.csv, r26_screen_agreement.csv, r27_assembly_counts.csv,
run_manifest.json."""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy.stats import beta, norm

HERE = Path(__file__).resolve().parent
R = HERE.parent
N_SIM, N_BOOT_SIM, POWER, CUT = 1000, 1000, 0.80, 0.50
AUCS = (0.65, 0.70, 0.739)
COHORTS = [("CPTAC lung", "one collection per class", 37, 36), ("NSCLC Radiogenomics", "both classes in one collection", 166, 33),
           ("NSCLC-Radiomics", "both classes in one collection", 51, 152),
           ("Lung-PET-CT-Dx", "both classes in one collection", 251, 61),
           ("C4KC-KiTS", "both classes in one collection", 143, 21)]
INPUTS = {"p05": R / "05_image_check/real/image_probe.csv",
          "u07": R / "07_screen_validation/real/image_probe_unflagged_tasks.csv",
          "r17_real": R / "13_transfer_checks/real/r17_crossed_assembly.csv",
          "r17_rehearsal": R / "13_transfer_checks/rehearsal/r17_crossed_assembly.csv"}


def lower_bounds(auc_true, n0, n1):
    """Lower 95% percentile-bootstrap bound of the AUC in N_SIM simulated data sets (code of check R14)."""
    d = np.sqrt(2) * norm.ppf(auc_true)
    n = n0 + n1
    lab = np.r_[np.zeros(n0, bool), np.ones(n1, bool)]
    out = np.empty(N_SIM)
    for s in range(N_SIM):
        rng = np.random.default_rng([2026, s])
        z = rng.standard_normal(n)
        idx = rng.integers(0, n, size=(N_BOOT_SIM, n))
        W = np.bincount((idx + np.arange(N_BOOT_SIM)[:, None] * n).ravel(), minlength=N_BOOT_SIM * n
                        ).reshape(N_BOOT_SIM, n).astype(float)
        score = z + d * lab
        o = np.argsort(score, kind="stable")
        Wo, pos = W[:, o], lab[o]
        negw = Wo * ~pos
        below = np.cumsum(negw, axis=1) - negw
        wp, wn = Wo[:, pos].sum(1), Wo[:, ~pos].sum(1)
        ok = (wp > 0) & (wn > 0)
        a = (Wo * pos * below).sum(1)[ok] / (wp[ok] * wn[ok])
        out[s] = np.percentile(a, 2.5)
    return out


def r25_cohort(name, kind, n0, n1):
    row = {"cohort": name, "assembly": kind, "n_class_0": n0, "n_class_1": n1, "n_patients": n0 + n1}
    for a in AUCS:
        row[f"probability_ci_above_0.50_at_auc_{a}"] = float((lower_bounds(a, n0, n1) > CUT).mean())
    lo, hi = CUT, 0.999
    while hi - lo > 0.001:
        mid = (lo + hi) / 2
        if (lower_bounds(mid, n0, n1) > CUT).mean() >= POWER:
            hi = mid
        else:
            lo = mid
    row["auc_detectable_with_80pct_probability"] = round(hi, 3)
    row["large_enough_for_cptac_size_transfer"] = row["probability_ci_above_0.50_at_auc_0.739"] >= POWER
    return row


def exact_ci(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return (float(lo), float(hi))


def r26():
    p05, u07 = pd.read_csv(INPUTS["p05"]), pd.read_csv(INPUTS["u07"])
    p05["metadata_flag"], u07["metadata_flag"] = True, False
    cols = ["task_id", "encoder", "metadata_flag", "background_flagged", "image_flagged"]
    both = pd.concat([p05[cols], u07[cols]], ignore_index=True)
    assert both.task_id.nunique() == 19 and len(both) == 38
    rows = []
    for image_input, col in (("background only", "background_flagged"), ("full image", "image_flagged")):
        per_task = both.pivot_table(index="task_id", columns="encoder", values=col, aggfunc="first")
        per_task["either"] = per_task.any(axis=1)
        meta = both.groupby("task_id").metadata_flag.first()
        for enc in ("ctfm", "merlin", "either"):
            img = per_task[enc].astype(bool)
            tp = int((meta & img).sum()); fp = int((meta & ~img).sum())
            fn = int((~meta & img).sum()); tn = int((~meta & ~img).sum())
            row = {"image_input": image_input, "encoder": enc, "both_flagged": tp, "metadata_only": fp,
                   "image_only": fn, "neither": tn}
            for name, k, n in (("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp),
                               ("positive_predictive_value", tp, tp + fp), ("negative_predictive_value", tn, tn + fn)):
                lo, hi = exact_ci(k, n)
                row.update({name: k / n if n else float("nan"), f"{name}_k_of_n": f"{k}/{n}",
                            f"{name}_ci_low": lo, f"{name}_ci_high": hi})
            rows.append(row)
    return pd.DataFrame(rows)


def r27():
    rows = []
    for run in ("real", "rehearsal"):
        d = pd.read_csv(INPUTS[f"r17_{run}"])
        d["group"] = np.where(d.set == "crossed", "crossed (classes from two cohorts)",
                              np.where(d.adenocarcinoma_from == "CPTAC lung", "CPTAC lung (one collection per class)",
                                       "both classes in one collection"))
        for (enc, inp, grp), g in d.groupby(["encoder", "input", "group"]):
            rows.append({"run": run, "encoder": enc, "input": inp, "test_sets": grp, "n_sets": len(g),
                         "ci_above_0.50": int((g.ci_low > 0.5).sum()), "ci_below_0.50": int((g.ci_high < 0.5).sum()),
                         "auc_min": round(float(g.auc.min()), 3), "auc_max": round(float(g.auc.max()), 3)})
    return pd.DataFrame(rows)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    t0 = time.time()
    r25 = pd.DataFrame(Parallel(n_jobs=5)(delayed(r25_cohort)(*c) for c in COHORTS))
    out = {"r25_external_cohort_power.csv": r25, "r26_screen_agreement.csv": r26(), "r27_assembly_counts.csv": r27()}
    for name, frame in out.items():
        frame.to_csv(HERE / name, index=False)
        pd.set_option("display.width", 250)
        print(f"== {name}\n{frame.to_string(index=False)}\n")
    manifest = {"check": "R25-R27", "plan": "protocol/analysis_plan_7q.md (25 November 2026)", "post_hoc": True,
                "settings": {"n_sim": N_SIM, "n_boot": N_BOOT_SIM, "ci_cut": CUT, "power": POWER, "aucs": AUCS,
                             "seed": "default_rng([2026, s])"},
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__},
                "inputs": {k: {"path": v.relative_to(R.parent).as_posix(), "md5": md5(v)} for k, v in INPUTS.items()},
                "outputs": {name: md5(HERE / name) for name in out}, "seconds": round(time.time() - t0)}
    (HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"finished in {time.time() - t0:.0f} s")
