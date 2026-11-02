"""
Which constraint do the network's unrepaired lists violate?

For the highlighted configurations (most feasible + top 3 of the quality
ranking, as in analyze_rcpsp.py) the network is re-run on every instance with
full activations (the stored results only hold repaired lists). At three
moments the strict list is decoded (argmax per position, consecutive
duplicates merged) and checked against the three constraints:

  uniqueness : some activity appears more than once (non-consecutively)
  cover      : some activity does not appear at all
  precedence : among the activities present, a successor's first occurrence
               comes before its predecessor's first occurrence

Moments:  final      the last-0.5 s average output
          best seen  the decode moment with the lowest makespan (after repair)
          all decodes every decode moment (t = T_DECODE_FROM ... T_SIM)

The raw (assignment) decoder always gives a permutation, so it can only
violate precedence; it is not analysed here.

Outputs: violations.csv, figures/fig_violations.png
Run from the same folder as analyze_rcpsp.py.
"""
import multiprocessing as mp
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis as ar

sw = ar.sw
MOMENTS = ["final", "best seen", "all decodes"]
FLAGS = ["uniqueness", "cover", "precedence"]


def collapse(seq):
    seq = [int(c) for c in seq]
    return [c for k, c in enumerate(seq) if k == 0 or c != seq[k - 1]]


def classify(winners, C, N):
    """winners: argmax activity per position. Returns
    (uniqueness, cover, precedence, n_missing, n_duplicates)."""
    tr = collapse(winners)
    counts = np.bincount(tr, minlength=N)
    uniq = bool((counts > 1).any())
    cover = bool((counts == 0).any())
    first = {}
    for pos, a in enumerate(tr):
        first.setdefault(a, pos)
    prec = any(C[i, j] == 1 and i in first and j in first and first[j] < first[i]
               for i in range(N) for j in range(N))
    return uniq, cover, prec, int((counts == 0).sum()), int(np.clip(counts - 1, 0, None).sum())


def analyse_config(params, metas, all_results):
    jobs = [(i, m["instance"], True, params) for i, m in enumerate(metas)]
    with mp.Pool(sw.N_WORKERS) as pool:
        runs = pool.map(sw.run_network, jobs)

    recs = {m: [] for m in MOMENTS}
    mismatch = 0
    step = max(1, int(sw.DECODE_EVERY / sw.DT))
    for meta, r in zip(metas, runs):
        inst = meta["instance"]
        N = len(inst["durations"])
        C = sw.precedence_matrix(inst)
        act, tt = r["act"], r["tt"]

        fin = classify(r["final"].argmax(axis=1), C, N)
        recs["final"].append(fin)
        k_best = int(np.argmin(np.abs(tt - r["best_time"])))
        recs["best seen"].append(classify(act[:, :, k_best].argmax(axis=1), C, N))
        for k in range(int(sw.T_DECODE_FROM / sw.DT), act.shape[2], step):
            recs["all decodes"].append(classify(act[:, :, k].argmax(axis=1), C, N))
        # sanity: agrees with the sweep's own feasibility check of the final list
        mismatch += (not any(fin[:3])) != bool(r["final_strict_valid"])
    if mismatch:
        print(f"  WARNING: {mismatch} final lists disagree with final_strict_valid")

    out = {}
    for m in MOMENTS:
        a = np.array(recs[m], float)
        viol = a[:, :3]
        out[m] = {
            "uniqueness": 100 * viol[:, 0].mean(),
            "cover": 100 * viol[:, 1].mean(),
            "precedence": 100 * viol[:, 2].mean(),
            "feasible": 100 * (viol.sum(axis=1) == 0).mean(),
            "only_precedence": 100 * ((viol[:, 2] == 1) & (viol[:, :2].sum(axis=1) == 0)).mean(),
            "mean_missing": a[:, 3].mean(),
            "mean_duplicates": a[:, 4].mean(),
            "n_lists": len(a),
        }
    return out


def fig_violations(table, hl, path):
    uniq = list(hl.items())
    colors = {"uniqueness": "#e4572e", "cover": "#f3a712", "precedence": "#4c78a8",
              "feasible": "#59a14f"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    x = np.arange(len(uniq)); w = 0.2
    for ax, mom in zip(axes, MOMENTS):
        for k, key in enumerate(FLAGS + ["feasible"]):
            vals = [table[(p, mom)][key] for p, _ in uniq]
            ax.bar(x + (k - 1.5) * w, vals, w, color=colors[key], edgecolor="k", label=key)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{ar.hl_label(lab)}\n{ar.cfg_str(p)}" for p, lab in uniq], fontsize=7)
        ax.set_title({"final": "Final list", "best seen": "Best-seen list",
                      "all decodes": "All decodes"}[mom])
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("% of lists violating the constraint")
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(ar.FIG_DIR, exist_ok=True)
    metas, all_results = ar.load()
    df = ar.build_table(metas, all_results)
    feas, qual, hl = ar.pick_highlights(df)

    table, rows = {}, []
    for params, labels in hl.items():
        print(f"\n{ar.hl_label(labels)}  ({ar.cfg_str(params)})")
        res = analyse_config(params, metas, all_results)
        for mom, vals in res.items():
            table[(params, mom)] = vals
            rows.append({"selected_as": ar.hl_label(labels), "config": ar.cfg_str(params),
                         "moment": mom, **vals})
            print(f"  {mom:12s} uniq {vals['uniqueness']:5.1f}%  cover {vals['cover']:5.1f}%  "
                  f"prec {vals['precedence']:5.1f}%  feasible {vals['feasible']:5.1f}%  "
                  f"(only prec {vals['only_precedence']:5.1f}%, "
                  f"missing {vals['mean_missing']:.2f}, dup {vals['mean_duplicates']:.2f})")
    pd.DataFrame(rows).to_csv("violations.csv", index=False)
    fig_violations(table, hl, f"{ar.FIG_DIR}/fig_violations.png")
    print(f"\nWrote violations.csv and {ar.FIG_DIR}/fig_violations.png")


if __name__ == "__main__":
    main()