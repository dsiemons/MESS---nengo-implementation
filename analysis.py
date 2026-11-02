"""
Feasibility-focused analysis of the RCPSP / Nengo full-factorial sweep.

Reads rcpsp_results.pkl = (metas, {params_tuple: results_list}) written by the
sweep script. A list is "feasible" if it is a precedence-feasible permutation.
We measure this in three ways (all without any repair step):
  strict  : % of decodes where the winners of the thalamus positions, with
            consecutive duplicates merged, form a feasible list
  raw     : % of decodes where the assignment-problem list is feasible
  final_* : % of instances for which the final (last 0.5 s average) output is
            feasible (strict / raw)

Two kinds of configurations are highlighted:
  * "Most feasible": highest strict rate (ties: raw rate, then final strict)
  * "Rank 1-3"     : top 3 of the quality ranking: most optimal solutions,
                     then lowest mean gap, then strict rate, then raw rate

Outputs
  figures/fig_overview.png   highlighted configurations + all configurations
  figures/fig_params.png     effect of each parameter on feasibility
  figures/fig_showcase.png   one instance in detail (most feasible config)
  results_section.tex        LaTeX results section with numbers filled in
  config_summary.csv         one row per configuration

Run from the folder with the sweep script, SGS.py and the pickle. Set
SWEEP_MODULE to the sweep script's file name (without .py).
"""
import importlib
import multiprocessing as mp
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd

from SGS import serial_schedule_generation

SWEEP_MODULE = "batchRCPSP_Experiments"          # <-- name of the sweep script (no .py)
PKL = "rcpsp_results.pkl"
FIG_DIR = "figures"
TEX_OUT = "results_section.tex"

# order must match the sweep's CONFIGS tuple
PARAM_COLS = ["w_prec", "w_unique", "w_cover", "neurons", "n_rest"]
PARAM_TEX = {"w_prec": r"$W_{\text{prec}}$", "w_unique": r"$W_{\text{unique}}$",
             "w_cover": r"$W_{\text{cover}}$", "neurons": "Neurons",
             "n_rest": r"$n_{\text{rest}}$"}

# "Most feasible": strict rate, then raw rate, then final strict share
FEAS_KEYS, FEAS_ASC = ["strict", "raw", "final_strict"], [False, False, False]
# Quality ranking (criteria 1-3); RANK_ON: "best" = best list seen, "final" = final list
RANK_ON = "best"
QUAL_KEYS = [f"{RANK_ON}_opt", f"{RANK_ON}_gap", "strict", "raw"]
QUAL_ASC = [False, True, False, False]
N_TOP = 3
PALETTE = ["#e4572e", "#4c78a8", "#f3a712", "#59a14f", "#af7aa1"]

sw = importlib.import_module(SWEEP_MODULE)


# --------------------------------------------------------------------------
# Loading, per-configuration metrics, rankings
# --------------------------------------------------------------------------
def load():
    with open(PKL, "rb") as f:
        metas, all_results = pickle.load(f)
    if not isinstance(all_results, dict):
        raise SystemExit("The pickle holds a single configuration (a list). This "
                         "script expects the sweep output {params: results}.")
    return metas, all_results


def build_table(metas, all_results):
    opt = np.array([m["opt"] for m in metas], float)
    rows = []
    for params, res in all_results.items():
        fm = np.array([r["final_ms"] for r in res], float)
        bm = np.array([r["best_ms"] for r in res], float)
        row = dict(zip(PARAM_COLS, params))
        row.update(
            strict=100 * np.mean([r["strict_rate"] for r in res]),
            raw=100 * np.mean([r["raw_rate"] for r in res]),
            final_strict=100 * np.mean([bool(r["final_strict_valid"]) for r in res]),
            final_raw=100 * np.mean([bool(r["final_raw_feasible"]) for r in res]),
            any_strict=100 * np.mean([r["strict_rate"] > 0 for r in res]),
            best_opt=100 * (bm == opt).mean(), best_gap=(100 * (bm - opt) / opt).mean(),
            final_opt=100 * (fm == opt).mean(), final_gap=(100 * (fm - opt) / opt).mean(),
            params=params)
        rows.append(row)
    return pd.DataFrame(rows)


def sort_by(df, keys, asc):
    # values are rounded so that float noise does not break ties
    return df.sort_values(keys, ascending=asc, key=lambda s: s.round(6)).reset_index(drop=True)


def cfg_str(p):
    return "/".join(str(int(v)) if k in ("neurons", "n_rest") else str(v)
                    for k, v in zip(PARAM_COLS, p))


def pick_highlights(df):
    """Returns (feasibility ordering, quality ordering, {params: [labels]})."""
    feas = sort_by(df, FEAS_KEYS, FEAS_ASC)
    qual = sort_by(df, QUAL_KEYS, QUAL_ASC)
    hl = {}
    hl.setdefault(feas.loc[0, "params"], []).append("Most feasible")
    for k in range(N_TOP):
        hl.setdefault(qual.loc[k, "params"], []).append(f"Rank {k + 1}")
    return feas, qual, hl


def hl_label(labels):
    return " / ".join(labels)


# --------------------------------------------------------------------------
# Figure 1: highlighted configurations and all configurations
# --------------------------------------------------------------------------
def fig_overview(df, hl, rows, metas, all_results, feas_params, path):
    uniq = list(hl.items())
    xt = [f"{hl_label(lab)}\n{cfg_str(p)}" for p, lab in uniq]
    x = np.arange(len(uniq)); w = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.bar(x - w / 2, [rows[p].strict for p, _ in uniq], w, color="#59a14f",
           edgecolor="k", label="strict (network alone)")
    ax.bar(x + w / 2, [rows[p].raw for p, _ in uniq], w, color="#af7aa1",
           edgecolor="k", label="raw assignment")
    ax.axhline(df.strict.mean(), color="#59a14f", ls="--", lw=1, label="mean all configs (strict)")
    ax.axhline(df.raw.mean(), color="#af7aa1", ls="--", lw=1, label="mean all configs (raw)")
    ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=7)
    ax.set_ylabel("% of decodes feasible"); ax.set_ylim(0, 105)
    ax.set_title("(a) Feasible decodes (W_prec/W_unique/W_cover/neurons/n_rest)", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.bar(x - w / 2, [rows[p].final_strict for p, _ in uniq], w, color="#59a14f",
           edgecolor="k", label="strict")
    ax.bar(x + w / 2, [rows[p].final_raw for p, _ in uniq], w, color="#af7aa1",
           edgecolor="k", label="raw assignment")
    ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=7)
    ax.set_ylabel("% of instances"); ax.set_ylim(0, 105)
    ax.set_title("(b) Final list feasible", fontsize=10); ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.scatter(df.strict, df.best_opt, s=14, color="lightgrey", edgecolor="grey",
               linewidth=0.3, label="other configs")
    for k, (p, lab) in enumerate(uniq):
        ax.scatter(rows[p].strict, rows[p].best_opt, s=130, color=PALETTE[k], edgecolor="k",
                   zorder=3, label=hl_label(lab))
    ax.set_xlabel("% of decodes strictly feasible")
    ax.set_ylabel("% instances solved to optimality (best seen)")
    ax.set_title("(c) Feasibility versus solution quality, all configurations", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    res = all_results[feas_params]
    n_arcs = np.array([sum(len(v) for v in m["instance"]["predecessors"].values()) for m in metas])
    sr = np.array([100 * r["strict_rate"] for r in res])
    rng = np.random.default_rng(0)
    ax.scatter(n_arcs + rng.uniform(-0.2, 0.2, len(n_arcs)), sr, s=25, color=PALETTE[0],
               edgecolor="k", linewidth=0.3)
    if n_arcs.std() > 0 and sr.std() > 0:
        r_ = np.corrcoef(n_arcs, sr)[0, 1]
        ax.text(0.97, 0.95, f"Pearson r = {r_:.2f}", transform=ax.transAxes, ha="right", va="top")
    ax.set_xlabel("number of precedence arcs in instance")
    ax.set_ylabel("% of decodes strictly feasible")
    ax.set_title(f"(d) Most feasible config ({cfg_str(feas_params)}), per instance", fontsize=10)

    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: effect of each parameter on feasibility
# --------------------------------------------------------------------------
def fig_params(df, hl, rows, path):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharey="row")
    rng = np.random.default_rng(1)
    for j, p in enumerate(PARAM_COLS):
        levels = sorted(df[p].unique())
        pos = {lv: k for k, lv in enumerate(levels)}
        xs = df[p].map(pos).to_numpy(float)
        marg = df.groupby(p)[["strict", "raw"]].mean()
        for i, (metric, name) in enumerate([("strict", "strict"), ("raw", "raw assignment")]):
            ax = axes[i, j]
            ax.scatter(xs + rng.uniform(-0.12, 0.12, len(xs)), df[metric], s=10, alpha=0.25,
                       color="grey")
            ax.plot(range(len(levels)), marg[metric].loc[levels], "o-", color="k", lw=2,
                    label="mean over configs")
            for k, (cp, lab) in enumerate(hl.items()):
                ax.scatter(pos[rows[cp]._asdict()[p]], getattr(rows[cp], metric), marker="*",
                           s=220, color=PALETTE[k], edgecolor="k", zorder=4,
                           label=hl_label(lab) if j == 0 else None)
            ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels)
            if i == 0:
                ax.set_title(p)
            else:
                ax.set_xlabel(p)
            if j == 0:
                ax.set_ylabel(f"% of decodes feasible ({name})")
                if i == 0:
                    ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3: one instance in detail (re-runs the network with activations)
# --------------------------------------------------------------------------
def clarity(final):
    """Mean over positions of (highest - second highest) final output."""
    s = np.sort(final, axis=1)
    return float((s[:, -1] - s[:, -2]).mean())


def pick_showcase(params, metas):
    """Re-run the configuration on every instance (stored results hold no
    activations); pick the instance with a strictly feasible final list and the
    clearest choice (largest mean margin between winner and runner-up)."""
    jobs = [(i, m["instance"], True, params) for i, m in enumerate(metas)]
    with mp.Pool(sw.N_WORKERS) as pool:
        runs = pool.map(sw.run_network, jobs)
    for r in runs:
        r["clarity"] = clarity(r["final"])
    best = max(runs, key=lambda r: (bool(r["final_strict_valid"]), r["clarity"]))
    n_valid = sum(bool(r["final_strict_valid"]) for r in runs)
    print(f"{n_valid}/{len(runs)} instances have a strictly feasible final list; "
          f"picked #{best['idx']} (clarity {best['clarity']:.2f}, "
          f"feasible: {bool(best['final_strict_valid'])})")
    return best["idx"], best


def draw_dag(ax, inst, cmap):
    acts = sorted(inst["durations"])
    lev = {}
    for a in acts:                                   # labels are topological
        ps = inst["predecessors"][a]
        lev[a] = 0 if not ps else max(lev[p] for p in ps) + 1
    by = {}
    for a, l in lev.items():
        by.setdefault(l, []).append(a)
    pos = {a: (l, k - (len(v) - 1) / 2) for l, v in by.items() for k, a in enumerate(sorted(v))}
    for a, ps in inst["predecessors"].items():
        for p in ps:
            ax.annotate("", xy=pos[a], xytext=pos[p],
                        arrowprops=dict(arrowstyle="->", color="grey", shrinkA=11, shrinkB=11))
    for a, (x, y) in pos.items():
        ax.scatter(x, y, s=600, color=cmap((a - 1) % 10), edgecolor="k", zorder=3)
        ax.text(x, y, str(a), ha="center", va="center", color="w", fontweight="bold")
        ax.text(x, y - 0.42, f"d={inst['durations'][a]}", ha="center", fontsize=7)
    ax.set_title(f"Precedence graph (capacities {inst['capacities']})", fontsize=10)
    ax.axis("off")


def gantt(ax, inst, order, title, xmax):
    s, e, ms = serial_schedule_generation(inst, order)
    cmap = plt.get_cmap("tab10")
    for a in order:
        ax.barh(a, e[a] - s[a], left=s[a], color=cmap((a - 1) % 10), edgecolor="k")
        ax.text((s[a] + e[a]) / 2, a, str(a), ha="center", va="center", color="w", fontsize=8)
    ax.set_yticks(sorted(inst["durations"])); ax.invert_yaxis()
    ax.set_xlim(0, xmax); ax.axvline(ms, color="k", ls="--")
    ax.set_title(f"{title} (makespan {ms})", fontsize=10); ax.set_ylabel("activity")


def fig_showcase(meta, params, idx, stored, res, path):
    inst = meta["instance"]
    acts = sorted(inst["durations"]); N = len(acts)
    if res["final_ms"] != stored["final_ms"]:
        print("WARNING: re-run differs from stored result "
              f"({res['final_ms']} vs {stored['final_ms']}).")
    act, tt, final = res["act"], res["tt"], res["final"]
    NP = act.shape[0]
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(18, max(11, 1.2 * NP)))
    gs = GridSpec(1, 2, width_ratios=[1.15, 1], figure=fig, wspace=0.18)
    left = GridSpecFromSubplotSpec(NP, 1, subplot_spec=gs[0], hspace=0.12)
    right = GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[1], hspace=0.7,
                                    height_ratios=[1.4, 1.4, 0.9, 0.9, 1, 1])

    for t in range(NP):                              # thalamus output per position
        ax = fig.add_subplot(left[t])
        for c in range(N):
            ax.plot(tt, act[t, c], color=cmap((acts[c] - 1) % 10), label=f"activity {acts[c]}")
        ax.axvspan(0, sw.T_DECODE_FROM, color="grey", alpha=0.12)
        ax.axvspan(tt[-1] - 0.5, tt[-1], color="gold", alpha=0.25)
        ax.set_ylim(-0.2, 1.3); ax.set_xlim(tt[0], tt[-1])
        ax.set_ylabel(f"pos {t + 1}\n-> {acts[int(final[t].argmax())]}", fontsize=8)
        if t < NP - 1:
            ax.set_xticklabels([])
        if t == 0:
            ax.legend(ncol=min(N, 8), fontsize=7, loc="upper right")
            ax.set_title("Thalamus output per list position  (grey: before decoding, "
                         "gold: averaged for final list)")
    ax.set_xlabel("time (s)")

    draw_dag(fig.add_subplot(right[0]), inst, cmap)

    ax = fig.add_subplot(right[1])
    im = ax.imshow(final, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    for t in range(NP):
        ax.text(int(final[t].argmax()), t, "*", ha="center", va="center", color="w", fontsize=14)
    ax.set_xticks(range(N)); ax.set_xticklabels(acts)
    ax.set_yticks(range(NP)); ax.set_yticklabels(range(1, NP + 1))
    ax.set_xlabel("activity"); ax.set_ylabel("position")
    ax.set_title("Final activation (mean over last 0.5 s), * = winner", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)

    ax = fig.add_subplot(right[2])
    ax.step(res["trace_t"], res["trace_ms"], where="post", color="k", label="decoded list")
    ax.axhline(meta["opt"], color="g", ls="--", label="optimum")
    ax.axhline(meta["lb_cpm"], color="grey", ls=":", label="critical path bound")
    ax.set_ylabel("makespan"); ax.set_xlabel("time (s)")
    ax2 = ax.twinx()
    ax2.plot(tt, sw.noise_scale(tt), color="#e4572e", alpha=0.6)
    ax2.set_ylabel("noise rms", color="#e4572e")
    ax.set_title("Makespan of decoded list (full repair) and noise level", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    ax = fig.add_subplot(right[3]); ax.axis("off")
    lines = ["Winner per position over time ('-' = no clear winner, output < 0.3):"]
    for ts in [sw.T_DECODE_FROM, 2.0, 3.0, tt[-1] - 0.25]:
        k = min(int(ts / sw.DT), act.shape[2] - 1)
        wnr = [str(acts[int(act[t, :, k].argmax())]) if act[t, :, k].max() >= 0.3 else "-"
               for t in range(NP)]
        lines.append(f"  t={ts:4.2f}s:  " + " ".join(wnr))
    lines.append(f"  final list (repair): {res['final_list']}")
    lines.append(f"  optimal list:        {meta['opt_list']}")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)

    xmax = max(res["final_ms"], meta["opt"], 1) + 1
    gantt(fig.add_subplot(right[4]), inst, res["final_list"], "Network final list", xmax)
    gantt(fig.add_subplot(right[5]), inst, meta["opt_list"], "Optimal list", xmax)

    fig.suptitle(f"Instance #{idx}, most feasible configuration {dict(zip(PARAM_COLS, params))}: "
                 f"optimum {meta['opt']}, network final {res['final_ms']}, "
                 f"best seen {res['best_ms']} (t={res['best_time']:.2f}s)", fontsize=13)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return res


# --------------------------------------------------------------------------
# LaTeX results section (numbers filled in; check the wording)
# --------------------------------------------------------------------------
def write_results(df, qual, hl, rows, feas_params, metas, show_idx, show_res):
    n_cfg, n_inst = len(df), len(metas)
    mf = rows[feas_params]
    qual_params = list(qual["params"])
    L = ["% Generated by analyze_rcpsp.py",
         r"\section{Results}"]

    L += [r"\begin{table}[t]", r"\centering", r"\small",
          r"\begin{tabular}{lccccc|cccccc}",
          r"Selected as & $W_{\text{prec}}$ & $W_{\text{unique}}$ & $W_{\text{cover}}$ & Neurons & "
          r"$n_{\text{rest}}$ & Strict (\%) & Raw (\%) & Final strict (\%) & Final raw (\%) & "
          r"Opt.\ best (\%) & Gap best (\%) \\ \hline"]
    for cp, lab in hl.items():
        r = rows[cp]
        L.append(rf"{hl_label(lab)} & {r.w_prec} & {r.w_unique} & {r.w_cover} & {int(r.neurons)} & "
                 rf"{int(r.n_rest)} & {r.strict:.1f} & {r.raw:.1f} & {r.final_strict:.0f} & "
                 rf"{r.final_raw:.0f} & {r.best_opt:.0f} & {r.best_gap:.1f} \\")
    L += [r"\end{tabular}",
          r"\caption{The most feasible configuration and the top three of the quality ranking. Strict and raw: "
          r"share of decodes that are feasible; final: share of instances whose final list is feasible; "
          r"Opt.\ = share of instances solved to optimality.}",
          r"\label{tab:highlight}", r"\end{table}"]

    L.append(r"\paragraph{Effect of the parameters on feasibility.}")
    spans = {p: df.groupby(p).strict.mean().agg(["min", "max"]) for p in PARAM_COLS}
    big = max(PARAM_COLS, key=lambda p: spans[p]["max"] - spans[p]["min"])
    L.append(
        rf"Table~\ref{{tab:marginal}} and Figure~\ref{{fig:params}} show the mean feasibility per parameter "
        rf"level, averaged over all other parameters. The parameter with the largest effect on the strict "
        rf"decode rate is {PARAM_TEX[big]}, whose level means range from {spans[big]['min']:.1f}\% to "
        rf"{spans[big]['max']:.1f}\%.")
    L += [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{ll|ccc}",
          r"Parameter & Value & Strict (\%) & Raw (\%) & Final strict (\%) \\ \hline"]
    for p in PARAM_COLS:
        g = df.groupby(p)[["strict", "raw", "final_strict"]].mean()
        for k, (lv, r) in enumerate(g.iterrows()):
            name = PARAM_TEX[p] if k == 0 else ""
            L.append(rf"{name} & {lv} & {r.strict:.1f} & {r.raw:.1f} & {r.final_strict:.1f} \\")
        L.append(r"\hline")
    L += [r"\end{tabular}",
          r"\caption{Mean feasibility per parameter level, averaged over all other parameters.}",
          r"\label{tab:marginal}", r"\end{table}"]

    inst = metas[show_idx]
    tp = show_res["final"].max(axis=1)
    with open(TEX_OUT, "w") as f:
        f.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    metas, all_results = load()
    df = build_table(metas, all_results)
    df.drop(columns="params").to_csv("config_summary.csv", index=False)
    feas, qual, hl = pick_highlights(df)
    rows = {r.params: r for r in df.itertuples()}
    feas_params = feas.loc[0, "params"]
    print(f"{len(df)} configurations, {len(metas)} instances")

    show_cols = PARAM_COLS + ["strict", "raw", "final_strict", "final_raw", "best_opt", "best_gap"]
    print("\nTop 5 by feasibility:")
    print(feas[show_cols].head(5).round(2).to_string())
    print(f"\nTop {N_TOP} of the quality ranking:")
    print(qual[show_cols].head(N_TOP).round(2).to_string())

    fig_overview(df, hl, rows, metas, all_results, feas_params, f"{FIG_DIR}/fig_overview.png")
    fig_params(df, hl, rows, f"{FIG_DIR}/fig_params.png")

    print("\nRe-running the most feasible configuration on all instances to pick the showcase ...")
    idx, res = pick_showcase(feas_params, metas)
    show_res = fig_showcase(metas[idx], feas_params, idx, all_results[feas_params][idx], res,
                            f"{FIG_DIR}/fig_showcase.png")
    write_results(df, qual, hl, rows, feas_params, metas, idx, show_res)
    print(f"Wrote {TEX_OUT} and figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()