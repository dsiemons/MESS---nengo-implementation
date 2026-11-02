"""
Test: do the utilities that reach the basal ganglia leave its working range?

Hypothesis (see Results): the summed input of a basal ganglia ensemble
(baseline + noise + cover + uniqueness + precedence) leaves the range in which
the Nengo basal ganglia selects reliably, mainly because the precedence signal
accumulates over positions.

For the highlighted configurations (most feasible + top 3 of the quality
ranking, as in analyze_rcpsp.py) the sweep's own network is re-run on a random
set of instances, with a probe on the input of every basal ganglia. This is
done by wrapping nengo.networks.BasalGanglia and nengo.Simulator, so the model
is exactly the one of the sweep script (run_network is not copied).

Measured in the decoding window (t >= T_DECODE_FROM), per configuration:
  * share of utilities above THR_SAT / THR_RADIUS and below THR_LOW
  * share of (position, time) points where >= 2 activities are above THR_SAT
    (saturated tie: the winner can no longer be told apart)
  * size of each input component (precedence, uniqueness, cover), computed
    from the thalamus output; the residual to the probed input is the noise
  * Spearman correlation between saturation and strict decode rate over runs

Thresholds come from the Nengo 3.0.0 BasalGanglia source: internal radius 1.5,
D1 striatum input scaled by 1.2 (so it saturates near 1.5 / 1.2 = 1.25),
striatal threshold 0.2 (lower inputs give no striatal output). Saturation is
soft, so treat the thresholds as indicative. Check your Nengo version.

Outputs: input_range_test.csv (per run), figures/fig_input_range.png
Run from the folder with analyze_rcpsp.py.
"""
import multiprocessing as mp
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nengo
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import analysis as ar

sw = ar.sw

N_TEST = 10                 # random instances
TEST_SEED = 7
ADD_PREC_CONTRAST = True   # True: also run the most feasible config with the other W_prec levels
THR_LOW, THR_SAT, THR_RADIUS = 0.2, 1.25, 1.5
BINS = np.linspace(-2.5, 3.5, 121)

# --------------------------------------------------------------------------
# Probe the input of every basal ganglia without touching the sweep code
# --------------------------------------------------------------------------
_ORIG_BG = nengo.networks.BasalGanglia
BG_PROBES, LAST = [], {}


def _bg_with_probe(*args, **kwargs):
    bg = _ORIG_BG(*args, **kwargs)
    BG_PROBES.append(nengo.Probe(bg.input))      # created inside the current model
    return bg


class _RecordingSim(nengo.Simulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LAST["sim"] = self


nengo.networks.BasalGanglia = _bg_with_probe
nengo.Simulator = _RecordingSim


def run_job(job):
    BG_PROBES.clear()
    res = sw.run_network(job)                    # job = (idx, inst, keep_act=True, params)
    sim = LAST["sim"]
    util = np.array([sim.data[p].T for p in BG_PROBES], dtype=np.float32)   # (P, n, T)
    return job[0], res, util


# --------------------------------------------------------------------------
# Statistics of one run
# --------------------------------------------------------------------------
def components(act, inst, params):
    """Input components reconstructed from the thalamus output (same synapse
    time constant as the connections): precedence, uniqueness, cover."""
    wp, wu, wc, _, n_rest = params
    NP, N, T = act.shape
    C = sw.precedence_matrix(inst)
    limit = 0 if n_rest < 1 else 1               # as in the sweep script
    prec = np.zeros((NP, N, T)); uniq = np.zeros((NP, N, T))
    for t in range(NP):
        for t2 in range(NP):
            if t2 == t:
                continue
            if abs(t - t2) > limit:
                uniq[t2] -= wu * act[t]
            prec[t2] += wp * ((C.T if t2 > t else C) @ act[t])
    cover = wc * (1.0 - np.clip(act.sum(axis=0), 0.0, 1.0))      # (N, T), same for all positions
    return prec, uniq, np.broadcast_to(cover, (NP, N, T))


def run_stats(util, res, inst, params):
    k0 = int(sw.T_DECODE_FROM / sw.DT)
    u = util[:, :, k0:].astype(float)
    act = res["act"].astype(float)
    prec, uniq, cover = components(act, inst, params)
    pred = sw.BASELINE + prec + uniq + cover
    resid = util.astype(float) - pred
    sat = u > THR_SAT
    return {
        "strict_rate": 100 * res["strict_rate"],
        "mean_u": u.mean(), "p05": np.percentile(u, 5), "p95": np.percentile(u, 95),
        "above_sat": 100 * sat.mean(),
        "above_radius": 100 * (u > THR_RADIUS).mean(),
        "below_low": 100 * (u < THR_LOW).mean(),
        "tie_sat": 100 * (sat.sum(axis=1) >= 2).mean(),     # (position, time) points
        "prec_p95": np.percentile(np.abs(prec[:, :, k0:]), 95),
        "prec_max": np.abs(prec[:, :, k0:]).max(),
        "uniq_mean": np.abs(uniq[:, :, k0:]).mean(),
        "cover_mean": cover[:, :, k0:].mean(),
        "resid_rms": np.sqrt((resid[:, :, k0:] ** 2).mean()),
        "noise_expected": float(np.mean(sw.noise_scale(res["tt"][k0:]))),
        "hist": np.histogram(np.clip(u, BINS[0], BINS[-1] - 1e-9), bins=BINS)[0],
    }


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(runs, hists, summary, labels, path):
    cmap = plt.get_cmap("tab10")
    keys = list(labels)
    colors = {k: cmap(i) for i, k in enumerate(keys)}
    xt = [labels[k] for k in keys]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))

    ax = axes[0, 0]
    centers = (BINS[:-1] + BINS[1:]) / 2
    for k in keys:
        ax.step(centers, hists[k] / hists[k].sum(), where="mid", color=colors[k],
                label=labels[k].replace("\n", " "))
    for v, name in [(THR_LOW, "0.2"), (THR_SAT, "1.25"), (THR_RADIUS, "1.5")]:
        ax.axvline(v, color="k", ls=":", lw=1); ax.text(v, 0.9, name, transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("utility (input of the basal ganglia)")
    ax.set_ylabel("share of samples"); ax.set_title("(a) Distribution of utilities", fontsize=10)
    ax.legend(fontsize=6)

    ax = axes[0, 1]
    w = 0.26
    for j, (col, name, c) in enumerate([("above_sat", f"> {THR_SAT}", "#e4572e"),
                                        ("above_radius", f"> {THR_RADIUS}", "#f3a712"),
                                        ("below_low", f"< {THR_LOW}", "#4c78a8")]):
        ax.bar(x + (j - 1) * w, [summary.loc[k, col] for k in keys], w, color=c,
               edgecolor="k", label=name)
    ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=7)
    ax.set_ylabel("% of utility samples"); ax.legend(fontsize=8)
    ax.set_title("(b) Utilities outside the working range", fontsize=10)

    ax = axes[1, 0]
    for k in keys:
        d = runs[runs.cfg == k]
        ax.scatter(d.above_sat, d.strict_rate, s=40, color=colors[k], edgecolor="k",
                   label=labels[k].replace("\n", " "))
    rho, p = spearmanr(runs.above_sat, runs.strict_rate)
    ax.set_xlabel(f"% of utilities above {THR_SAT} (decoding window)")
    ax.set_ylabel("% of decodes strictly feasible")
    ax.set_title(f"(c) Saturation versus feasibility, runs (Spearman rho = {rho:.2f}, p = {p:.3f})",
                 fontsize=10)

    ax = axes[1, 1]
    for j, (col, name, c) in enumerate([("prec_p95", "|precedence| (95th pct)", "#4c78a8"),
                                        ("uniq_mean", "|uniqueness| (mean)", "#e4572e"),
                                        ("cover_mean", "cover (mean)", "#f3a712")]):
        ax.bar(x + (j - 1) * w, [summary.loc[k, col] for k in keys], w, color=c,
               edgecolor="k", label=name)
    ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=7)
    ax.set_ylabel("size of input component"); ax.legend(fontsize=8)
    ax.set_title("(d) Components of the input", fontsize=10)

    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
def main():
    os.makedirs(ar.FIG_DIR, exist_ok=True)
    metas, all_results = ar.load()
    df = ar.build_table(metas, all_results)
    feas, qual, hl = ar.pick_highlights(df)
    configs = {p: list(lab) for p, lab in hl.items()}
    if ADD_PREC_CONTRAST:
        base = feas.loc[0, "params"]
        for lv in sorted(df.w_prec.unique()):
            if lv != base[0]:
                configs.setdefault((lv,) + tuple(base[1:]), []).append(
                    f"Most feasible with W_prec={lv}")

    rng = np.random.default_rng(TEST_SEED)
    inst_idx = sorted(int(i) for i in rng.choice(len(metas), N_TEST, replace=False))
    print(f"Instances: {inst_idx}")

    rows, hists, labels = [], {}, {}
    with mp.Pool(sw.N_WORKERS) as pool:
        for params, lab in configs.items():
            key = ar.cfg_str(params)
            labels[key] = f"{ar.hl_label(lab)}\n{key}"
            print(f"\n{labels[key].replace(chr(10), '  ')}")
            jobs = [(i, metas[i]["instance"], True, params) for i in inst_idx]
            out = pool.map(run_job, jobs)
            hists[key] = np.zeros(len(BINS) - 1)
            mismatch = 0
            for i, res, util in out:
                stored = all_results[params][i]
                mismatch += (res["final_ms"] != stored["final_ms"]
                             or bool(res["final_strict_valid"]) != bool(stored["final_strict_valid"]))
                st = run_stats(util, res, metas[i]["instance"], params)
                hists[key] += st.pop("hist")
                rows.append({"cfg": key, "idx": i, **st})
            if mismatch:
                print(f"  WARNING: {mismatch}/{len(out)} re-runs differ from the stored results "
                      "(the probes may have changed the random seeds)")

    runs = pd.DataFrame(rows)
    runs.to_csv("input_range_test.csv", index=False)
    summary = runs.groupby("cfg", sort=False).mean(numeric_only=True)
    show = ["strict_rate", "mean_u", "p05", "p95", "above_sat", "above_radius", "below_low",
            "tie_sat", "prec_p95", "prec_max", "uniq_mean", "cover_mean"]
    print("\nPer configuration (mean over instances):")
    print(summary[show].round(3).to_string())
    print("\nDecomposition check (residual of probed input minus reconstructed components "
          "should be about the noise level):")
    print(summary[["resid_rms", "noise_expected"]].round(3).to_string())
    for col in ["above_sat", "tie_sat", "prec_p95"]:
        rho, p = spearmanr(runs[col], runs.strict_rate)
        print(f"Spearman({col}, strict_rate) over {len(runs)} runs: rho={rho:.2f}, p={p:.3f}")

    make_figure(runs, hists, summary, labels, f"{ar.FIG_DIR}/fig_input_range.png")
    print(f"\nWrote input_range_test.csv and {ar.FIG_DIR}/fig_input_range.png")


if __name__ == "__main__":
    main()