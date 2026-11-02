"""
Summary of the most feasible configuration, averaged over all instances.

  * the table row (Selected as, parameters, strict / raw decode rate, final strict /
    raw feasible, optimal best, gap best) from the stored sweep results
  * the averages of the showcase criteria over all instances: in how many of the
    P positions the winning thalamus output of the final list is at least
    THR_WIN, mean margin between winner and runner-up (clarity), mean winning
    output, and the share of strictly feasible final lists. These need the
    activations, so the configuration is re-run once on every instance.

Outputs: best_config_summary.csv, best_config_summary.tex (and console output)
Run from the folder with the sweep script, SGS.py, the pickle and the module
named in AR_MODULE.
"""
import importlib
import multiprocessing as mp

import numpy as np
import pandas as pd

AR_MODULE = "analysis"                 # file name (no .py) of analyze_rcpsp.py in your folder
THR_WIN = 0.9                       # "winning thalamus output is at least ..."
N_REST_FILTER = None                # e.g. 0 or 2 to restrict the choice to that n_rest
CSV_OUT = "best_config_summary.csv"
TEX_OUT = "best_config_summary.tex"

ar = importlib.import_module(AR_MODULE)
sw = ar.sw


def find_best(df):
    sub = df if N_REST_FILTER is None else df[df.n_rest == N_REST_FILTER]
    ranked = ar.sort_by(sub, ar.FEAS_KEYS, ar.FEAS_ASC)      # strict, then raw, then final strict
    return ranked.loc[0]


def showcase_stats(runs):
    """Showcase criteria per instance, averaged over instances."""
    n_win, n_pos, clar, win, valid = [], [], [], [], []
    for r in runs:
        final = np.asarray(r["final"], float)                 # (positions, activities)
        s = np.sort(final, axis=1)
        top = s[:, -1]
        n_win.append(int((top >= THR_WIN).sum()))
        n_pos.append(final.shape[0])
        clar.append(float((s[:, -1] - s[:, -2]).mean()))
        win.append(float(top.mean()))
        valid.append(bool(r["final_strict_valid"]))
    n_win, n_pos = np.array(n_win), np.array(n_pos)
    return {
        "n_instances": len(runs),
        "positions": float(n_pos.mean()),
        "winners_above_thr": float(n_win.mean()),
        "winners_above_thr_min": int(n_win.min()),
        "winners_above_thr_max": int(n_win.max()),
        "all_positions_above_thr": int((n_win == n_pos).sum()),
        "clarity": float(np.mean(clar)),
        "mean_winner_output": float(np.mean(win)),
        "final_strict_valid": 100 * float(np.mean(valid)),
    }


def write_tex(row, st, path):
    p = lambda x: f"{x:.1f}"
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lccccc|cccccc}",
         r"Selected as & $W_{\text{prec}}$ & $W_{\text{unique}}$ & $W_{\text{cover}}$ & Neurons & $n_{\text{rest}}$ & "
         r"Strict (\%) & Raw (\%) & Final strict (\%) & Final raw (\%) & Opt.\ best (\%) & Gap best (\%) \\ \hline",
         rf"Most feasible & {row['w_prec']} & {row['w_unique']} & {row['w_cover']} & {int(row['neurons'])} & "
         rf"{int(row['n_rest'])} & {p(row['strict'])} & {p(row['raw'])} & {row['final_strict']:.0f} & "
         rf"{row['final_raw']:.0f} & {row['best_opt']:.0f} & {p(row['best_gap'])} \\",
         r"\end{tabular}",
         rf"\caption{{The most feasible configuration (mean over {st['n_instances']} instances).}}",
         r"\label{tab:best-config}", r"\end{table}", "",
         rf"Averaged over the {st['n_instances']} instances, the winning thalamus output of the final list is at "
         rf"least ${THR_WIN}$ in {st['winners_above_thr']:.1f} of {st['positions']:.0f} positions "
         rf"(range {st['winners_above_thr_min']}--{st['winners_above_thr_max']}); in "
         rf"{st['all_positions_above_thr']} of {st['n_instances']} instances this holds for all positions. The mean "
         rf"margin between the winning and the second highest output is {st['clarity']:.2f}, the mean winning output "
         rf"is {st['mean_winner_output']:.2f}, and {st['final_strict_valid']:.0f}\% of the final lists are strictly "
         rf"feasible."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    metas, all_results = ar.load()
    df = ar.build_table(metas, all_results)
    row = find_best(df)
    params = row["params"]
    print(f"Most feasible configuration: {ar.cfg_str(params)}")

    jobs = [(i, m["instance"], True, params) for i, m in enumerate(metas)]
    with mp.Pool(sw.N_WORKERS) as pool:
        runs = pool.map(sw.run_network, jobs)
    mismatch = sum(bool(r["final_strict_valid"]) != bool(all_results[params][i]["final_strict_valid"])
                   for i, r in enumerate(runs))
    if mismatch:
        print(f"WARNING: {mismatch} re-runs differ from the stored results")
    st = showcase_stats(runs)

    out = {"config": ar.cfg_str(params), **{k: row[k] for k in
           ["strict", "raw", "final_strict", "final_raw", "best_opt", "best_gap"]}, **st}
    pd.DataFrame([out]).to_csv(CSV_OUT, index=False)
    print(pd.Series(out).to_string())
    write_tex(row, st, TEX_OUT)
    print(f"\nWrote {CSV_OUT} and {TEX_OUT}")


if __name__ == "__main__":
    main()
    