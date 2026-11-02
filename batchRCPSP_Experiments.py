"""
RCPSP priority lists with a Nengo basal-ganglia network: 100 instances.

1. Generate N_INST "interesting" instances (resource constraints bind, and
   random priority lists are rarely optimal).
2. Solve each exactly (branch and bound over precedence-feasible lists).
3. Run the neuromorphic network on each instance (in parallel).
4. Compare: neuro (final list), neuro (best list seen), random lists with the
   same decode budget, and the topological-level rule alone.
5. Save a summary dashboard and one in-depth figure for a showcase instance.

Outputs: rcpsp_summary.png, rcpsp_showcase.png, rcpsp_results.pkl
"""
import math
import multiprocessing as mp
import os
import pickle
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import nengo
import numpy as np
from scipy.optimize import linear_sum_assignment

from SGS import serial_schedule_generation

# --------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------
SEED = 42
N_INST = 40
N_ACT = 8                   # activities per instance (network size grows with this)
N_RES = 2                   # resource types
P_EDGE = 0.25               # probability of a precedence arc i -> j (i < j)
N_PROBE = 100               # random lists used to judge how "interesting" an instance is
MAX_RAND_OPT_FRAC = 0.50   # interesting: at most 20% of random lists are optimal
MIN_SPREAD = 2              # interesting: random lists differ by >= 2 in makespan
MAX_TRIES = 10000
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
USE_CACHE = False           # True: skip simulation, only redraw plots from the pickle
CACHE = "rcpsp_results.pkl"
FUN_XKCD = False             # hand-drawn style for the summary figure
INST_FILE = "rcpsp_instances.pkl"
REGENERATE = False          # True: ignore the stored instances and generate new ones

# Full-factorial grid: 3 x 3 x 3 x 2 x 2 = 108 configurations
W_VALUES = [0.2, 0.4, 0.6]      # W_PREC, W_UNIQUE, W_COVER
NEURON_VALUES = [50, 100]       # N_NEURONS
REST_VALUES = [0, 2]            # N_REST
CONFIGS = list(itertools.product(W_VALUES, W_VALUES, W_VALUES,
                                 NEURON_VALUES, REST_VALUES))

def instance_config():
    return {"SEED": SEED, "N_INST": N_INST, "N_ACT": N_ACT, "N_RES": N_RES,
            "P_EDGE": P_EDGE, "N_PROBE": N_PROBE,
            "MAX_RAND_OPT_FRAC": MAX_RAND_OPT_FRAC, "MIN_SPREAD": MIN_SPREAD,
            "N_DECODES": N_DECODES}


def load_instances():
    cfg = instance_config()
    if os.path.exists(INST_FILE) and not REGENERATE:
        with open(INST_FILE, "rb") as f:
            saved_cfg, metas = pickle.load(f)
        if saved_cfg == cfg:
            print(f"Loaded {len(metas)} stored instances from {INST_FILE}")
            return metas
        diff = {k: (saved_cfg.get(k), cfg[k]) for k in cfg if saved_cfg.get(k) != cfg[k]}
        print(f"Stored instances were made with other settings {diff}; regenerating.")

    return metas
# --------------------------------------------------------------------------
# Network parameters (same model as before)
# --------------------------------------------------------------------------
N_REST = 2
BASELINE = 0.5
W_PREC = 0.15
W_UNIQUE = 0.5
W_COVER = 0.3
W_LEVEL = 0.15
NOISE_START = 0.30
NOISE_END = 0.02
T_ANNEAL = 1.5
TAU = 0.05
N_NEURONS = 50
T_SIM = 4.0
T_DECODE_FROM = 1.0
DECODE_EVERY = 0.05
DT = 0.001
N_DECODES = len(range(int(T_DECODE_FROM / DT), int(T_SIM / DT),
                      max(1, int(DECODE_EVERY / DT))))


def noise_scale(t):
    return NOISE_END + (NOISE_START - NOISE_END) * np.exp(-t / T_ANNEAL)


# --------------------------------------------------------------------------
# Instance helpers
# --------------------------------------------------------------------------
def makespan_of(inst, order):
    return serial_schedule_generation(inst, order)[2]


def precedence_matrix(inst):
    """C[i,j] = +1 if i precedes j (transitively), -1 if reversed, else 0."""
    acts = sorted(inst["durations"])
    n = len(acts)
    idx = {a: k for k, a in enumerate(acts)}
    P = np.zeros((n, n), dtype=bool)
    for succ, preds in inst["predecessors"].items():
        for p in preds:
            P[idx[p], idx[succ]] = True
    for k in range(n):                       # Warshall closure
        P |= P[:, [k]] & P[[k], :]
    C = np.zeros((n, n))
    C[P] = 1.0
    C[P.T] = -1.0
    return C


def levels(inst):
    """Topological level (longest chain of predecessors) per activity label."""
    lev = {}
    for a in sorted(inst["durations"]):      # labels are in topological order
        ps = inst["predecessors"][a]
        lev[a] = 0 if not ps else max(lev[p] for p in ps) + 1
    return lev


def level_rule_list(inst):
    lev = levels(inst)
    return sorted(inst["durations"], key=lambda a: (lev[a], a))


def random_feasible_list(inst, rng):
    acts = sorted(inst["durations"])
    placed, out = set(), []
    while len(out) < len(acts):
        elig = [a for a in acts if a not in placed
                and all(p in placed for p in inst["predecessors"][a])]
        a = elig[int(rng.integers(len(elig)))]
        placed.add(a)
        out.append(a)
    return out


# --------------------------------------------------------------------------
# Neuromorphic model for one instance
# --------------------------------------------------------------------------
def run_network(job):
    global W_PREC, W_UNIQUE, W_COVER, N_NEURONS, N_REST
    idx_inst, inst, keep_act, params = job
    W_PREC, W_UNIQUE, W_COVER, N_NEURONS, N_REST = params
    acts = sorted(inst["durations"])
    N = len(acts)
    NP = N + N_REST
    idx = {a: k for k, a in enumerate(acts)}
    preds = {k: [idx[p] for p in inst["predecessors"][a]] for a, k in idx.items()}
    C = precedence_matrix(inst)

    level = np.zeros(N)
    for j in range(N):                       # labels are in topological order
        level[j] = 0 if not preds[j] else max(level[p] for p in preds[j]) + 1
    level_n = level / max(level.max(), 1)

    def to_labels(tr):
        return [acts[k] for k in tr]

    def makespan(tr):
        return makespan_of(inst, to_labels(tr))

    model = nengo.Network(seed=1000)
    with model:
        bgs, thals, probes = [], [], []
        for t in range(NP):
            bg = nengo.networks.BasalGanglia(N, n_neurons_per_ensemble=N_NEURONS, label=f"BG_step{t}")
            thal = nengo.networks.Thalamus(N, n_neurons_per_ensemble=N_NEURONS, label=f"Thal_step{t}")
            nengo.Connection(bg.output, thal.input, synapse=0.008)
            bgs.append(bg)
            thals.append(thal)
            probes.append(nengo.Probe(thal.output, synapse=0.05))

        baseline = nengo.Node(np.full(N, BASELINE))
        for t in range(NP):
            nengo.Connection(baseline, bgs[t].input, synapse=None)
            white = nengo.Node(
                nengo.processes.WhiteSignal(T_SIM, high=10, rms=1.0, seed=100 + t),
                size_out=N, label = f"whitenoise{t}")
            scaled = nengo.Node(lambda t_, x: x * noise_scale(t_),
                            size_in=N, size_out=N, label=f"scaled{t}")
            nengo.Connection(white, scaled, synapse=None)
            nengo.Connection(scaled, bgs[t].input, synapse=None)
        # Global coverage feedback: aggregate activity outputs
        global_cover = nengo.Node(
            lambda t_, x: W_COVER * (1.0 - np.clip(x, 0.0, 1.0)),
            size_in=N,
            size_out=N,
            label="global_coverage"
        )
        # Sum each selection-step network's output into the coverage node
        for t in range(NP):
            nengo.Connection(
                thals[t].output,
                global_cover,
                synapse=TAU
            )
        # Feed the same coverage signal back to every selection network
        for t in range(NP):
            nengo.Connection(
                global_cover,
                bgs[t].input,
                synapse=None
            )
        if N_REST < 1:
            limit = 0
        if N_REST > 1:
            limit = 1
        for t in range(NP):
            for t2 in range(NP):
                if t2 == t:
                    continue
                if abs(t - t2) > limit:          # uniqueness, neighbours exempt (resting)
                    nengo.Connection(thals[t].output, bgs[t2].input, transform=-W_UNIQUE * np.eye(N), synapse=TAU)
                T = C.T if t2 > t else C     # precedence
                nengo.Connection(thals[t].output, bgs[t2].input, 
                                 transform=W_PREC * T, synapse=TAU)


    with nengo.Simulator(model, dt=DT, progress_bar=False) as sim:
        sim.run(T_SIM)
    act = np.array([sim.data[p].T for p in probes])      # (position, activity, time)
    tt = sim.trange()

    def collapse(seq):
        seq = list(seq)
        return [c for k, c in enumerate(seq) if k == 0 or c != seq[k - 1]]

    def is_feasible(tr):
        if sorted(tr) != list(range(N)):
            return False
        pos = {a: k for k, a in enumerate(tr)}
        return all(pos[p] < pos[j] for j in range(N) for p in preds[j])

    def decode_strict(seq):
        tr = [int(c) for c in collapse(seq)]
        return tr if is_feasible(tr) else None


    def raw_order(a):
        """Assignment order BEFORE greedy repair; returns (slot per activity, list)."""
        a = np.clip(a, 0.0, None)
        rows, cols = linear_sum_assignment(-a.T)
        slot = np.empty(N, dtype=int)
        slot[rows] = cols
        return slot, [int(j) for j in np.argsort(slot)]

    def decode_repair(a):
        slot, _ = raw_order(a)
        placed, out = np.zeros(N, dtype=bool), []
        while len(out) < N:
            elig = [j for j in range(N)
                    if not placed[j] and all(placed[p] for p in preds[j])]
            j = min(elig, key=lambda j: (slot[j], j))
            placed[j] = True
            out.append(j)
        return out

    win = int(0.5 / DT)
    final = act[:, :, -win:].mean(axis=2)
    final_list = decode_repair(final)
    final_strict = decode_strict(final.argmax(axis=1).tolist()) is not None
    final_raw_feasible = is_feasible(raw_order(final)[1])         

    best_list, best_ms, best_time = None, np.inf, None
    trace_t, trace_ms, strict_hits, raw_hits = [], [], 0, 0
    step = max(1, int(DECODE_EVERY / DT))
    for k in range(int(T_DECODE_FROM / DT), act.shape[2], step):
        tr = decode_repair(act[:, :, k])
        raw_hits += is_feasible(raw_order(act[:, :, k])[1])
        ms = makespan(tr)
        trace_t.append(tt[k])
        trace_ms.append(ms)
        strict_hits += decode_strict(act[:, :, k].argmax(axis=1).tolist()) is not None
        if ms < best_ms:
            best_list, best_ms, best_time = tr, ms, tt[k]

    res = {"idx": idx_inst,
           "final_list": to_labels(final_list), "final_ms": makespan(final_list),
           "final_strict_valid": final_strict,
           "best_list": to_labels(best_list), "best_ms": best_ms, "best_time": best_time,
           "strict_rate": strict_hits / len(trace_t),
           "final_raw_feasible": final_raw_feasible,
           "raw_rate": raw_hits / len(trace_t),
           "trace_t": np.array(trace_t), "trace_ms": np.array(trace_ms)}
    if keep_act:
        res.update({"act": act.astype(np.float32), "tt": tt, "final": final})
    return res

def summarize(metas, results):
    opt = np.array([m["opt"] for m in metas], dtype=float)
    ms = {
        "Neuro\n(final list)": np.array([r["final_ms"] for r in results], dtype=float),
        "Neuro\n(best seen)": np.array([r["best_ms"] for r in results], dtype=float),
        f"Random\n(best of {N_DECODES})": np.array([m["rand_best"] for m in metas], float),
        "Level rule\n(1 list)": np.array([m["rule_ms"] for m in metas], dtype=float),
    }
    return opt, ms



def main():

    rng = np.random.default_rng(SEED)
    print("Loading instances and solving them exactly ...")
    metas = load_instances()
    all_results = {}
    with mp.Pool(N_WORKERS) as pool:
        for c, params in enumerate(CONFIGS, 1):
            print(f"\n[{c}/{len(CONFIGS)}] W_PREC, W_UNIQUE, W_COVER, "
                  f"N_NEURONS, N_REST = {params}")
            jobs = [(i, m["instance"], False, params) for i, m in enumerate(metas)]
            results = [None] * len(jobs)
            for r in pool.imap_unordered(run_network, jobs):
                results[r["idx"]] = r
            all_results[params] = results
            with open(CACHE, "wb") as f:          # saved after every config
                pickle.dump((metas, all_results), f)

            opt, ms = summarize(metas, results)
            for k, v in ms.items():
                print(f"  {k.replace(chr(10), ' '):28s} optimal: {(v == opt).sum():3d}/{len(opt)}"
                      f"   mean gap: {(100 * (v - opt) / opt).mean():5.2f}%")
            print(f"  Network alone gave a valid list in "
                  f"{100 * np.mean([r['strict_rate'] for r in results]):.1f}% of decodes")
            print(f"  Assignment feasible before repair in "
                  f"{100 * np.mean([r['raw_rate'] for r in results]):.1f}% of decodes")

if __name__ == "__main__":
    main()