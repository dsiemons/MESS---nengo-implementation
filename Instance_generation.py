import math
import multiprocessing as mp
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import nengo
import numpy as np

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
INST_FILE = "rcpsp_instances.pkl"
REGENERATE = True          # True: ignore the stored instances and generate new ones
T_SIM = 4.0
T_DECODE_FROM = 1.0
DECODE_EVERY = 0.05
DT = 0.001
N_DECODES = len(range(int(T_DECODE_FROM / DT), int(T_SIM / DT),
                      max(1, int(DECODE_EVERY / DT))))


def makespan_of(inst, order):
    return serial_schedule_generation(inst, order)[2]

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

def instance_config():
    return {"SEED": SEED, "N_INST": N_INST, "N_ACT": N_ACT, "N_RES": N_RES,
            "P_EDGE": P_EDGE, "N_PROBE": N_PROBE,
            "MAX_RAND_OPT_FRAC": MAX_RAND_OPT_FRAC, "MIN_SPREAD": MIN_SPREAD,
            "N_DECODES": N_DECODES}


def load_or_generate_instances():
    cfg = instance_config()
    if os.path.exists(INST_FILE) and not REGENERATE:
        with open(INST_FILE, "rb") as f:
            saved_cfg, metas = pickle.load(f)
        if saved_cfg == cfg:
            print(f"Loaded {len(metas)} stored instances from {INST_FILE}")
            return metas
        diff = {k: (saved_cfg.get(k), cfg[k]) for k in cfg if saved_cfg.get(k) != cfg[k]}
        print(f"Stored instances were made with other settings {diff}; regenerating.")

    rng = np.random.default_rng(SEED)
    print("Generating instances and solving them exactly ...")
    metas = generate_instances(N_INST, rng)
    for m in metas:                           # baselines (same decode budget)
        inst = m["instance"]
        m["rand_best"] = min(makespan_of(inst, random_feasible_list(inst, rng))
                             for _ in range(N_DECODES))
        m["rule_ms"] = makespan_of(inst, level_rule_list(inst))
    with open(INST_FILE, "wb") as f:
        pickle.dump((cfg, metas), f)
    print(f"Saved {len(metas)} instances to {INST_FILE}")
    return metas


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


def random_instance(rng, resource_num):
    """Arcs only go from lower to higher label, so labels are a topological order."""
    n = N_ACT
    acts = list(range(1, n + 1))
    durations = {a: int(rng.integers(1, 7)) for a in acts}
    demands = {a: [int(rng.integers(1, 5))] +
                  [int(rng.integers(0, 5)) for _ in range(resource_num - 1)] for a in acts}
    capacities = []
    for k in range(resource_num):
        col = np.array([demands[a][k] for a in acts])
        capacities.append(int(max(col.max(), math.ceil(2.0 * col.mean()))))
    predecessors = {a: [i for i in range(1, a) if rng.random() < P_EDGE] for a in acts}
    return {"durations": durations, "demands": demands,
            "capacities": capacities, "predecessors": predecessors}


# --------------------------------------------------------------------------
# Exact solver: DFS over precedence-feasible lists with serial-SGS placement.
# The serial SGS over all precedence-feasible lists generates all active
# schedules, which contain an optimal one, so this is exact.
# --------------------------------------------------------------------------
def solve_exact(inst):
    acts = sorted(inst["durations"])
    n = len(acts)
    idx = {a: k for k, a in enumerate(acts)}
    d = [inst["durations"][a] for a in acts]
    preds = [[idx[p] for p in inst["predecessors"][a]] for a in acts]
    succs = [[] for _ in range(n)]
    for j in range(n):
        for p in preds[j]:
            succs[p].append(j)
    dem = np.array([inst["demands"][a] for a in acts])
    cap = np.array(inst["capacities"])
    H = sum(d)

    tail = [0] * n                           # longest path from start of j to the end
    for j in reversed(range(n)):
        tail[j] = d[j] + max((tail[s] for s in succs[j]), default=0)
    lb_cpm = max(tail)
    lb_work = max(math.ceil((dem[:, k] * np.array(d)).sum() / cap[k])
                  for k in range(len(cap)))
    lb = max(lb_cpm, lb_work)

    usage = np.zeros((H + 1, len(cap)), dtype=int)
    start, end = [-1] * n, [0] * n
    best = {"ms": H + 1, "order": None}
    order = []

    def dfs(cur_ms):
        if best["ms"] <= lb:
            return
        if len(order) == n:
            if cur_ms < best["ms"]:
                best["ms"], best["order"] = cur_ms, list(order)
            return
        est, bound = [0] * n, cur_ms
        for j in range(n):
            if start[j] < 0:
                est[j] = max((end[p] if start[p] >= 0 else est[p] + d[p]
                              for p in preds[j]), default=0)
                bound = max(bound, est[j] + tail[j])
        if bound >= best["ms"]:
            return
        cands = sorted((est[j], j) for j in range(n)
                       if start[j] < 0 and all(start[p] >= 0 for p in preds[j]))
        for e, j in cands:
            t = e
            while not (usage[t:t + d[j]] + dem[j] <= cap).all():
                t += 1
            usage[t:t + d[j]] += dem[j]
            start[j], end[j] = t, t + d[j]
            order.append(j)
            dfs(max(cur_ms, end[j]))
            order.pop()
            start[j], end[j] = -1, 0
            usage[t:t + d[j]] -= dem[j]
            if best["ms"] <= lb:
                return

    dfs(0)
    return best["ms"], [acts[k] for k in best["order"]], lb_cpm, lb


def generate_instances(n_inst, rng):
    out, tries = [], 0
    type_one = round(n_inst / 2)
    while len(out) < type_one and tries < MAX_TRIES:
        tries += 1
        inst = random_instance(rng, N_RES)
        opt, opt_list, lb_cpm, lb = solve_exact(inst)
        if opt <= lb_cpm:                    # resources do not bind
            continue
        assert makespan_of(inst, opt_list) == opt, "exact solver / SGS mismatch"
        ms = np.array([makespan_of(inst, random_feasible_list(inst, rng))
                       for _ in range(N_PROBE)])
        frac = float((ms == opt).mean())
        if frac > MAX_RAND_OPT_FRAC or ms.max() - ms.min() < MIN_SPREAD:
            continue
        out.append({"instance": inst, "opt": opt, "opt_list": opt_list,
                    "lb_cpm": lb_cpm, "rand_frac_opt": frac})
        if len(out) % 10 == 0:
            print(f"  generated {len(out)}/{n_inst} instances ({tries} candidates)")
    tries_2 = 0
    while len(out)  < n_inst and tries_2 < MAX_TRIES:
        tries_2 += 1
        new_res = N_RES + 1
        inst = random_instance(rng, new_res)
        opt, opt_list, lb_cpm, lb = solve_exact(inst)
        if opt <= lb_cpm:                    # resources do not bind
            continue
        assert makespan_of(inst, opt_list) == opt, "exact solver / SGS mismatch"
        ms = np.array([makespan_of(inst, random_feasible_list(inst, rng))
                       for _ in range(N_PROBE)])
        frac = float((ms == opt).mean())
        if frac > MAX_RAND_OPT_FRAC or ms.max() - ms.min() < MIN_SPREAD:
            continue
        out.append({"instance": inst, "opt": opt, "opt_list": opt_list,
                    "lb_cpm": lb_cpm, "rand_frac_opt": frac})
        if len(out) % 10 == 0:
            print(f"  generated {len(out)}/{n_inst} instances ({tries} candidates)")
    if len(out) < n_inst:
        print(f"WARNING: only {len(out)} interesting instances found; "
              f"relax MAX_RAND_OPT_FRAC / MIN_SPREAD or change P_EDGE.")
    return out

def main():
    rng = np.random.default_rng(SEED)
    print("Generating or loading instances and solving them exactly ...")
    metas = load_or_generate_instances()

if __name__ == "__main__":
    main()
