"""Relaxed minimal-CZ search: the BE gates only need to be correct up to phases that are COMMON TO A
WHOLE UNARY LEVEL, because such phases commute to the front of U_n and are absorbed for free into the
amplitude-loading network W (single-qubit prefix phases in the unary encoding).

Groups of reachable inputs whose relative phase is physical (must agree) vs free:
  gate (ii)_ell of SCS_l on (a,b,c):  {000, 001, 010} share one phase (each lower level has two
     components among them); {111} alone (levels > ell, one component at this time); {011} alone
     (level ell, rotated into its two components with one common phase).
  gate (i) of SCS_l on (a,c): {00}, {01}, {11} each alone.
Objective: 1 - sum_groups |sum_{x in group} <t_x|U|x>| / N.

Usage: python min_cz_relaxed.py
"""
import itertools
import json
import math

import numpy as np
from scipy.optimize import minimize

import min_cz_search as S

rng = np.random.default_rng(99)


def cost_groups(params, nq, placement, groups):
    U = S.build(params, nq, placement)
    tot = 0.0
    n = 0
    for grp in groups:
        ov = sum(np.vdot(t, U[:, x]) for x, t in grp)
        tot += abs(ov)
        n += len(grp)
    return 1.0 - tot / n


def search(nq, groups, maxc, restarts=10, tag=""):
    pairs = list(itertools.combinations(range(nq), 2))
    for c in range(0, maxc + 1):
        best = (1.0, None, None)
        for placement in itertools.product(pairs, repeat=c):
            npar = 3 * nq * (c + 1)
            for _ in range(restarts):
                r = minimize(cost_groups, rng.uniform(-math.pi, math.pi, npar),
                             args=(nq, placement, groups), method="BFGS",
                             options={"gtol": 1e-12, "maxiter": 4000})
                if r.fun < best[0]:
                    best = (r.fun, placement, r.x)
                if r.fun < 1e-13:
                    break
            if best[0] < 1e-12:
                break
        print(f"  [{tag}] c={c}: best residual {best[0]:.3e} placement {best[1]}", flush=True)
        if best[0] < 1e-12:
            return c, best
    return None, best


if __name__ == "__main__":
    out = {}
    for l in (5, 17):
        nq, ins, tg = S.spec_gate_i(l)
        groups = [[(x, t)] for x, t in zip(ins, tg)]
        c, best = search(nq, groups, 2, tag=f"gate_i relaxed l={l}")
        out[f"gate_i_relaxed(l={l})"] = {"min_cz_found": c, "residual": best[0], "placement": best[1]}
    for (l, ell) in ((9, 4), (17, 2)):
        nq, ins, tg = S.spec_gate_ii(l, ell)
        pairs = list(zip(ins, tg))
        groups = [pairs[0:3], [pairs[4]], [pairs[3]]]
        c, best = search(nq, groups, 4, tag=f"gate_ii relaxed l={l} ell={ell}")
        out[f"gate_ii_relaxed(l={l},ell={ell})"] = {"min_cz_found": c, "residual": best[0], "placement": best[1]}
    with open("min_cz_relaxed.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
