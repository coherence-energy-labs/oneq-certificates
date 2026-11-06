"""Tabulate 4-CZ implementations of the Baertschi-Eidenbenz gate (ii)_ell of SCS_l for every
(l, ell) with 3 <= l <= LMAX, 2 <= ell <= l-1, on the reachable input subspace
{000, 001, 010, 011, 111} of (a, b, c) = (l-ell, l-ell+1, l) (positions, 1-indexed), exact up to one
global phase. Placement found by min_cz_search.py: CZ(a,c) CZ(a,b) CZ(b,c) CZ(a,c), U3 layers between.
3 CZ was not reachable in 27 placements x 6 restarts (min_cz_search.log).

Output: gate_ii_num4.json {"l,ell": [45 U3 angles], residual}.
Usage: python gate_tables.py [LMAX=24]
"""
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import minimize

import min_cz_search as S

PLACEMENT = ((0, 2), (0, 1), (1, 2), (0, 2))
rng = np.random.default_rng(2024)


def solve(l, ell, warm=None, tries=40):
    nq, ins, tg = S.spec_gate_ii(l, ell)
    best = (1.0, None)
    starts = ([warm] if warm is not None else []) + [None] * tries
    for x0 in starts:
        if x0 is None:
            x0 = rng.uniform(-math.pi, math.pi, 3 * nq * (len(PLACEMENT) + 1))
        r = minimize(S.cost, x0, args=(nq, PLACEMENT, ins, tg), method="BFGS",
                     options={"gtol": 1e-14, "maxiter": 6000})
        if r.fun < best[0]:
            best = (r.fun, r.x)
        if best[0] < 1e-14:
            break
    return best


if __name__ == "__main__":
    LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    out = {}
    t0 = time.time()
    warm = None
    worst = 0.0
    for l in range(3, LMAX + 1):
        for ell in range(2, l):
            res, x = solve(l, ell, warm)
            if res > 1e-12:
                res, x = solve(l, ell, None, tries=200)
            warm = x
            worst = max(worst, res)
            out[f"{l},{ell}"] = {"params": list(map(float, x)), "residual": float(res)}
        print(f"l={l} done, worst residual so far {worst:.2e} ({time.time() - t0:.0f}s)", flush=True)
    with open("gate_ii_num4.json", "w") as fh:
        json.dump({"placement": PLACEMENT, "gates": out, "worst_residual": worst}, fh)
    print("worst residual", worst)
