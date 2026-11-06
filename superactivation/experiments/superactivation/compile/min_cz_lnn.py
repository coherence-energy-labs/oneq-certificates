"""Minimal-CZ search for a LINE-COMPATIBLE BE gate: on three qubits in line order a - b - c (CZ allowed
only on (a,b) and (b,c)), implement  SWAP(b,c) o gate(ii)_ell  on the reachable inputs
{000, 001, 010, 011, 111}. The trailing SWAP moves the travelling qubit c next to the next gate's
a', so a whole SCS_l runs on nearest neighbours (see dicke_lnn.py). Also: SWAP(a,c) o SWAP(b,c)
o gate(ii) for the LAST gate of an SCS (the travelling qubit must end in front of qubit 1).

Usage: python min_cz_lnn.py [maxc]  -> min_cz_lnn.json (minimal counts + placements)
"""
import itertools
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import minimize

import min_cz_search as S

rng = np.random.default_rng(77)
LINE_PAIRS = [(0, 1), (1, 2)]


def swap_bc(v):
    T = v.reshape(2, 2, 2)          # axes (c, b, a) in little-endian reshape
    return T.transpose(1, 0, 2).reshape(-1)


def swap_ac(v):
    T = v.reshape(2, 2, 2)
    return T.transpose(2, 1, 0).reshape(-1)


def spec_lnn(l, ell, last=False):
    nq, ins, tg = S.spec_gate_ii(l, ell)
    tg2 = [swap_bc(t) for t in tg]
    if last:
        tg2 = [swap_ac(t) for t in tg2]
    return nq, ins, tg2


def search(nq, ins, tg, maxc, restarts=8, tag=""):
    for c in range(0, maxc + 1):
        best = (1.0, None, None)
        for placement in itertools.product(LINE_PAIRS, repeat=c):
            npar = 3 * nq * (c + 1)
            for _ in range(restarts):
                r = minimize(S.cost, rng.uniform(-math.pi, math.pi, npar), args=(nq, placement, ins, tg),
                             method="BFGS", options={"gtol": 1e-12, "maxiter": 4000})
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
    maxc = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    out = {}
    t0 = time.time()
    for (l, ell, last) in ((9, 4, False), (17, 2, False), (9, 8, True)):
        nq, ins, tg = spec_lnn(l, ell, last)
        c, best = search(nq, ins, tg, maxc, tag=f"lnn l={l} ell={ell} last={last}")
        out[f"lnn(l={l},ell={ell},last={last})"] = {"min_cz_found": c, "residual": best[0], "placement": best[1]}
        print(f"lnn l={l} ell={ell} last={last}: minimal CZ = {c} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open("min_cz_lnn.json", "w"), indent=1, default=str)
