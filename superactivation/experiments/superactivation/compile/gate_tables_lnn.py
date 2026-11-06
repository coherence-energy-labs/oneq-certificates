"""Tabulate LINE-COMPATIBLE 7-CZ implementations of  SWAP(b,c) o gate(ii)_ell  for every (l, ell),
3 <= l <= LMAX, 2 <= ell <= l-1, CZs only on (a,b) and (b,c). Placement found by min_cz_lnn.py
(7 CZ; 6 not reachable in 64 placements x 8 restarts):
   CZ(a,b) CZ(b,c) CZ(a,b) CZ(b,c) CZ(a,b) CZ(b,c) CZ(a,b)
Fitted by Levenberg-Marquardt on the AMPLITUDE residual (warm-started along ell), accepted < 1e-13.

Usage: python gate_tables_lnn.py [LMAX=24]  -> gate_ii_lnn7.json
"""
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import least_squares

import min_cz_lnn as M
import min_cz_search as S

PL = ((0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2), (0, 1))
rng = np.random.default_rng(4242)


def fit(l, ell, warm=None, tries=200):
    nq, ins, tg = M.spec_lnn(l, ell)

    def res(z):
        U = S.build(z[:-1], nq, PL)
        r = np.concatenate([U[:, x] - np.exp(1j * z[-1]) * t for x, t in zip(ins, tg)])
        return np.concatenate([r.real, r.imag])

    from scipy.optimize import minimize
    best = (1.0, None)
    # 1) direct LM from the nearest solved gates (in ell/l): cheap and usually enough
    for w in (warm or []):
        sol = least_squares(res, np.array(w), method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
        err = float(np.abs(res(sol.x)).max())
        if err < best[0]:
            best = (err, sol.x)
        if err < 1e-13:
            return best
    starts = [np.array(w)[:-1] for w in (warm or [])] + [None] * tries
    npar = 3 * nq * (len(PL) + 1)
    for x0 in starts:
        if x0 is None:
            x0 = rng.uniform(-math.pi, math.pi, npar)
        # stage 1: fidelity objective (fast, as in the search); stage 2: amplitude-level LM polish
        r = minimize(S.cost, x0, args=(nq, PL, ins, tg), method="BFGS", options={"gtol": 1e-12, "maxiter": 4000})
        if r.fun > 1e-9:
            continue
        U = S.build(r.x, nq, PL)
        ov = sum(np.vdot(t, U[:, x]) for x, t in zip(ins, tg))
        sol = least_squares(res, np.concatenate([r.x, [np.angle(ov)]]), method="lm", xtol=1e-15, ftol=1e-15,
                            gtol=1e-15, max_nfev=6000)
        err = float(np.abs(res(sol.x)).max())
        if err < best[0]:
            best = (err, sol.x)
        if err < 1e-13:
            break
    return best


if __name__ == "__main__":
    LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    out, worst, t0 = {}, 0.0, time.time()
    solved = []                       # (ell/l, full parameter vector incl. phase)
    for l in range(3, LMAX + 1):
        for ell in range(2, l):
            ratio = ell / l
            cand = [w for _, w in sorted(solved, key=lambda t: abs(t[0] - ratio))[:6]]
            err, x = fit(l, ell, cand)
            solved.append((ratio, list(map(float, x))))
            worst = max(worst, err)
            out[f"{l},{ell}"] = {"params": list(map(float, x[:-1])), "amplitude_error": err}
        print(f"l={l}: worst amplitude error so far {worst:.2e} ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"placement": PL, "gates": out, "worst_amplitude_error": worst}, open("gate_ii_lnn7.json", "w"))
    print("worst", worst)
