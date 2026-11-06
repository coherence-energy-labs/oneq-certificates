"""Polish gate_ii_num4.json with a Levenberg-Marquardt fit of the AMPLITUDE residual
(U|x> - e^{i phi} t_x over the reachable inputs). The fidelity objective used by the search bottoms
out at ~1e-15, which only certifies amplitudes to ~1e-7; this step drives the amplitude error itself
to machine precision so the downstream statevector checks at 1e-9 are meaningful.

Usage: python polish_gates.py
"""
import json
import time

import numpy as np
from scipy.optimize import least_squares

import min_cz_search as S

with open("gate_ii_num4.json") as fh:
    tab = json.load(fh)
PL = [tuple(p) for p in tab["placement"]]
worst = 0.0
t0 = time.time()
for key, ent in tab["gates"].items():
    l, ell = map(int, key.split(","))
    nq, ins, tg = S.spec_gate_ii(l, ell)
    x0 = np.array(ent["params"])
    U = S.build(x0, nq, PL)
    ov = sum(np.vdot(t, U[:, x]) for x, t in zip(ins, tg))
    phi0 = np.angle(ov)

    def res(z):
        U = S.build(z[:-1], nq, PL)
        r = np.concatenate([U[:, x] - np.exp(1j * z[-1]) * t for x, t in zip(ins, tg)])
        return np.concatenate([r.real, r.imag])

    sol = least_squares(res, np.concatenate([x0, [phi0]]), method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                        max_nfev=20000)
    amp = float(np.max(np.abs(res(sol.x))))
    ent["params"] = list(map(float, sol.x[:-1]))
    ent["amplitude_error"] = amp
    worst = max(worst, amp)
tab["worst_amplitude_error"] = worst
with open("gate_ii_num4.json", "w") as fh:
    json.dump(tab, fh)
print(f"polished {len(tab['gates'])} gates; worst amplitude error {worst:.2e} ({time.time() - t0:.0f}s)")
