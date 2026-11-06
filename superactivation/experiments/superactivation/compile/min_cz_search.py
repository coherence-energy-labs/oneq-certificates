"""Minimal-CZ search for the small gates that dominate the counts, on their REACHABLE input subspace.

A gate only has to be correct on the inputs that can reach it inside the larger circuit (the rest
are don't-cares). For each spec we search circuits  [U3 layer] (CZ on pair_t, [U3 layer])_t=1..c  over
every placement of c CZs on undirected pairs, minimizing 1 - |sum_x <target_x| U |x>| / N_inputs
(i.e. exact up to ONE global phase). A placement is accepted only at residual < 1e-12 after
polishing; the minimal c found is an upper bound on the true minimum, and failure at c-1 over all
placements and restarts is evidence (not proof) that c-1 is impossible.

Specs
  gate_i(l)        : BE split-and-cyclic-shift gate (i) on (a, c), reachable {00, 01, 11}
  gate_ii(l, ell)  : BE gate (ii)_ell on (a, b, c), reachable {000, 001, 010, 011, 111}
  move             : controlled move |f, x, 0> -> f=0: |0, x, 0>, f=1: |1, 0, x> (erasure dilation)
Index convention: qiskit little-endian, qubit 0 = a (least significant).

Usage: python min_cz_search.py [maxc]
"""
import itertools
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(1234)


def u3(t, p, l):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -np.exp(1j * l) * s], [np.exp(1j * p) * s, np.exp(1j * (p + l)) * c]])


def layer(params, nq):
    M = np.array([[1.0 + 0j]])
    for q in reversed(range(nq)):  # little-endian: qubit nq-1 most significant
        M = np.kron(M, u3(*params[3 * q:3 * q + 3]))
    return M


def cz_mat(nq, a, b):
    d = 2 ** nq
    v = np.ones(d, dtype=complex)
    for x in range(d):
        if (x >> a) & 1 and (x >> b) & 1:
            v[x] = -1
    return np.diag(v)


def build(params, nq, placement):
    U = layer(params[:3 * nq], nq)
    off = 3 * nq
    for (a, b) in placement:
        U = layer(params[off:off + 3 * nq], nq) @ cz_mat(nq, a, b) @ U
        off += 3 * nq
    return U


def cost(params, nq, placement, ins, targets):
    U = build(params, nq, placement)
    ov = sum(np.vdot(t, U[:, x]) for x, t in zip(ins, targets))
    return 1.0 - abs(ov) / len(ins)


def search(nq, ins, targets, maxc, restarts=6, tag=""):
    pairs = list(itertools.combinations(range(nq), 2))
    for c in range(0, maxc + 1):
        best = (1.0, None, None)
        for placement in itertools.product(pairs, repeat=c):
            if False:
                continue
            npar = 3 * nq * (c + 1)
            for _ in range(restarts):
                x0 = rng.uniform(-math.pi, math.pi, npar)
                r = minimize(cost, x0, args=(nq, placement, ins, targets), method="BFGS",
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


def basis(nq, idx):
    v = np.zeros(2 ** nq, dtype=complex)
    v[idx] = 1
    return v


def idx3(a, b, c):
    return a + 2 * b + 4 * c


def spec_gate_i(l):
    # qubits (a, c) = (0, 1); index = a + 2c
    cth, sth = math.sqrt(1 / l), math.sqrt((l - 1) / l)
    ins = [0b00, 0b10, 0b11]          # (a,c) = 00, (a=0,c=1), 11
    tg = [basis(2, 0), cth * basis(2, 0b10) + sth * basis(2, 0b01), basis(2, 0b11)]
    return 2, ins, tg


def spec_gate_ii(l, ell):
    cth, sth = math.sqrt(ell / l), math.sqrt((l - ell) / l)
    ins = [idx3(0, 0, 0), idx3(0, 0, 1), idx3(0, 1, 0), idx3(0, 1, 1), idx3(1, 1, 1)]
    tg = [basis(3, idx3(0, 0, 0)), basis(3, idx3(0, 0, 1)), basis(3, idx3(0, 1, 0)),
          cth * basis(3, idx3(0, 1, 1)) + sth * basis(3, idx3(1, 1, 0)), basis(3, idx3(1, 1, 1))]
    return 3, ins, tg


def spec_move():
    # qubits (f, x, e) = (0, 1, 2)
    ins, tg = [], []
    for f in (0, 1):
        for x in (0, 1):
            ins.append(idx3(f, x, 0))
            tg.append(basis(3, idx3(f, x, 0)) if f == 0 else basis(3, idx3(1, 0, x)))
    return 3, ins, tg


if __name__ == "__main__":
    maxc = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = {}
    t0 = time.time()
    for name, spec in [("gate_i(l=5)", spec_gate_i(5)), ("gate_i(l=17)", spec_gate_i(17)),
                       ("gate_ii(l=9,ell=4)", spec_gate_ii(9, 4)), ("gate_ii(l=17,ell=2)", spec_gate_ii(17, 2)),
                       ("move", spec_move())]:
        nq, ins, tg = spec
        c, best = search(nq, ins, tg, maxc, tag=name)
        out[name] = {"min_cz_found": c, "residual": best[0], "placement": best[1]}
        print(f"{name}: minimal CZ found = {c}  ({time.time() - t0:.0f}s)", flush=True)
    with open("min_cz_search.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
