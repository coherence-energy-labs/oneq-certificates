"""Coherent P with the select-phase synthesized as a PHASE POLYNOMIAL (parity network, AAM), instead of
gate-by-gate CCZ / CCP. The phase function over (k, s, ea, eb, ec) is the one derived in channels.py;
its parity expansion is computed exactly here, the parity network is synthesized by qiskit's
synth_cnot_phase_aam, and the full Stinespring circuit is Choi-verified.

Usage: python p_phasepoly.py
"""
import itertools
import json
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.synthesis import synth_cnot_phase_aam

import channels as C
import sclib as L

p = L.P_FLIP
NQ = 5   # qubit order: k=0, s=1, ea=2, eb=3, ec=4


def phi(x):
    k, s, a, b, c = x
    return (math.pi * b * s + math.pi * a * k * s + math.pi * b * k - math.pi * a * b * k
            + (math.pi / 2) * b * c * k - (math.pi / 4) * c * k - (math.pi / 2) * c * s)


xs = list(itertools.product((0, 1), repeat=NQ))
subsets = [S for S in itertools.product((0, 1), repeat=NQ) if any(S)]
A = np.array([[sum(xi * si for xi, si in zip(x, S)) % 2 for S in subsets] for x in xs], float)
y = np.array([phi(x) for x in xs])
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
assert np.abs(A @ coef - y).max() < 1e-12
terms = [(S, c) for S, c in zip(subsets, coef) if abs(math.remainder(c, 2 * math.pi)) > 1e-12]
print(f"phase polynomial: {len(terms)} parity terms, {sum(1 for S, _ in terms if sum(S) > 1)} non-linear")


def build(order_seed=None):
    tt = list(terms)
    if order_seed is not None:
        np.random.default_rng(order_seed).shuffle(tt)
    cnots = [[S[q] for S, _ in tt] for q in range(NQ)]
    angles = [float(c % (2 * math.pi)) for _, c in tt]   # qiskit AAM mishandles negative angles (measured: off by pi)
    pp = synth_cnot_phase_aam(cnots, angles)
    # MEASURED (debug_aam.py): qiskit 2.5's synth_cnot_phase_aam output can differ from its own input
    # phase polynomial by pi on a parity. Correct it exactly: expand the residual diagonal error in
    # parities; only single-qubit (degree-1) residuals are accepted and fixed with p gates.
    from qiskit.quantum_info import Operator
    U = Operator(pp).data
    if not np.allclose(np.abs(U), np.eye(2 ** NQ)):
        raise RuntimeError("AAM output is not diagonal")
    got = np.array([np.angle(U[sum(b << q for q, b in enumerate(x)), sum(b << q for q, b in enumerate(x))])
                    for x in xs])
    err = np.mod((y - got) - (y[0] - got[0]), 2 * math.pi)
    err[np.abs(err - 2 * math.pi) < 1e-9] = 0.0     # one consistent branch of the residual
    ce, *_ = np.linalg.lstsq(A, err, rcond=None)
    for S, c in zip(subsets, ce):
        cr = math.remainder(c, 2 * math.pi)
        if abs(cr) > 1e-9:
            if abs(abs(cr) - math.pi) < 1e-9:      # e^{i pi parity_S} = prod_{j in S} Z_j : local
                for j, bit in enumerate(S):
                    if bit:
                        pp.z(j)
            elif sum(S) == 1:
                pp.p(float(cr), S.index(1))
            else:
                raise RuntimeError(f"non-local AAM residual on parity {S}")
    th_a = 2 * math.asin(math.sqrt((1 + p) / 2))
    th_c = 2 * math.acos(math.sqrt(2 * p / (1 + p)))
    qc = QuantumCircuit(NQ)
    k, s, ea, eb, ec = range(NQ)
    qc.ry(th_a, ea)
    qc.h(eb)
    qc.ry(-th_c / 2, ec); qc.cx(ea, ec); qc.ry(th_c / 2, ec)
    qc.compose(pp, inplace=True)
    qc.x(s); qc.cx(ea, s)
    qc.cx(ec, k)
    return qc


JP = L.choi_from_kraus(L.horodecki_kraus())
best = None
skipped = 0
for seed in [None] + list(range(40)):
    try:
        qc = build(seed)
    except RuntimeError:
        skipped += 1
        continue
    qcc = L.compile_cz(qc, level=3)
    cz = L.count_ops(qcc).get("cz", 0)
    if best is None or cz < best[0]:
        d = L.choi_dist(L.channel_choi(qcc, [0, 1], [0, 1]), JP)
        best = (cz, seed, d)
print(f"coherent P via phase polynomial: best {best[0]} CZ (term order seed {best[1]}), Choi trace distance {best[2]:.2e};"
      f" {skipped} of 41 term orders rejected (non-local AAM residual)")
print(f"hand gate-by-gate version: {L.count_ops(L.compile_cz(C.p_coherent_circuit(), level=3)).get('cz', 0)} CZ")
json.dump({"terms": len(terms), "cz_best": best[0], "seed": best[1], "choi_dist": best[2]},
          open("p_phasepoly.json", "w"), indent=1)
