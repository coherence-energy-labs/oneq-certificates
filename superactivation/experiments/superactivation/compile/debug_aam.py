"""Debug: does qiskit's synth_cnot_phase_aam implement the requested phase polynomial? Compares the
diagonal of its output against sum_S theta_S parity_S(x), for the P select-phase terms.

Usage: python debug_aam.py
"""
import itertools
import math

import numpy as np
from qiskit.quantum_info import Operator
from qiskit.synthesis import synth_cnot_phase_aam

NQ = 5


def phi(x):
    k, s, a, b, c = x
    return (math.pi * b * s + math.pi * a * k * s + math.pi * b * k - math.pi * a * b * k
            + (math.pi / 2) * b * c * k - (math.pi / 4) * c * k - (math.pi / 2) * c * s)


xs = list(itertools.product((0, 1), repeat=NQ))
subsets = [S for S in itertools.product((0, 1), repeat=NQ) if any(S)]
A = np.array([[sum(xi * si for xi, si in zip(x, S)) % 2 for S in subsets] for x in xs], float)
y = np.array([phi(x) for x in xs])
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
terms = [(S, c % (2 * math.pi)) for S, c in zip(subsets, coef) if abs(math.remainder(c, 2 * math.pi)) > 1e-12]
cn = [[S[q] for S, _ in terms] for q in range(NQ)]
ang = [float(c) for _, c in terms]
pp = synth_cnot_phase_aam(cn, ang)
U = Operator(pp).data
idx = lambda x: sum(b << q for q, b in enumerate(x))
tgt_terms = np.array([sum(c * (sum(xi * si for xi, si in zip(x, S)) % 2) for S, c in terms) for x in xs])
got = np.array([np.angle(U[idx(x), idx(x)]) for x in xs])
e1 = np.angle(np.exp(1j * (got - tgt_terms)))
e2 = np.angle(np.exp(1j * (tgt_terms - y)))
print("AAM output vs its own term list (should be constant):", np.round(e1 - e1[0], 3))
print("term list vs phi (should be constant):", np.round(e2 - e2[0], 3))
print("permutation part identity:", np.allclose(np.abs(U), np.eye(32)))
