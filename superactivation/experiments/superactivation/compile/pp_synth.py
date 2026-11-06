"""Coherent-P select phase as a phase polynomial, synthesized with qiskit's AAM parity network.

The phase over (k, s, ea, eb, ec) (qubits 0..4) is the one derived in channels.p_coherent_circuit;
its exact parity expansion has 14 terms (10 non-linear). MEASURED: qiskit 2.5's synth_cnot_phase_aam
returned a circuit whose diagonal differs from its own input polynomial by pi on one parity; since
e^{i pi parity_S} = prod_{j in S} Z_j is local, the residual is computed and cancelled exactly with
single-qubit gates (anything non-local raises). The caller verifies the Choi matrix.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import synth_cnot_phase_aam

NQ = 5


def phi(x):
    k, s, a, b, c = x
    return (math.pi * b * s + math.pi * a * k * s + math.pi * b * k - math.pi * a * b * k
            + (math.pi / 2) * b * c * k - (math.pi / 4) * c * k - (math.pi / 2) * c * s)


XS = list(itertools.product((0, 1), repeat=NQ))
SUBSETS = [S for S in itertools.product((0, 1), repeat=NQ) if any(S)]
A = np.array([[sum(xi * si for xi, si in zip(x, S)) % 2 for S in SUBSETS] for x in XS], float)
Y = np.array([phi(x) for x in XS])
_coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
TERMS = [(S, c) for S, c in zip(SUBSETS, _coef) if abs(math.remainder(c, 2 * math.pi)) > 1e-12]


def select_phase_circuit(order_seed=None) -> QuantumCircuit:
    tt = list(TERMS)
    if order_seed is not None:
        np.random.default_rng(order_seed).shuffle(tt)
    cnots = [[S[q] for S, _ in tt] for q in range(NQ)]
    angles = [float(c % (2 * math.pi)) for _, c in tt]
    pp = synth_cnot_phase_aam(cnots, angles)
    U = Operator(pp).data
    if not np.allclose(np.abs(U), np.eye(2 ** NQ)):
        raise RuntimeError("AAM output is not diagonal")
    ix = [sum(b << q for q, b in enumerate(x)) for x in XS]
    got = np.array([np.angle(U[i, i]) for i in ix])
    err = np.mod((Y - got) - (Y[0] - got[0]), 2 * math.pi)
    err[np.abs(err - 2 * math.pi) < 1e-9] = 0.0
    ce, *_ = np.linalg.lstsq(A, err, rcond=None)
    for S, c in zip(SUBSETS, ce):
        cr = math.remainder(c, 2 * math.pi)
        if abs(cr) > 1e-9:
            if abs(abs(cr) - math.pi) < 1e-9:
                for j, bit in enumerate(S):
                    if bit:
                        pp.z(j)
            elif sum(S) == 1:
                pp.p(float(cr), S.index(1))
            else:
                raise RuntimeError(f"non-local AAM residual on parity {S}")
    return pp
