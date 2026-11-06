"""The LEAN two-use protocol: every erasure branch needs at most ONE quantum decode.

Same maps as hardware_design.py, compiled per branch by exact identities (each one checked below by exact simulation):
  both kept   decode use 0 (measure-and-correct); use 1's key COPY reaches Bob intact through A, so its Z bit is
              exact. Bob never uses use 1's P output or shields, and tracing an unused output of a channel equals
              never applying it (P is trace preserving), so they are dropped.
  one erased  decode the kept use; the erased use's key goes into P dephased (its copy was lost) with a maximally
              mixed shield, and Bob only reads its Z value. P's ONLY effect on that bit is the sector-1 key flip,
              probability p: an exact classical equivalent with zero two-qubit gates.
  both erased the two Z bits, each flipped with probability p: only the code's CNOT remains.
The decoded use runs P's REAL gates (the randomized Clifford(+T) family), its Bell pairs, the key copy and the
decoder. Bob's logical read-out uses single-qubit measurements (Z_L = X_k1, X_L = Z_k1 Z_k2, Y_L = -Y_k1 Z_k2), so no
two-qubit decoding gates.

  python lean_protocol.py check       exact F_e = (7 - 6p + 2p^2)/8 at m = 1, 2 (the lean maps are the same maps)
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hardware_design as hd  # noqa: E402

PAULI = hd.PAULI
H2 = hd.H2
PHI = hd.PHI


class Lean:
    """qubits: R (reference) = 0; key0 = 1, key1 = 2, copy0 = 3, copy1 = 4, s[t] = 5 + t, a[t] = 5 + m + t
    (only use 0 is ever decoded quantumly; when z = (1, 0) the roles of the uses are swapped by symmetry)"""

    def __init__(self, m):
        self.m = m
        self.n = 5 + 2 * m

    def s(self, t):
        return 5 + t

    def a(self, t):
        return 5 + self.m + t


def lean_circuit(m, kind, choice, beta, flips=(0, 0)):
    """kind: 'kept_kept' or 'kept_erased' (use 0 decoded) or 'erased_erased'. choice: P's sampled unitary on use 0.
    flips: the classical sector-1 key flips of the erased uses (sampled with probability p each)."""
    from qiskit import QuantumCircuit
    L = Lean(m)
    qc = QuantumCircuit(L.n)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.h(1)
    qc.h(2)
    if kind == "erased_erased":
        for q, f in ((1, flips[0]), (2, flips[1])):
            if f:
                qc.x(q)
        return qc
    # use 0: Bell pairs, key copy, P (real gates), then the decoder
    for t in range(m):
        qc.h(L.a(t))
        qc.cx(L.a(t), L.s(t))
    qc.cx(1, 3)
    _apply_p_mapped(qc, m, choice, key=1, shields=[L.s(t) for t in range(m)])
    if kind == "kept_kept":
        qc.cx(2, 4)                                      # use 1: the copy only (Bob reads it in Z)
    elif flips[1]:
        qc.x(2)                                          # use 1 erased: P's sector-1 flip of the dephased key
    qc.cx(1, 3)
    for t in range(m):
        qc.cx(L.a(t), L.s(t))
        qc.ch(L.s(t), L.a(t))
        if beta:
            qc.ry(-math.pi / 4, L.a(t))
    return qc


def _apply_p_mapped(qc, m, choice, key, shields):
    kind, lab = choice
    if kind == "s0":
        for t, g in enumerate(lab):
            if g != "I":
                getattr(qc, g.lower())(shields[t])
        if lab.count("Y") % 2:
            qc.z(key)
        for t, g in enumerate(lab):
            if g in "IZ":
                qc.cz(key, shields[t])
    else:
        phase = 0.0
        for t, ct in enumerate(lab):
            (qc.s if ct == 0 else qc.sdg)(shields[t])
            phase += math.pi / 4 if ct == 0 else -math.pi / 4
            qc.cz(key, shields[t])
        qc.p(phase, key)
        qc.x(key)


def lean_fidelity(m, kind, psi, beta):
    """sum over Bob's selected outcomes of P x F_e. Measured: copy0 (l), a[t] (eigenvalues), and the bit of use 1
    (copy1 if kept, key1 if erased); both erased: key0, key1. Kept: R and key0. Traced: the rest."""
    L = Lean(m)
    n = L.n
    T = np.moveaxis(psi.reshape([2] * n), list(range(n)), list(reversed(range(n))))
    if kind == "erased_erased":
        meas, keep = [1, 2], [0]
    else:
        meas = [3] + [L.a(t) for t in range(m)] + [4 if kind == "kept_kept" else 2]
        keep = [0, 1]
    rest = [q for q in range(n) if q not in meas and q not in keep]
    A = np.transpose(T, meas + keep + rest).reshape(2 ** len(meas), 2 ** len(keep), 2 ** len(rest))
    rho = np.einsum("xkt,xlt->xkl", A, A.conj())
    total = 0.0
    for x in range(2 ** len(meas)):
        pr = float(np.real(np.trace(rho[x])))
        if pr < 1e-15:
            continue
        bits = [(x >> (len(meas) - 1 - i)) & 1 for i in range(len(meas))]
        r = rho[x] / pr
        if kind == "erased_erased":
            out = np.array([1, 1 if (bits[0] ^ bits[1]) == 0 else -1]) / math.sqrt(2)
            rL = np.kron(r, np.outer(out, out))
        else:
            ell, eig_bits, b1 = bits[0], bits[1:1 + m], bits[-1]
            if ell != beta:
                continue
            frame = (PAULI["X"] if ell else np.eye(2)) @ (PAULI["Z"] if sum(eig_bits) % 2 else np.eye(2))
            U = np.kron(np.eye(2), H2 @ (PAULI["X"] if b1 else np.eye(2)) @ frame)
            rL = U @ r @ U.conj().T
        total += pr * float(np.real(PHI @ rL @ PHI))
    return total


def simulate(m):
    from qiskit.quantum_info import Statevector
    d = 2 ** m
    p = 1 / (math.sqrt(d) + 1)
    Ps = hd.p_unitaries(m)
    F = {}
    acc = 0.0
    for w, choice, _ in Ps:
        for beta in (0, 1):
            acc += w * lean_fidelity(m, "kept_kept", np.asarray(Statevector(lean_circuit(m, "kept_kept", choice, beta)).data), beta)
    F["kept_kept"] = acc
    acc = 0.0
    for w, choice, _ in Ps:
        for f1, pf in ((0, 1 - p), (1, p)):
            for beta in (0, 1):
                qc = lean_circuit(m, "kept_erased", choice, beta, flips=(0, f1))
                acc += w * pf * lean_fidelity(m, "kept_erased", np.asarray(Statevector(qc).data), beta)
    F["kept_erased"] = acc
    acc = 0.0
    for (f0, p0), (f1, p1) in itertools.product(((0, 1 - p), (1, p)), repeat=2):
        qc = lean_circuit(m, "erased_erased", None, 0, flips=(f0, f1))
        acc += p0 * p1 * lean_fidelity(m, "erased_erased", np.asarray(Statevector(qc).data), 0)
    F["erased_erased"] = acc
    return 0.25 * (F["kept_kept"] + 2 * F["kept_erased"] + F["erased_erased"]), F


if __name__ == "__main__":
    if sys.argv[1] == "check":
        for m in (1, 2):
            p = 1 / (math.sqrt(2 ** m) + 1)
            F, Fb = simulate(m)
            print(f"m = {m}: LEAN protocol F_e = {F:.12f}   closed form {(7 - 6 * p + 2 * p * p) / 8:.12f}   "
                  f"{ {k: round(v, 10) for k, v in Fb.items()} }")
        sys.exit(0)
    raise SystemExit(__doc__)
