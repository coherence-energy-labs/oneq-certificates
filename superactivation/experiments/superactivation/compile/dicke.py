"""Symmetric-subspace machinery: Baertschi-Eidenbenz (BE) Dicke unitary U_n and unary Givens networks.

Unary ("thermometer") encoding on m qubits at POSITIONS 1..m (position p -> qubit qs[p-1]):
  |unary_w> = |0^{m-w} 1^w>  (the last w positions are 1),   w = 0..m.
U_n (BE, arXiv:1904.07358) maps |unary_w> -> |D^n_w> for every w simultaneously, as
  U_n = SCS_2 ... SCS_{n-1} SCS_n  (SCS_n applied first), SCS_l acting on positions 1..l:
  gate (i) on (l-1, l), then gate (ii)_ell on (l-ell, l-ell+1, l) for ell = 2..l-1.
Gate implementations (all exact on the REACHABLE inputs, see min_cz_search.py):
  gate (i):  'be'  CX . CRy . CX            (4 CZ)
             'givens' the full Givens completion, KAK-synthesized   (2 CZ; 1 CZ impossible)
  gate (ii): 'be'  CX . CCRy(4 CX) . CX    (6 CZ)
             'mux5' CX . [3-CX multiplexed Ry, analytic, uses don't-cares] . CX   (5 CZ)
             'num4' tabulated 4-CZ circuit (gate_ii_num4.json)                    (4 CZ; 3 not found)

unary_isometry_ops: Givens-rotation network (on adjacent unary levels) preparing s orthonormal
columns from |unary_0..s-1>; each rotation is uncontrolled / singly / doubly controlled according to
which levels are populated at that moment (support tracking), complex phases are moved into
single-qubit prefix phases, and rotations into an empty top level use a 1-CZ 'CRy on |0>'.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

HERE = os.path.dirname(os.path.abspath(__file__))
_NUM4 = None


def _num4():
    global _NUM4
    if _NUM4 is None:
        with open(os.path.join(HERE, "gate_ii_num4.json")) as fh:
            _NUM4 = json.load(fh)
    return _NUM4


def ccry(qc, theta, c1, c2, t):
    """Doubly controlled Ry(theta), exact (no relative phase), 4 CX."""
    q = theta / 4
    qc.ry(q, t); qc.cx(c1, t); qc.ry(-q, t); qc.cx(c2, t)
    qc.ry(q, t); qc.cx(c1, t); qc.ry(-q, t); qc.cx(c2, t)


def gate_i(qc, a, c, l, impl="givens"):
    cth, sth = math.sqrt(1 / l), math.sqrt((l - 1) / l)
    if impl == "be":
        qc.cx(a, c)
        qc.cry(2 * math.acos(cth), c, a)
        qc.cx(a, c)
    elif impl == "givens":
        G = np.eye(4, dtype=complex)
        # little-endian over [a, c]: idx = a + 2c ; (a0,c1)=2, (a1,c0)=1
        G[2, 2] = cth; G[1, 2] = sth
        G[2, 1] = -sth; G[1, 1] = cth
        qc.append(UnitaryGate(G, label=f"G{l}"), [a, c])
    else:
        raise ValueError(impl)


def gate_ii(qc, a, b, c, l, ell, impl="num4"):
    cth, sth = math.sqrt(ell / l), math.sqrt((l - ell) / l)
    if impl == "be":
        qc.cx(a, c)
        ccry(qc, 2 * math.acos(cth), c, b, a)
        qc.cx(a, c)
    elif impl == "mux5":
        S1, S2, S3 = 0.0, math.pi, 0.0
        S4 = 2 * math.atan2(cth, sth)
        al = (S1 + S2 + S3 + S4) / 4
        be = (S1 + S2 - S3 - S4) / 4
        ga = (S1 - S2 - S3 + S4) / 4
        de = (S1 - S2 + S3 - S4) / 4
        qc.cx(a, c)
        qc.ry(al, a); qc.cx(b, a); qc.ry(be, a); qc.cx(c, a); qc.ry(ga, a); qc.cx(b, a); qc.ry(de, a)
        qc.cx(a, c)
    elif impl == "num4":
        tab = _num4()
        ent = tab["gates"][f"{l},{ell}"]
        prm = ent["params"]
        qmap = [a, b, c]
        def layer(off):
            for qq in range(3):
                t, ph, lm = prm[off + 3 * qq: off + 3 * qq + 3]
                qc.u(t, ph, lm, qmap[qq])
        layer(0)
        off = 9
        for (x, y) in tab["placement"]:
            qc.cz(qmap[x], qmap[y])
            layer(off)
            off += 9
    else:
        raise ValueError(impl)


def scs(qc, qs, l, impl_i="givens", impl_ii="num4"):
    P = lambda p: qs[p - 1]
    gate_i(qc, P(l - 1), P(l), l, impl_i)
    for ell in range(2, l):
        gate_ii(qc, P(l - ell), P(l - ell + 1), P(l), l, ell, impl_ii)


def dicke_unitary(qc, qs, impl_i="givens", impl_ii="num4"):
    """U_n on qubits qs (positions 1..n): |unary_w> -> |D^n_w> for all w."""
    n = len(qs)
    for l in range(n, 1, -1):
        scs(qc, qs, l, impl_i, impl_ii)


def dicke_unitary_cz(n, impl_i="givens", impl_ii="num4") -> int:
    ci = {"be": 4, "givens": 2}[impl_i]
    cii = {"be": 6, "mux5": 5, "num4": 4}[impl_ii]
    return sum(ci + cii * (l - 2) for l in range(2, n + 1))


# ----------------------------------------------------------------------------------------------
# nearest-neighbour (line) version of U_n
# ----------------------------------------------------------------------------------------------
_LNN7 = None


def _lnn7():
    global _LNN7
    if _LNN7 is None:
        with open(os.path.join(HERE, "gate_ii_lnn7.json")) as fh:
            _LNN7 = json.load(fh)
    return _LNN7


def gate_ii_lnn(qc, a, b, c, l, ell):
    """SWAP(b,c) o gate(ii)_ell with CZs only on (a,b), (b,c): 7 CZ (gate_tables_lnn.py)."""
    tab = _lnn7()
    prm = tab["gates"][f"{l},{ell}"]["params"]
    qmap = [a, b, c]

    def layer(off):
        for qq in range(3):
            t, ph, lm = prm[off + 3 * qq: off + 3 * qq + 3]
            qc.u(t, ph, lm, qmap[qq])
    layer(0)
    off = 9
    for (x, y) in tab["placement"]:
        qc.cz(qmap[x], qmap[y])
        layer(off)
        off += 9


def dicke_unitary_lnn(qc, qs):
    """U_n with every two-qubit gate between NEIGHBOURS of the line qs[0] - qs[1] - ... - qs[n-1].
    Logical position p starts on qs[p-1]; in SCS_l the qubit carrying position l travels down the
    block (each gate (ii) is fused with the swap that moves it), and one extra SWAP parks it in front
    of position 1 so that positions 1..l-1 are again contiguous for SCS_{l-1}. The output is in
    Sym^n, so the final permutation of positions over qubits is irrelevant."""
    n = len(qs)
    where = {p: qs[p - 1] for p in range(1, n + 1)}
    for l in range(n, 1, -1):
        gate_i(qc, where[l - 1], where[l], l, "givens")
        if l == 2:
            break
        for ell in range(2, l):
            a, b, c = where[l - ell], where[l - ell + 1], where[l]
            gate_ii_lnn(qc, a, b, c, l, ell)
            where[l], where[l - ell + 1] = where[l - ell + 1], where[l]
        qc.swap(where[1], where[l])
        where[1], where[l] = where[l], where[1]
    return where


def dicke_unitary_lnn_cz(n) -> int:
    return sum(2 + (7 * (l - 2) + 3 if l >= 3 else 0) for l in range(2, n + 1))


def encoder_circuit_lnn(V: np.ndarray):
    """Line-compatible encoder: unary network W (already nearest-neighbour) then dicke_unitary_lnn."""
    n = V.shape[0] - 1
    qc = QuantumCircuit(n, name=f"enc_lnn{n}")
    qs = list(range(n))
    ops, alphas, resid = unary_isometry_ops(V)
    apply_unary_isometry(qc, qs, ops, alphas, 2)
    dicke_unitary_lnn(qc, qs)
    return qc, ops, resid


# ----------------------------------------------------------------------------------------------
# unary Givens network
# ----------------------------------------------------------------------------------------------
def unary_isometry_ops(Psi: np.ndarray, tol: float = 1e-14):
    """QR of Psi ((m+1) x s, orthonormal columns) by adjacent-level Givens rotations with prefix
    phases. Returns (ops, alphas): ops in QR order, each ('phase', i, phi) or ('rot', i, c, s, lo, up, zero_top),
    where lo/up say whether the forward rotation needs the lower/upper control and zero_top whether
    level i (and above) is empty when the forward rotation acts (1-CZ case)."""
    M = np.array(Psi, dtype=complex)
    L, s = M.shape
    ops = []
    scale = max(1.0, np.abs(M).max())
    for j in range(s):
        for i in range(L - 1, j, -1):
            x, y = M[i - 1, j], M[i, j]
            if abs(y) <= tol * scale:
                M[i, j] = 0.0 if abs(y) <= 1e-300 else M[i, j]
                continue
            phi = (np.angle(x) - np.angle(y)) if abs(x) > tol * scale else -np.angle(y)
            if abs(phi) > 1e-15:
                M[i:, :] *= np.exp(1j * phi)
                ops.append(["phase", i, float(phi)])
            x, y = M[i - 1, j], M[i, j]
            r = math.hypot(abs(x), abs(y))
            c, sn = abs(x) / r, abs(y) / r
            ri1, ri = M[i - 1, :].copy(), M[i, :].copy()
            M[i - 1, :] = c * ri1 + sn * ri
            M[i, :] = -sn * ri1 + c * ri
            M[i, j] = 0.0
            sup = np.where(np.max(np.abs(M), axis=1) > tol * scale)[0]
            lo = bool(np.any(sup < i - 1))
            up = bool(np.any(sup > i))
            zero_top = not bool(np.any(sup >= i))
            ops.append(["rot", i, float(c), float(sn), lo, up, zero_top])
    alphas = [float(np.angle(M[j, j])) for j in range(s)]
    resid = float(np.abs(np.abs(np.diag(M[:s, :s])) - 1).max() + (np.abs(M[s:, :]).max() if L > s else 0.0))
    return ops, alphas, resid


def apply_unary_isometry(qc, qs, ops, alphas, s):
    """Forward network on positions 1..m (qs): |unary_j> -> sum_w Psi[w, j] |unary_w>, j < s
    (up to one global phase)."""
    m = len(qs)
    P = lambda p: qs[p - 1]
    # initial level phases alphas[j] on levels 0..s-1 as prefix phases
    for t in range(1, s):
        d = alphas[t] - alphas[t - 1]
        if abs(d) > 1e-15:
            qc.p(d, P(m - t + 1))
    for op in reversed(ops):
        if op[0] == "phase":
            _, i, phi = op
            qc.p(-phi, P(m - i + 1))
            continue
        _, i, c, sn, lo, up, zero_top = op
        theta = 2 * math.atan2(sn, c)
        tgt = P(m - i + 1)
        lo_q = P(m - i + 2) if (lo and i - 1 >= 1) else None
        up_q = P(m - i) if (up and i <= m - 1) else None
        if lo and lo_q is None or up and up_q is None:
            raise RuntimeError("control requested beyond the register")
        if lo_q is None and up_q is None:
            qc.ry(theta, tgt)
        elif zero_top and up_q is None:
            # target is |0> on every populated level: Ry(phi/2) CX Ry(-phi/2) gives Ry(phi) X on ctrl=1;
            # on |0>: -sin(phi/2)|0> + cos(phi/2)|1> must equal c|0> + sn|1>
            ph = 2 * math.atan2(-c, sn)
            qc.ry(-ph / 2, tgt); qc.cx(lo_q, tgt); qc.ry(ph / 2, tgt)
        elif lo_q is not None and up_q is None:
            qc.cry(theta, lo_q, tgt)
        elif up_q is not None and lo_q is None:
            qc.x(up_q); qc.cry(theta, up_q, tgt); qc.x(up_q)
        else:
            qc.x(up_q); ccry(qc, theta, lo_q, up_q, tgt); qc.x(up_q)


def unary_isometry_cz(ops) -> int:
    n = 0
    for op in ops:
        if op[0] != "rot":
            continue
        _, i, c, sn, lo, up, zero_top = op
        if not lo and not up:
            n += 0
        elif zero_top and not up:
            n += 1
        elif lo and up:
            n += 4
        else:
            n += 2
    return n


# ----------------------------------------------------------------------------------------------
# Dicke / unary statevector helpers (little-endian over qs)
# ----------------------------------------------------------------------------------------------
def dicke_vector(n, w):
    x = np.arange(2 ** n, dtype=np.uint64)
    v = (np.bitwise_count(x) == w).astype(float)
    return v / math.sqrt(math.comb(n, w))


def symmetric_vector(n, amps):
    """sum_w amps[w] |D^n_w> as a dense little-endian vector (vectorized)."""
    x = np.arange(2 ** n, dtype=np.uint64)
    wt = np.bitwise_count(x).astype(np.int64)
    coef = np.array([amps[w] / math.sqrt(math.comb(n, w)) for w in range(n + 1)], dtype=complex)
    return coef[wt]


def unary_index(m, w):
    """Index (little-endian over positions 1..m = qubits 0..m-1) of |unary_w>."""
    idx = 0
    for p in range(m - w + 1, m + 1):
        idx |= 1 << (p - 1)
    return idx


def encoder_circuit(V: np.ndarray, impl_i="givens", impl_ii="num4") -> QuantumCircuit:
    """Encoder for an isometry V: C^2 -> Sym^n given as an (n+1) x 2 matrix of Dicke amplitudes.
    The input qubit is position n (qubit n-1); the other n-1 qubits start in |0>."""
    n = V.shape[0] - 1
    qc = QuantumCircuit(n, name=f"enc{n}")
    qs = list(range(n))
    ops, alphas, resid = unary_isometry_ops(V)
    apply_unary_isometry(qc, qs, ops, alphas, 2)
    dicke_unitary(qc, qs, impl_i, impl_ii)
    return qc, ops, resid
