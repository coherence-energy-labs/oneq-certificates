"""The expected score of an ISA circuit under the calibrated noise model -- exact where the circuit's structure allows,
sampled only where it does not.

THE IDEA. Every error in the model is a Pauli. Push a candidate Pauli forward through the gates after it: through a
Clifford it stays a Pauli, and a Pauli that reaches the terminal measurements does one thing -- it XORs a fixed
pattern x into the measured bits. Such an error is REACHABLE. One that meets a gate turning it into a non-Pauli (an
X on a T, anything but Y on an ry(theta)) is BLOCKED: its effect depends on the state and must be simulated.
Propagation depends only on the circuit, never on the state, so every candidate error is classified once per circuit.

Reachable errors at different operations are independent, so the XOR of their patterns has characteristic function
    chi(s) = prod_i f_i(s),   f_i(s) = E[(-1)^{s.x_i}]   (the Walsh-Hadamard transform of op i's pattern law)
and with q the error-free outcome law over the measured bits and g the readout-smoothed score g(b) = sum_o C(o|b)
score(o), the expected score is EXACTLY
    E[S] = 2^{-m} sum_s chi(s) q^(s) g^(s)                       (^ = Walsh-Hadamard transform)
-- every reachable error pattern summed in closed form, no sampling. A fully Clifford circuit (every QEC memory, every
stabilizer protocol) gets its noisy expectation EXACTLY, with zero variance, from ONE statevector run.

Blocked errors are sampled, Rao-Blackwellized over the reachable ones: a trajectory draws only which blocked errors
fire (conditioned on at least one, with the exact no-blocked-error term P0 S0 added analytically), simulates them,
and scores the result with the reachable errors' characteristic function given the drawn set. Unbiased by
construction; its variance comes only from the operations that a non-Clifford gate makes state-dependent.

Readout is the asymmetric per-qubit confusion (p01, p10), applied exactly. The protocol supplies score(bits) with
bits in CLASSICAL-BIT order; the engine never sees the protocol's meaning, so the same function grades simulated
expectations and real counts. exact_expectation evolves the same channels on the density matrix -- a different
representation, the oracle the fast path is tested against.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .noise import PAULIS2, NoiseParams, compile_ops
from .snapshot import DeviceSnapshot

PAULIS1 = ("X", "Y", "Z")
MAX_ACTIVE = 24                      # numpy statevector: 2^24 amplitudes = 256 MiB, ~1 GiB peak with temporaries
EXACT_MAX_ACTIVE = 10                # density-matrix oracle: 4^10 entries
MAX_MEASURED = 22                    # the score vector and its transforms hold 2^m entries

_P1 = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
       "Z": np.diag([1.0, -1.0])}
_CODE1 = {"I": 0, "Z": 1, "X": 2, "Y": 3}            # code = 2 x + z
_NAME1 = {v: k for k, v in _CODE1.items()}


class EngineError(ValueError):
    """the request cannot be simulated faithfully within the engine's limits"""


@dataclass(frozen=True)
class Expectation:
    mean: float
    se: float                        # 0 when exact
    exact: bool                      # no error is blocked: the mean is the model's exact value
    p_unblocked: float               # probability that no blocked error fires
    s_unblocked: float               # exact score given no blocked error (all reachable errors included)
    p_no_error: float                # probability that no error at all occurs
    trajectories: int
    active_qubits: int
    measured_bits: int
    blocked_mass: float              # expected number of blocked errors per shot


# ------------------------------------------------------------------ statevectors
def score_vector(score, n_bits: int) -> np.ndarray:
    """score(bits) over every outcome, C order over (bit 0, ..., bit n-1)"""
    return np.array([float(score(b)) for b in itertools.product((0, 1), repeat=n_bits)])


def _apply(psi, U, qs):
    if len(qs) == 1:
        psi = np.tensordot(U, psi, axes=([1], [qs[0]]))
        return np.moveaxis(psi, 0, qs[0])
    q0, q1 = qs                                         # qiskit little-endian: index = b(q1) * 2 + b(q0)
    Ut = U.reshape(2, 2, 2, 2)                          # (out q1, out q0, in q1, in q0)
    psi = np.tensordot(Ut, psi, axes=([3, 2], [q0, q1]))
    return np.moveaxis(psi, [0, 1], [q1, q0])


def _pauli(psi, g, q):
    if g in ("X", "Y"):
        psi = np.flip(psi, axis=q)
    if g in ("Z", "Y"):
        idx = [slice(None)] * psi.ndim
        idx[q] = 1
        psi = psi.copy()
        psi[tuple(idx)] *= -1                            # applies ZX = iY: the global phase is irrelevant
    return psi


def _kernel(U: np.ndarray, qs: tuple, k: int):
    """an in-place applier for one gate on a (2,)*k state, chosen from the matrix's structure. Global phases are
    dropped (every output of this engine is a probability): a diagonal gate multiplies only the slices whose phase
    differs from the first entry's."""
    def view(bits):
        idx = [slice(None)] * k
        for q, b in zip(qs, bits):
            idx[q] = slice(b, b + 1)                          # a width-1 slice is ALWAYS a view; an integer that
        return tuple(idx)                                     # indexes every axis returns a scalar copy instead
    n = len(qs)
    off = U - np.diag(np.diag(U))
    if not off.any():
        d = np.diag(U) / U[0, 0]
        muls = []
        for j in range(1, 2 ** n):
            if abs(d[j] - 1) > 1e-15:
                bits = [(j >> t) & 1 for t in range(n)]       # qiskit order: bit t of the index is qs[t]
                muls.append((view(bits), d[j]))

        def diag_apply(psi):
            for v, c in muls:
                psi[v] *= c
        return diag_apply
    if n == 1:
        v0, v1 = view([0]), view([1])
        u00, u01, u10, u11 = U[0, 0], U[0, 1], U[1, 0], U[1, 1]

        def one_apply(psi):
            a, b = psi[v0], psi[v1]
            na = u00 * a + u01 * b
            b *= u11
            b += u10 * a
            a[...] = na
        return one_apply

    def general(psi):
        psi[...] = _apply(psi, U, qs)
    return general


class _NumpySV:
    name = "numpy"
    CHECKPOINT_BYTES = 64 << 20

    def __init__(self, k: int):
        self.k = k
        self._kernels: list | None = None
        self._ckpt: dict = {}

    def _prepare(self, ops) -> list:
        if self._kernels is None:
            self._kernels = [None if op.matrix is None else _kernel(op.matrix, op.qubits, self.k) for op in ops]
        return self._kernels

    def _fresh(self):
        psi = np.zeros((2,) * self.k, dtype=complex)
        psi[(0,) * self.k] = 1.0
        return psi

    def run(self, ops: list, errors: dict) -> np.ndarray:
        """the state after every op, with errors[i] applied right after op i. The prefix before the first error is
        the clean run: it resumes from the latest stored clean state at or before that point."""
        kernels = self._prepare(ops)
        first = min(errors) if errors else len(ops)
        starts = [c for c in self._ckpt if c <= first]
        if starts:
            c = max(starts)
            psi = self._ckpt[c].copy()
        else:
            c, psi = 0, self._fresh()
        stride = max(1, -(-len(ops) * psi.nbytes // self.CHECKPOINT_BYTES))
        for i in range(c, len(ops)):
            if not errors and i % stride == 0 and i not in self._ckpt:
                self._ckpt[i] = psi.copy()                    # the clean state BEFORE op i
            kern = kernels[i]
            if kern is not None:
                kern(psi)
            for q, g in errors.get(i, ()):
                psi = _pauli(psi, g, q)
        return psi


class _AerSV:
    name = "aer"

    def __init__(self, k: int):
        from qiskit_aer import AerSimulator
        self.k = k
        self.sim = AerSimulator(method="statevector")

    def run(self, ops: list, errors: dict) -> np.ndarray:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        qc = QuantumCircuit(self.k)
        for i, op in enumerate(ops):
            if op.matrix is not None:
                qc.append(UnitaryGate(op.matrix, check_input=False), list(op.qubits))
            for q, g in errors.get(i, ()):
                getattr(qc, g.lower())(q)
        qc.save_statevector()
        sv = np.asarray(self.sim.run(qc).result().get_statevector())
        return np.moveaxis(sv.reshape((2,) * self.k), list(range(self.k)), list(reversed(range(self.k))))


def simulator(k: int, prefer: str = "auto"):
    if prefer not in ("auto", "aer", "numpy"):
        raise EngineError(f"unknown simulator {prefer!r}")
    if prefer in ("auto", "aer"):
        try:
            return _AerSV(k)
        except ImportError:
            if prefer == "aer":
                raise
    return _NumpySV(k)


# ------------------------------------------------------------------ outcome laws
def _true_probs(psi, meas) -> np.ndarray:
    """error-free-readout outcome law over the measured qubits, clbit order, flattened C order"""
    axes = [a for a, _ in meas]
    p = np.abs(psi) ** 2
    rest = tuple(a for a in range(psi.ndim) if a not in axes)
    p = p.sum(axis=rest) if rest else p
    order = sorted(axes)                                # remaining axes are in increasing active order
    return np.transpose(p, [order.index(a) for a in axes]).reshape(-1)


def _confusions(meas, snap, active, params) -> list:
    out = []
    for a, _ in meas:
        q = active[a]
        f = params.readout_factor
        p01, p10 = (min(0.5, f * snap.p01[q]), min(0.5, f * snap.p10[q])) if params.readout else (0.0, 0.0)
        out.append(np.array([[1 - p01, p10], [p01, 1 - p10]]))     # rows: observed, columns: true
    return out


def _smoothed_score(svec, confusions) -> np.ndarray:
    """g(b) = sum_o C(o|b) score(o): the expected score of TRUE outcome b once readout has acted"""
    m = len(confusions)
    g = svec.reshape((2,) * m) if m else svec
    for j, C in enumerate(confusions):
        g = np.moveaxis(np.tensordot(C.T, g, axes=([1], [j])), 0, j)
    return g.reshape(-1)


def _readout_probs(psi, meas, snap, active, params) -> np.ndarray:
    """observed outcome law: the true law pushed through each measured qubit's confusion matrix"""
    p = _true_probs(psi, meas)
    m = len(meas)
    p = p.reshape((2,) * m)
    for j, C in enumerate(_confusions(meas, snap, active, params)):
        p = np.moveaxis(np.tensordot(C, p, axes=([1], [j])), 0, j)
    return p.reshape(-1)


def wht(v: np.ndarray) -> np.ndarray:
    """f^(s) = sum_x f(x) (-1)^{popcount(s & x)}, both indexed in the same C-order bit convention"""
    n = len(v)
    m = n.bit_length() - 1
    if n != 1 << m:
        raise EngineError("Walsh-Hadamard transform needs a power-of-two length")
    t = np.asarray(v, dtype=float).reshape((2,) * m) if m else np.asarray(v, dtype=float)
    for ax in range(m):
        a, b = np.take(t, 0, axis=ax), np.take(t, 1, axis=ax)
        t = np.stack([a + b, a - b], axis=ax)
    return t.reshape(-1)


# ------------------------------------------------------------------ Pauli propagation
def _pauli_matrix(codes: tuple) -> np.ndarray:
    """codes for (q0, q1, ...) -> the qiskit-ordered matrix (q0 least significant)"""
    M = np.eye(1)
    for c in codes:
        M = np.kron(_P1[_NAME1[c]], M)
    return M


def _conjugation_table(U: np.ndarray) -> np.ndarray:
    """table[code] = code of U P U^dagger when that is a Pauli (up to sign), else -1 (the error is BLOCKED).
    1q code = 2x + z; 2q code = 4 code(q0) + code(q1)."""
    n = int(round(math.log2(U.shape[0])))
    allc = list(itertools.product(range(4), repeat=n))
    mats = {c: _pauli_matrix(c) for c in allc}
    table = np.full(4 ** n, -1, dtype=np.int64)
    for c in allc:
        M = U @ mats[c] @ U.conj().T
        for d in allc:
            # 1 - overlap ~ delta^2 / 8 for a gate delta radians from Clifford: 1e-12 admits only delta < 3e-6, whose
            # effect is O(delta^2); anything further is treated as non-Clifford (slower, never wrong)
            if abs(abs(np.trace(mats[d].conj().T @ M)) / U.shape[0] - 1) < 1e-12:
                table[_pack(c)] = _pack(d)
                break
    return table


def _pack(codes: tuple) -> int:
    v = 0
    for c in codes:
        v = 4 * v + c
    return v


@dataclass
class _Plan:
    ops: list
    active: list
    meas: list
    tables: list                     # per op: conjugation table (None = identity)
    # candidate errors, one row per (op, Pauli)
    e_op: np.ndarray                 # op index
    e_prob: np.ndarray               # probability of this Pauli at this op
    e_blocked: np.ndarray            # bool
    e_flip: np.ndarray               # measured-bit XOR pattern (valid when not blocked)
    e_paulis: list                   # [(active qubit, 'X'|'Y'|'Z'), ...] for the statevector


def _candidates(op):
    if len(op.qubits) == 1:
        return [([(op.qubits[0], g)], pp) for g, pp in zip(PAULIS1, op.pauli_probs) if pp > 0]
    if op.pauli_probs:                  # a structured CZ channel (noise.cz_pauli_probs), in PAULIS2 order
        return [([(q, g) for q, g in zip(op.qubits, pair) if g != "I"], pp)
                for pair, pp in zip(PAULIS2, op.pauli_probs) if pp > 0]
    return [([(q, g) for q, g in zip(op.qubits, pair) if g != "I"], op.p_error / 15) for pair in PAULIS2]


def plan(qc, snap: DeviceSnapshot, params: NoiseParams) -> _Plan:
    ops, active, meas = compile_ops(qc, snap, params)
    k = len(active)
    cache: dict = {}
    tables: list[np.ndarray | None] = []
    for op in ops:
        if op.matrix is None:
            tables.append(None)
            continue
        key = (op.matrix.shape, op.matrix.round(12).tobytes())
        if key not in cache:
            t = _conjugation_table(op.matrix)
            cache[key] = None if np.array_equal(t, np.arange(len(t))) else t
        tables.append(cache[key])
    e_op, e_prob, e_paulis = [], [], []
    for i, op in enumerate(ops):
        if op.p_error <= 0:
            continue
        for paulis, pp in _candidates(op):
            e_op.append(i)
            e_prob.append(pp)
            e_paulis.append(paulis)
    E = len(e_op)
    X = np.zeros((E, k), dtype=bool)
    Z = np.zeros((E, k), dtype=bool)
    for e, paulis in enumerate(e_paulis):
        for q, g in paulis:
            c = _CODE1[g]
            X[e, q], Z[e, q] = bool(c >> 1), bool(c & 1)
    start = np.array(e_op, dtype=np.int64)
    blocked = np.zeros(E, dtype=bool)
    for j, op in enumerate(ops):                         # an error AT op i follows op i's gate: propagate through j > i
        tab = tables[j]
        if tab is None or E == 0:
            continue
        rows = np.nonzero((start < j) & ~blocked)[0]
        if rows.size == 0:
            continue
        if len(op.qubits) == 1:
            q = op.qubits[0]
            code = 2 * X[rows, q] + Z[rows, q]
            new = tab[code]
            bad = new < 0
            blocked[rows[bad]] = True
            ok, nw = rows[~bad], new[~bad]
            X[ok, q], Z[ok, q] = (nw >> 1) & 1, nw & 1
        else:
            q0, q1 = op.qubits
            code = 4 * (2 * X[rows, q0] + Z[rows, q0]) + (2 * X[rows, q1] + Z[rows, q1])
            new = tab[code]
            bad = new < 0
            blocked[rows[bad]] = True
            ok, nw = rows[~bad], new[~bad]
            c0, c1 = nw >> 2, nw & 3
            X[ok, q0], Z[ok, q0], X[ok, q1], Z[ok, q1] = (c0 >> 1) & 1, c0 & 1, (c1 >> 1) & 1, c1 & 1
    m = len(meas)
    flip = np.zeros(E, dtype=np.int64)
    for j, (a, _) in enumerate(meas):
        flip |= X[:, a].astype(np.int64) << (m - 1 - j)
    return _Plan(ops, active, meas, tables, start, np.array(e_prob, dtype=float), blocked, flip, e_paulis)


def _parity_signs(flips: np.ndarray, m: int) -> np.ndarray:
    """rows (-1)^{popcount(s & x)} over every s, one row per pattern x"""
    s = np.arange(1 << m, dtype=np.int64)
    return 1.0 - 2.0 * (np.bitwise_count(flips[:, None] & s[None, :]) & 1)


def _op_factors(P: _Plan, n_ops: int, m: int):
    """per op: f_i(s) for its reachable errors given that none of its blocked errors fired; and its blocked mass"""
    pB = np.zeros(n_ops)
    np.add.at(pB, P.e_op[P.e_blocked], P.e_prob[P.e_blocked])
    factors = {}
    reach = ~P.e_blocked & (P.e_flip != 0)
    for i in np.unique(P.e_op[reach]):
        sel = reach & (P.e_op == i)
        signs = _parity_signs(P.e_flip[sel], m)
        w = P.e_prob[sel]
        # f_i(s) = [P(no reachable flip) + sum_x w_x (-1)^{s.x}] / (1 - pB_i); at s = 0 it is exactly 1
        factors[int(i)] = 1.0 + (w @ (signs - 1.0)) / (1.0 - pB[i])
    return factors, pB


# ------------------------------------------------------------------ the expectation
def expectation(qc, score, snap: DeviceSnapshot, params: NoiseParams, trajectories: int,
                rng: np.random.Generator, prefer: str = "numpy") -> Expectation:
    P = plan(qc, snap, params)
    k, m, n_ops = len(P.active), len(P.meas), len(P.ops)
    if k > MAX_ACTIVE:
        raise EngineError(f"{k} active qubits exceed the engine limit {MAX_ACTIVE}")
    if m > MAX_MEASURED:
        raise EngineError(f"{m} measured bits exceed the engine limit {MAX_MEASURED}")
    sv = simulator(k, prefer)
    g_hat = wht(_smoothed_score(score_vector(score, m), _confusions(P.meas, snap, P.active, params)))
    factors, pB = _op_factors(P, n_ops, m)
    chi_all = np.ones(1 << m)
    for f in factors.values():
        chi_all = chi_all * f
    p_err = np.array([op.p_error for op in P.ops])

    def value(errors: dict, blocked_ops: set) -> float:
        chi = chi_all
        hit = [i for i in blocked_ops if i in factors]
        if hit:
            chi = np.ones(1 << m)
            for i, f in factors.items():
                if i not in blocked_ops:
                    chi = chi * f
        q_hat = wht(_true_probs(sv.run(P.ops, errors), P.meas))
        return float(np.dot(chi, q_hat * g_hat)) / (1 << m)

    s0 = value({}, set())
    survive = np.cumprod(np.concatenate([[1.0], 1 - pB]))
    p0 = float(survive[-1])
    common: dict[str, Any] = dict(p_unblocked=p0, s_unblocked=s0, p_no_error=float(np.prod(1 - p_err)), active_qubits=k,
                  measured_bits=m, blocked_mass=float(pB.sum()))
    if p0 >= 1.0:
        return Expectation(mean=s0, se=0.0, exact=True, trajectories=0, **common)
    if trajectories <= 0:
        raise EngineError("this circuit has blocked errors: it needs trajectories > 0 (or report s_unblocked as a "
                          "conditional value, never as the expectation)")
    first = survive[:-1] * pB
    first = first / first.sum()
    blocked_rows: dict[int, list[int]] = {}
    for e in np.nonzero(P.e_blocked)[0]:
        blocked_rows.setdefault(int(P.e_op[e]), []).append(int(e))
    vals = np.empty(trajectories)
    for t in range(trajectories):
        start = int(rng.choice(n_ops, p=first))
        fired = [start] + [i for i in range(start + 1, n_ops) if pB[i] > 0 and rng.random() < pB[i]]
        errors = {}
        for i in fired:
            rows = blocked_rows[i]
            w = P.e_prob[rows] / pB[i]
            errors[i] = P.e_paulis[rows[int(rng.choice(len(rows), p=w))]]
        vals[t] = value(errors, set(fired))
    mean = p0 * s0 + (1 - p0) * float(vals.mean())
    se = (1 - p0) * float(vals.std(ddof=1) / math.sqrt(trajectories)) if trajectories > 1 else float("nan")
    return Expectation(mean=mean, se=se, exact=False, trajectories=trajectories, **common)


# ------------------------------------------------------------------ the oracle
def _rho_apply(rho, U, qs, k):
    """rho -> U rho U^dagger, rho stored as a (2,)*2k tensor (ket axes 0..k-1, bra axes k..2k-1)"""
    rho = _apply(rho, U, qs)
    return _apply(rho, U.conj(), tuple(q + k for q in qs))


def _rho_pauli(rho, g, q, k):
    return _pauli(_pauli(rho, g, q), g, q + k)          # P rho P^dagger; the real matrix ZX = iY cancels its phase


def exact_expectation(qc, score, snap: DeviceSnapshot, params: NoiseParams, unblocked_only: bool = False) -> float:
    """the SAME Pauli channels evolved exactly on the density matrix (tiny circuits only): no propagation, no
    transforms, no sampling -- the oracle the fast path is tested against.

    unblocked_only: condition every operation on "none of its BLOCKED errors fired" (each op's channel restricted to
    no-error + its unblocked Paulis, renormalized by 1 - pB_i). That is exactly the engine's s_unblocked, so the
    closed-form reachable sum can be checked to rounding even on a circuit that is not Clifford. Only the blocked /
    unblocked LABELS come from the engine's classification; the evolution itself shares nothing with it."""
    ops, active, meas = compile_ops(qc, snap, params)
    k = len(active)
    if k > EXACT_MAX_ACTIVE:
        raise EngineError(f"{k} active qubits: the density-matrix oracle stops at {EXACT_MAX_ACTIVE}")
    blocked_of: dict = {}
    if unblocked_only:
        P = plan(qc, snap, params)
        for e in range(len(P.e_op)):
            blocked_of.setdefault(int(P.e_op[e]), []).append(bool(P.e_blocked[e]))
    rho = np.zeros((2,) * (2 * k), dtype=complex)
    rho[(0,) * (2 * k)] = 1.0
    for i, op in enumerate(ops):
        if op.matrix is not None:
            rho = _rho_apply(rho, op.matrix, op.qubits, k)
        if op.p_error <= 0:
            continue
        cands = _candidates(op)
        flags = blocked_of.get(i, [False] * len(cands))
        p_blocked = sum(w for (_, w), b in zip(cands, flags) if b)
        out = (1 - op.p_error) * rho
        for (paulis, w), b in zip(cands, flags):
            if b:
                continue
            r = rho
            for q, g in paulis:
                r = _rho_pauli(r, g, q, k)
            out = out + w * r
        rho = out / (1 - p_blocked)
    diag = np.einsum(rho.reshape(2 ** k, 2 ** k), [0, 0], [0]).real.reshape((2,) * k)
    amp = np.sqrt(np.clip(diag, 0, None))               # |amplitude| suffices: readout reads only probabilities
    return float(_readout_probs(amp, meas, snap, active, params) @ score_vector(score, len(meas)))
