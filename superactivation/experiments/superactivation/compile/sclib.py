"""Shared library for the superactivation circuit compilation (arXiv 2604.27042, item E).

Contents
  * paper objects, transcribed from supplemental.tex (lines cited inline): the private Horodecki
    PPT channel P (Kraus, eq. ppt_channel, and Choi state, eq. choi-state-horo), the 50% erasure
    channel A (d_A = 4, d_B = 5), the flagged effective channel N~ (Prop. 1);
  * an exact BRANCH SIMULATOR for qiskit circuits with mid-circuit measurement, reset and
    if_else feed-forward: every measurement splits the (unnormalized) statevector into branches,
    so the channel a dynamic circuit implements is computed exactly (no sampling);
  * Choi utilities and the gate counter used for every number in RESULTS.md.

Conventions
  Choi matrix J(Phi) = sum_{ij} |i><j| (x) Phi(|i><j|), INPUT FIRST, unnormalized (Tr = d_in).
  Statevectors are numpy tensors with axis q <-> qubit q; qiskit gate matrices are little-endian
  (qarg 0 least significant) and are applied by tensordot on the reversed qarg list.
"""
from __future__ import annotations

import math
from functools import reduce

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, IfElseOp, Clbit, ClassicalRegister
from qiskit.quantum_info import Operator

SQ2 = math.sqrt(2.0)
P_FLIP = 1.0 / (1.0 + SQ2)  # p = 1/(1+sqrt 2) = sqrt(2) - 1

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
K0 = np.array([[1, 0], [0, 0]], dtype=complex)
K1 = np.array([[0, 0], [0, 1]], dtype=complex)


def kron(*ops):
    return reduce(np.kron, ops)


# ----------------------------------------------------------------------------------------------
# paper objects (supplemental.tex)
# ----------------------------------------------------------------------------------------------
def horodecki_kraus(p: float = P_FLIP):
    """Six Kraus operators of P on C^4 = key (x) shield (first tensor factor = key), lines 371-387.
    M_1 carries the minus sign the paper says Smith-Yard's supplement misprinted (footnote, l.381)."""
    c = math.sqrt(2 + SQ2) / 2
    s = math.sqrt(2 - SQ2) / 2
    M0 = np.diag([c, s]).astype(complex)
    M1 = np.diag([s, -c]).astype(complex)
    return [math.sqrt((1 - p) / 2) * kron(I2, K0),
            math.sqrt((1 - p) / 4) * kron(I2, X),
            math.sqrt((1 - p) / 2) * kron(Z, K1),
            math.sqrt((1 - p) / 4) * kron(Z, Y),
            math.sqrt(p) * kron(X, M0),
            math.sqrt(p) * kron(Y, M1)]


def horodecki_choi_state(p: float = P_FLIP) -> np.ndarray:
    """Normalized Choi state Phi^P_{RB}, eq. choi-state-horo (l.333-361), returned on R'R''B'B''
    (R = R'R'' first, then B = B'B'')."""
    def ket(v):
        return np.array(v, dtype=complex).reshape(-1, 1)
    psi = [ket([1, 0, 0, 1]) / SQ2, ket([1, 0, 0, -1]) / SQ2,
           ket([0, 1, 1, 0]) / SQ2, ket([0, 1, -1, 0]) / SQ2]
    e00, e11 = ket([1, 0, 0, 0]), ket([0, 0, 0, 1])
    chip = 0.5 * (math.sqrt(2 + SQ2) * e00 + math.sqrt(2 - SQ2) * e11)
    chim = 0.5 * (math.sqrt(2 - SQ2) * e00 - math.sqrt(2 + SQ2) * e11)
    rho = [0.5 * (e00 @ e00.T + psi[2] @ psi[2].conj().T),
           0.5 * (e11 @ e11.T + psi[3] @ psi[3].conj().T),
           chip @ chip.conj().T, chim @ chim.conj().T]
    q = [(1 - p) / 2, (1 - p) / 2, p / 2, p / 2]
    # sum_i q_i psi^i_{R'B'} (x) rho^(i)_{R''B''}: built on (R', B', R'', B'')
    S = sum(q[i] * np.kron(psi[i] @ psi[i].conj().T, rho[i]) for i in range(4))
    # permute (R', B', R'', B'') -> (R', R'', B', B'')
    T = S.reshape([2] * 8).transpose(0, 2, 1, 3, 4, 6, 5, 7).reshape(16, 16)
    return T


def choi_from_kraus(Ks) -> np.ndarray:
    """Unnormalized Choi, input first: J = sum_k vec-style (I (x) K)|Omega><Omega|(I (x) K)^dag."""
    din = Ks[0].shape[1]
    J = 0
    for K in Ks:
        v = np.zeros((din * K.shape[0],), dtype=complex)
        for i in range(din):
            v += np.kron(np.eye(din)[i], K[:, i])
        J = J + np.outer(v, v.conj())
    return J


def erasure_kraus(d: int = 4, q: float = 0.5):
    """Kraus of A^q: C^d -> C^{d+1}, erasure symbol |e> = basis vector d (l.311-316)."""
    Ks = [math.sqrt(1 - q) * np.vstack([np.eye(d), np.zeros((1, d))]).astype(complex)]
    for i in range(d):
        K = np.zeros((d + 1, d), dtype=complex)
        K[d, i] = math.sqrt(q)
        Ks.append(K)
    return Ks


def ntilde_choi(p: float = P_FLIP) -> np.ndarray:
    """J(N~) on R (x) B (x) Z, N~(rho) = 1/2 rho (x) |0><0| + 1/2 (X^p o Delta)(rho) (x) |1><1|."""
    J = np.zeros((8, 8), dtype=complex)
    for i in range(2):
        for j in range(2):
            Eij = np.zeros((2, 2), dtype=complex)
            Eij[i, j] = 1
            out0 = 0.5 * Eij
            D = np.diag(np.diag(Eij))
            out1 = 0.5 * ((1 - p) * D + p * X @ D @ X)
            J += np.kron(Eij, np.kron(out0, K0) + np.kron(out1, K1))
    return J


# ----------------------------------------------------------------------------------------------
# exact branch simulator for dynamic qiskit circuits
# ----------------------------------------------------------------------------------------------
_MAT_CACHE: dict = {}


def _gate_tensor(op):
    key = None
    try:
        if not op.params or all(isinstance(x, (int, float, complex)) for x in op.params):
            key = (op.name, tuple(complex(x) for x in op.params) if op.params else ())
    except Exception:
        key = None
    if key is not None and key in _MAT_CACHE:
        return _MAT_CACHE[key]
    U = Operator(op).data
    k = int(round(math.log2(U.shape[0])))
    T = U.reshape([2] * (2 * k))
    if key is not None:
        _MAT_CACHE[key] = T
    return T


class Branch:
    __slots__ = ("psi", "cl")

    def __init__(self, psi, cl):
        self.psi = psi
        self.cl = cl


def _apply_gate(psi, T, qubits):
    k = len(qubits)
    n = psi.ndim
    if k == 1 and n > 12:   # fast in-place path for large registers
        q = qubits[0]
        i0 = tuple([slice(None)] * q + [0])
        i1 = tuple([slice(None)] * q + [1])
        a, b, c, d = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
        s0 = psi[i0].copy()
        s1 = psi[i1]
        if b == 0 and c == 0:
            if a != 1:
                psi[i0] *= a
            if d != 1:
                psi[i1] *= d
            return psi
        psi[i0] = a * s0 + b * s1
        psi[i1] = c * s0 + d * s1
        return psi
    if k == 2 and n > 12:
        U = T.reshape(4, 4)
        if np.allclose(U, np.diag(np.diag(U))):
            # qiskit little-endian over qargs [q0, q1]: index = b0 + 2 b1
            dg = np.diag(U)
            q0, q1 = qubits
            for b0 in (0, 1):
                for b1 in (0, 1):
                    ph = dg[b0 + 2 * b1]
                    if ph != 1:
                        idx = [slice(None)] * n
                        idx[q0] = b0
                        idx[q1] = b1
                        psi[tuple(idx)] *= ph
            return psi
    rev = list(reversed(qubits))  # qiskit: last qarg is the most significant index
    out = np.tensordot(T, psi, axes=(list(range(k, 2 * k)), rev))
    # out axes: (rev[0], ..., rev[k-1], remaining axes in order)
    rest = [a for a in range(psi.ndim) if a not in rev]
    order = rev + rest
    inv = np.argsort(order)
    return np.transpose(out, inv)


def _eval_condition(cond, cl, clmap):
    if isinstance(cond, tuple):
        target, val = cond
        if isinstance(target, Clbit):
            return cl[clmap[target]] == int(val)
        if isinstance(target, ClassicalRegister):
            v = sum(cl[clmap[b]] << i for i, b in enumerate(target))
            return v == int(val)
    raise NotImplementedError(f"condition {cond!r}")


def run_branches(qc: QuantumCircuit, branches, qmap=None, clmap=None, prune=1e-28):
    """Apply qc to a list of Branch (psi tensors over the GLOBAL qubit axes)."""
    if qmap is None:
        qmap = {q: i for i, q in enumerate(qc.qubits)}
    if clmap is None:
        clmap = {c: i for i, c in enumerate(qc.clbits)}
    for inst in qc.data:
        op = inst.operation
        qs = [qmap[q] for q in inst.qubits]
        cs = [clmap[c] for c in inst.clbits]
        name = op.name
        if name == "barrier" or name == "delay" or name == "id":
            continue
        if name == "measure":
            nb = []
            for br in branches:
                for v in (0, 1):
                    idx = [slice(None)] * br.psi.ndim
                    idx[qs[0]] = 1 - v
                    psi = br.psi.copy()
                    psi[tuple(idx)] = 0
                    if np.vdot(psi, psi).real > prune:
                        cl = list(br.cl)
                        cl[cs[0]] = v
                        nb.append(Branch(psi, cl))
            branches = nb
            continue
        if name == "reset":
            nb = []
            for br in branches:
                for v in (0, 1):
                    idx = [slice(None)] * br.psi.ndim
                    idx[qs[0]] = 1 - v
                    psi = br.psi.copy()
                    psi[tuple(idx)] = 0
                    if np.vdot(psi, psi).real > prune:
                        if v == 1:
                            psi = _apply_gate(psi, X.reshape(2, 2), [qs[0]])
                        nb.append(Branch(psi, list(br.cl)))
            branches = nb
            continue
        if isinstance(op, IfElseOp):
            true_body, false_body = op.params
            nb = []
            for br in branches:
                ok = _eval_condition(op.condition, br.cl, clmap)
                body = true_body if ok else false_body
                if body is None:
                    nb.append(br)
                    continue
                sub_q = {bq: qmap[oq] for bq, oq in zip(body.qubits, inst.qubits)}
                sub_c = {bc: clmap[oc] for bc, oc in zip(body.clbits, inst.clbits)}
                nb.extend(run_branches(body, [br], sub_q, sub_c, prune))
            branches = nb
            continue
        if isinstance(op, ControlFlowOp):
            raise NotImplementedError(name)
        T = _gate_tensor(op)
        for br in branches:
            br.psi = _apply_gate(br.psi, T, qs)
    return branches


def statevector(qc: QuantumCircuit, init_le: int = 0, psi0=None) -> np.ndarray:
    """Pure-state simulation of a measurement-free circuit; returns the little-endian vector
    (qiskit convention: index = sum_q bit_q 2^q). init_le: little-endian basis index to start from."""
    n = qc.num_qubits
    if psi0 is None:
        psi = np.zeros([2] * n, dtype=complex)
        psi[tuple((init_le >> q) & 1 for q in range(n))] = 1.0
    else:
        psi = np.asarray(psi0, dtype=complex).reshape([2] * n).transpose(list(range(n))[::-1])
    br = run_branches(qc, [Branch(psi, [0] * qc.num_clbits)])
    if len(br) != 1:
        raise RuntimeError("statevector(): circuit branched")
    return np.ascontiguousarray(np.transpose(br[0].psi, list(range(n))[::-1])).reshape(-1)


def reduced_dm(psi, keep):
    """Reduced density matrix on qubits `keep` (first in list = most significant row index)."""
    n = psi.ndim
    rest = [a for a in range(n) if a not in keep]
    M = np.transpose(psi, list(keep) + rest).reshape(2 ** len(keep), -1)
    return M @ M.conj().T


def channel_choi(qc: QuantumCircuit, in_qubits, out_qubits, cl_out=(), extra_init=None):
    """Exact Choi of the channel qc implements from `in_qubits` to (`out_qubits`, classical `cl_out`).

    All other qubits start in |0>. The input register is maximally entangled with a reference of the
    same size. Returns J on R (x) OUT (x) CL with R ordered like in_qubits (first = most significant),
    OUT like out_qubits, CL as a diagonal block in the order of cl_out. Unnormalized (Tr = 2^len(in))."""
    nq = qc.num_qubits
    nin = len(in_qubits)
    N = nq + nin
    psi = np.zeros([2] * N, dtype=complex)
    # reference axes nq..nq+nin-1 paired with in_qubits
    for idx in range(2 ** nin):
        bits = [(idx >> (nin - 1 - t)) & 1 for t in range(nin)]
        sl = [0] * N
        for t in range(nin):
            sl[in_qubits[t]] = bits[t]
            sl[nq + t] = bits[t]
        psi[tuple(sl)] = 1.0
    if extra_init is not None:
        psi = extra_init(psi)
    br = [Branch(psi, [0] * qc.num_clbits)]
    qmap = {q: i for i, q in enumerate(qc.qubits)}
    clmap = {c: i for i, c in enumerate(qc.clbits)}
    br = run_branches(qc, br, qmap, clmap)
    keep = [nq + t for t in range(nin)] + list(out_qubits)
    dk = 2 ** len(keep)
    ncl = len(cl_out)
    J = np.zeros((dk * 2 ** ncl, dk * 2 ** ncl), dtype=complex)
    for b in br:
        rho = reduced_dm(b.psi, keep)
        cval = 0
        for t, c in enumerate(cl_out):
            cval = (cval << 1) | b.cl[c]
        # place as rho (x) |c><c| : index = row*2^ncl + cval
        for i in range(dk):
            J[i * 2 ** ncl + cval, cval::2 ** ncl][:dk] += rho[i]
    return J


def channel_choi_basis(qc: QuantumCircuit, vectors, in_qubits, out_qubits, cl_out=()):
    """Choi of the channel restricted to the input subspace spanned by `vectors` (orthonormal,
    little-endian over in_qubits): J = sum_ij |i><j| (x) Phi(|v_i><v_j|), on R (x) OUT (x) CL."""
    nq = qc.num_qubits
    d = len(vectors)
    nr = max(1, int(math.ceil(math.log2(d))))
    N = nq + nr
    nin = len(in_qubits)
    psi = np.zeros([2] * N, dtype=complex)
    for i, v in enumerate(vectors):
        rbits = [(i >> (nr - 1 - t)) & 1 for t in range(nr)]
        for x in np.nonzero(np.abs(v) > 0)[0]:
            sl = [0] * N
            for t in range(nin):
                sl[in_qubits[t]] = (int(x) >> t) & 1
            for t in range(nr):
                sl[nq + t] = rbits[t]
            psi[tuple(sl)] += v[x]
    br = [Branch(psi, [0] * qc.num_clbits)]
    br = run_branches(qc, br)
    keep = [nq + t for t in range(nr)] + list(out_qubits)
    dout = 2 ** len(out_qubits)
    ncl = len(cl_out)
    J = np.zeros((d * dout * 2 ** ncl, d * dout * 2 ** ncl), dtype=complex)
    for b in br:
        rho = reduced_dm(b.psi, keep)
        rho = rho.reshape(2 ** nr, dout, 2 ** nr, dout)[:d, :, :d, :].reshape(d * dout, d * dout)
        cval = 0
        for c in cl_out:
            cval = (cval << 1) | b.cl[c]
        idx = np.arange(d * dout) * 2 ** ncl + cval
        J[np.ix_(idx, idx)] += rho
    return J


def choi_dist(A, B) -> float:
    """Trace norm of the difference (and we also report the max entry)."""
    return float(np.sum(np.abs(np.linalg.eigvalsh((A - B + (A - B).conj().T) / 2))))


def max_abs(A) -> float:
    return float(np.max(np.abs(A)))


# ----------------------------------------------------------------------------------------------
# counting
# ----------------------------------------------------------------------------------------------
TWO_Q = {"cz", "cx", "ecr", "swap", "rzz", "cp", "crz", "cry", "ch", "cswap", "ccx", "rxx", "ryy"}


def count_ops(qc: QuantumCircuit) -> dict:
    """Straight-line op counts; if_else bodies are counted separately as 'cond:<name>' (max of the
    two bodies), so the caller sees what is unconditional and what is feed-forward."""
    out: dict = {}
    for inst in qc.data:
        op = inst.operation
        if isinstance(op, IfElseOp):
            best: dict = {}
            for body in op.params:
                if body is None:
                    continue
                c = count_ops(body)
                for k, v in c.items():
                    best[k] = max(best.get(k, 0), v)
            for k, v in best.items():
                kk = k if k.startswith("cond:") else "cond:" + k
                out[kk] = out.get(kk, 0) + v
            out["feedforward_blocks"] = out.get("feedforward_blocks", 0) + 1
            continue
        out[op.name] = out.get(op.name, 0) + 1
    return out


def twoq(counts: dict, include_cond=True) -> int:
    t = sum(v for k, v in counts.items() if k in TWO_Q)
    if include_cond:
        t += sum(v for k, v in counts.items() if k.startswith("cond:") and k[5:] in TWO_Q)
    return t


BASIS = ["cz", "sx", "rz", "x", "measure", "reset", "if_else"]


def compile_cz(qc: QuantumCircuit, level: int = 1, seed: int = 7, coupling_map=None, backend=None,
               initial_layout=None):
    from qiskit import transpile
    if backend is not None:
        return transpile(qc, backend=backend, optimization_level=level, seed_transpiler=seed,
                         initial_layout=initial_layout)
    return transpile(qc, basis_gates=BASIS, optimization_level=level, seed_transpiler=seed,
                     coupling_map=coupling_map, initial_layout=initial_layout)
