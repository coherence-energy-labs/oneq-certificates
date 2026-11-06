"""R170: the MONOGAMY (partition) relaxation -- an upper bound on F_c(A_D^{(x)m}, 2) valid for EVERY input dimension D.

Theorem (DERIVED here; the steps are elementary and each is tested in tests/test_r170_bar.py):
  Let a code for m uses of the 50% erasure channel give Bob, holding the set S of kept outputs, the optimal fidelity
  F_S. Then
    (a) F_empty = 1/4 (Bob's output is independent of R), F_[m] <= 1, and S <= T  =>  F_S <= F_T;
    (b) for every partition {T_1, ..., T_r} of [m], the vector (F_{T_1}, ..., F_{T_r}) lies in the 1 -> r
        universal-cloning region C_r = { (<Pi_{0i}>_sigma)_i : sigma a state of r + 1 qubits },  Pi_{0i} the
        projector onto Phi+ between qubit 0 (R) and qubit i:
        apply the r optimal decoders to the r disjoint systems B_{T_i} simultaneously; the result is such a state.
  C_r is convex with support function h(p) = lambda_max(sum_i p_i Pi_{0i}) for p >= 0, and down-closed above 1/4.
  The constraints are convex and permutation invariant, so the maximum of sum_S 2^-m F_S over them is attained by a
  symmetric profile F_k (k = |S|), and for a partition lambda of m only directions p that are constant on parts of
  equal size are needed. Hence
      F_c(A_D^{(x)m}, 2) <= R(m) := max sum_k C(m, k) 2^-m F_k  s.t.  F_0 = 1/4, F_m <= 1, F_k <= F_{k+1},
                                     sum_j q_j n_j F_{a_j} <= h_lambda(q)  for all lambda = (a_1^n_1 ... a_t^n_t), q >= 0.
  Any FINITE subset of these cuts gives a valid (looser) bound, so a cutting-plane LP is sound at every iteration.

Support function, exactly reduced (DERIVED): with Y on qubit 0, Phi+ -> Psi-, and P^-_{0i} = 1/4 - s_0.s_i, so
      sum_j q_j sum_{i in G_j} P^-_{0i} = sum_j q_j n_j / 4 - s_0 . (sum_j q_j S_{G_j}),
which only sees each group's total spin J_j: h = sum_j q_j n_j/4 - min_{J_1..J_t} lambda_min(s_0 . sum_j q_j S^{(J_j)})
on C^2 (x) V_{J_1} (x) ... (x) V_{J_t}. For one group this is (n + 1)/2 per unit weight: the symmetric cloning bound.
"""
from __future__ import annotations

import functools
import itertools
import math

import numpy as np


# ------------------------------------------------------------------ spin algebra

@functools.lru_cache(maxsize=None)
def spin_matrices(J2: int):
    """(Sx, Sy, Sz) for spin J = J2/2 in the |J, M> basis, M = J, J-1, ..., -J"""
    J = J2 / 2
    d = J2 + 1
    Ms = [J - k for k in range(d)]
    Sp = np.zeros((d, d))
    for k in range(1, d):                               # S+ |J, M> = sqrt(J(J+1) - M(M+1)) |J, M+1>
        M = Ms[k]
        Sp[k - 1, k] = math.sqrt(J * (J + 1) - M * (M + 1))
    Sm = Sp.T
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / 2j
    Sz = np.diag(Ms)
    return Sx, Sy.astype(complex), Sz


def _kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def coupling_min(q, J2s) -> float:
    """lambda_min( s_0 . sum_j q_j S^{(J_j)} ) on C^2 (x) V_{J_1} (x) ... ; J2s = (2 J_1, ..., 2 J_t)"""
    dims = [j2 + 1 for j2 in J2s]
    s0 = spin_matrices(1)
    H = None
    for a in range(3):
        T = None
        for j, (qj, j2) in enumerate(zip(q, J2s)):
            if qj == 0:
                continue
            ops = [np.eye(d) for d in dims]
            ops[j] = spin_matrices(j2)[a]
            term = qj * _kron_all(ops)
            T = term if T is None else T + term
        if T is None:
            return 0.0
        part = np.kron(s0[a], T)
        H = part if H is None else H + part
    return float(np.linalg.eigvalsh((H + H.conj().T) / 2)[0])


def group_spins(n: int):
    """the total spins (as 2J) that n qubits can carry: n, n - 2, ..., (0 or 1)"""
    return list(range(n, -1, -2))


def support(parts_sizes, mults, q) -> float:
    """h_lambda(q) = lambda_max( sum_j q_j sum_{i in group j} Pi_{0i} ) for a partition with distinct part sizes
    parts_sizes[j] occurring mults[j] times; q[j] >= 0 is the weight on every part of size parts_sizes[j]."""
    q = [float(x) for x in q]
    if any(x < 0 for x in q):
        raise ValueError("support function is only used for q >= 0")
    base = sum(qj * nj for qj, nj in zip(q, mults)) / 4
    best = 0.0
    for J2s in itertools.product(*[group_spins(n) for n in mults]):
        best = min(best, coupling_min(q, J2s))
    return base - best


def integer_partitions(m: int, maxpart: int | None = None):
    if maxpart is None:
        maxpart = m
    if m == 0:
        yield ()
        return
    for a in range(min(m, maxpart), 0, -1):
        for rest in integer_partitions(m - a, a):
            yield (a,) + rest


def partition_groups(lam):
    sizes = sorted(set(lam), reverse=True)
    return sizes, [lam.count(a) for a in sizes]


# ------------------------------------------------------------------ the cutting-plane LP

def _violation_direction(sizes, mults, F, iters=400):
    """maximize g(q) = sum_j q_j n_j F_{a_j} - h(q) over the simplex (g concave: h is convex). Returns (g*, q*)."""
    t = len(sizes)
    c = np.array([n * F[a] for a, n in zip(sizes, mults)])

    def g(qv):
        return float(c @ qv) - support(sizes, mults, qv)

    best_q = np.ones(t) / t
    best = g(best_q)
    for j in range(t):                                  # vertices
        e = np.zeros(t)
        e[j] = 1.0
        v = g(e)
        if v > best:
            best, best_q = v, e
    if t == 1:
        return best, best_q
    # projected (sub)gradient ascent with numerical gradients on the simplex, then a local polish
    qv = best_q.copy()
    step = 0.2
    for it in range(iters):
        grad = np.zeros(t)
        h0 = g(qv)
        for j in range(t):
            d = np.zeros(t)
            d[j] = 1e-6
            grad[j] = (g(_proj_simplex(qv + d)) - h0) / 1e-6
        qn = _proj_simplex(qv + step * grad)
        vn = g(qn)
        if vn > h0:
            qv = qn
            if vn > best:
                best, best_q = vn, qn.copy()
        else:
            step *= 0.5
            if step < 1e-9:
                break
    return best, best_q


def _proj_simplex(v):
    """Euclidean projection onto {q >= 0, sum q = 1}"""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)


def relaxation(m: int, rounds: int = 60, tol: float = 1e-9, log=None):
    """R(m) by cutting planes. Returns (value, profile F_0..F_m, cuts) where every cut is
    (lam, q, h) meaning sum_j q_j n_j F_{a_j} <= h. The value is an upper bound at every round."""
    from scipy.optimize import linprog
    w = np.array([math.comb(m, k) * 2.0 ** -m for k in range(m + 1)])
    parts = [lam for lam in integer_partitions(m) if len(lam) >= 2]
    cuts = []
    for lam in parts:                                   # seed: each group alone, and all groups equally weighted
        sizes, mults = partition_groups(lam)
        for j in range(len(sizes)):
            q = np.zeros(len(sizes))
            q[j] = 1.0
            cuts.append((lam, q, support(sizes, mults, q)))
        if len(sizes) > 1:
            q = np.ones(len(sizes))
            cuts.append((lam, q, support(sizes, mults, q)))
    value = None
    for rnd in range(rounds):
        A, b = [], []
        for lam, q, h in cuts:
            sizes, mults = partition_groups(lam)
            row = np.zeros(m + 1)
            for qj, a, n in zip(q, sizes, mults):
                row[a] += qj * n
            A.append(row)
            b.append(h)
        for k in range(m):                              # monotone: F_k - F_{k+1} <= 0
            row = np.zeros(m + 1)
            row[k], row[k + 1] = 1.0, -1.0
            A.append(row)
            b.append(0.0)
        bounds = [(0.25, 0.25)] + [(0.25, 1.0)] * m
        res = linprog(-w, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed: {res.message}")
        F = res.x
        value = float(w @ F)
        worst = 0.0
        added = 0
        for lam in parts:
            sizes, mults = partition_groups(lam)
            g, q = _violation_direction(sizes, mults, F)
            if g > tol:
                cuts.append((lam, q, support(sizes, mults, q)))
                added += 1
            worst = max(worst, g)
        if log:
            log(f"   m={m} round {rnd}: LP value {value:.10f}, worst violation {worst:.2e}, cuts {len(cuts)}")
        if added == 0:
            break
    return value, F, cuts
