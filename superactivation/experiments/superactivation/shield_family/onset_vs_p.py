"""The onset of superactivation as a function of the PPT channel's flip noise p.

For the shield-dimension family (ppt_family.py), p = 1/(sqrt d + 1), and the paper's general effective-channel theorem
gives N~_p = 1/2 id (x) |0><0| + 1/2 (X^p o Delta) (x) |1><1| per use of P_d (x) A_{2d}. So F_c((P_d (x) A)^n) >=
F_c(N~_p^n), and superactivation at n uses holds as soon as F_c(N~_p^n) > 3/4 (Result 1's 3/4 bound holds for any PPT
channel and any 50% erasure channel). n = 1 is impossible for every p (F = 1/2 + (1-p)/4 <= 3/4); this finds, for
n = 2, 3, the largest p at which some code crosses 3/4.

Dense model over ALL encoders (general_codes.branch_states with its p set), power-iteration see-saw on both sides.

  python onset_vs_p.py explicit         the hand code {|++>, |-->} at n = 2: F = (7 - 6p + 2p^2)/8, numerically
                                        (both kept 1, one erased 1 - p, both erased 1/2 - p(1 - p) from the
                                        Z-parity of the two flagged outcomes). NOT optimal: asymmetric codes do
                                        better for p in [0.15, 0.2] (independent adversary 2026-10-14; certified
                                        crossings at d = 20, 21 in evidence/superactivation/shield_family/).
                                        The power-iteration encoder in best() under-converges at n >= 3.
  python onset_vs_p.py scan N R S       best F_c(N~_p^N) over a p grid; threshold by bisection
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent / "general_codes"))
from oneq import superactivation as sa  # noqa: E402

import general_codes as gcm  # noqa: E402


def branch_states(X, n, p):
    # p is PASSED, never written into general_codes.P: that global leaked into every later caller (2026-10-19)
    return {z: 2 * s for z, s in gcm.branch_states(X / 2, n, p).items()}     # Choi normalization: Tr = 2


def decode(X, n, p, warm=None, iters=300):
    decs, F = {}, 0.0
    for z, R in branch_states(X, n, p).items():
        d = 2 ** n
        D = warm[z] if warm is not None else np.eye(2 * d)[None] / 2
        R3 = R[None]
        for _ in range(iters):
            D = sa.power_step(R3, D, d)
        decs[z] = D
        F += 2.0 ** -n * float(np.real(np.einsum("bij,bji->", R3, D))) / 4
    return decs, F


def adjoint_states(decs, n, p):
    """gradient G with Tr(X G) = F: the dual of each branch map applied to its decoder (the channel is a Kraus map)"""
    nk = gcm.noisy_kraus(p)
    G = 0
    for z, D in decs.items():
        # dual of applying nk on each flagged qubit q: sum_k K^dag (.) K on that qubit, R untouched
        T = D[0]
        for q, f in enumerate(z):
            if f:
                T = apply_dual(T, nk, q, n)
        G = G + 2.0 ** -n * T / 4
    return sa.herm(G)


def apply_dual(Y, kraus, q, n):
    """sum_k K^dag Y K on qubit q of the A register (R untouched): the Heisenberg dual of general_codes.apply_on.

    The bra side contracts Y's column index with K's ROW index (Y K). Until 2026-10-18 it contracted with K^T
    (`k.T.conj().conj()`), i.e. computed K^dag Y K^T: identical for the diagonal dephasing Kraus operators, wrong for
    the flip ones sqrt(p) X|a><a|. Tr(X G) then missed F by 0.013-0.045 at the two-use optima, and best()'s see-saw
    climbed that wrong functional onto the closed form {|++>, |-->} from every start, even from a better code
    (tests/test_superactivation_shield_family.py::test_the_encoder_gradient_is_the_fidelitys_linear_functional)."""
    dA = 2 ** n
    T = Y.reshape([2] + [2] * n + [2] + [2] * n)
    out = np.zeros_like(T)
    for k in kraus:
        kd = k.conj().T
        t = np.tensordot(kd, T, axes=([1], [1 + q]))
        t = np.moveaxis(t, 0, 1 + q)
        t = np.tensordot(t, k, axes=([2 + n + q], [0]))
        t = np.moveaxis(t, -1, 2 + n + q)
        out = out + t
    return out.reshape(2 * dA, 2 * dA)


def random_encoder(n, rng):
    m = 2 ** n
    K = rng.normal(size=(2 * m, 2)) + 1j * rng.normal(size=(2 * m, 2))
    X = K @ K.conj().T
    T = X.reshape(2, m, 2, m).trace(axis1=1, axis2=3)
    tw, tv = np.linalg.eigh(sa.herm(T))
    Ti = np.kron((tv / np.sqrt(tw)) @ tv.conj().T, np.eye(m))
    return sa.herm(Ti @ X @ Ti)


def best(n, p, restarts, steps, seed=5):
    import twogroup as tg
    rng = np.random.default_rng(seed)
    top = (0.0, None)
    for _ in range(restarts):
        X = random_encoder(n, rng)
        decs, F = decode(X, n, p, iters=200)
        for _ in range(steps):
            X = tg.encoder_power(adjoint_states(decs, n, p), X, 2 ** n, iters=10)
            decs, F = decode(X, n, p, warm=decs, iters=30)
        decs, F = decode(X, n, p, warm=decs, iters=2000)
        if F > top[0]:
            top = (F, X)
    return top


def explicit_code_n2():
    s = 1 / math.sqrt(2)
    plus, minus = np.array([s, s]), np.array([s, -s])
    vec = np.concatenate([np.kron(plus, plus), np.kron(minus, minus)])
    return np.outer(vec, vec.conj()).astype(complex)


if __name__ == "__main__":
    if sys.argv[1] == "explicit":
        X = explicit_code_n2()
        for p in (0.0, 0.05, 0.125, 0.2, 1 / (1 + math.sqrt(2))):
            _, F = decode(X, 2, p, iters=4000)
            print(f"p = {p:.6f}: F = {F:.10f}   (7 - 6p + 2p^2)/8 = {(7 - 6 * p + 2 * p * p) / 8:.10f}")
        sys.exit(0)
    if sys.argv[1] == "scan":
        n, R, S = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        for p in (0.10, 0.15, 0.20, 0.25, 0.30, 1 / (1 + math.sqrt(8)), 1 / (1 + math.sqrt(4)), 1 / (1 + math.sqrt(2))):
            F, _ = best(n, p, R, S)
            d_needed = (1 / p - 1) ** 2
            print(f"n = {n}, p = {p:.6f} (shield d = {d_needed:.2f}): best F = {F:.10f} "
                  f"{'ABOVE 3/4' if F > 0.75 else 'below 3/4'}", flush=True)
        sys.exit(0)
    raise SystemExit(__doc__)
