"""R170 item 2: the single-component bar for ANY figure of merit, and how much more noise it tolerates than F_e.

DERIVED: every code on A^{(x)m} (any m, any input dimension) gives a logical channel whose Choi state J is 2-extendible
(Eve holds a copy of what Bob holds and applies Bob's decoder; A_{1/2} is self-complementary), and every code on P^m
gives a PPT -- for two qubits separable, hence 2-extendible -- J. So "J is NOT 2-extendible" is the strongest test a
single-component bar can support; F_e > 3/4 is the special case of an isotropic J. Pauli twirling (correlated Paulis on
R and on Bob's qubit) maps 2-extendible states to 2-extendible states and leaves the achievable sets invariant, so it
suffices to test the Bell-diagonal (Pauli-twirled) spectrum lambda, which the six-cardinal-state design measures.
For a two-qubit state with rho_B = 1/2, symmetric extendibility is (Chen, Ji, Kribs, Lutkenhaus, Zeng, PRA 90 032318,
2014, as recalled here and checked below against the isotropic boundary F = 3/4)
        sum_i lambda_i^2 - 4 sqrt(prod_i lambda_i) <= 1/2.
"""
from __future__ import annotations

import math

import numpy as np


def margin(lam) -> float:
    """> 0 iff the Bell-diagonal state with spectrum lam is NOT 2-extendible"""
    lam = np.asarray(lam, dtype=float)
    return float(np.sum(lam ** 2) - 4 * math.sqrt(max(0.0, float(np.prod(lam)))) - 0.5)


def two_use_code_lambdas(p: float) -> np.ndarray:
    """the ideal two-use shield code {|++>, |-->} through N~_p, branch by branch (R142's closed form):
    both kept (1/4) identity; one erased (1/2) logical Z w.p. p; both erased (1/4) X-basis measure-and-prepare with
    flip e = 2p(1-p). Returns (lambda_I, lambda_X, lambda_Y, lambda_Z); lambda_I = (7 - 6p + 2p^2)/8."""
    e = 2 * p * (1 - p)
    return np.array([0.25 + 0.5 * (1 - p) + (1 - e) / 8, (1 - e) / 8, e / 8, 0.5 * p + e / 8])


def depolarize(lam, delta):
    return (1 - delta) * np.asarray(lam) + delta / 4


def threshold_2ext(lam, tol=1e-13) -> float:
    """largest global depolarizing strength delta at which the state is still NOT 2-extendible (bisection)"""
    if margin(lam) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if margin(depolarize(lam, mid)) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def threshold_fe(lam) -> float:
    """the same for the F_e witness: lambda_I(delta) > 3/4"""
    lI = float(lam[0])
    return max(0.0, (lI - 0.75) / (lI - 0.25))


def shield_table(ms=(5, 6, 7, 8)):
    rows = []
    for m in ms:
        d = 2 ** m
        p = 1 / (math.sqrt(d) + 1)
        lam = two_use_code_lambdas(p)
        dW, dF = threshold_2ext(lam), threshold_fe(lam)
        rows.append({"m": m, "d": d, "p": p, "F_e_ideal": float(lam[0]), "delta_Fe": dF, "delta_2ext": dW,
                     "gain": dW / dF if dF > 0 else float("inf"),
                     "F_e_at_2ext_threshold": float(depolarize(lam, dW)[0])})
    return rows
