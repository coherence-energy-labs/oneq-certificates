"""R225 -- the antidegradability witness the Helios run's mixture test uses: its exact bound, and its betting
constants on a package.

THE BOUND (DERIVED; certified in exact arithmetic). For w >= 0 and EVERY antidegradable qubit channel,
    w . diag(T) <= lambda_max(M_w),   M_w = (W_BA (x) I_B' + W_B'A (x) I_B) / 2,   W = sum_i w_i s_i (x) s_i^T,
T the Pauli transfer matrix. An AD channel's Choi state J is the BA marginal of a state rho on (B, B', A) symmetric
under B <-> B', so Tr(W J) = Tr(M_w rho) <= lambda_max(M_w); and Tr((s (x) s^T) J) = T_ss = 2 (the mean fidelity of
s's two eigenstates, read in s's basis) - 1 for any qubit channel. M_w is a real 8 x 8 matrix (Y (x) Y^T is real), so
a rational hbar with hbar I - M_w positive definite -- shown by an exact LDL^T -- is a PROVEN upper bound.
w = (1, 1, 1) gives 2: the cloning bound F_e <= 3/4.

THE TEST ON A PACKAGE. The design's per-basis diagonal is lam_b = lam_b(ideal) - 6 sum_g width_g W_{g,b} e_{g,b}
(errors of low units only raise it, so with w >= 0 dropping them is conservative). A shot labelled g (probability
pi_g), its unit of input basis b (probability W_{g,b} / W_g under the design law), scores kappa_{g,b} X with
    kappa_{g,b} = 6 w_b width_g W_g / pi_g,      and the null is  mean E[S] >= m_w = w . lam(ideal) - hbar.
kappa is computed in exact arithmetic from the package's own W_g (a double the freeze rounded UP from the exact law)
and the preregistered pi (as the rational it names), then rounded UP; m_w is rounded DOWN. Both directions are the
conservative ones, and hbar's certificate slack (1e-12) sits below the AD bound by far more than any double's ulp.
w = (1, 1, 1) / 6 is R203's F_e test.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

BASES = "XYZ"
CATS = ("ee_x/high|X", "kept/high|X", "kept/high|Y", "kept/high|Z")
WIDTH = {"ee_x/high": Fraction(1), "kept/high": Fraction(2)}


class WitnessRefused(ValueError):
    pass


def _pauli_terms() -> list:
    """for X, Y, Z: (s (x) I (x) s^T + I (x) s (x) s^T) / 2 on (B, B', A), as exact integer/2 matrices"""
    i2 = np.eye(2, dtype=complex)
    sig = (np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]).astype(complex))
    out = []
    for s in sig:
        t = np.kron(np.kron(s, i2), s.T) + np.kron(np.kron(i2, s), s.T)
        if np.abs(t.imag).max() > 0 or np.abs(t.real - np.round(t.real)).max() > 0:
            raise WitnessRefused("a Pauli term of M_w is not an integer matrix")
        out.append([[Fraction(int(round(x)), 2) for x in row] for row in t.real])
    return out


TERMS = _pauli_terms()


def m_matrix(w) -> list:
    """M_w exactly: w is three Fractions (or ints)"""
    w = [Fraction(x) for x in w]
    return [[sum(w[k] * TERMS[k][i][j] for k in range(3)) for j in range(8)] for i in range(8)]


def lambda_max(w) -> float:
    return float(np.linalg.eigvalsh(np.array([[float(x) for x in row] for row in m_matrix(w)])).max())


def positive_definite(a: list) -> bool:
    """exact LDL^T: True iff the symmetric rational matrix a is positive definite (every pivot > 0)"""
    n = len(a)
    L = [[Fraction(0)] * n for _ in range(n)]
    d = [Fraction(0)] * n
    for j in range(n):
        d[j] = a[j][j] - sum(L[j][k] ** 2 * d[k] for k in range(j))
        if d[j] <= 0:
            return False
        for i in range(j + 1, n):
            L[i][j] = (a[i][j] - sum(L[i][k] * L[j][k] * d[k] for k in range(j))) / d[j]
    return True


def is_bound(w, hbar) -> bool:
    """True iff hbar I - M_w is positive definite: hbar is then a proven upper bound on w . diag(T) over AD"""
    m = m_matrix(w)
    h = Fraction(hbar)
    return positive_definite([[(h if i == j else 0) - m[i][j] for j in range(8)] for i in range(8)])


def certify_bound(w, slack: Fraction = Fraction(1, 10**12)) -> Fraction:
    """a rational hbar >= lambda_max(M_w), PROVEN by hbar I - M_w positive definite in exact arithmetic"""
    w = [Fraction(x) for x in w]
    if min(w) < 0:
        raise WitnessRefused(f"w {w} has a negative entry: dropping low units is conservative only for w >= 0")
    hbar = Fraction(math.ceil(lambda_max(w) * 10**12), 10**12) + slack
    if not is_bound(w, hbar):
        raise WitnessRefused(f"hbar {hbar} is not a certified bound for w {w}")
    return hbar


def lam_ideal(m: int = 8) -> tuple:
    """the design channel's per-basis diagonal, closed form (R213): both kept 1/4 (identity), one erased 1/2 (Z flip
    with p), both erased 1/4 (X-basis readout wrong with q = 2p(1 - p))"""
    root = math.isqrt(2 ** m)
    if root * root != 2 ** m:
        raise WitnessRefused(f"m = {m} is odd: p_flip is irrational")
    p = Fraction(1, root + 1)
    q = 2 * p * (1 - p)
    quarter, half = Fraction(1, 4), Fraction(1, 2)
    return (quarter + half * (1 - 2 * p) + quarter * (1 - 2 * q), quarter + half * (1 - 2 * p), 3 * quarter)


def round_toward(x: Fraction, up: bool) -> float:
    f = float(x)
    if up and Fraction(f) < x:
        return math.nextafter(f, math.inf)
    if not up and Fraction(f) > x:
        return math.nextafter(f, -math.inf)
    return f


def constants(w, hbar, group_w: dict, pi_rare, m: int = 8) -> dict:
    """the witness test's betting constants: kappa per category (rounded UP) and m_w (rounded DOWN).
    group_w: {group: W_g} (Fractions or doubles -- a double is taken at its exact value); pi_rare: the rare label's
    probability as the rational it names"""
    w = [Fraction(x) for x in w]
    hbar = Fraction(hbar)
    if min(w) < 0:
        raise WitnessRefused(f"w {w} has a negative entry")
    if not is_bound(w, hbar):
        raise WitnessRefused(f"hbar {hbar} is not a certified antidegradability bound for w {w}")
    pi_rare = Fraction(pi_rare)
    pi = {"ee_x/high": pi_rare, "kept/high": 1 - pi_rare}
    gw = {g: Fraction(group_w[g]) for g in WIDTH}
    m_w = sum(wi * li for wi, li in zip(w, lam_ideal(m))) - hbar
    if m_w <= 0:
        raise WitnessRefused(f"w {w}: the ideal design does not clear its own bound (margin {float(m_w)})")
    kappa = {}
    for cat in CATS:
        g, b = cat.split("|")
        kappa[cat] = round_toward(6 * w[BASES.index(b)] * WIDTH[g] * gw[g] / pi[g], up=True)
    return dict(m=round_toward(m_w, up=False), m_exact=m_w, kappa=kappa)
