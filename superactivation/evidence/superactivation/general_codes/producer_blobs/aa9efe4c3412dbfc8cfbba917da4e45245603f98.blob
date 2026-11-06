r"""A rigorous certificate that a symmetric-subspace code superactivates N~ at blocklength n.

THE CLAIM IT CHECKS. Given a witness -- an encoder Choi X on R (x) Sym^n and adjoint-decoder Chois
D_{k,b} on R (x) Sym^{n-k} -- it proves a lower bound L on the entanglement fidelity of an EXACTLY
CPTP encoder/decoder pair built from them, for N~^{(x)n} (oneq.superactivation). When L > 3/4,
that is finite-blocklength superactivation of quantum capacity, given two theorems imported from
arXiv 2604.27042 and NOT re-proved here: F_c((P (x) A)^{(x)n}, 2) >= F_c(N~^{(x)n}, 2) (their
Prop. 1, data processing through the effective-channel protocol), and F_c <= 3/4 for each of the
PPT and 50%-erasure channels alone at every n (their Result 1).

THE PROOF, step by step -- no floating-point operation carries it:
 1. Every witness entry is rounded to the dyadic grid 2^-96 and shifted by eta I (eta dyadic).
    This DEFINES the certified maps; it is not an approximation inside the proof.
 2. Complete positivity, EXACTLY: each block is proved positive definite by Sylvester's criterion
    on its real 2d x 2d form, every leading principal minor an exact integer determinant.
 3. Normalization without square roots: T = Tr_Sym X and S_{k,b} = Tr_R D_{k,b} are exact
    rationals; a Gershgorin bound lambda_max <= 1 + e (exact) makes X/(1+e_E) and D/(1+e_{k,b})
    trace non-increasing; completing each to a CPTP map adds a completely positive term, whose
    fidelity contribution is >= 0. So F(exact CPTP pair) >= sum ... / ((1+e_E)(1+e_{k,b})).
 4. That sum is evaluated in 256-bit ball arithmetic (python-flint acb), with p = sqrt(2) - 1 and
    every sqrt(C(n-k,i)/C(n,i+a)) as rigorous balls; L is the lower end of the final ball.

The checker never calls the solver that produced the witness and never trusts its numbers: a
witness that is wrong in any way either fails a positivity proof (refused) or certifies less.
Requires python-flint (the `certify` extra).
"""
from __future__ import annotations

import math
import pathlib
import time
from typing import Any

import numpy as np

GRID_BITS = 96
PRECISION_BITS = 256
THRESHOLD_NUM, THRESHOLD_DEN = 3, 4


class CertificateRefusal(ValueError):
    """The witness does not define CPTP maps the proof can stand on."""


def _flint():
    try:
        import flint
    except ImportError as e:   # pragma: no cover - exercised only without the extra
        raise CertificateRefusal("python-flint is required (pip install oneq[certify])") from e
    flint.ctx.prec = PRECISION_BITS
    return flint


def _q(flint, x: float):
    return flint.fmpq(int(round(float(x) * 2.0 ** GRID_BITS)), 2 ** GRID_BITS)


def exact_block(flint, Z: np.ndarray, eta: float):
    """(re, im) exact rational Hermitian parts of round(herm Z) + eta I."""
    if not np.all(np.isfinite(Z)):
        raise CertificateRefusal("non-finite witness entry")
    H = (Z + Z.conj().T) / 2
    m = H.shape[0]
    re = [[_q(flint, H[i, j].real) for j in range(m)] for i in range(m)]
    im = [[_q(flint, H[i, j].imag) for j in range(m)] for i in range(m)]
    e = _q(flint, eta)
    for i in range(m):
        for j in range(i + 1, m):
            re[j][i] = re[i][j]
            im[j][i] = -im[i][j]
        im[i][i] = flint.fmpq(0)
        re[i][i] = re[i][i] + e
    return re, im


def positive_definite(flint, re, im) -> int:
    """-1 if proved positive definite; else the index of the first non-positive leading minor."""
    m = len(re)
    order = [x for i in range(m) for x in (i, i + m)]

    def big(a, b):
        ia, ib = a % m, b % m
        if a < m and b < m:
            return re[ia][ib]
        if a < m:
            return -im[ia][ib]
        if b < m:
            return im[ia][ib]
        return re[ia][ib]
    B = [[big(order[a], order[b]) for b in range(2 * m)] for a in range(2 * m)]
    den = 1
    for row in B:
        for v in row:
            den = math.lcm(den, int(v.q))
    Z = [[int(v.p) * (den // int(v.q)) for v in row] for row in B]
    for s in range(1, 2 * m + 1):
        if flint.fmpz_mat([r[:s] for r in Z[:s]]).det() <= 0:
            return s
    return -1


def gershgorin_excess(flint, re, im):
    """Exact e >= 0 with lambda_max <= 1 + e (|z| <= |Re z| + |Im z|)."""
    m = len(re)
    best = None
    for i in range(m):
        r = re[i][i]
        for j in range(m):
            if j != i:
                r += abs(re[i][j]) + abs(im[i][j])
        best = r if best is None or r > best else best
    e = best - 1
    return e if e > 0 else flint.fmpq(0)


def partial_trace_exact(flint, re, im, d1: int, d2: int, trace_first: bool):
    out = d2 if trace_first else d1

    def idx(i, j):
        return ([(r * d2 + i, r * d2 + j) for r in range(d1)] if trace_first
                else [(i * d2 + s, j * d2 + s) for s in range(d2)])
    z = flint.fmpq(0)
    ore = [[sum((re[a][b] for a, b in idx(i, j)), z) for j in range(out)] for i in range(out)]
    oim = [[sum((im[a][b] for a, b in idx(i, j)), z) for j in range(out)] for i in range(out)]
    return ore, oim


def _acb(flint, re, im):
    m = len(re)
    return flint.acb_mat([[flint.acb(flint.arb(re[i][j]), flint.arb(im[i][j])) for j in range(m)]
                          for i in range(m)])


def _selection(flint, n: int, k: int, a: int):
    d = n - k + 1
    rows = []
    for r in range(2):
        for i in range(d):
            row = [flint.acb(0)] * (2 * (n + 1))
            row[r * (n + 1) + i + a] = flint.acb(
                flint.arb(flint.fmpq(math.comb(n - k, i), math.comb(n, i + a))).sqrt())
            rows.append(row)
    return flint.acb_mat(rows)


def _coeff(flint, k: int, b: int, a: int, p):
    tot = flint.arb(0)
    for f1 in range(a + 1):
        f0 = b - a + f1
        if 0 <= f0 <= k - a:
            tot += (math.comb(a, f1) * math.comb(k - a, f0)) * p ** (f1 + f0) * (1 - p) ** (k - f1 - f0)
    return tot * math.comb(k, a)


def certify(n: int, X: np.ndarray, decoders: dict, eta: float = 2.0 ** -34, log=None) -> dict:
    """Rigorous lower bound on the fidelity of the exactly-CPTP pair derived from the witness."""
    flint = _flint()
    t0 = time.time()
    m = n + 1
    if X.shape != (2 * m, 2 * m):
        raise CertificateRefusal(f"encoder must be {2 * m} square for n={n}")
    reX, imX = exact_block(flint, X, eta)
    bad = positive_definite(flint, reX, imX)
    if bad != -1:
        raise CertificateRefusal(f"encoder not proved positive definite (leading minor {bad})")
    eE = gershgorin_excess(flint, *partial_trace_exact(flint, reX, imX, 2, m, trace_first=False))
    Xb = _acb(flint, reX, imX)
    p = flint.arb(2).sqrt() - 1
    total = flint.arb(0)
    worst = flint.fmpq(0)
    proved = 0
    for k in range(n + 1):
        d = n - k + 1
        Dk = decoders[k]
        if len(Dk) != k + 1:
            raise CertificateRefusal(f"branch k={k} needs {k + 1} decoders, got {len(Dk)}")
        sel = [_selection(flint, n, k, a) for a in range(k + 1)]
        Fk = flint.arb(0)
        for b in range(k + 1):
            if Dk[b].shape != (2 * d, 2 * d):
                raise CertificateRefusal(f"decoder (k={k}, b={b}) must be {2 * d} square")
            reD, imD = exact_block(flint, Dk[b], eta)
            bad = positive_definite(flint, reD, imD)
            if bad != -1:
                raise CertificateRefusal(f"decoder (k={k}, b={b}) not proved positive definite (minor {bad})")
            eD = gershgorin_excess(flint, *partial_trace_exact(flint, reD, imD, 2, d, trace_first=True))
            worst = max(worst, eD)
            Db = _acb(flint, reD, imD)
            val = flint.acb(0)
            for a in range(k + 1):
                c = _coeff(flint, k, b, a, p)
                val += c * (sel[a] * Xb * sel[a].transpose() * Db).trace()
            Fk += val.real / (4 * (1 + flint.arb(eD)))
            proved += 1
        total += flint.arb(flint.fmpq(math.comb(n, k), 2 ** n)) * Fk
        if log:
            log(f"  k={k:2d} ({time.time() - t0:.0f}s)")
    F = total / (1 + flint.arb(eE))
    lower = F.lower()
    third = flint.arb(flint.fmpq(THRESHOLD_NUM, THRESHOLD_DEN))
    return {
        "n": n,
        "certified_lower_bound": lower.str(40),
        "certified_lower_bound_float": float(lower),
        "exceeds_three_quarters": bool(lower > third),
        "certified_margin_lower_bound": float(lower - third),
        "fidelity_ball": F.str(30),
        "encoder_trace_excess": float(eE),
        "max_decoder_trace_excess": float(worst),
        "blocks_proved_positive_definite": proved + 1,
        "grid_bits": GRID_BITS, "precision_bits": PRECISION_BITS, "eta": eta,
        "seconds": round(time.time() - t0, 1),
        "imported_theorems": [
            "arXiv 2604.27042 Prop. 1: F_c((P x A)^n, 2) >= F_c(N~^n, 2)",
            "arXiv 2604.27042 Result 1: F_c(P^n, 2) <= 1/2 and F_c(A^n, 2) <= 3/4 for every n",
        ],
    }


def save_witness(path: pathlib.Path, n: int, X: np.ndarray, decoders: dict) -> None:
    # typed Any: numpy's stub reads a **dict[str, ndarray] as the bool `allow_pickle` keyword
    arrays: dict[str, Any] = {"n": np.array(n), "encoder": X}
    for k, Dk in decoders.items():
        arrays[f"decoders_k{k}"] = np.stack(Dk)
    np.savez_compressed(path, **arrays)


def load_witness(path: pathlib.Path):
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n"])
        X = np.array(z["encoder"])
        decs = {k: list(np.array(z[f"decoders_k{k}"])) for k in range(n + 1)}
    return n, X, decs
