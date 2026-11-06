"""The learnable Pauli channel of a twirled CZ, from product-basis ladders -- exact algebra, no simulator.

A layer is a CZ followed by a Pauli channel {p_Q}. Its Pauli fidelities are f_P = sum_Q p_Q s(P, Q) (s = +1 when P
and Q commute, -1 otherwise), and a correlator P measured after d layers (d even) from a P eigenstate reads
SPAM_P (f_P f_G(P))^(d/2), G the CZ's conjugation action. So a ladder learns, SPAM-free:
  - f_P for the 3 Paulis the CZ fixes (ZI, IZ, ZZ);
  - the products f_P f_G(P) for its 6 pairs ((XI, XZ), (IX, ZX), (YI, YZ), (IY, ZY), (XX, YY), (XY, YX)).
That is 9 of the 15. The other 6 are a gauge: they move error between the gate and SPAM, and no experiment can see
them. Here they are fixed by the standard symmetric choice f_P = f_G(P). It is nearly free for the total error: the
asymmetry within a pair enters 1 - p_I only at second order (<= 0.02% on uniform, dephasing, T1-type, ZZ and random
channels at realistic rates).

Seven product bases measure all nine (each basis reads a(x)I, I(x)b and a(x)b): ZZ, XZ, ZX, YZ, ZY, XX, XY.
The estimator this replaces -- uniform + ZI/IZ + ZZ from Bell and |++> ladders -- missed the total error by up to 19%
on the same channels (it cannot represent T1-type X/Y errors or an unequal pair)."""
from __future__ import annotations

import itertools
import math

import numpy as np

LABELS = tuple("".join(t) for t in itertools.product("IXYZ", repeat=2))     # "II", "IX", ..., "ZZ"; qubit a first
NONI = LABELS[1:]
BASES = ("ZZ", "XZ", "ZX", "YZ", "ZY", "XX", "XY")


def _sym(p: str):
    return [((c in "XY"), (c in "ZY")) for c in p]


def commute(p: str, q: str) -> bool:
    return sum(x1 * z2 + z1 * x2 for (x1, z1), (x2, z2) in zip(_sym(p), _sym(q))) % 2 == 0


def cz_image(p: str) -> str:
    """the CZ's conjugation action, sign dropped: an X on one qubit picks up a Z on the other"""
    (xa, za), (xb, zb) = _sym(p)
    lab = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    return lab[(int(xa), int(za ^ xb))] + lab[(int(xb), int(zb ^ xa))]


WHT = np.array([[1.0 if commute(p, q) else -1.0 for q in LABELS] for p in LABELS])   # WHT @ WHT = 16 I
FIXED = tuple(p for p in NONI if cz_image(p) == p)
PAIRS = tuple(sorted({tuple(sorted((p, cz_image(p)))) for p in NONI if cz_image(p) != p}))


def observables(basis: str) -> tuple:
    a, b = basis[0], basis[1]
    return (a + "I", "I" + b, a + b)


def learnable_key(p: str):
    """the learnable quantity a measured correlator P decays by: P itself if fixed, else its pair"""
    return p if cz_image(p) == p else tuple(sorted((p, cz_image(p))))


def fidelities(probs: dict) -> dict:
    """{Pauli: f_P} from error probabilities {Pauli: p} over NONI (the identity takes the rest)"""
    v = np.array([0.0] + [probs.get(q, 0.0) for q in NONI])
    v[0] = 1.0 - v[1:].sum()
    return dict(zip(LABELS, WHT @ v))


def probabilities(f: dict) -> dict:
    """the inverse: {Pauli: p} including "II", from fidelities over NONI (f_II = 1)"""
    fv = np.array([1.0] + [f[p] for p in NONI])
    return dict(zip(LABELS, WHT @ fv / 16))


def learnable(f: dict) -> dict:
    """the nine quantities a ladder can see: {fixed P: f_P^2, pair: f_P f_G(P)} -- per TWO layers, the form a decay
    over an even depth measures"""
    out: dict = {p: f[p] ** 2 for p in FIXED}             # keys: a fixed Pauli (str) or a pair (tuple)
    out.update({pq: f[pq[0]] * f[pq[1]] for pq in PAIRS})
    return out


def symmetric_channel(g: dict) -> dict:
    """the channel in the symmetric gauge from the nine learnable quantities (each per two layers): f_P = g^(1/2) for
    a pair member, and f_P = g^(1/2) for a fixed P too (g = f_P^2). Probabilities, including "II"."""
    f = {}
    for p in FIXED:
        f[p] = math.sqrt(max(g[p], 0.0))
    for a, b in PAIRS:
        f[a] = f[b] = math.sqrt(max(g[(a, b)], 0.0))
    return probabilities(f)


def project_physical(p: dict) -> dict:
    """the nearest physical channel: negative probabilities (shot noise on a near-zero rate) clipped, the identity
    renormalized. Reports nothing silently -- the caller records the clipped mass."""
    q = {k: max(v, 0.0) for k, v in p.items() if k != "II"}
    q["II"] = 1.0 - sum(q.values())
    return q
