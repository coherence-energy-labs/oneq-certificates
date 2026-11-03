r"""Small exact GF(2) helpers on Python-int bit vectors (bit q = qubit q), for morph2026.py (R198).

rank / kernel / logical bases, and an EXACT minimum-weight-logical search by meet in the middle: every vector of
weight <= h is tabulated by its syndrome; a logical of weight <= 2h is u ^ v for two tabulated vectors with equal
syndrome (split any such logical's support into halves of size <= h). The search therefore decides "is there a
nontrivial logical of weight <= 2h" exactly, and returns the minimum weight found or None.
"""
from __future__ import annotations

from itertools import combinations


def rank(rows: list[int]) -> int:
    basis: dict[int, int] = {}
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h not in basis:
                basis[h] = v
                break
            v ^= basis[h]
    return len(basis)


def reduce(v: int, basis: dict[int, int]) -> int:
    while v:
        h = v.bit_length() - 1
        if h not in basis:
            return v
        v ^= basis[h]
    return 0


def span_basis(rows: list[int]) -> dict[int, int]:
    basis: dict[int, int] = {}
    for v in rows:
        v = reduce(v, basis)
        if v:
            basis[v.bit_length() - 1] = v
    return basis


def kernel(rows: list[int], n: int) -> list[int]:
    """A basis of {x in GF(2)^n : popcount(row & x) even for every row}."""
    piv: dict[int, int] = {}  # pivot column -> row, fully reduced (RREF keyed by lowest set bit)
    for v in rows:
        for c, r in piv.items():
            if v >> c & 1:
                v ^= r
        if v:
            c = (v & -v).bit_length() - 1
            for k in list(piv):
                if piv[k] >> c & 1:
                    piv[k] ^= v
            piv[c] = v
    out = []
    for f in range(n):
        if f in piv:
            continue
        x = 1 << f
        for c, r in piv.items():
            if r >> f & 1:
                x |= 1 << c
        out.append(x)
    return out


def logical_basis(h_same: list[int], h_other: list[int], n: int) -> list[int]:
    """Logical operators of h_same's type: ker(h_other) modulo rowspace(h_same)."""
    basis = span_basis(h_same)
    out = []
    for v in kernel(h_other, n):
        w = reduce(v, basis)
        if w:
            basis[w.bit_length() - 1] = w
            out.append(v)
    return out


def min_logical(h_same: list[int], h_other: list[int], n: int, half: int) -> int | None:
    """Minimum weight of a nontrivial logical of h_same's type (commutes with h_other, not in rowspace(h_same)),
    if it is <= 2 * half; else None. Exact (meet in the middle over syndromes of h_other)."""
    other_logicals = logical_basis(h_other, h_same, n)   # the conjugate logicals: a logical is nontrivial iff it
    col = [0] * n                                         # anticommutes with one of them
    for i, r in enumerate(h_other):
        for q in range(n):
            if r >> q & 1:
                col[q] |= 1 << i
    table: dict[int, list[int]] = {}
    for w in range(half + 1):
        for sub in combinations(range(n), w):
            s, v = 0, 0
            for q in sub:
                s ^= col[q]
                v |= 1 << q
            table.setdefault(s, []).append(v)
    best = None
    for vecs in table.values():
        if len(vecs) < 2 and vecs[0] != 0:
            continue
        for i in range(len(vecs)):
            for j in range(i, len(vecs)):
                x = vecs[i] ^ vecs[j]
                if not x:
                    continue
                wt = x.bit_count()
                if best is not None and wt >= best:
                    continue
                if any((x & z).bit_count() & 1 for z in other_logicals):
                    best = wt
    return best
