#!/usr/bin/env python3
"""Second branch-dual tree checker -- separately implemented from the schema.

Written from the WRITTEN SPECIFICATION of the branch-dual receipt (paper
SS2 tier BD + the serialized-tree schema of qldpc_heldout_addendum.json),
without importing or consulting the production checker (qldpc_check.py).
Shares no code, no sign-convention derivation, and no data structures
with it; arithmetic is fractions over Python ints only.

Serialized tree schema (JSON):
  ["branch", k, child_for_xk_eq_0, child_for_xk_eq_1]
  ["bound",  [[rows, F, num, den], ...]]     # facet duals on the REDUCED
                                             # instance at this node
  ["infeasible", rows]                       # single row index or list:
                                             # reduced supports XOR to
                                             # empty with odd parity

Instance document (JSON):
  {"checks": [[i,...],...], "weights": [int,...], "syndrome": [0/1,...],
   "candidate": [i,...], "tree": <tree>}

Acceptance := candidate feasible AND every root-to-leaf path closes:
each leaf either exhibits a parity-infeasible row combination of its
reduced instance, or carries facet duals whose safe-dual bound L
satisfies W_fixed1 + L > U - 1/D (D = lcm of weight denominators; here
weights are ints so D = 1). Exit 0 accept, 2 refuse, 1 malformed.

Input domain: docs/QLDPC_RECEIPT_INPUT_DOMAIN.md, the contract all seven
qLDPC checkers implement (vectors: conformance/qldpc_input_domain_vectors.json).
Every integer slot holds a JSON integer literal; syndrome entries are exactly
0 or 1; a combined row list has at least two distinct rows; multipliers are in
lowest terms with den >= 1; a root bound above U is refused.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from math import gcd


def refuse(msg):
    print(f"REFUSE: {msg}")
    raise SystemExit(2)


def _exact_int(v, what: str) -> int:
    """An integer, or a refusal. NEVER a silent truncation.

    EXTERNAL AUDIT 2026-08-04, finding 6 (CONFIRMED and fixed here).
    This file parsed untrusted evidence with bare `int()`, which is a
    LOSSY COERCION dressed as a cast: `int(1.9)` is 1, so a weight vector
    of [1.9, 1.0] became [1, 1] and a genuinely suboptimal candidate
    certified against weights nobody supplied. Booleans coerce too, and
    numeric strings.

    The mathematics in this file was sound. The failure was entirely at
    the untrusted-input boundary -- which is the only boundary an
    independent checker has, and therefore the one place it cannot afford
    to be relaxed.

    INPUT-DOMAIN CONTRACT 2026-11-03 (docs/QLDPC_RECEIPT_INPUT_DOMAIN.md):
    an integer slot holds a JSON integer literal and nothing else. A float
    is refused even when integral: the JavaScript checkers cannot tell
    `1.0` from `1` without reading the token, so the one policy all seven
    implementations can enforce is "the token is an integer literal", and
    json.loads hands this file a float for every other token.
    """
    if isinstance(v, bool):
        refuse(f"{what} is a boolean, not an integer: {v!r}")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        refuse(f"{what} is not an integer literal: {v!r}")
    refuse(f"{what} is not a number: {v!r}")


def _rows(spec, m: int, what: str) -> list[int]:
    """A single row (an integer), or a GF(2) combination: a list of at least
    two distinct rows. A one-element list is refused, because a single row
    is written as an integer (the contract, section 3)."""
    if isinstance(spec, list):
        rows = [_exact_int(r, f"{what} row") for r in spec]
        if len(rows) < 2:
            refuse(f"{what} combines fewer than two rows: {spec!r}")
        if len(set(rows)) != len(rows):
            refuse(f"duplicate row in {what} combination")
    else:
        rows = [_exact_int(spec, f"{what} row")]
    for r in rows:
        if not (0 <= r < m):
            refuse(f"{what} row out of range: {r}")
    return rows


def _index(v, n: int, what: str) -> int:
    """A VALID index in [0, n). Python's negative indexing is a gift to
    a forger: index -1 silently means 'the last element', so an
    out-of-range value that should be refused instead selects real data.
    """
    i = _exact_int(v, what)
    if not (0 <= i < n):
        refuse(f"{what} out of range: {i} not in [0, {n})")
    return i


def main(path: str) -> int:
    doc = json.loads(open(path, encoding="utf-8").read())
    if not isinstance(doc.get("weights"), list):
        refuse("weights is not a list")
    nvar = len(doc["weights"])
    checks = [tuple(_index(i, nvar, f"check {j} index") for i in c)
              for j, c in enumerate(doc["checks"])]
    for j, c in enumerate(checks):
        if len(set(c)) != len(c):
            refuse(f"duplicate index in check {j}")
    w = [_exact_int(v, f"weight[{i}]") for i, v in
         enumerate(doc["weights"])]
    if any(v < 0 for v in w):
        refuse("negative weight")
    # A GF(2) entry is exactly 0 or 1. This line read `_exact_int(b) & 1`
    # until 2026-11-03, so a syndrome entry 2 was read as 0 and -1 as 1 and
    # the malformed instance certified (external audit 2, finding F05).
    syn = [_exact_int(b, f"syndrome[{i}]") for i, b in
           enumerate(doc["syndrome"])]
    for i, b in enumerate(syn):
        if b not in (0, 1):
            refuse(f"syndrome[{i}] = {b} is not 0 or 1")
    if len(syn) != len(checks):
        refuse("syndrome length mismatch")
    cand = sorted(_index(i, nvar, "candidate index")
                  for i in doc["candidate"])
    cs = set(cand)
    if len(cs) != len(cand):
        refuse("duplicate candidate index")
    for j, c in enumerate(checks):
        if sum(1 for i in c if i in cs) % 2 != syn[j]:
            refuse(f"candidate violates check {j}")
    U = sum(w[i] for i in cand)
    step = Fraction(1, 1)               # integer weights

    def reduced(j, fix):                # (support tuple, parity) at node
        sup = tuple(i for i in checks[j] if i not in fix)
        par = (syn[j] ^ (sum(1 for i in checks[j]
                             if fix.get(i) == 1) & 1)) & 1
        return sup, par

    def leaf_bound(duals, fix):
        load: dict[int, Fraction] = {}
        by = Fraction(0)
        seen = set()
        for entry in duals:
            if not (isinstance(entry, list) and len(entry) == 4):
                refuse("malformed dual entry")
            rows, F, num, den = entry
            num = _exact_int(num, "multiplier numerator")
            den = _exact_int(den, "multiplier denominator")
            if den <= 0:
                refuse("non-integer rational: denominator must be >= 1")
            if gcd(abs(num), den) != 1:
                refuse(f"multiplier {num}/{den} is not in lowest terms")
            y = Fraction(num, den)
            if y < 0:
                refuse("negative multiplier")
            # a zero multiplier is skipped only AFTER the facet is validated
            # (final referee, 2026-11-03): every listed facet is checked
            rlist = _rows(rows, len(checks), "facet")
            sup: set = set()
            par = 0
            for r in rlist:
                s2, p2 = reduced(r, fix)
                sup ^= set(s2)
                par ^= p2
            if not isinstance(F, list):
                refuse("F is not a list")
            Fs = set(_index(i, nvar, "F index") for i in F)
            if len(Fs) != len(F):
                refuse("duplicate F index")
            if not Fs <= sup:
                refuse("F escapes reduced support")
            if len(Fs) % 2 == par:
                refuse("F has satisfiable parity: not a facet")
            key = (tuple(sorted(rlist)), tuple(sorted(Fs)))
            if key in seen:
                refuse("duplicate facet")
            seen.add(key)
            if y == 0:                      # validated above; adds nothing
                continue
            for i in sup:
                load[i] = load.get(i, Fraction(0)) + \
                    (-y if i in Fs else y)
            by += y * (1 - len(Fs))
        z = Fraction(0)
        for i, t in load.items():
            excess = t - w[i]
            if excess > 0:
                z += excess
        return by - z

    budget = [200000]

    def walk(node, fix):
        budget[0] -= 1
        if budget[0] < 0:
            refuse("node budget")
        if not (isinstance(node, list) and node):
            refuse("malformed node")
        kind = node[0]
        if kind == "branch":
            if len(node) != 4:
                refuse("malformed branch")
            k = _exact_int(node[1], "branch variable")
            if k in fix or not (0 <= k < len(w)):
                refuse(f"bad branch variable {k}")
            walk(node[2], {**fix, k: 0})
            walk(node[3], {**fix, k: 1})
            return
        if kind == "infeasible":
            if len(node) != 2:
                refuse("malformed infeasible leaf")
            rlist = _rows(node[1], len(checks), "infeasible leaf")
            sup: set = set()
            par = 0
            for r in rlist:
                s2, p2 = reduced(r, fix)
                sup ^= set(s2)
                par ^= p2
            if sup:
                refuse("claimed-infeasible combo has support")
            if par == 0:
                refuse("combo is 0=0: feasible")
            return
        if kind == "bound":
            if len(node) != 2:
                refuse("malformed bound leaf")
            W1 = sum(w[i] for i, a in fix.items() if a == 1)
            if not isinstance(node[1], list):
                refuse("bound leaf duals are not a list")
            L = leaf_bound(node[1], fix)
            # THE ROOT GUARD (2026-11-03). With nothing fixed the leaf bound
            # is a flat bound on the whole instance, and L <= OPT <= U for a
            # valid dual and a verified-feasible candidate: L > U is a broken
            # bound, refused as an instrument failure -- as the flat checkers
            # and verify-qldpc-tree.mjs (depth 0) refuse it.
            if not fix and L > U:
                refuse(f"VERIFIER FAILURE: root bound {L} exceeds the weight "
                       f"{U} of a valid correction")
            if not (W1 + L > U - step):
                refuse(f"leaf bound {W1}+{L} fails vs {U}-{step}")
            return
        refuse(f"unknown node kind {kind}")

    try:
        walk(doc["tree"], {})
    except RecursionError:
        refuse("recursion limit")
    print(f"ACCEPT: candidate weight {U} proven minimum by branch duals")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: tree_checker_b.py <document.json>")
        raise SystemExit(1)
    try:
        raise SystemExit(main(sys.argv[1]))
    except SystemExit:
        raise
    except Exception as e:                          # malformed input
        print(f"MALFORMED: {type(e).__name__}: {e}")
        raise SystemExit(1)
